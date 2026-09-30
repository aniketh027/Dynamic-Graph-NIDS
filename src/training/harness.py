"""Stage E.7 production training harness for the frozen HTGNN classifier.

The harness owns optimization, validation, early stopping, checkpointing, and
bounded batch traversal.  It deliberately does not own data preprocessing,
graph construction, model architecture, test evaluation, pruning, or
hyperparameter search.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Final, Iterable, Iterator, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score
import torch
from torch import nn
from torch.optim import AdamW, Optimizer
from torch_geometric.data import HeteroData

from src.graph import adapter as d4
from src.graph import batching as d6
from src.graph import htgnn_classifier as e5
from src.graph import pyg_adapter as e2
from src.graph import resource_pilot as e6
from src.preprocessing.labels import LABEL_TO_INT
from src.preprocessing import splitting as stage_c_splitting
from src.training import baseline_integration
from src.training import baselines


HARNESS_VERSION: Final = "stage-e7-htgnn-training-harness-v2"
CHECKPOINT_SCHEMA_VERSION: Final = 2
LEGACY_HTGNN_CHECKPOINT_SCHEMA_VERSION: Final = 1
TASKS: Final = ("E1", "E2", "E3")
APPROVED_SEEDS: Final = (2025, 2026, 2027)
LEARNING_RATE_CANDIDATES: Final = (1e-4, 3e-4, 1e-3)
WEIGHT_DECAY: Final = 1e-4
MAX_EPOCHS: Final = 30
MIN_EPOCHS: Final = 3
PATIENCE: Final = 5
MIN_DELTA: Final = 1e-4
DEFAULT_BATCH_SIZE: Final = 32
MAX_BATCH_SIZE: Final = 256
EXPECTED_PARAMETERS: Final = {"E1": 7_183, "E2": 7_183, "E3": 6_721}
HTGNN_MODEL_FAMILY: Final = "htgnn"
MODEL_FAMILIES: Final = (HTGNN_MODEL_FAMILY, *baselines.BASELINE_KINDS)
MODEL_INPUT_CONTRACTS: Final = {
    HTGNN_MODEL_FAMILY: "frozen_e2_heterodata_78_feature_8_category_1_sample",
    baselines.FINAL_FLOW: "frozen_e2_window_float32_Bx5x78_final_position_to_Bx78",
    baselines.FLATTENED_FIVE_FLOW: (
        "frozen_e2_window_float32_Bx5x78_temporal_major_to_Bx390"
    ),
}
SELECTION_NAMES: Final = {
    "E1": "support_restricted_macro_f1",
    "E2": "validation_selection_macro_f1",
    "E3": "pr_auc_average_precision",
}


class TrainingContractError(RuntimeError):
    """Raised when an E.7 safety or training contract is violated."""


@dataclass(frozen=True, slots=True)
class ModelContract:
    """Identity derived from an actual supported production model instance."""

    family: str
    architecture_version: str
    input_contract: str
    task: str
    parameter_count: int


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """One explicit, reproducible E.7 run configuration."""

    task: str
    learning_rate: float
    seed: int
    batch_size: int = DEFAULT_BATCH_SIZE
    max_epochs: int = MAX_EPOCHS
    max_train_batches: int | None = None
    max_validation_batches: int | None = None

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValueError("task must be exactly one of E1, E2, or E3")
        if self.seed not in APPROVED_SEEDS:
            raise ValueError(f"seed must be one of {APPROVED_SEEDS}")
        if self.learning_rate not in LEARNING_RATE_CANDIDATES:
            raise ValueError(
                f"learning_rate must be one of {LEARNING_RATE_CANDIDATES}"
            )
        _require_int_between("batch_size", self.batch_size, 1, MAX_BATCH_SIZE)
        _require_int_between("max_epochs", self.max_epochs, 1, MAX_EPOCHS)
        _require_optional_positive_int("max_train_batches", self.max_train_batches)
        _require_optional_positive_int(
            "max_validation_batches", self.max_validation_batches
        )

    def checkpoint_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "weight_decay": WEIGHT_DECAY,
            "minimum_epochs": MIN_EPOCHS,
            "patience": PATIENCE,
            "min_delta": MIN_DELTA,
            "optimizer": "AdamW",
            "scheduler": None,
            "weighted_loss": False,
            "train_order": "seeded bounded affine permutation per epoch",
            "validation_order": "sequential",
        }


@dataclass(frozen=True, slots=True)
class EpochStats:
    """Compact accounting for one train or validation pass."""

    loss: float
    batches: int
    examples: int
    order_sha256: str
    target_codes: tuple[int, ...]
    selection_value: float | None = None
    selection_name: str | None = None
    selection_support_complete: bool | None = None
    selection_class_codes: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Summary returned after a completed or early-stopped run."""

    task: str
    checkpoint_path: str
    epochs_completed: int
    stopped_early: bool
    best_epoch: int
    best_selection_value: float
    selection_name: str
    history: tuple[Mapping[str, Any], ...]
    parameter_update_observed: bool
    device: str
    model_family: str


@dataclass(slots=True)
class EarlyStopping:
    """Validation-only maximization with the approved E.7 thresholds."""

    minimum_epochs: int = MIN_EPOCHS
    patience: int = PATIENCE
    min_delta: float = MIN_DELTA
    best_value: float = -math.inf
    best_epoch: int = 0
    epochs_without_improvement: int = 0

    def __post_init__(self) -> None:
        _require_int_between("minimum_epochs", self.minimum_epochs, 1, MAX_EPOCHS)
        _require_int_between("patience", self.patience, 1, MAX_EPOCHS)
        if not math.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError("min_delta must be finite and nonnegative")

    def update(self, epoch: int, value: float) -> tuple[bool, bool]:
        """Return ``(is_new_best, should_stop)`` for one validation value."""

        _require_int_between("epoch", epoch, 1, MAX_EPOCHS)
        if not math.isfinite(value):
            raise TrainingContractError("validation selection value is nonfinite")
        improved = value > self.best_value + self.min_delta
        if improved:
            self.best_value = value
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        should_stop = (
            epoch >= self.minimum_epochs
            and self.epochs_without_improvement >= self.patience
        )
        return improved, should_stop


def _require_int_between(name: str, value: object, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")


def _require_optional_positive_int(name: str, value: object) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer or None")
    if value <= 0:
        raise ValueError(f"{name} must be positive when provided")


def seed_everything(seed: int) -> dict[str, Any]:
    """Seed Python, NumPy, Torch CPU, and every visible CUDA device.

    CUDA is best-effort reproducible: PyG scatter/aggregation kernels may remain
    nondeterministic even with fixed seeds and deterministic cuDNN settings.
    """

    if seed not in APPROVED_SEEDS:
        raise ValueError(f"seed must be one of {APPROVED_SEEDS}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return {
        "seed": seed,
        "python_random": True,
        "numpy": True,
        "torch_cpu": True,
        "torch_cuda": cuda_available,
        "cuda_determinism": (
            "best effort only; fixed seeds and deterministic cuDNN settings do "
            "not guarantee bitwise determinism for PyG/CUDA scatter operations"
        ),
    }


def resolve_device(requested: str) -> torch.device:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of auto, cpu, or cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise TrainingContractError("CUDA was requested but is not available")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def create_model(
    task: str, model_family: str = HTGNN_MODEL_FAMILY
) -> nn.Module:
    if task not in TASKS:
        raise ValueError("task must be exactly one of E1, E2, or E3")
    if model_family == HTGNN_MODEL_FAMILY:
        model: nn.Module = e5.HTGNNClassifier(task)
    elif model_family in baselines.BASELINE_KINDS:
        dense = baselines.create_baseline(model_family, task)
        model = baseline_integration.BaselineTrainingAdapter(
            dense, kind=model_family, task=task
        )
    else:
        raise ValueError(f"model_family must be one of {MODEL_FAMILIES}")
    model_contract(model)
    return model


def model_contract(model: nn.Module) -> ModelContract:
    """Derive model identity from the instance and reject relabelled models."""

    if isinstance(model, e5.HTGNNClassifier):
        task = model.task
        family = HTGNN_MODEL_FAMILY
        architecture_version = e5.MODEL_VERSION
        expected = EXPECTED_PARAMETERS[task]
    elif type(model) is baseline_integration.BaselineTrainingAdapter:
        try:
            family = baselines.model_family(model.model)
        except (TypeError, ValueError) as exc:
            raise TrainingContractError(f"invalid baseline model identity: {exc}") from exc
        task = model.model.task
        if model.kind != family or model.task != task:
            raise TrainingContractError(
                "baseline adapter identity does not match its actual model"
            )
        architecture_version = baselines.BASELINE_VERSION
        expected = baselines.EXPECTED_PARAMETER_COUNTS[family][task]
    else:
        raise TrainingContractError("unsupported production model class")
    if task not in TASKS:
        raise TrainingContractError("production model has an invalid task identity")
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != expected:
        raise TrainingContractError(
            f"{family}/{task} parameter count changed: {count} != {expected}"
        )
    return ModelContract(
        family=family,
        architecture_version=architecture_version,
        input_contract=MODEL_INPUT_CONTRACTS[family],
        task=task,
        parameter_count=count,
    )


def create_loss(task: str) -> nn.Module:
    """Return the unweighted loss that consumes frozen E.5 raw logits."""

    if task in {"E1", "E2"}:
        return nn.CrossEntropyLoss()
    if task == "E3":
        return nn.BCEWithLogitsLoss()
    raise ValueError("task must be exactly one of E1, E2, or E3")


def create_optimizer(model: nn.Module, learning_rate: float) -> AdamW:
    if learning_rate not in LEARNING_RATE_CANDIDATES:
        raise ValueError(
            f"learning_rate must be one of {LEARNING_RATE_CANDIDATES}"
        )
    return AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)


def compute_loss(
    task: str, criterion: nn.Module, logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Apply the loss directly to raw logits, with strict task shapes."""

    if task in {"E1", "E2"}:
        if tuple(logits.shape) != (targets.numel(), 15):
            raise TrainingContractError(
                f"{task} logits must have shape [B,15], got {tuple(logits.shape)}"
            )
        if targets.dtype != torch.long or targets.ndim != 1:
            raise TrainingContractError(f"{task} targets must be one-dimensional long")
    elif task == "E3":
        if logits.ndim != 1 or tuple(logits.shape) != tuple(targets.shape):
            raise TrainingContractError(
                f"E3 logits and targets must both have shape [B], got "
                f"{tuple(logits.shape)} and {tuple(targets.shape)}"
            )
        if targets.dtype != torch.float32:
            raise TrainingContractError("E3 targets must be float32")
    else:
        raise ValueError("task must be exactly one of E1, E2, or E3")
    return criterion(logits, targets)


def _affine_permutation_parameters(length: int, seed: int) -> tuple[int, int]:
    """Choose a seeded O(1)-state permutation ``(a*i+b) mod length``."""

    _require_int_between("dataset_length", length, 1, 2**63 - 1)
    rng = random.Random(seed)
    if length == 1:
        return 1, 0
    candidate = rng.randrange(1, length)
    while math.gcd(candidate, length) != 1:
        candidate = (candidate + 1) % length
        if candidate == 0:
            candidate = 1
    return candidate, rng.randrange(length)


def iter_index_batches(
    dataset_length: int,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int | None = None,
    max_batches: int | None = None,
) -> Iterator[tuple[int, ...]]:
    """Yield bounded index tuples without materializing a split-sized index list."""

    _require_int_between("dataset_length", dataset_length, 1, 2**63 - 1)
    _require_int_between("batch_size", batch_size, 1, MAX_BATCH_SIZE)
    _require_optional_positive_int("max_batches", max_batches)
    if shuffle and seed is None:
        raise ValueError("a seed is required for shuffled traversal")
    if not shuffle and seed is not None:
        raise ValueError("sequential traversal must not receive a seed")
    multiplier, offset = (
        _affine_permutation_parameters(dataset_length, int(seed))
        if shuffle
        else (1, 0)
    )
    total_batches = (dataset_length + batch_size - 1) // batch_size
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)
    for batch_number in range(total_batches):
        start = batch_number * batch_size
        stop = min(start + batch_size, dataset_length)
        yield tuple(
            (multiplier * position + offset) % dataset_length
            for position in range(start, stop)
        )


def iter_production_batches(
    dataset: d4.LazyGraphDataset,
    *,
    task: str,
    split: str,
    batch_size: int,
    epoch_seed: int | None,
    max_batches: int | None,
) -> Iterator[HeteroData]:
    """Use only the authoritative D4 -> D6 -> E.2 bounded production path."""

    if not isinstance(dataset, d4.LazyGraphDataset):
        raise TypeError("dataset must be a frozen D4 LazyGraphDataset")
    if split not in {"train", "validation"}:
        raise TrainingContractError("E.7 accepts only train and validation splits")
    if dataset.experiment != task or dataset.split != split:
        raise TrainingContractError("cross-task or cross-split dataset substitution")
    if not dataset.is_lazy_read_only:
        raise TrainingContractError("source dataset must remain lazy and read-only")
    shuffle = split == "train"
    if shuffle != (epoch_seed is not None):
        raise TrainingContractError(
            "training requires a seed and validation must remain sequential"
        )
    for indices in iter_index_batches(
        len(dataset),
        batch_size,
        shuffle=shuffle,
        seed=epoch_seed,
        max_batches=max_batches,
    ):
        logical = d6.build_logical_batch(dataset, indices)
        if logical.batch_size > batch_size:
            raise TrainingContractError("D6 exceeded the configured batch bound")
        yield e2.logical_batch_to_heterodata(logical)


def _require_batch_audit(data: HeteroData, task: str, split: str) -> tuple[int, ...]:
    if not isinstance(data, HeteroData):
        raise TypeError("training batches must be PyG HeteroData objects")
    audit = getattr(data, "e2_audit", None)
    if not isinstance(audit, Mapping):
        raise TrainingContractError("batch is missing frozen E.2 audit metadata")
    if audit.get("experiment") != task or audit.get("split") != split:
        raise TrainingContractError(
            f"expected {task}/{split} batch, got "
            f"{audit.get('experiment')}/{audit.get('split')}"
        )
    raw_indices = audit.get("ordered_source_sample_indices")
    if not isinstance(raw_indices, list) or not raw_indices:
        raise TrainingContractError("batch audit has no ordered source indices")
    indices = tuple(int(value) for value in raw_indices)
    if len(set(indices)) != len(indices):
        raise TrainingContractError("batch audit contains duplicate source indices")
    return indices


def _finite_gradients(model: nn.Module) -> None:
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is not None and not torch.isfinite(gradient).all():
            raise TrainingContractError(f"nonfinite gradient detected in {name}")


class _MetricAccumulator:
    def __init__(self, task: str) -> None:
        self.task = task
        self.confusion = np.zeros((15, 15), dtype=np.int64)
        self.binary_scores: list[np.ndarray] = []
        self.binary_targets: list[np.ndarray] = []
        self.target_codes: set[int] = set()

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        target_array = targets.detach().cpu().numpy()
        self.target_codes.update(int(value) for value in np.unique(target_array))
        if self.task in {"E1", "E2"}:
            predictions = logits.detach().argmax(dim=1).cpu().numpy()
            for target, prediction in zip(target_array, predictions, strict=True):
                self.confusion[int(target), int(prediction)] += 1
        else:
            self.binary_scores.append(logits.detach().cpu().numpy().astype(np.float64))
            self.binary_targets.append(target_array.astype(np.int8))

    def selection(
        self,
        *,
        train_target_codes: Sequence[int],
        validation_selection_class_codes: Sequence[int],
    ) -> tuple[float, bool, tuple[int, ...]]:
        if self.task == "E3":
            targets = np.concatenate(self.binary_targets)
            scores = np.concatenate(self.binary_scores)
            complete = set(int(value) for value in np.unique(targets)) == {0, 1}
            if not complete:
                return 0.0, False, ()
            return float(average_precision_score(targets, scores)), True, ()
        if self.task == "E1":
            selected = sorted(set(train_target_codes) & self.target_codes)
        else:
            selected = list(validation_selection_class_codes)
        if not selected:
            raise TrainingContractError("no authorized classes support validation F1")
        scores: list[float] = []
        for code in selected:
            true_positive = int(self.confusion[code, code])
            false_positive = int(self.confusion[:, code].sum()) - true_positive
            false_negative = int(self.confusion[code, :].sum()) - true_positive
            denominator = 2 * true_positive + false_positive + false_negative
            scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
        complete = all(int(self.confusion[code, :].sum()) > 0 for code in selected)
        return float(sum(scores) / len(scores)), complete, tuple(selected)


def _run_epoch(
    *,
    task: str,
    model: nn.Module,
    batches: Iterable[HeteroData],
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optimizer | None,
    expected_split: str,
    train_target_codes: Sequence[int] = (),
    validation_selection_class_codes: Sequence[int] = (),
) -> EpochStats:
    training = optimizer is not None
    if training != (expected_split == "train"):
        raise TrainingContractError("optimizer/split mode mismatch")
    model.train(training)
    loss_total = 0.0
    example_total = 0
    batch_total = 0
    order_digest = hashlib.sha256()
    metric = _MetricAccumulator(task)

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for data in batches:
            indices = _require_batch_audit(data, task, expected_split)
            for index in indices:
                order_digest.update(int(index).to_bytes(8, "little", signed=False))
            data = data.to(device)
            targets = data["sample"].y
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(data)
            if isinstance(logits, tuple):
                raise TrainingContractError("ordinary training must not collect attention")
            loss = compute_loss(task, criterion, logits, targets)
            if not torch.isfinite(loss):
                raise TrainingContractError(
                    f"nonfinite {expected_split} loss at batch {batch_total + 1}"
                )
            if training:
                loss.backward()
                _finite_gradients(model)
                optimizer.step()
                for name, parameter in model.named_parameters():
                    if not torch.isfinite(parameter).all():
                        raise TrainingContractError(
                            f"optimizer produced nonfinite parameter {name}"
                        )
            size = int(targets.numel())
            loss_total += float(loss.detach().cpu()) * size
            example_total += size
            batch_total += 1
            metric.update(logits, targets)
            del logits, loss, targets, data

    if batch_total == 0 or example_total == 0:
        raise TrainingContractError(f"{expected_split} iterator produced no batches")
    selection_value: float | None = None
    support_complete: bool | None = None
    selection_codes: tuple[int, ...] = ()
    if not training:
        selection_value, support_complete, selection_codes = metric.selection(
            train_target_codes=train_target_codes,
            validation_selection_class_codes=validation_selection_class_codes,
        )
    return EpochStats(
        loss=loss_total / example_total,
        batches=batch_total,
        examples=example_total,
        order_sha256=order_digest.hexdigest(),
        target_codes=tuple(sorted(metric.target_codes)),
        selection_value=selection_value,
        selection_name=None if training else SELECTION_NAMES[task],
        selection_support_complete=support_complete,
        selection_class_codes=selection_codes,
    )


def train_one_epoch(
    *,
    task: str,
    model: nn.Module,
    batches: Iterable[HeteroData],
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> EpochStats:
    return _run_epoch(
        task=task,
        model=model,
        batches=batches,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        expected_split="train",
    )


def validate_one_epoch(
    *,
    task: str,
    model: nn.Module,
    batches: Iterable[HeteroData],
    criterion: nn.Module,
    device: torch.device,
    train_target_codes: Sequence[int],
    validation_selection_class_codes: Sequence[int] = (),
) -> EpochStats:
    return _run_epoch(
        task=task,
        model=model,
        batches=batches,
        criterion=criterion,
        optimizer=None,
        device=device,
        expected_split="validation",
        train_target_codes=train_target_codes,
        validation_selection_class_codes=validation_selection_class_codes,
    )


def _derive_e2_validation_selection_class_codes(
    train_dataset: d4.LazyGraphDataset,
    validation_dataset: d4.LazyGraphDataset,
) -> tuple[int, ...]:
    """Derive E2 selection classes from train/validation targets only."""
    try:
        policy = stage_c_splitting.validation_selection_eligibility(
            train_dataset.y_memmap,
            validation_dataset.y_memmap,
        )
    except Exception as exc:
        raise TrainingContractError(
            f"E2 train/validation selection authority is invalid: {exc}"
        ) from exc
    if policy.get("test_support_consulted") is not False:
        raise TrainingContractError("E2 validation selection consulted test support")
    codes = policy.get("validation_selection_class_codes")
    if not isinstance(codes, list) or not codes:
        raise TrainingContractError(
            "E2 train/validation selection class authority is missing"
        )
    return tuple(int(code) for code in codes)


def _load_e1_train_support(project_root: Path) -> tuple[int, ...]:
    """Read only the frozen E1 train-support field used by validation selection."""

    path = (
        project_root
        / "data"
        / "processed"
        / "preprocessed"
        / "exp1"
        / "split_metadata.json"
    )
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise TrainingContractError(f"cannot load frozen E1 train support: {exc}") from exc
    distribution = metadata.get("train_class_distribution")
    if (
        metadata.get("experiment") != "1_global_chronological"
        or not isinstance(distribution, Mapping)
        or not distribution
        or metadata.get("preprocessing", {}).get("fit_split") != "train"
        or metadata.get("preprocessing", {}).get("validation_or_test_used_for_fit")
        is not False
    ):
        raise TrainingContractError("frozen E1 train-support authority is invalid")
    try:
        return tuple(sorted(LABEL_TO_INT[str(name)] for name in distribution))
    except KeyError as exc:
        raise TrainingContractError("E1 train support contains an unknown class") from exc


def model_state_schema_sha256(model: nn.Module) -> str:
    schema = [
        {"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in model.state_dict().items()
    ]
    payload = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def checkpoint_metadata(
    *,
    config: TrainingConfig,
    model: nn.Module,
    epoch: int,
    validation: EpochStats,
    frozen_source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    identity = model_contract(model)
    if identity.task != config.task:
        raise TrainingContractError("checkpoint model task does not match configuration")
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "harness_version": HARNESS_VERSION,
        "task": config.task,
        "model_family": identity.family,
        "model_contract_version": identity.architecture_version,
        "model_input_contract": identity.input_contract,
        "model_version": identity.architecture_version,
        "model_state_schema_sha256": model_state_schema_sha256(model),
        "parameter_count": identity.parameter_count,
        "epoch": epoch,
        "selection_name": SELECTION_NAMES[config.task],
        "selection_value": validation.selection_value,
        "validation_loss": validation.loss,
        "selection_class_codes": list(validation.selection_class_codes),
        "training_config": config.checkpoint_dict(),
        "frozen_source_sha256": dict(frozen_source_hashes),
    }


def save_checkpoint(
    path: Path,
    *,
    config: TrainingConfig,
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    validation: EpochStats,
    frozen_source_hashes: Mapping[str, str],
) -> None:
    """Atomically save state dictionaries and plain contract metadata only."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": checkpoint_metadata(
            config=config,
            model=model,
            epoch=epoch,
            validation=validation,
            frozen_source_hashes=frozen_source_hashes,
        ),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _preflight_model_state_dict(
    model: nn.Module, model_state: Mapping[str, Any]
) -> None:
    """Validate the complete model payload without touching the target model."""

    expected = model.state_dict()
    expected_keys = set(expected)
    received_keys = set(model_state)
    missing = sorted(repr(key) for key in expected_keys - received_keys)
    unexpected = sorted(repr(key) for key in received_keys - expected_keys)
    if missing or unexpected:
        raise TrainingContractError(
            "checkpoint model state_dict keys are incompatible: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for name, target in expected.items():
        value = model_state[name]
        if not isinstance(value, torch.Tensor):
            raise TrainingContractError(
                f"checkpoint model tensor {name!r} is not a tensor"
            )
        if value.shape != target.shape:
            raise TrainingContractError(
                f"checkpoint model tensor {name!r} has shape {tuple(value.shape)}; "
                f"expected {tuple(target.shape)}"
            )
        if value.dtype != target.dtype or value.layout != target.layout:
            raise TrainingContractError(
                f"checkpoint model tensor {name!r} has incompatible dtype/layout"
            )
        if value.device.type == "meta" or target.device.type == "meta":
            raise TrainingContractError(
                f"checkpoint model tensor {name!r} cannot use the meta device"
            )


def _checkpoint_optimizer_learning_rate(metadata: Mapping[str, Any]) -> float:
    """Return the metadata-bound LR after validating the frozen AdamW contract."""

    training_config = metadata.get("training_config")
    if not isinstance(training_config, Mapping):
        raise TrainingContractError("checkpoint training configuration is missing")
    learning_rate = training_config.get("learning_rate")
    weight_decay = training_config.get("weight_decay")
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(float(learning_rate))
        or float(learning_rate) not in LEARNING_RATE_CANDIDATES
        or training_config.get("optimizer") != "AdamW"
        or isinstance(weight_decay, bool)
        or not isinstance(weight_decay, (int, float))
        or not math.isfinite(float(weight_decay))
        or float(weight_decay) != WEIGHT_DECAY
        or training_config.get("scheduler") is not None
    ):
        raise TrainingContractError(
            "checkpoint training configuration violates the frozen AdamW contract"
        )
    return float(learning_rate)


def _preflight_optimizer_state_dict(
    optimizer: Optimizer,
    optimizer_state: Mapping[str, Any],
    *,
    checkpoint_learning_rate: float,
) -> None:
    """Validate complete frozen-AdamW semantics on a disposable copy."""

    if set(optimizer_state) != {"state", "param_groups"}:
        raise TrainingContractError(
            "checkpoint optimizer state_dict fields are incompatible"
        )
    saved_state = optimizer_state.get("state")
    saved_groups = optimizer_state.get("param_groups")
    current = optimizer.state_dict()
    current_groups = current["param_groups"]
    if not isinstance(saved_state, Mapping) or not isinstance(saved_groups, list):
        raise TrainingContractError("checkpoint optimizer state_dict is malformed")
    if type(optimizer) is not AdamW:
        raise TrainingContractError("target optimizer must be the frozen AdamW type")
    if len(current_groups) != 1:
        raise TrainingContractError(
            "target optimizer must have the single frozen production parameter group"
        )
    authoritative_optimizer = AdamW(
        list(optimizer.param_groups[0]["params"]),
        lr=checkpoint_learning_rate,
        weight_decay=WEIGHT_DECAY,
    )
    authoritative_group = authoritative_optimizer.state_dict()["param_groups"][0]
    if len(saved_groups) != len(current_groups):
        raise TrainingContractError(
            "checkpoint optimizer parameter-group count is incompatible"
        )

    parameter_by_saved_id: dict[int, torch.Tensor] = {}
    for saved_group, current_group, live_group in zip(
        saved_groups, current_groups, optimizer.param_groups, strict=True
    ):
        if not isinstance(saved_group, Mapping):
            raise TrainingContractError("checkpoint optimizer parameter group is malformed")
        if set(saved_group) != set(current_group):
            raise TrainingContractError(
                "checkpoint optimizer parameter-group fields are incompatible"
            )
        saved_learning_rate = saved_group.get("lr")
        saved_weight_decay = saved_group.get("weight_decay")
        if (
            isinstance(saved_learning_rate, bool)
            or not isinstance(saved_learning_rate, (int, float))
            or not math.isfinite(float(saved_learning_rate))
            or float(saved_learning_rate) != checkpoint_learning_rate
        ):
            raise TrainingContractError(
                "checkpoint optimizer learning rate contradicts checkpoint metadata"
            )
        if (
            isinstance(saved_weight_decay, bool)
            or not isinstance(saved_weight_decay, (int, float))
            or not math.isfinite(float(saved_weight_decay))
            or float(saved_weight_decay) != WEIGHT_DECAY
            or current_group.get("weight_decay") != WEIGHT_DECAY
        ):
            raise TrainingContractError(
                "checkpoint or target optimizer weight decay violates the frozen contract"
            )
        current_learning_rate = current_group.get("lr")
        if (
            isinstance(current_learning_rate, bool)
            or not isinstance(current_learning_rate, (int, float))
            or not math.isfinite(float(current_learning_rate))
            or float(current_learning_rate) not in LEARNING_RATE_CANDIDATES
        ):
            raise TrainingContractError(
                "target optimizer learning rate violates the frozen contract"
            )
        for field in set(saved_group) - {"params", "lr", "weight_decay"}:
            if (
                saved_group[field] != authoritative_group[field]
                or current_group[field] != authoritative_group[field]
            ):
                raise TrainingContractError(
                    f"checkpoint optimizer parameter-group field {field!r} "
                    "violates the frozen production optimizer contract"
                )
        saved_ids = saved_group.get("params")
        current_ids = current_group.get("params")
        live_parameters = live_group.get("params")
        if not all(isinstance(values, list) for values in (saved_ids, current_ids, live_parameters)):
            raise TrainingContractError(
                "checkpoint optimizer parameter-group membership is malformed"
            )
        if len(saved_ids) != len(current_ids) or len(saved_ids) != len(live_parameters):
            raise TrainingContractError(
                "checkpoint optimizer parameter-group size is incompatible"
            )
        if saved_ids != current_ids:
            raise TrainingContractError(
                "checkpoint optimizer serialized parameter mapping is incompatible"
            )
        for saved_id, parameter in zip(saved_ids, live_parameters, strict=True):
            if isinstance(saved_id, bool) or not isinstance(saved_id, int):
                raise TrainingContractError(
                    "checkpoint optimizer parameter identifier is invalid"
                )
            if saved_id in parameter_by_saved_id:
                raise TrainingContractError(
                    "checkpoint optimizer parameter membership is duplicated"
                )
            if not isinstance(parameter, torch.Tensor):
                raise TrainingContractError(
                    "target optimizer parameter membership is invalid"
                )
            parameter_by_saved_id[saved_id] = parameter

    saved_state_ids = set(saved_state)
    for saved_id in saved_state_ids:
        if isinstance(saved_id, bool) or not isinstance(saved_id, int):
            raise TrainingContractError(
                "checkpoint optimizer state parameter identifier is invalid"
            )
    expected_state_ids = set(parameter_by_saved_id)
    if saved_state_ids != expected_state_ids:
        missing = sorted(expected_state_ids - saved_state_ids)
        unexpected = sorted(saved_state_ids - expected_state_ids)
        raise TrainingContractError(
            "checkpoint AdamW parameter-state coverage is incomplete or incompatible: "
            f"missing={missing}, unexpected={unexpected}"
        )

    for saved_id, state in saved_state.items():
        if saved_id not in parameter_by_saved_id or not isinstance(state, Mapping):
            raise TrainingContractError(
                "checkpoint optimizer per-parameter state is malformed"
            )
        parameter = parameter_by_saved_id[saved_id]
        group = next(group for group in saved_groups if saved_id in group["params"])
        required = {"step", "exp_avg", "exp_avg_sq"}
        if group.get("amsgrad") is True:
            required.add("max_exp_avg_sq")
        if set(state) != required:
            raise TrainingContractError(
                "checkpoint AdamW per-parameter state fields are incompatible"
            )
        step = state["step"]
        if (
            not isinstance(step, torch.Tensor)
            or step.shape != torch.Size([])
            or step.dtype != torch.float32
            or step.layout != torch.strided
            or step.device.type == "meta"
        ):
            raise TrainingContractError("checkpoint AdamW step state is invalid")
        step_value = float(step.item())
        if (
            not math.isfinite(step_value)
            or step_value < 0
            or not step_value.is_integer()
        ):
            raise TrainingContractError("checkpoint AdamW step value is invalid")
        for field in required - {"step"}:
            value = state[field]
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != parameter.shape
                or value.dtype != parameter.dtype
                or value.layout != parameter.layout
                or value.device.type == "meta"
            ):
                raise TrainingContractError(
                    f"checkpoint AdamW state {field!r} is incompatible"
                )
            if not bool(torch.isfinite(value).all().item()):
                raise TrainingContractError(
                    f"checkpoint AdamW state {field!r} contains nonfinite values"
                )
            if field in {"exp_avg_sq", "max_exp_avg_sq"} and bool(
                (value < 0).any().item()
            ):
                raise TrainingContractError(
                    f"checkpoint AdamW state {field!r} contains negative values"
                )

    try:
        probe = copy.deepcopy(optimizer)
        probe.load_state_dict(copy.deepcopy(dict(optimizer_state)))
    except Exception as exc:
        raise TrainingContractError(
            f"checkpoint optimizer state_dict is incompatible: {exc}"
        ) from exc


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    expected_task: str,
    optimizer: Optimizer | None = None,
    expected_config: TrainingConfig | None = None,
    map_location: str | torch.device = "cpu",
) -> Mapping[str, Any]:
    """Load a shared state-dict checkpoint after identity checks, then weights."""

    if expected_task not in TASKS:
        raise ValueError("expected_task must be exactly one of E1, E2, or E3")
    try:
        payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    except Exception as exc:
        raise TrainingContractError(f"cannot load E.7 checkpoint: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise TrainingContractError("checkpoint payload must be a mapping")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise TrainingContractError("checkpoint metadata is missing")
    identity = model_contract(model)
    if identity.task != expected_task:
        raise TrainingContractError("target model task does not match requested task")
    schema_version = metadata.get("schema_version")
    if schema_version not in {
        LEGACY_HTGNN_CHECKPOINT_SCHEMA_VERSION,
        CHECKPOINT_SCHEMA_VERSION,
    }:
        raise TrainingContractError(
            f"checkpoint schema_version mismatch: {schema_version!r}"
        )
    if (
        schema_version == LEGACY_HTGNN_CHECKPOINT_SCHEMA_VERSION
        and identity.family != HTGNN_MODEL_FAMILY
    ):
        raise TrainingContractError(
            "legacy checkpoints are valid only for the HTGNN model family"
        )
    expected = {
        "harness_version": HARNESS_VERSION,
        "task": expected_task,
        "model_version": identity.architecture_version,
        "parameter_count": identity.parameter_count,
        "model_state_schema_sha256": model_state_schema_sha256(model),
        "frozen_source_sha256": dict(e6.FROZEN_SOURCE_SHA256),
    }
    if schema_version == CHECKPOINT_SCHEMA_VERSION:
        expected.update(
            {
                "model_family": identity.family,
                "model_contract_version": identity.architecture_version,
                "model_input_contract": identity.input_contract,
            }
        )
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise TrainingContractError(
                f"checkpoint {key} mismatch: {metadata.get(key)!r} != {value!r}"
            )
    if metadata.get("selection_name") != SELECTION_NAMES[expected_task]:
        raise TrainingContractError("checkpoint validation selection contract mismatch")
    if (
        isinstance(metadata.get("epoch"), bool)
        or not isinstance(metadata.get("epoch"), int)
        or not 1 <= int(metadata["epoch"]) <= MAX_EPOCHS
        or not isinstance(metadata.get("selection_value"), (int, float))
        or not math.isfinite(float(metadata["selection_value"]))
        or not isinstance(metadata.get("validation_loss"), (int, float))
        or not math.isfinite(float(metadata["validation_loss"]))
    ):
        raise TrainingContractError("checkpoint epoch or validation values are invalid")
    if expected_config is not None and metadata.get(
        "training_config"
    ) != expected_config.checkpoint_dict():
        raise TrainingContractError("checkpoint training configuration mismatch")
    checkpoint_learning_rate = _checkpoint_optimizer_learning_rate(metadata)
    model_state = payload.get("model_state_dict")
    optimizer_state = payload.get("optimizer_state_dict")
    if not isinstance(model_state, Mapping):
        raise TrainingContractError("checkpoint model state_dict is missing")
    _preflight_model_state_dict(model, model_state)
    if not isinstance(optimizer_state, Mapping):
        raise TrainingContractError("checkpoint optimizer state_dict is missing")
    validation_optimizer = (
        optimizer
        if optimizer is not None
        else create_optimizer(model, checkpoint_learning_rate)
    )
    _preflight_optimizer_state_dict(
        validation_optimizer,
        optimizer_state,
        checkpoint_learning_rate=checkpoint_learning_rate,
    )

    model_snapshot = copy.deepcopy(model.state_dict())
    optimizer_snapshot = (
        copy.deepcopy(optimizer.state_dict()) if optimizer is not None else None
    )
    try:
        model.load_state_dict(model_state, strict=True)
        if optimizer is not None:
            assert isinstance(optimizer_state, Mapping)
            optimizer.load_state_dict(optimizer_state)
    except Exception as exc:
        rollback_errors: list[str] = []
        try:
            model.load_state_dict(model_snapshot, strict=True)
        except Exception as rollback_exc:  # pragma: no cover - catastrophic target failure
            rollback_errors.append(f"model rollback failed: {rollback_exc}")
        if optimizer is not None:
            assert optimizer_snapshot is not None
            try:
                optimizer.load_state_dict(optimizer_snapshot)
            except Exception as rollback_exc:  # pragma: no cover - catastrophic target failure
                rollback_errors.append(f"optimizer rollback failed: {rollback_exc}")
        if rollback_errors:
            raise TrainingContractError(
                "checkpoint load failed and transactional rollback was incomplete: "
                + "; ".join(rollback_errors)
            ) from exc
        raise TrainingContractError(
            f"checkpoint state_dict is incompatible; target state was restored: {exc}"
        ) from exc
    return metadata


def _parameter_snapshot(model: nn.Module) -> tuple[torch.Tensor, ...]:
    return tuple(parameter.detach().cpu().clone() for parameter in model.parameters())


def _parameters_changed(
    before: Sequence[torch.Tensor], model: nn.Module
) -> bool:
    after = tuple(parameter.detach().cpu() for parameter in model.parameters())
    return any(not torch.equal(left, right) for left, right in zip(before, after, strict=True))


def _resolve_output_path(path: Path, description: str) -> Path:
    try:
        return Path(path).resolve(strict=False)
    except OSError as exc:
        raise TrainingContractError(
            f"cannot resolve {description} path {path}: {exc}"
        ) from exc


def _reserve_run_outputs(
    checkpoint_path: Path, log_path: Path | None
) -> tuple[Path, Path | None]:
    """Normalize outputs and atomically reserve one fresh log for one run.

    ``open('x')`` maps to exclusive creation and therefore combines the
    existence check and file creation without a check-then-create race.  Every
    existing path, including an empty file, is rejected.  The checkpoint/log
    alias check occurs before directory or file creation.
    """

    resolved_checkpoint = _resolve_output_path(checkpoint_path, "checkpoint")
    if log_path is None:
        return resolved_checkpoint, None
    resolved_log = _resolve_output_path(log_path, "log")
    if resolved_checkpoint == resolved_log:
        raise TrainingContractError(
            "checkpoint and log paths must resolve to different targets"
        )
    try:
        resolved_log.parent.mkdir(parents=True, exist_ok=True)
        with resolved_log.open("x", encoding="utf-8", newline="\n"):
            pass
    except FileExistsError as exc:
        raise TrainingContractError(
            f"log path already exists; one log file belongs to one run: {resolved_log}"
        ) from exc
    except OSError as exc:
        raise TrainingContractError(
            f"cannot reserve log path {resolved_log}: {exc}"
        ) from exc
    return resolved_checkpoint, resolved_log


def _append_jsonl(path: Path | None, record: Mapping[str, Any]) -> None:
    if path is None:
        return
    try:
        with path.open("r+", encoding="utf-8", newline="\n") as handle:
            handle.seek(0, 2)
            handle.write(
                json.dumps(dict(record), sort_keys=True, separators=(",", ":"))
            )
            handle.write("\n")
            handle.flush()
    except OSError as exc:
        raise TrainingContractError(f"cannot write reserved log path {path}: {exc}") from exc


def run_training(
    config: TrainingConfig,
    *,
    project_root: Path,
    checkpoint_path: Path,
    log_path: Path | None = None,
    device: str = "auto",
    model_family: str = HTGNN_MODEL_FAMILY,
) -> TrainingResult:
    """Train one supported family on train and select only on validation."""

    project_root = Path(project_root).resolve()
    if model_family not in MODEL_FAMILIES:
        raise ValueError(f"model_family must be one of {MODEL_FAMILIES}")
    checkpoint_path, log_path = _reserve_run_outputs(checkpoint_path, log_path)
    frozen_hashes = e6.require_frozen_sources(project_root)
    seed_everything(config.seed)
    selected_device = resolve_device(device)
    processed_root = project_root / "data" / "processed" / "preprocessed"
    train_dataset = d4.LazyGraphDataset(
        config.task, "train", processed_root=processed_root
    )
    validation_dataset = d4.LazyGraphDataset(
        config.task, "validation", processed_root=processed_root
    )
    model = create_model(config.task, model_family).to(selected_device)
    identity = model_contract(model)
    criterion = create_loss(config.task).to(selected_device)
    optimizer = create_optimizer(model, config.learning_rate)
    validation_selection_class_codes = (
        _derive_e2_validation_selection_class_codes(
            train_dataset, validation_dataset
        )
        if config.task == "E2"
        else ()
    )
    e1_train_support = (
        _load_e1_train_support(project_root) if config.task == "E1" else ()
    )
    state_schema_before = model_state_schema_sha256(model)
    parameters_before = _parameter_snapshot(model)
    early_stopping = EarlyStopping()
    history: list[Mapping[str, Any]] = []
    stopped_early = False

    for epoch in range(1, config.max_epochs + 1):
        train_stats = train_one_epoch(
            task=config.task,
            model=model,
            batches=iter_production_batches(
                train_dataset,
                task=config.task,
                split="train",
                batch_size=config.batch_size,
                epoch_seed=config.seed + epoch - 1,
                max_batches=config.max_train_batches,
            ),
            criterion=criterion,
            optimizer=optimizer,
            device=selected_device,
        )
        validation_stats = validate_one_epoch(
            task=config.task,
            model=model,
            batches=iter_production_batches(
                validation_dataset,
                task=config.task,
                split="validation",
                batch_size=config.batch_size,
                epoch_seed=None,
                max_batches=config.max_validation_batches,
            ),
            criterion=criterion,
            device=selected_device,
            train_target_codes=(
                e1_train_support if config.task == "E1" else train_stats.target_codes
            ),
            validation_selection_class_codes=validation_selection_class_codes,
        )
        if validation_stats.selection_value is None:
            raise TrainingContractError("validation did not produce a selection value")
        if (
            config.max_validation_batches is None
            and validation_stats.selection_support_complete is not True
        ):
            raise TrainingContractError(
                "complete validation traversal lacks required selection support"
            )
        is_best, should_stop = early_stopping.update(
            epoch, validation_stats.selection_value
        )
        if is_best:
            save_checkpoint(
                checkpoint_path,
                config=config,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                validation=validation_stats,
                frozen_source_hashes=frozen_hashes,
            )
        record: Mapping[str, Any] = {
            "task": config.task,
            "model_family": identity.family,
            "model_contract_version": identity.architecture_version,
            "model_input_contract": identity.input_contract,
            "seed": config.seed,
            "learning_rate": config.learning_rate,
            "epoch": epoch,
            "train_loss": train_stats.loss,
            "validation_loss": validation_stats.loss,
            "validation_selection_name": validation_stats.selection_name,
            "validation_selection_value": validation_stats.selection_value,
            "validation_selection_support_complete": (
                validation_stats.selection_support_complete
            ),
            "train_batches": train_stats.batches,
            "validation_batches": validation_stats.batches,
            "train_examples": train_stats.examples,
            "validation_examples": validation_stats.examples,
            "train_order_sha256": train_stats.order_sha256,
            "validation_order_sha256": validation_stats.order_sha256,
            "best_checkpoint_updated": is_best,
        }
        history.append(record)
        _append_jsonl(log_path, record)
        if should_stop:
            stopped_early = True
            break

    if early_stopping.best_epoch == 0 or not Path(checkpoint_path).is_file():
        raise TrainingContractError("training produced no best validation checkpoint")
    if model_state_schema_sha256(model) != state_schema_before:
        raise TrainingContractError("training mutated the frozen model architecture")
    return TrainingResult(
        task=config.task,
        checkpoint_path=str(Path(checkpoint_path)),
        epochs_completed=len(history),
        stopped_early=stopped_early,
        best_epoch=early_stopping.best_epoch,
        best_selection_value=early_stopping.best_value,
        selection_name=SELECTION_NAMES[config.task],
        history=tuple(history),
        parameter_update_observed=_parameters_changed(parameters_before, model),
        device=str(selected_device),
        model_family=identity.family,
    )


def evaluate_saved_validation_checkpoint(
    config: TrainingConfig,
    *,
    project_root: Path,
    checkpoint_path: Path,
    device: str = "auto",
    model_family: str = HTGNN_MODEL_FAMILY,
) -> tuple[Mapping[str, Any], EpochStats, nn.Module]:
    """Reload a best checkpoint and rerun the same bounded validation path."""

    selected_device = resolve_device(device)
    e6.require_frozen_sources(Path(project_root).resolve())
    model = create_model(config.task, model_family).to(selected_device)
    metadata = load_checkpoint(
        checkpoint_path,
        model=model,
        expected_task=config.task,
        expected_config=config,
        map_location=selected_device,
    )
    criterion = create_loss(config.task).to(selected_device)
    processed_root = Path(project_root).resolve() / "data" / "processed" / "preprocessed"
    validation_dataset = d4.LazyGraphDataset(
        config.task, "validation", processed_root=processed_root
    )
    train_codes = (
        tuple(int(value) for value in metadata.get("selection_class_codes", ()))
        if config.task == "E1"
        else ()
    )
    validation_selection_class_codes: tuple[int, ...] = ()
    if config.task == "E2":
        train_dataset = d4.LazyGraphDataset(
            config.task, "train", processed_root=processed_root
        )
        validation_selection_class_codes = (
            _derive_e2_validation_selection_class_codes(
                train_dataset, validation_dataset
            )
        )
        checkpoint_codes = tuple(
            int(value) for value in metadata.get("selection_class_codes", ())
        )
        if checkpoint_codes != validation_selection_class_codes:
            raise TrainingContractError(
                "checkpoint E2 validation-selection classes do not match "
                "current train/validation authority"
            )
    validation = validate_one_epoch(
        task=config.task,
        model=model,
        batches=iter_production_batches(
            validation_dataset,
            task=config.task,
            split="validation",
            batch_size=config.batch_size,
            epoch_seed=None,
            max_batches=config.max_validation_batches,
        ),
        criterion=criterion,
        device=selected_device,
        train_target_codes=train_codes,
        validation_selection_class_codes=validation_selection_class_codes,
    )
    return metadata, validation, model
