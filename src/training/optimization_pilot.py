"""Stage E.8 bounded learning-rate pilot around the frozen E.7 harness.

This module owns candidate enumeration, isolated artifact paths, result
validation, and compact observational reporting.  It deliberately delegates
every optimization step to :mod:`src.training.harness`; it does not implement
another training loop, accept split names, evaluate test data, or finalize a
production configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import secrets
import time
from typing import Any, Final, Mapping, Sequence

from src.graph import resource_pilot as e6
from src.graph import schema as d2
from src.training import harness as e7


PILOT_VERSION: Final = "stage-e8-optimization-pilot-v2"
PILOT_TASKS: Final = ("E1", "E2", "E3")
PILOT_LEARNING_RATES: Final = (1e-4, 3e-4, 1e-3)
PILOT_SEED: Final = 2025
PILOT_BATCH_SIZE: Final = 32
PILOT_MAX_EPOCHS: Final = 3
PILOT_MAX_TRAIN_BATCHES: Final = 16
PILOT_MAX_VALIDATION_BATCHES: Final = 128
PILOT_LOSS_MODE: Final = "unweighted"
ALLOWED_DATA_PARTITIONS: Final = ("train", "validation")
OBSERVATIONAL_RANKING_RULE: Final = (
    "higher_validation_objective_then_approved_learning_rate_order"
)
EXPECTED_SELECTION_NAMES: Final = {
    "E1": "support_restricted_macro_f1",
    "E2": "validation_selection_macro_f1",
    "E3": "pr_auc_average_precision",
}
CORRECTED_E7_SOURCE_SHA256: Final = {
    "src/preprocessing/splitting.py": (
        "11c11605e59015fb32b9fc4634363ba2979918ac9de9025088ea70da2e084445"
    ),
    "src/training/harness.py": (
        "da3175de7a17659cfbbf665d189fa0a78223eb6b9feb99faae1cea7d72e0b584"
    ),
    "src/training/validation.py": (
        "4e1e116dfad6adc1514bba6a242619bea2404c2abc51847a030ce8bf12b67c9f"
    ),
    "src/training/__init__.py": (
        "d699309aa3e24338d7926d6bd2018367f16b1ea837eb53ba612a42b6ca671e7a"
    ),
    "scripts/train_htgnn.py": (
        "5421759cdbfcac33e384638f723308b053c4346f99150b83465c7565f30fff8d"
    ),
    "tests/test_preprocessing_splitting.py": (
        "142a3d76b2ef6a64e920d849d4dc0b1ea946728caf1a6ffe84753da3eaac9aeb"
    ),
    "tests/test_training_harness.py": (
        "2f3e273a863b18e1da766cf45c284e0910e7c98b85df40cbd0f4beab08030373"
    ),
}
_RUN_ID_PATTERN: Final = re.compile(r"^[a-z0-9][a-z0-9._-]{7,95}$")
_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
_LR_TOKENS: Final = {1e-4: "1e-4", 3e-4: "3e-4", 1e-3: "1e-3"}


class OptimizationPilotError(RuntimeError):
    """Raised when the bounded E.8 pilot contract is violated."""


@dataclass(frozen=True, slots=True)
class PilotCandidate:
    """One approved E.8 candidate; only the task and learning rate vary."""

    task: str
    learning_rate: float

    def __post_init__(self) -> None:
        if self.task not in PILOT_TASKS:
            raise ValueError(f"task must be one of {PILOT_TASKS}")
        if self.learning_rate not in PILOT_LEARNING_RATES:
            raise ValueError(
                f"learning_rate must be one of {PILOT_LEARNING_RATES}"
            )

    @property
    def loss_mode(self) -> str:
        return PILOT_LOSS_MODE

    @property
    def selection_name(self) -> str:
        return EXPECTED_SELECTION_NAMES[self.task]

    def training_config(self) -> e7.TrainingConfig:
        return e7.TrainingConfig(
            task=self.task,
            learning_rate=self.learning_rate,
            seed=PILOT_SEED,
            batch_size=PILOT_BATCH_SIZE,
            max_epochs=PILOT_MAX_EPOCHS,
            max_train_batches=PILOT_MAX_TRAIN_BATCHES,
            max_validation_batches=PILOT_MAX_VALIDATION_BATCHES,
        )

    def contract_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "learning_rate": self.learning_rate,
            "loss_mode": self.loss_mode,
            "seed": PILOT_SEED,
            "batch_size": PILOT_BATCH_SIZE,
            "max_epochs": PILOT_MAX_EPOCHS,
            "max_train_batches_per_epoch": PILOT_MAX_TRAIN_BATCHES,
            "max_validation_batches_per_epoch": PILOT_MAX_VALIDATION_BATCHES,
            "max_train_examples_per_epoch": (
                PILOT_BATCH_SIZE * PILOT_MAX_TRAIN_BATCHES
            ),
            "max_validation_examples_per_epoch": (
                PILOT_BATCH_SIZE * PILOT_MAX_VALIDATION_BATCHES
            ),
            "validation_selection": self.selection_name,
        }

    @property
    def identity_sha256(self) -> str:
        return d2.canonical_sha256(self.contract_dict())

    @property
    def candidate_id(self) -> str:
        token = _LR_TOKENS[self.learning_rate]
        return (
            f"e8-{self.task.lower()}-lr{token}-s{PILOT_SEED}"
            f"-b{PILOT_BATCH_SIZE}-t{PILOT_MAX_TRAIN_BATCHES}"
            f"-v{PILOT_MAX_VALIDATION_BATCHES}-e{PILOT_MAX_EPOCHS}"
            f"-{self.identity_sha256[:12]}"
        )


@dataclass(frozen=True, slots=True)
class CandidatePaths:
    checkpoint: Path
    log: Path
    result: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(project_root: Path, path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(Path(project_root).resolve()).as_posix()
    except ValueError as exc:
        raise OptimizationPilotError(
            f"pilot artifact is outside the project root: {path}"
        ) from exc


def _validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "run_id must be 8-96 lowercase alphanumeric, dot, underscore, or "
            "hyphen characters and start with an alphanumeric character"
        )
    return run_id


def new_run_id() -> str:
    """Return a human-readable, collision-resistant run identifier."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%Sz")
    return f"pilot-{timestamp}-{secrets.token_hex(4)}"


def require_frozen_contracts(project_root: Path) -> dict[str, str]:
    """Fail closed if frozen E.2-E.6 or corrected E.7 sources changed."""

    root = Path(project_root).resolve()
    e6.require_frozen_sources(root)
    actual: dict[str, str] = {}
    for relative, expected in CORRECTED_E7_SOURCE_SHA256.items():
        path = root / relative
        if not path.is_file():
            raise OptimizationPilotError(f"corrected E.7 source is missing: {relative}")
        actual[relative] = _sha256(path)
        if actual[relative] != expected:
            raise OptimizationPilotError(
                f"corrected E.7 source changed: {relative}: "
                f"{actual[relative]} != {expected}"
            )
    expected_constants = {
        "tasks": tuple(e7.TASKS),
        "learning_rates": tuple(e7.LEARNING_RATE_CANDIDATES),
        "seeds": tuple(e7.APPROVED_SEEDS),
        "weight_decay": e7.WEIGHT_DECAY,
        "selection_names": dict(e7.SELECTION_NAMES),
        "parameter_counts": dict(e7.EXPECTED_PARAMETERS),
        "minimum_epochs": e7.MIN_EPOCHS,
        "patience": e7.PATIENCE,
        "min_delta": e7.MIN_DELTA,
    }
    approved = {
        "tasks": PILOT_TASKS,
        "learning_rates": PILOT_LEARNING_RATES,
        "seeds": (2025, 2026, 2027),
        "weight_decay": 1e-4,
        "selection_names": EXPECTED_SELECTION_NAMES,
        "parameter_counts": {"E1": 7_183, "E2": 7_183, "E3": 6_721},
        "minimum_epochs": 3,
        "patience": 5,
        "min_delta": 1e-4,
    }
    if expected_constants != approved:
        raise OptimizationPilotError(
            f"frozen E.7 constants changed: {expected_constants!r}"
        )
    return actual


def enumerate_candidates() -> tuple[PilotCandidate, ...]:
    """Enumerate exactly three approved LRs for each task in fixed order."""

    if tuple(e7.LEARNING_RATE_CANDIDATES) != PILOT_LEARNING_RATES:
        raise OptimizationPilotError("E.7 approved learning-rate space changed")
    return tuple(
        PilotCandidate(task, learning_rate)
        for task in PILOT_TASKS
        for learning_rate in PILOT_LEARNING_RATES
    )


def candidate_paths(
    project_root: Path, run_id: str, candidate: PilotCandidate
) -> CandidatePaths:
    """Build isolated ignored paths for one run and one candidate."""

    root = Path(project_root).resolve()
    run_id = _validate_run_id(run_id)
    return CandidatePaths(
        checkpoint=(
            root / "checkpoints" / "stage_e8" / run_id / f"{candidate.candidate_id}.pt"
        ),
        log=root / "logs" / "stage_e8" / run_id / f"{candidate.candidate_id}.jsonl",
        result=(
            root
            / "logs"
            / "stage_e8"
            / run_id
            / f"{candidate.candidate_id}.result.json"
        ),
    )


def _candidate_record_base(
    project_root: Path,
    run_id: str,
    candidate: PilotCandidate,
    paths: CandidatePaths,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "candidate_identity_sha256": candidate.identity_sha256,
        **candidate.contract_dict(),
        "run_id": run_id,
        "checkpoint_path": _relative_path(project_root, paths.checkpoint),
        "log_path": _relative_path(project_root, paths.log),
        "candidate_result_path": _relative_path(project_root, paths.result),
        "test_accessed": False,
        "test_metrics_recorded": False,
    }


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(d2.canonical_json_bytes(payload))
    except FileExistsError as exc:
        raise OptimizationPilotError(f"refusing to overwrite existing file: {path}") from exc


def _load_log(path: Path) -> list[dict[str, Any]]:
    try:
        records = [
            json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except Exception as exc:
        raise OptimizationPilotError(f"cannot read candidate JSONL log: {exc}") from exc
    if not all(isinstance(record, dict) for record in records):
        raise OptimizationPilotError("candidate JSONL contains a non-object record")
    return records


def _validate_history(
    candidate: PilotCandidate, history: Sequence[Mapping[str, Any]]
) -> None:
    if len(history) != PILOT_MAX_EPOCHS:
        raise OptimizationPilotError("candidate history has an invalid epoch count")
    for expected_epoch, record in enumerate(history, start=1):
        expected = {
            "task": candidate.task,
            "seed": PILOT_SEED,
            "learning_rate": candidate.learning_rate,
            "epoch": expected_epoch,
            "validation_selection_name": candidate.selection_name,
            "train_batches": PILOT_MAX_TRAIN_BATCHES,
            "validation_batches": PILOT_MAX_VALIDATION_BATCHES,
            "train_examples": PILOT_BATCH_SIZE * PILOT_MAX_TRAIN_BATCHES,
            "validation_examples": (
                PILOT_BATCH_SIZE * PILOT_MAX_VALIDATION_BATCHES
            ),
        }
        for key, value in expected.items():
            if record.get(key) != value:
                raise OptimizationPilotError(
                    f"candidate history {key} mismatch: {record.get(key)!r} != {value!r}"
                )
        for key in (
            "train_loss",
            "validation_loss",
            "validation_selection_value",
        ):
            value = record.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise OptimizationPilotError(f"candidate history has invalid {key}")
        for key in ("train_order_sha256", "validation_order_sha256"):
            if not isinstance(record.get(key), str) or not _SHA256_PATTERN.fullmatch(
                str(record[key])
            ):
                raise OptimizationPilotError(f"candidate history has invalid {key}")
        if not isinstance(record.get("validation_selection_support_complete"), bool):
            raise OptimizationPilotError(
                "candidate history lacks validation support-completeness accounting"
            )


def _verify_checkpoint(
    candidate: PilotCandidate, paths: CandidatePaths, best_epoch: int, best_value: float
) -> Mapping[str, Any]:
    config = candidate.training_config()
    model = e7.create_model(candidate.task)
    metadata = e7.load_checkpoint(
        paths.checkpoint,
        model=model,
        expected_task=candidate.task,
        expected_config=config,
        map_location="cpu",
    )
    if metadata.get("epoch") != best_epoch:
        raise OptimizationPilotError("checkpoint best epoch disagrees with run result")
    if not math.isclose(
        float(metadata.get("selection_value")), best_value, rel_tol=0.0, abs_tol=0.0
    ):
        raise OptimizationPilotError(
            "checkpoint validation objective disagrees with run result"
        )
    if metadata.get("parameter_count") != e7.EXPECTED_PARAMETERS[candidate.task]:
        raise OptimizationPilotError("checkpoint parameter count changed")
    return metadata


def run_candidate(
    candidate: PilotCandidate,
    *,
    project_root: Path,
    run_id: str,
    device: str,
    executor: str,
) -> dict[str, Any]:
    """Run one candidate through E.7 and persist complete or failed accounting."""

    root = Path(project_root).resolve()
    paths = candidate_paths(root, run_id, candidate)
    base = _candidate_record_base(root, run_id, candidate, paths)
    started = time.perf_counter()
    try:
        result = e7.run_training(
            candidate.training_config(),
            project_root=root,
            checkpoint_path=paths.checkpoint,
            log_path=paths.log,
            device=device,
        )
        history = [dict(record) for record in result.history]
        _validate_history(candidate, history)
        log_records = _load_log(paths.log)
        if log_records != history:
            raise OptimizationPilotError("candidate log does not match returned history")
        if result.task != candidate.task or result.selection_name != candidate.selection_name:
            raise OptimizationPilotError("candidate result task/objective mismatch")
        if result.epochs_completed != len(history):
            raise OptimizationPilotError("candidate epoch accounting mismatch")
        if not result.parameter_update_observed:
            raise OptimizationPilotError("candidate observed no parameter update")
        metadata = _verify_checkpoint(
            candidate,
            paths,
            result.best_epoch,
            result.best_selection_value,
        )
        record = {
            **base,
            "status": "COMPLETE",
            "executor": executor,
            "device": result.device,
            "epochs_executed": result.epochs_completed,
            "stopped_early": result.stopped_early,
            "best_epoch": result.best_epoch,
            "best_validation_objective": result.best_selection_value,
            "history": history,
            "parameter_update_observed": True,
            "parameter_count": metadata["parameter_count"],
            "model_state_schema_sha256": metadata[
                "model_state_schema_sha256"
            ],
            "validation_selection_class_codes": list(
                metadata.get("selection_class_codes", ())
            ) if candidate.task == "E2" else [],
            "checkpoint_sha256": _sha256(paths.checkpoint),
            "log_sha256": _sha256(paths.log),
            "elapsed_seconds": round(time.perf_counter() - started, 6),
        }
    except Exception as exc:
        record = {
            **base,
            "status": "FAILED",
            "executor": executor,
            "failure_type": type(exc).__name__,
            "failure_message": str(exc),
            "elapsed_seconds": round(time.perf_counter() - started, 6),
        }
    _write_json_exclusive(paths.result, record)
    return record


def validate_candidate_record(
    record: Mapping[str, Any],
    candidate: PilotCandidate,
    *,
    allow_failed: bool,
) -> None:
    """Reject incomplete, cross-candidate, or malformed stored results."""

    base = candidate.contract_dict()
    if record.get("candidate_id") != candidate.candidate_id:
        raise OptimizationPilotError("candidate result ID mismatch")
    if record.get("candidate_identity_sha256") != candidate.identity_sha256:
        raise OptimizationPilotError("candidate result identity hash mismatch")
    for key, value in base.items():
        if record.get(key) != value:
            raise OptimizationPilotError(f"candidate result {key} mismatch")
    if record.get("test_accessed") is not False:
        raise OptimizationPilotError("candidate result indicates forbidden test access")
    if record.get("test_metrics_recorded") is not False:
        raise OptimizationPilotError("candidate result contains test metrics")
    status = record.get("status")
    if status == "FAILED":
        if not allow_failed:
            raise OptimizationPilotError("failed candidate cannot be aggregated as complete")
        if not record.get("failure_type") or not record.get("failure_message"):
            raise OptimizationPilotError("failed candidate accounting is incomplete")
        return
    if status != "COMPLETE":
        raise OptimizationPilotError("candidate result status is incomplete")
    required = (
        "run_id",
        "checkpoint_path",
        "log_path",
        "candidate_result_path",
        "executor",
        "device",
        "epochs_executed",
        "best_epoch",
        "best_validation_objective",
        "history",
        "parameter_count",
        "model_state_schema_sha256",
        "validation_selection_class_codes",
        "checkpoint_sha256",
        "log_sha256",
        "elapsed_seconds",
    )
    if any(key not in record for key in required):
        raise OptimizationPilotError("complete candidate result is missing required fields")
    history = record.get("history")
    if not isinstance(history, list):
        raise OptimizationPilotError("complete candidate result history is malformed")
    _validate_history(candidate, history)
    if record.get("epochs_executed") != len(history):
        raise OptimizationPilotError("complete candidate epoch count is inconsistent")
    if not isinstance(record.get("best_epoch"), int) or not (
        1 <= int(record["best_epoch"]) <= len(history)
    ):
        raise OptimizationPilotError("complete candidate best epoch is invalid")
    objective = record.get("best_validation_objective")
    if not isinstance(objective, (int, float)) or not math.isfinite(float(objective)):
        raise OptimizationPilotError("complete candidate objective is invalid")
    best_record = history[int(record["best_epoch"]) - 1]
    if float(best_record["validation_selection_value"]) != float(objective):
        raise OptimizationPilotError(
            "complete candidate best objective does not match its best epoch"
        )
    for key in ("checkpoint_sha256", "log_sha256", "model_state_schema_sha256"):
        if not isinstance(record.get(key), str) or not _SHA256_PATTERN.fullmatch(
            str(record[key])
        ):
            raise OptimizationPilotError(f"complete candidate {key} is invalid")
    if record.get("parameter_count") != e7.EXPECTED_PARAMETERS[candidate.task]:
        raise OptimizationPilotError("complete candidate parameter count changed")
    selection_codes = record.get("validation_selection_class_codes")
    if not isinstance(selection_codes, list):
        raise OptimizationPilotError(
            "complete candidate validation-selection classes are malformed"
        )
    if candidate.task == "E2":
        if not selection_codes or not all(
            isinstance(code, int) and 0 <= code < 15 for code in selection_codes
        ):
            raise OptimizationPilotError(
                "E2 candidate lacks validation-selection class authority"
            )
    elif selection_codes:
        raise OptimizationPilotError(
            "non-E2 candidate unexpectedly records E2 selection classes"
        )


def _verify_live_artifacts(
    project_root: Path, candidate: PilotCandidate, record: Mapping[str, Any]
) -> None:
    root = Path(project_root).resolve()
    paths = candidate_paths(root, str(record.get("run_id")), candidate)
    expected_paths = {
        "checkpoint_path": _relative_path(root, paths.checkpoint),
        "log_path": _relative_path(root, paths.log),
        "candidate_result_path": _relative_path(root, paths.result),
    }
    for key, expected in expected_paths.items():
        if record.get(key) != expected:
            raise OptimizationPilotError(f"candidate {key} does not match its identity")
    if not paths.checkpoint.is_file() or not paths.log.is_file():
        raise OptimizationPilotError("candidate heavy artifacts are missing")
    if _sha256(paths.checkpoint) != record.get("checkpoint_sha256"):
        raise OptimizationPilotError("candidate checkpoint hash mismatch")
    if _sha256(paths.log) != record.get("log_sha256"):
        raise OptimizationPilotError("candidate log hash mismatch")
    _verify_checkpoint(
        candidate,
        paths,
        int(record["best_epoch"]),
        float(record["best_validation_objective"]),
    )


def _validate_fairness(records: Sequence[Mapping[str, Any]]) -> None:
    by_task: dict[str, list[Mapping[str, Any]]] = {task: [] for task in PILOT_TASKS}
    for record in records:
        by_task[str(record["task"])].append(record)
    for task, task_records in by_task.items():
        if len(task_records) != len(PILOT_LEARNING_RATES):
            raise OptimizationPilotError(f"{task} candidate coverage is incomplete")
        complete = [record for record in task_records if record.get("status") == "COMPLETE"]
        if not complete:
            continue
        baseline = complete[0]
        fairness_keys = (
            "loss_mode",
            "seed",
            "batch_size",
            "max_epochs",
            "max_train_batches_per_epoch",
            "max_validation_batches_per_epoch",
            "validation_selection",
        )
        for record in complete[1:]:
            if any(record.get(key) != baseline.get(key) for key in fairness_keys):
                raise OptimizationPilotError(f"{task} candidates have unequal budgets")
            left_history = baseline.get("history", [])
            right_history = record.get("history", [])
            if len(left_history) != len(right_history):
                raise OptimizationPilotError(f"{task} candidates have unequal epochs")
            for left, right in zip(left_history, right_history, strict=True):
                for key in ("train_order_sha256", "validation_order_sha256"):
                    if left.get(key) != right.get(key):
                        raise OptimizationPilotError(
                            f"{task} candidates used unequal {key} membership/order"
                        )


def _validate_stored_paths(
    records: Sequence[Mapping[str, Any]], run_id: str
) -> None:
    observed: list[str] = []
    candidates = enumerate_candidates()
    for candidate, record in zip(candidates, records, strict=True):
        expected = {
            "checkpoint_path": (
                f"checkpoints/stage_e8/{run_id}/{candidate.candidate_id}.pt"
            ),
            "log_path": f"logs/stage_e8/{run_id}/{candidate.candidate_id}.jsonl",
            "candidate_result_path": (
                f"logs/stage_e8/{run_id}/{candidate.candidate_id}.result.json"
            ),
        }
        for key, value in expected.items():
            if record.get(key) != value:
                raise OptimizationPilotError(
                    f"candidate stored {key} does not match its identity"
                )
            observed.append(value)
    if len(observed) != len(set(observed)):
        raise OptimizationPilotError("candidate stored paths are not unique")


def rank_task_records(records: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    """Rank complete same-task observations without finalizing a configuration."""

    if len(records) != len(PILOT_LEARNING_RATES):
        raise OptimizationPilotError("task ranking requires all approved candidates")
    if any(record.get("status") != "COMPLETE" for record in records):
        raise OptimizationPilotError("task ranking rejects failed candidates")
    tasks = {record.get("task") for record in records}
    if len(tasks) != 1:
        raise OptimizationPilotError("task ranking cannot mix tasks")
    lr_order = {value: index for index, value in enumerate(PILOT_LEARNING_RATES)}
    ordered = sorted(
        records,
        key=lambda record: (
            -float(record["best_validation_objective"]),
            lr_order[float(record["learning_rate"])],
        ),
    )
    return tuple(str(record["candidate_id"]) for record in ordered)


def _e2_validation_selection_class_codes(
    records: Sequence[Mapping[str, Any]],
) -> list[int]:
    observed = {
        tuple(record.get("validation_selection_class_codes", ()))
        for record in records
        if record.get("task") == "E2" and record.get("status") == "COMPLETE"
    }
    if len(observed) > 1:
        raise OptimizationPilotError(
            "E2 candidates used different validation-selection class sets"
        )
    return list(next(iter(observed), ()))


def build_observations(
    *,
    run_id: str,
    executor: str,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    candidates = enumerate_candidates()
    if len(records) != len(candidates):
        raise OptimizationPilotError("pilot result set is incomplete")
    for candidate, record in zip(candidates, records, strict=True):
        validate_candidate_record(record, candidate, allow_failed=True)
        if record.get("run_id") != run_id:
            raise OptimizationPilotError("pilot mixes run identifiers")
    _validate_fairness(records)
    all_complete = all(record.get("status") == "COMPLETE" for record in records)
    task_observations: dict[str, Any] = {}
    for task in PILOT_TASKS:
        task_records = [record for record in records if record.get("task") == task]
        if all(record.get("status") == "COMPLETE" for record in task_records):
            ranking = rank_task_records(task_records)
            task_observations[task] = {
                "status": "OBSERVED",
                "ranked_candidate_ids": list(ranking),
                "strongest_bounded_observation": ranking[0],
                "production_configuration_finalized": False,
            }
        else:
            task_observations[task] = {
                "status": "NOT_RANKED_DUE_TO_FAILURE",
                "ranked_candidate_ids": [],
                "strongest_bounded_observation": None,
                "production_configuration_finalized": False,
            }
    report: dict[str, Any] = {
        "stage": "E.8",
        "pilot_version": PILOT_VERSION,
        "scope": "bounded train/validation learning-rate optimization pilot",
        "run_id": _validate_run_id(run_id),
        "executor": executor,
        "candidate_results": [dict(record) for record in records],
        "task_observations": task_observations,
        "observational_ranking_rule": OBSERVATIONAL_RANKING_RULE,
        "production_configuration_finalized": False,
        "test_accessed": False,
        "test_metrics_recorded": False,
        "e2_validation_selection_policy": {
            "thresholds": {"min_train": 100, "min_val": 30},
            "validation_selection_class_codes": (
                _e2_validation_selection_class_codes(records)
            ),
            "test_labels_consulted": False,
            "test_support_consulted": False,
            "test_membership_consulted": False,
            "test_metadata_consulted": False,
            "final_reporting_eligibility_consulted": False,
        },
        "interpretation": (
            "Bounded pilot observations only; no multi-seed robustness, full-data "
            "training result, locked-test result, or production configuration."
        ),
        "overall_status": "PASS" if all_complete else "FAIL",
    }
    validate_observations(report)
    return report


def validate_observations(report: Mapping[str, Any]) -> None:
    errors: list[str] = []
    if report.get("stage") != "E.8" or report.get("pilot_version") != PILOT_VERSION:
        errors.append("stage or pilot version mismatch")
    try:
        _validate_run_id(str(report.get("run_id")))
    except ValueError as exc:
        errors.append(str(exc))
    if report.get("test_accessed") is not False:
        errors.append("observations indicate forbidden test access")
    if report.get("test_metrics_recorded") is not False:
        errors.append("observations contain test metrics")
    if report.get("production_configuration_finalized") is not False:
        errors.append("pilot observations incorrectly finalize a configuration")
    e2_policy = report.get("e2_validation_selection_policy", {})
    if (
        e2_policy.get("thresholds") != {"min_train": 100, "min_val": 30}
        or any(
            e2_policy.get(key) is not False
            for key in (
                "test_labels_consulted",
                "test_support_consulted",
                "test_membership_consulted",
                "test_metadata_consulted",
                "final_reporting_eligibility_consulted",
            )
        )
    ):
        errors.append("E2 validation-selection policy is not test-blind")
    records = report.get("candidate_results")
    candidates = enumerate_candidates()
    if not isinstance(records, list) or len(records) != len(candidates):
        errors.append("candidate result coverage is incomplete")
    else:
        try:
            for candidate, record in zip(candidates, records, strict=True):
                if not isinstance(record, Mapping):
                    raise OptimizationPilotError("candidate result is not a mapping")
                validate_candidate_record(record, candidate, allow_failed=True)
                if record.get("run_id") != report.get("run_id"):
                    raise OptimizationPilotError("pilot mixes run identifiers")
            _validate_fairness(records)
            _validate_stored_paths(records, str(report.get("run_id")))
            expected_codes = _e2_validation_selection_class_codes(records)
            if e2_policy.get("validation_selection_class_codes") != expected_codes:
                raise OptimizationPilotError(
                    "E2 validation-selection class evidence is inconsistent"
                )
            expected_task_observations: dict[str, Any] = {}
            for task in PILOT_TASKS:
                task_records = [
                    record for record in records if record.get("task") == task
                ]
                if all(
                    record.get("status") == "COMPLETE" for record in task_records
                ):
                    ranking = rank_task_records(task_records)
                    expected_task_observations[task] = {
                        "status": "OBSERVED",
                        "ranked_candidate_ids": list(ranking),
                        "strongest_bounded_observation": ranking[0],
                        "production_configuration_finalized": False,
                    }
                else:
                    expected_task_observations[task] = {
                        "status": "NOT_RANKED_DUE_TO_FAILURE",
                        "ranked_candidate_ids": [],
                        "strongest_bounded_observation": None,
                        "production_configuration_finalized": False,
                    }
            if report.get("task_observations") != expected_task_observations:
                raise OptimizationPilotError(
                    "task observations do not match deterministic ranking"
                )
        except (OptimizationPilotError, ValueError) as exc:
            errors.append(str(exc))
    if report.get("observational_ranking_rule") != OBSERVATIONAL_RANKING_RULE:
        errors.append("observational ranking rule changed")
    if not isinstance(report.get("executor"), str) or not report.get("executor"):
        errors.append("executor accounting is missing")
    if report.get("overall_status") not in {"PASS", "FAIL"}:
        errors.append("overall status is invalid")
    if isinstance(records, list):
        expected = (
            "PASS"
            if len(records) == len(candidates)
            and all(
                isinstance(record, Mapping) and record.get("status") == "COMPLETE"
                for record in records
            )
            else "FAIL"
        )
        if report.get("overall_status") != expected:
            errors.append("overall status disagrees with candidate completion")
    if errors:
        raise OptimizationPilotError("; ".join(errors))


def _preflight_run(
    project_root: Path, run_id: str, output_path: Path
) -> tuple[PilotCandidate, ...]:
    root = Path(project_root).resolve()
    candidates = enumerate_candidates()
    if Path(output_path).exists():
        raise OptimizationPilotError(
            f"refusing to overwrite existing observations: {output_path}"
        )
    all_paths: list[Path] = []
    for candidate in candidates:
        paths = candidate_paths(root, run_id, candidate)
        if paths.checkpoint.resolve() == paths.log.resolve():
            raise OptimizationPilotError("checkpoint and log paths alias")
        all_paths.extend((paths.checkpoint.resolve(), paths.log.resolve(), paths.result.resolve()))
    if len(set(all_paths)) != len(all_paths):
        raise OptimizationPilotError("candidate output paths are not unique")
    existing = [path for path in all_paths if path.exists()]
    if existing:
        raise OptimizationPilotError(
            f"fresh output location required; existing path: {existing[0]}"
        )
    reservation = root / "logs" / "stage_e8" / run_id / ".pilot-reservation"
    reservation.parent.mkdir(parents=True, exist_ok=True)
    try:
        reservation.open("x", encoding="ascii").close()
    except FileExistsError as exc:
        raise OptimizationPilotError(
            f"run_id has already been reserved: {run_id}"
        ) from exc
    return candidates


def run_pilot(
    *,
    project_root: Path,
    run_id: str,
    output_path: Path,
    device: str = "auto",
    executor: str = "Codex",
) -> dict[str, Any]:
    """Execute all nine bounded candidates and write one compact report."""

    root = Path(project_root).resolve()
    run_id = _validate_run_id(run_id)
    require_frozen_contracts(root)
    candidates = _preflight_run(root, run_id, output_path)
    records = [
        run_candidate(
            candidate,
            project_root=root,
            run_id=run_id,
            device=device,
            executor=executor,
        )
        for candidate in candidates
    ]
    report = build_observations(
        run_id=run_id,
        executor=executor,
        records=records,
    )
    _write_json_exclusive(output_path, report)
    return report
def aggregate_existing_run(
    *,
    project_root: Path,
    run_id: str,
    output_path: Path,
    executor: str,
) -> dict[str, Any]:
    """Aggregate only complete, identity-matching results from one run."""

    root = Path(project_root).resolve()
    run_id = _validate_run_id(run_id)
    require_frozen_contracts(root)
    if Path(output_path).exists():
        raise OptimizationPilotError(
            f"refusing to overwrite existing observations: {output_path}"
        )
    records: list[Mapping[str, Any]] = []
    for candidate in enumerate_candidates():
        paths = candidate_paths(root, run_id, candidate)
        if not paths.result.is_file():
            raise OptimizationPilotError(
                f"candidate result set is incomplete: {candidate.candidate_id}"
            )
        try:
            record = json.loads(paths.result.read_text(encoding="utf-8"))
        except Exception as exc:
            raise OptimizationPilotError(
                f"cannot load candidate result {paths.result}: {exc}"
            ) from exc
        if not isinstance(record, Mapping):
            raise OptimizationPilotError("candidate result is not a mapping")
        validate_candidate_record(record, candidate, allow_failed=False)
        _verify_live_artifacts(root, candidate, record)
        records.append(record)
    report = build_observations(
        run_id=run_id,
        executor=executor,
        records=records,
    )
    _write_json_exclusive(output_path, report)
    return report


__all__ = [
    "OptimizationPilotError",
    "PilotCandidate",
    "aggregate_existing_run",
    "build_observations",
    "enumerate_candidates",
    "new_run_id",
    "run_pilot",
    "validate_observations",
]
