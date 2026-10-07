"""Stage E.10 full train/validation orchestration.

This module is deliberately additive.  It delegates every optimization step,
validation metric, early-stopping decision, and checkpoint write to the frozen
Stage E.7/E.9 production harness.  E.10 owns only the full-run matrix,
validation-only learning-rate selection, run isolation, resource observation,
test-array lockout, and compact reporting.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import threading
import time
from typing import Any, Final, Iterator, Mapping, Sequence

import psutil
import torch
import torch_geometric

from src.graph import adapter as d4
from src.graph import schema as d2
from src.training import batch_size_pilot as e9
from src.training import e10_provenance_policy as provenance_policy
from src.training import harness as e7


STAGE: Final = "E.10"
FULL_TRAINING_VERSION: Final = "stage-e10-full-training-v2"
FROZEN_START_COMMIT: Final = "2fa823943ba318fe32b2867bc8e2b1f799d78ac1"
FROZEN_START_TAG: Final = "v0.12-stage-e9-model-binding-resource-pilot"
MODEL_FAMILIES: Final = tuple(e7.MODEL_FAMILIES)
TASKS: Final = tuple(e7.TASKS)
LEARNING_RATES: Final = tuple(e7.LEARNING_RATE_CANDIDATES)
SELECTION_SEED: Final = 2025
FINAL_SEEDS: Final = tuple(e7.APPROVED_SEEDS)
BATCH_SIZE: Final = 256
MAX_EPOCHS: Final = 30
LR_SELECTION_RULE: Final = (
    "higher_best_validation_objective_then_approved_learning_rate_order"
)
RUN_ID_PATTERN: Final = re.compile(r"^[a-z0-9][a-z0-9._-]{7,95}$")
OFFICIAL_STUDY_ID_PATTERN: Final = re.compile(
    r"^e10-official-[0-9]{8}-v[1-9][0-9]*$"
)
SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
LR_TOKENS: Final = {1e-4: "1e-4", 3e-4: "3e-4", 1e-3: "1e-3"}
RUN_CLASS_OFFICIAL: Final = "official_owner_execution"
EXECUTION_AUTHORITY: Final = "owner_manual_only"
PREAUTHORIZATION_DIAGNOSTIC_REASON: Final = (
    "pre-authorization diagnostic runs — excluded from official E.10 model selection"
)
EXCLUDED_DIAGNOSTIC_RUNS: Final = {
    "e10-20261001-fulltrain": PREAUTHORIZATION_DIAGNOSTIC_REASON,
}
RUN_STATUS_VALUES: Final = ("RUNNING", "COMPLETE", "INTERRUPTED", "FAILED")

UPSTREAM_EVIDENCE_PATHS: Final = (
    "results/stage_d1/feature_taxonomy.json",
    "results/stage_d2/graph_schema.json",
    "results/stage_d3/structural_validation.json",
    "results/stage_d4/lazy_adapter_validation.json",
    "results/stage_d5/experiment_boundary_validation.json",
    "results/stage_d6/template_batching_validation.json",
    "results/stage_d7/construction_sanity_validation.json",
    "results/stage_d8/full_split_integration.json",
    "results/stage_e1/environment-report.json",
    "results/stage_e1/environment-report.sha256",
    "results/stage_e2/heterodata_adapter_validation.json",
    "results/stage_e2/heterodata_adapter_validation.sha256",
    "results/stage_e2/provenance_cache_validation.json",
    "results/stage_e2/provenance_cache_validation.sha256",
    "results/stage_e2/provenance_cache_observations.json",
    "results/stage_e3/feature_encoder_validation.json",
    "results/stage_e3/feature_encoder_validation.sha256",
    "results/stage_e4/hierarchical_encoder_validation.json",
    "results/stage_e4/hierarchical_encoder_validation.sha256",
    "results/stage_e5/htgnn_classifier_validation.json",
    "results/stage_e5/htgnn_classifier_validation.sha256",
    "results/stage_e6/htgnn_resource_pilot_contract.json",
    "results/stage_e6/htgnn_resource_pilot_contract.sha256",
    "results/stage_e6/htgnn_resource_pilot_observations.json",
    "results/stage_e7/htgnn_training_harness_validation.json",
    "results/stage_e7/htgnn_training_harness_validation.sha256",
    "results/stage_e8/htgnn_optimization_pilot_contract.json",
    "results/stage_e8/htgnn_optimization_pilot_contract.sha256",
    "results/stage_e8/htgnn_optimization_pilot_observations.json",
    "results/stage_e9/stage_e9_contract.json",
    "results/stage_e9/stage_e9_contract.sha256",
    "results/stage_e9/htgnn_batch_size_pilot_observations.json",
)

E10_SOURCE_PATHS: Final = (
    "src/training/full_training.py",
    "scripts/run_stage_e10_full_training.py",
    "tests/test_stage_e10_full_training.py",
    "docs/stage_e10_full_training.md",
)


class FullTrainingError(RuntimeError):
    """Raised when an E.10 execution or evidence contract is violated."""


@dataclass(frozen=True, slots=True)
class FullTrainingCandidate:
    """One isolated model-family, task, seed, and LR training run."""

    model_family: str
    task: str
    learning_rate: float
    seed: int
    phase: str

    def __post_init__(self) -> None:
        if self.model_family not in MODEL_FAMILIES:
            raise ValueError(f"model_family must be one of {MODEL_FAMILIES}")
        if self.task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}")
        if self.learning_rate not in LEARNING_RATES:
            raise ValueError(f"learning_rate must be one of {LEARNING_RATES}")
        if self.seed not in FINAL_SEEDS:
            raise ValueError(f"seed must be one of {FINAL_SEEDS}")
        if self.phase not in {"lr_selection", "final_seed"}:
            raise ValueError("phase must be lr_selection or final_seed")
        if self.phase == "lr_selection" and self.seed != SELECTION_SEED:
            raise ValueError("learning-rate selection must use seed 2025")
        if self.phase == "final_seed" and self.seed == SELECTION_SEED:
            raise ValueError("seed 2025 winner is reused; final_seed must use 2026 or 2027")

    @property
    def training_config(self) -> e7.TrainingConfig:
        return e7.TrainingConfig(
            task=self.task,
            learning_rate=self.learning_rate,
            seed=self.seed,
            batch_size=BATCH_SIZE,
            max_epochs=MAX_EPOCHS,
            max_train_batches=None,
            max_validation_batches=None,
        )

    def config_dict(self) -> dict[str, Any]:
        return {
            "stage": STAGE,
            "version": FULL_TRAINING_VERSION,
            "run_class": RUN_CLASS_OFFICIAL,
            "execution_authority": EXECUTION_AUTHORITY,
            "phase": self.phase,
            "model_family": self.model_family,
            "task": self.task,
            "seed": self.seed,
            "learning_rate": self.learning_rate,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "max_train_batches": None,
            "max_validation_batches": None,
            "optimizer": "AdamW",
            "weight_decay": e7.WEIGHT_DECAY,
            "scheduler": None,
            "loss": (
                "CrossEntropyLoss"
                if self.task in {"E1", "E2"}
                else "BCEWithLogitsLoss"
            ),
            "class_weighting": False,
            "validation_selection": e7.SELECTION_NAMES[self.task],
            "minimum_epochs": e7.MIN_EPOCHS,
            "patience": e7.PATIENCE,
            "min_delta": e7.MIN_DELTA,
            "train_order": "seeded bounded affine permutation per epoch",
            "validation_order": "sequential",
            "allowed_partitions": ["train", "validation"],
            "test_predictive_array_access": False,
            "pruning": False,
        }

    @property
    def config_sha256(self) -> str:
        return d2.canonical_sha256(self.config_dict())

    @property
    def candidate_id(self) -> str:
        return (
            f"e10-{self.model_family}-{self.task.lower()}-"
            f"lr{LR_TOKENS[self.learning_rate]}-s{self.seed}-b{BATCH_SIZE}-"
            f"{self.config_sha256[:12]}"
        )


@dataclass(frozen=True, slots=True)
class CandidatePaths:
    checkpoint: Path
    log: Path
    manifest: Path
    training_history: Path
    validation_history: Path
    result: Path
    status: Path


@dataclass(frozen=True, slots=True)
class ReportPaths:
    final_report: Path
    final_report_sha256: Path
    lr_selection_report: Path
    selected_learning_rates: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None


def _relative_path(project_root: Path, path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(Path(project_root).resolve()).as_posix()
    except ValueError as exc:
        raise FullTrainingError(f"E.10 artifact is outside the project root: {path}") from exc


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(d2.canonical_json_bytes(dict(payload)).decode("utf-8"))
            handle.write("\n")
    except FileExistsError as exc:
        raise FullTrainingError(f"E.10 output already exists: {target}") from exc


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise FullTrainingError(f"cannot read E.10 JSON {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise FullTrainingError(f"E.10 JSON root must be a mapping: {path}")
    return payload


def validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError(
            "run_id must be 8-96 lowercase alphanumeric, dot, underscore, or "
            "hyphen characters and begin with an alphanumeric character"
        )
    return run_id


def validate_official_study_id(official_study_id: str) -> str:
    """Accept only owner-controlled official identities, never diagnostic IDs."""

    validate_run_id(official_study_id)
    if official_study_id in EXCLUDED_DIAGNOSTIC_RUNS:
        raise FullTrainingError(EXCLUDED_DIAGNOSTIC_RUNS[official_study_id])
    if OFFICIAL_STUDY_ID_PATTERN.fullmatch(official_study_id) is None:
        raise ValueError(
            "official_study_id must match e10-official-YYYYMMDD-vN"
        )
    return official_study_id


def new_run_id(*, revision: int = 1) -> str:
    """Return a suggested official study ID; owners must check it is fresh."""

    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
        raise ValueError("revision must be a positive integer")
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"e10-official-{date}-v{revision}"


def enumerate_lr_selection_candidates() -> tuple[FullTrainingCandidate, ...]:
    """Return exactly 27 seed-2025 full-training LR candidates."""

    return tuple(
        FullTrainingCandidate(family, task, lr, SELECTION_SEED, "lr_selection")
        for family in MODEL_FAMILIES
        for task in TASKS
        for lr in LEARNING_RATES
    )


def candidate_paths(
    project_root: Path,
    official_study_id: str,
    candidate: FullTrainingCandidate,
) -> CandidatePaths:
    root = Path(project_root).resolve()
    validate_official_study_id(official_study_id)
    relative = (
        Path(official_study_id)
        / candidate.phase
        / candidate.task.lower()
        / candidate.model_family
        / candidate.candidate_id
    )
    runtime = root / "logs" / "stage_e10" / relative
    return CandidatePaths(
        checkpoint=root / "checkpoints" / "stage_e10" / relative / "best.pt",
        log=runtime / "epochs.jsonl",
        manifest=runtime / "manifest.json",
        training_history=runtime / "training_history.json",
        validation_history=runtime / "validation_history.json",
        result=runtime / "result.json",
        status=runtime / "status.json",
    )


def report_paths(project_root: Path, official_study_id: str) -> ReportPaths:
    root = Path(project_root).resolve()
    validate_official_study_id(official_study_id)
    directory = root / "results" / "stage_e10" / official_study_id
    return ReportPaths(
        final_report=directory / "stage_e10_report.json",
        final_report_sha256=directory / "stage_e10_report.sha256",
        lr_selection_report=directory / "lr_selection_report.json",
        selected_learning_rates=directory / "selected_learning_rates.json",
    )


def _status_payload(
    official_study_id: str,
    candidate: FullTrainingCandidate,
    state: str,
    *,
    epochs_completed: int,
    paths: CandidatePaths,
    detail: str | None = None,
) -> dict[str, Any]:
    if state not in RUN_STATUS_VALUES:
        raise ValueError(f"invalid E.10 run state: {state}")
    payload: dict[str, Any] = {
        "stage": STAGE,
        "version": FULL_TRAINING_VERSION,
        "official_study_id": validate_official_study_id(official_study_id),
        "run_class": RUN_CLASS_OFFICIAL,
        "execution_authority": EXECUTION_AUTHORITY,
        "candidate_id": candidate.candidate_id,
        "phase": candidate.phase,
        "state": state,
        "epochs_completed": int(epochs_completed),
        "artifacts_present": {
            "checkpoint": paths.checkpoint.is_file(),
            "log": paths.log.is_file(),
            "manifest": paths.manifest.is_file(),
            "training_history": paths.training_history.is_file(),
            "validation_history": paths.validation_history.is_file(),
            "result": paths.result.is_file(),
        },
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if detail:
        payload["detail"] = detail
    payload["status_sha256"] = d2.canonical_sha256(payload)
    return payload


def _write_status(path: Path, payload: Mapping[str, Any], *, fresh: bool) -> None:
    """Create or atomically replace only the lifecycle status record."""

    target = Path(path)
    if fresh:
        _write_json_exclusive(target, payload)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(d2.canonical_json_bytes(dict(payload)).decode("utf-8"))
            handle.write("\n")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _completed_epoch_count(log_path: Path) -> int:
    if not Path(log_path).is_file():
        return 0
    try:
        return sum(
            1
            for line in Path(log_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    except (OSError, UnicodeError):
        return 0


def _git(project_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=project_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise FullTrainingError(
            f"git {' '.join(args)} failed: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def source_git_identity(project_root: Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    head = _git(root, "rev-parse", "HEAD")
    origin = _git(root, "rev-parse", "origin/main")
    tags = tuple(line for line in _git(root, "tag", "--points-at", "HEAD").splitlines() if line)
    status = tuple(line for line in _git(root, "status", "--short").splitlines() if line)
    if head != FROZEN_START_COMMIT or origin != FROZEN_START_COMMIT:
        raise FullTrainingError(
            f"frozen Git identity changed: HEAD={head}, origin/main={origin}"
        )
    if FROZEN_START_TAG not in tags:
        raise FullTrainingError("frozen Stage E.9 tag does not point at HEAD")
    return {
        "head": head,
        "origin_main": origin,
        "tags_at_head": list(tags),
        "dirty": bool(status),
        "status_short": list(status),
    }


def environment_identity() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": cuda_available,
        "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
        "gpu_total_memory_bytes": (
            int(torch.cuda.get_device_properties(0).total_memory)
            if cuda_available
            else None
        ),
        "platform": platform.platform(),
    }


def _evidence_hashes(project_root: Path) -> dict[str, str]:
    root = Path(project_root).resolve()
    hashes: dict[str, str] = {}
    for relative in UPSTREAM_EVIDENCE_PATHS:
        path = root / relative
        if not path.is_file():
            raise FullTrainingError(f"frozen upstream evidence is missing: {relative}")
        hashes[relative] = _sha256(path)
    return hashes


def _e10_source_hashes(project_root: Path) -> dict[str, str]:
    root = Path(project_root).resolve()
    hashes: dict[str, str] = {}
    for relative in E10_SOURCE_PATHS:
        path = root / relative
        if not path.is_file():
            raise FullTrainingError(f"E.10 source is missing: {relative}")
        hashes[relative] = _sha256(path)
    return hashes


def _validate_frozen_e9_evidence(project_root: Path) -> Mapping[str, Any]:
    root = Path(project_root).resolve()
    contract_path = root / "results" / "stage_e9" / "stage_e9_contract.json"
    report = _read_json(contract_path)
    unhashed = dict(report)
    stored_canonical = unhashed.pop("report_sha256", None)
    if stored_canonical != d2.canonical_sha256(unhashed):
        raise FullTrainingError("frozen E.9 canonical evidence hash is invalid")
    sidecar = (
        root / "results" / "stage_e9" / "stage_e9_contract.sha256"
    ).read_text(encoding="utf-8").split()[0]
    if not _valid_sha256(sidecar) or sidecar != _sha256(contract_path):
        raise FullTrainingError("frozen E.9 physical evidence hash is invalid")
    observations = report.get("batch_size_pilot", {}).get("observations_physical_sha256")
    if not _valid_sha256(observations):
        raise FullTrainingError("frozen E.9 observation binding is missing")
    e9.validate_observation_file(
        root / "results" / "stage_e9" / "htgnn_batch_size_pilot_observations.json",
        project_root=root,
        expected_sha256=str(observations),
        verify_live_device=True,
    )
    if report.get("finalized", {}).get("htgnn_training_batch_size") != BATCH_SIZE:
        raise FullTrainingError("frozen E.9 training batch size is not 256")
    if report.get("not_finalized") != [
        "learning_rate",
        "weighted_loss_configuration",
        "task_performance",
        "final_model_checkpoint",
        "test_performance",
    ]:
        raise FullTrainingError("frozen E.9 limitation statement changed")
    return report


def preflight(project_root: Path) -> dict[str, Any]:
    """Reproduce the frozen identity, source, evidence, and data checks once."""

    root = Path(project_root).resolve()
    git = source_git_identity(root)
    e9_contract = _validate_frozen_e9_evidence(root)
    frozen_sources = e9.require_frozen_upstream(root)
    expected_e9_sources = e9_contract.get("e9_source_sha256")
    if not isinstance(expected_e9_sources, Mapping):
        raise FullTrainingError("frozen E.9 source-hash authority is missing")
    for relative, expected in expected_e9_sources.items():
        path = root / str(relative)
        if not path.is_file() or _sha256(path) != expected:
            raise FullTrainingError(f"frozen E.9 source changed: {relative}")
    integrity = d4.verify_frozen_integrity(root)
    if not integrity["stage_c_sha256_matches"] or not integrity["raw_csv_sha256_matches"]:
        raise FullTrainingError("frozen Stage C or raw CSV integrity failed")
    result = {
        "stage": STAGE,
        "version": FULL_TRAINING_VERSION,
        "git": git,
        "environment": environment_identity(),
        "frozen_sources": frozen_sources,
        "upstream_evidence_sha256": _evidence_hashes(root),
        "e10_source_sha256": _e10_source_hashes(root),
        "stage_c_integrity": {
            "status": "PASS",
            "matched": integrity["stage_c_artifact_count"],
            "expected": integrity["stage_c_expected_count"],
        },
        "raw_csv_integrity": {
            "status": "PASS",
            "matched": integrity["raw_csv_count"],
            "expected": integrity["raw_csv_expected_count"],
        },
        "stage_e9_contract_canonical_sha256": e9_contract["report_sha256"],
        "stage_e9_contract_physical_sha256": _sha256(
            root / "results" / "stage_e9" / "stage_e9_contract.json"
        ),
        "stage_e9_observations_physical_sha256": e9_contract[
            "batch_size_pilot"
        ]["observations_physical_sha256"],
        "test_predictive_array_access": False,
    }
    result["preflight_sha256"] = d2.canonical_sha256(result)
    return result


def _test_array_paths(project_root: Path) -> frozenset[Path]:
    processed = Path(project_root).resolve() / "data" / "processed" / "preprocessed"
    return frozenset(
        (processed / experiment / name).resolve()
        for experiment in ("exp1", "exp2", "exp3")
        for name in ("X_test.npy", "y_test.npy")
    )


_AUDIT_LOCK = threading.Lock()
_AUDIT_INSTALLED = False
_ACTIVE_FORBIDDEN: frozenset[Path] = frozenset()
_ACTIVE_ACCESSES: list[str] | None = None


def _audit_hook(event: str, args: tuple[Any, ...]) -> None:
    if event != "open" or not args:
        return
    raw = args[0]
    if not isinstance(raw, (str, bytes, os.PathLike)):
        return
    try:
        path = Path(os.fsdecode(raw)).resolve()
    except (OSError, TypeError, ValueError):
        return
    with _AUDIT_LOCK:
        forbidden = _ACTIVE_FORBIDDEN
        accesses = _ACTIVE_ACCESSES
    if path in forbidden:
        if accesses is not None:
            accesses.append(str(path))
        raise FullTrainingError(f"predictive test-array access blocked: {path}")


def _ensure_audit_hook() -> None:
    global _AUDIT_INSTALLED
    with _AUDIT_LOCK:
        if not _AUDIT_INSTALLED:
            sys.addaudithook(_audit_hook)
            _AUDIT_INSTALLED = True


@contextmanager
def predictive_test_lockout(project_root: Path) -> Iterator[list[str]]:
    """Fail immediately if any E1/E2/E3 X_test or y_test array is opened."""

    global _ACTIVE_FORBIDDEN, _ACTIVE_ACCESSES
    _ensure_audit_hook()
    accesses: list[str] = []
    with _AUDIT_LOCK:
        if _ACTIVE_ACCESSES is not None:
            raise FullTrainingError("a predictive test lockout is already active")
        _ACTIVE_FORBIDDEN = _test_array_paths(project_root)
        _ACTIVE_ACCESSES = accesses
    try:
        yield accesses
    finally:
        with _AUDIT_LOCK:
            _ACTIVE_FORBIDDEN = frozenset()
            _ACTIVE_ACCESSES = None


def _print_informational(message: str) -> None:
    """Emit best-effort operator output without affecting training semantics."""

    try:
        print(message, flush=True)
    except (BrokenPipeError, OSError, ValueError):
        pass


class _RuntimeMonitor:
    """Observe RSS and report only complete authoritative epoch-log records."""

    def __init__(
        self,
        log_path: Path,
        candidate: FullTrainingCandidate,
        poll_seconds: float = 0.05,
    ) -> None:
        self.log_path = Path(log_path)
        self.candidate = candidate
        self.poll_seconds = poll_seconds
        self.started = time.perf_counter()
        self.peak_rss_bytes = int(psutil.Process().memory_info().rss)
        self.epoch_completed_seconds: list[float] = []
        self._seen_lines = 0
        self._best_metric: float | None = None
        self._patience_count = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _sample(self) -> None:
        self.peak_rss_bytes = max(
            self.peak_rss_bytes, int(psutil.Process().memory_info().rss)
        )
        if not self.log_path.is_file():
            return
        try:
            lines = [
                line
                for line in self.log_path.read_text(encoding="utf-8").splitlines(
                    keepends=True
                )
                if line.strip() and line.endswith("\n")
            ]
        except (OSError, UnicodeError):
            return
        while self._seen_lines < len(lines):
            try:
                record = json.loads(lines[self._seen_lines])
            except (json.JSONDecodeError, TypeError):
                return
            if not isinstance(record, Mapping):
                return
            now = time.perf_counter() - self.started
            previous = (
                self.epoch_completed_seconds[-1]
                if self.epoch_completed_seconds
                else 0.0
            )
            self.epoch_completed_seconds.append(now)
            self._seen_lines += 1
            if record.get("best_checkpoint_updated") is True:
                self._best_metric = float(record["validation_selection_value"])
                self._patience_count = 0
            else:
                if self._best_metric is None:
                    self._best_metric = float(record["validation_selection_value"])
                self._patience_count += 1
            _print_informational(
                self._progress_line(
                    record,
                    epoch_elapsed_seconds=now - previous,
                )
            )

    def _progress_line(
        self,
        record: Mapping[str, Any],
        *,
        epoch_elapsed_seconds: float,
    ) -> str:
        return (
            f"[{self.candidate.task}]"
            f"[{self.candidate.model_family}]"
            f"[lr={self.candidate.learning_rate:g}]"
            f"[seed={self.candidate.seed}] "
            f"Epoch {int(record['epoch'])}/{MAX_EPOCHS} | "
            f"train_loss={float(record['train_loss']):.6f} | "
            f"val_loss={float(record['validation_loss']):.6f} | "
            f"{record['validation_selection_name']}="
            f"{float(record['validation_selection_value']):.6f} | "
            f"best={float(self._best_metric):.6f} | "
            f"best_updated={'yes' if record['best_checkpoint_updated'] else 'no'} | "
            f"patience={self._patience_count}/{e7.PATIENCE} | "
            f"epoch_elapsed={epoch_elapsed_seconds:.3f}s"
        )

    def _loop(self) -> None:
        while not self._stop.wait(self.poll_seconds):
            self._sample()

    def finish(self) -> tuple[float, ...]:
        self._stop.set()
        self._thread.join(timeout=5)
        self._sample()
        return tuple(self.epoch_completed_seconds)


def _manifest(
    project_root: Path,
    official_study_id: str,
    candidate: FullTrainingCandidate,
    preflight_result: Mapping[str, Any],
) -> dict[str, Any]:
    model = e7.create_model(candidate.task, candidate.model_family)
    identity = e7.model_contract(model)
    manifest = {
        "stage": STAGE,
        "version": FULL_TRAINING_VERSION,
        "official_study_id": validate_official_study_id(official_study_id),
        "run_class": RUN_CLASS_OFFICIAL,
        "execution_authority": EXECUTION_AUTHORITY,
        "phase": candidate.phase,
        "candidate_id": candidate.candidate_id,
        "configuration": candidate.config_dict(),
        "config_sha256": candidate.config_sha256,
        "model_config_sha256": e7.model_state_schema_sha256(model),
        "model_contract": {
            "family": identity.family,
            "architecture_version": identity.architecture_version,
            "input_contract": identity.input_contract,
            "task": identity.task,
            "parameter_count": identity.parameter_count,
        },
        "source_git_identity": preflight_result["git"],
        "environment_identity": preflight_result["environment"],
        "stage_c_provenance": {
            "integrity": preflight_result["stage_c_integrity"],
            "split_artifact_sha256": {
                f"exp{index}/split_metadata.json": _sha256(
                    Path(project_root)
                    / "data"
                    / "processed"
                    / "preprocessed"
                    / f"exp{index}"
                    / "split_metadata.json"
                )
                for index in (1, 2, 3)
            },
        },
        "upstream_evidence_sha256": preflight_result["upstream_evidence_sha256"],
        "e10_source_sha256": preflight_result["e10_source_sha256"],
        "preflight_sha256": preflight_result["preflight_sha256"],
        "test_lockout": {
            "forbidden_predictive_arrays": [
                _relative_path(project_root, path)
                for path in sorted(_test_array_paths(project_root))
            ],
            "enforcement": "python_audit_hook_fail_closed",
        },
    }
    manifest["manifest_sha256"] = d2.canonical_sha256(manifest)
    return manifest


def _load_log(path: Path) -> list[Mapping[str, Any]]:
    try:
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except Exception as exc:
        raise FullTrainingError(f"cannot read E.10 JSONL log {path}: {exc}") from exc
    if not records or not all(isinstance(record, Mapping) for record in records):
        raise FullTrainingError("E.10 JSONL log is empty or malformed")
    return records


def _derived_histories(
    history: Sequence[Mapping[str, Any]], epoch_completed: Sequence[float]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[float]]:
    if len(history) != len(epoch_completed):
        raise FullTrainingError(
            "epoch timing count does not match the frozen training history"
        )
    previous = 0.0
    durations: list[float] = []
    train_history: list[dict[str, Any]] = []
    validation_history: list[dict[str, Any]] = []
    for record, completed in zip(history, epoch_completed, strict=True):
        duration = float(completed - previous)
        previous = float(completed)
        if not math.isfinite(duration) or duration <= 0:
            raise FullTrainingError("observed epoch duration is not finite and positive")
        durations.append(duration)
        shared = {
            "epoch": record["epoch"],
            "learning_rate": record["learning_rate"],
            "epoch_elapsed_seconds": round(duration, 6),
        }
        train_history.append(
            {
                **shared,
                "loss": record["train_loss"],
                "batches": record["train_batches"],
                "examples": record["train_examples"],
                "order_sha256": record["train_order_sha256"],
            }
        )
        validation_history.append(
            {
                **shared,
                "loss": record["validation_loss"],
                "selection_name": record["validation_selection_name"],
                "selection_value": record["validation_selection_value"],
                "selection_support_complete": record[
                    "validation_selection_support_complete"
                ],
                "batches": record["validation_batches"],
                "examples": record["validation_examples"],
                "order_sha256": record["validation_order_sha256"],
                "best_checkpoint_updated": record["best_checkpoint_updated"],
            }
        )
    return train_history, validation_history, durations


def execute_candidate(
    project_root: Path,
    official_study_id: str,
    candidate: FullTrainingCandidate,
    *,
    preflight_result: Mapping[str, Any],
    device: str = "cuda",
    owner_confirmed_manual_execution: bool = False,
) -> Mapping[str, Any]:
    """Execute exactly one owner-confirmed job through the frozen harness."""

    root = Path(project_root).resolve()
    validate_official_study_id(official_study_id)
    if not owner_confirmed_manual_execution:
        raise FullTrainingError(
            "training is owner-only; pass explicit owner confirmation for one job"
        )
    paths = candidate_paths(root, official_study_id, candidate)
    all_paths = (
        paths.checkpoint,
        paths.log,
        paths.manifest,
        paths.training_history,
        paths.validation_history,
        paths.result,
        paths.status,
    )
    existing = [str(path) for path in all_paths if path.exists()]
    if existing:
        raise FullTrainingError(f"candidate paths are not fresh: {existing}")

    manifest = _manifest(root, official_study_id, candidate, preflight_result)
    _write_json_exclusive(paths.manifest, manifest)
    _write_status(
        paths.status,
        _status_payload(
            official_study_id,
            candidate,
            "RUNNING",
            epochs_completed=0,
            paths=paths,
        ),
        fresh=True,
    )

    monitor: _RuntimeMonitor | None = None
    monitor_finished = False
    epoch_completed: tuple[float, ...] = ()
    started = time.perf_counter()
    test_accesses: list[str] = []
    try:
        if device == "cuda":
            if not torch.cuda.is_available():
                raise FullTrainingError("CUDA is required for the authorized E.10 run")
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        monitor = _RuntimeMonitor(paths.log, candidate)
        monitor.start()
        _print_informational(
            "Starting official E.10 job: "
            f"experiment={candidate.task} family={candidate.model_family} "
            f"lr={candidate.learning_rate:g} seed={candidate.seed} "
            f"batch={BATCH_SIZE} device={device}"
        )
        with predictive_test_lockout(root) as test_accesses:
            training_result = e7.run_training(
                candidate.training_config,
                project_root=root,
                checkpoint_path=paths.checkpoint,
                log_path=paths.log,
                device=device,
                model_family=candidate.model_family,
            )
        epoch_completed = monitor.finish()
        monitor_finished = True
        elapsed_seconds = time.perf_counter() - started
        if test_accesses:
            raise FullTrainingError("predictive test-array access occurred")

        log_records = _load_log(paths.log)
        history = [dict(record) for record in training_result.history]
        if log_records != history:
            raise FullTrainingError("frozen JSONL log does not match returned history")
        train_history, validation_history, durations = _derived_histories(
            history, epoch_completed
        )
        _write_json_exclusive(
            paths.training_history,
            {"candidate_id": candidate.candidate_id, "history": train_history},
        )
        _write_json_exclusive(
            paths.validation_history,
            {"candidate_id": candidate.candidate_id, "history": validation_history},
        )

        verification_model = e7.create_model(candidate.task, candidate.model_family)
        metadata = e7.load_checkpoint(
            paths.checkpoint,
            model=verification_model,
            expected_task=candidate.task,
            expected_config=candidate.training_config,
            map_location="cpu",
        )
        if (
            metadata.get("epoch") != training_result.best_epoch
            or metadata.get("selection_value") != training_result.best_selection_value
            or metadata.get("model_state_schema_sha256")
            != manifest["model_config_sha256"]
        ):
            raise FullTrainingError("best checkpoint provenance does not match the run")

        examples_per_epoch = [
            int(record["train_examples"]) + int(record["validation_examples"])
            for record in history
        ]
        throughput = [
            examples / duration
            for examples, duration in zip(examples_per_epoch, durations, strict=True)
        ]
        steady_values = throughput[1:] if len(throughput) > 1 else throughput
        resource = {
            "measurement_scope": "end_to_end_train_plus_validation_epoch",
            "elapsed_seconds": round(elapsed_seconds, 6),
            "epoch_elapsed_seconds": [round(value, 6) for value in durations],
            "epoch_end_to_end_windows_per_second": [
                round(value, 6) for value in throughput
            ],
            "steady_state_windows_per_second": round(
                statistics.median(steady_values), 6
            ),
            "steady_state_rule": "median end-to-end epoch throughput excluding epoch 1",
            "peak_cpu_rss_bytes": monitor.peak_rss_bytes,
            "peak_gpu_allocated_bytes": (
                int(torch.cuda.max_memory_allocated()) if device == "cuda" else None
            ),
            "peak_gpu_reserved_bytes": (
                int(torch.cuda.max_memory_reserved()) if device == "cuda" else None
            ),
            "checkpoint_size_bytes": paths.checkpoint.stat().st_size,
        }
        best_record = history[training_result.best_epoch - 1]
        result: dict[str, Any] = {
            "stage": STAGE,
            "version": FULL_TRAINING_VERSION,
            "official_study_id": official_study_id,
            "run_class": RUN_CLASS_OFFICIAL,
            "execution_authority": EXECUTION_AUTHORITY,
            "candidate_id": candidate.candidate_id,
            "phase": candidate.phase,
            "status": "PASS",
            "configuration": candidate.config_dict(),
            "config_sha256": candidate.config_sha256,
            "model_config_sha256": manifest["model_config_sha256"],
            "epochs_completed": training_result.epochs_completed,
            "stopped_early": training_result.stopped_early,
            "early_stopping_reason": (
                "patience_exhausted"
                if training_result.stopped_early
                else "max_epochs_reached"
            ),
            "best_epoch": training_result.best_epoch,
            "best_validation_metric": training_result.best_selection_value,
            "validation_selection_name": training_result.selection_name,
            "best_validation_loss": best_record["validation_loss"],
            "best_epoch_training_loss": best_record["train_loss"],
            "parameter_update_observed": training_result.parameter_update_observed,
            "device": training_result.device,
            "history": history,
            "resource_observation": resource,
            "artifacts": {
                "checkpoint": _relative_path(root, paths.checkpoint),
                "checkpoint_sha256": _sha256(paths.checkpoint),
                "log": _relative_path(root, paths.log),
                "log_sha256": _sha256(paths.log),
                "manifest": _relative_path(root, paths.manifest),
                "manifest_sha256": _sha256(paths.manifest),
                "training_history": _relative_path(root, paths.training_history),
                "training_history_sha256": _sha256(paths.training_history),
                "validation_history": _relative_path(root, paths.validation_history),
                "validation_history_sha256": _sha256(paths.validation_history),
            },
            "checkpoint_provenance": dict(metadata),
            "checkpoint_provenance_verified": True,
            "test_predictive_arrays_opened": [],
            "test_accessed": False,
            "test_metrics_recorded": False,
            "threshold_selected": False,
            "pruning_performed": False,
            "failed_or_retried": False,
        }
        result["result_sha256"] = d2.canonical_sha256(result)
        _write_json_exclusive(paths.result, result)
        _write_status(
            paths.status,
            _status_payload(
                official_study_id,
                candidate,
                "COMPLETE",
                epochs_completed=training_result.epochs_completed,
                paths=paths,
            ),
            fresh=False,
        )
        _print_informational(
            f"COMPLETE | epochs={training_result.epochs_completed} | "
            f"best_epoch={training_result.best_epoch} | "
            f"best_metric={training_result.best_selection_value:.6f} | "
            f"elapsed={elapsed_seconds:.3f}s"
        )
        return result
    except KeyboardInterrupt:
        if monitor is not None and not monitor_finished:
            monitor.finish()
        _write_status(
            paths.status,
            _status_payload(
                official_study_id,
                candidate,
                "INTERRUPTED",
                epochs_completed=_completed_epoch_count(paths.log),
                paths=paths,
                detail="owner_interrupt",
            ),
            fresh=False,
        )
        raise
    except Exception as exc:
        if monitor is not None and not monitor_finished:
            monitor.finish()
        _write_status(
            paths.status,
            _status_payload(
                official_study_id,
                candidate,
                "FAILED",
                epochs_completed=_completed_epoch_count(paths.log),
                paths=paths,
                detail=type(exc).__name__,
            ),
            fresh=False,
        )
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            raise FullTrainingError(
                "CUDA OOM during batch-256 full training; do not reduce batch size"
            ) from exc
        raise


def validate_candidate_result(
    project_root: Path,
    official_study_id: str,
    candidate: FullTrainingCandidate,
    record: Mapping[str, Any],
    *,
    verify_live_artifacts: bool,
) -> None:
    validate_official_study_id(official_study_id)
    if record.get("stage") != STAGE or record.get("version") != FULL_TRAINING_VERSION:
        raise FullTrainingError("candidate result stage/version identity is invalid")
    if record.get("status") != "PASS" or record.get("candidate_id") != candidate.candidate_id:
        raise FullTrainingError("candidate result identity or status is invalid")
    if (
        record.get("official_study_id") != official_study_id
        or record.get("run_class") != RUN_CLASS_OFFICIAL
        or record.get("execution_authority") != EXECUTION_AUTHORITY
        or record.get("phase") != candidate.phase
    ):
        raise FullTrainingError("candidate is not an official owner-executed result")
    if record.get("configuration") != candidate.config_dict():
        raise FullTrainingError("candidate result configuration mismatch")
    if record.get("config_sha256") != candidate.config_sha256:
        raise FullTrainingError("candidate result config hash mismatch")
    if record.get("test_accessed") is not False or record.get("test_metrics_recorded") is not False:
        raise FullTrainingError("candidate result violates the test lockout")
    if record.get("threshold_selected") is not False or record.get("pruning_performed") is not False:
        raise FullTrainingError("candidate result contains downstream work")
    unhashed = dict(record)
    stored_hash = unhashed.pop("result_sha256", None)
    if stored_hash != d2.canonical_sha256(unhashed):
        raise FullTrainingError("candidate result canonical hash is invalid")
    history = record.get("history")
    epochs = record.get("epochs_completed")
    if not isinstance(history, list) or not isinstance(epochs, int) or len(history) != epochs:
        raise FullTrainingError("candidate history length is invalid")
    best_epoch = record.get("best_epoch")
    if not isinstance(best_epoch, int) or not 1 <= best_epoch <= epochs:
        raise FullTrainingError("candidate best epoch is invalid")
    if not math.isfinite(float(record.get("best_validation_metric", math.nan))):
        raise FullTrainingError("candidate best validation metric is invalid")
    if verify_live_artifacts:
        paths = candidate_paths(project_root, official_study_id, candidate)
        artifacts = record.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise FullTrainingError("candidate artifact evidence is missing")
        checks = {
            paths.checkpoint: artifacts.get("checkpoint_sha256"),
            paths.log: artifacts.get("log_sha256"),
            paths.manifest: artifacts.get("manifest_sha256"),
            paths.training_history: artifacts.get("training_history_sha256"),
            paths.validation_history: artifacts.get("validation_history_sha256"),
        }
        for path, expected in checks.items():
            if not path.is_file() or not _valid_sha256(expected) or _sha256(path) != expected:
                raise FullTrainingError(f"candidate live artifact is missing or changed: {path}")

        manifest = _read_json(paths.manifest)
        manifest_unhashed = dict(manifest)
        manifest_hash = manifest_unhashed.pop("manifest_sha256", None)
        if manifest_hash != d2.canonical_sha256(manifest_unhashed):
            raise FullTrainingError("candidate manifest canonical hash is invalid")
        if (
            manifest.get("stage") != STAGE
            or manifest.get("version") != FULL_TRAINING_VERSION
            or manifest.get("official_study_id") != official_study_id
            or manifest.get("run_class") != RUN_CLASS_OFFICIAL
            or manifest.get("execution_authority") != EXECUTION_AUTHORITY
            or manifest.get("phase") != candidate.phase
            or manifest.get("candidate_id") != candidate.candidate_id
            or manifest.get("configuration") != candidate.config_dict()
            or manifest.get("config_sha256") != candidate.config_sha256
        ):
            raise FullTrainingError("candidate manifest identity is invalid")
        source_git = manifest.get("source_git_identity")
        source_hashes = manifest.get("e10_source_sha256")
        upstream_hashes = manifest.get("upstream_evidence_sha256")
        if (
            not isinstance(source_git, Mapping)
            or source_git.get("head") != FROZEN_START_COMMIT
            or source_git.get("origin_main") != FROZEN_START_COMMIT
            or FROZEN_START_TAG not in source_git.get("tags_at_head", [])
            or not isinstance(source_hashes, Mapping)
            or set(source_hashes) != set(E10_SOURCE_PATHS)
            or not all(_valid_sha256(value) for value in source_hashes.values())
            or not isinstance(upstream_hashes, Mapping)
            or set(upstream_hashes) != set(UPSTREAM_EVIDENCE_PATHS)
            or not all(_valid_sha256(value) for value in upstream_hashes.values())
            or not _valid_sha256(manifest.get("preflight_sha256"))
        ):
            raise FullTrainingError("candidate manifest provenance is incomplete")

        status = _read_json(paths.status)
        status_unhashed = dict(status)
        status_hash = status_unhashed.pop("status_sha256", None)
        if status_hash != d2.canonical_sha256(status_unhashed):
            raise FullTrainingError("candidate lifecycle status hash is invalid")
        if (
            status.get("stage") != STAGE
            or status.get("version") != FULL_TRAINING_VERSION
            or status.get("official_study_id") != official_study_id
            or status.get("run_class") != RUN_CLASS_OFFICIAL
            or status.get("execution_authority") != EXECUTION_AUTHORITY
            or status.get("candidate_id") != candidate.candidate_id
            or status.get("phase") != candidate.phase
            or status.get("state") != "COMPLETE"
            or status.get("epochs_completed") != epochs
            or status.get("artifacts_present", {}).get("result") is not True
        ):
            raise FullTrainingError("candidate lifecycle status is not COMPLETE")


def load_completed_results(
    project_root: Path,
    official_study_id: str,
    candidates: Sequence[FullTrainingCandidate],
) -> list[Mapping[str, Any]]:
    """Load an exact expected set; never discover or execute candidate runs."""

    validate_official_study_id(official_study_id)
    records: list[Mapping[str, Any]] = []
    provenance_identity: str | None = None
    for candidate in candidates:
        paths = candidate_paths(project_root, official_study_id, candidate)
        path = paths.result
        if not path.is_file():
            raise FullTrainingError(
                f"completed official result is missing: {candidate.candidate_id}"
            )
        record = _read_json(path)
        validate_candidate_result(
            project_root,
            official_study_id,
            candidate,
            record,
            verify_live_artifacts=True,
        )
        candidate_provenance = _candidate_provenance_identity(
            project_root, official_study_id, candidate
        )
        if provenance_identity is None:
            provenance_identity = candidate_provenance
        elif candidate_provenance != provenance_identity:
            raise FullTrainingError(
                "official candidate results do not share one provenance identity"
            )
        records.append(record)
    return records


def _provenance_compatibility_payload(
    manifest: Mapping[str, Any], official_study_id: str
) -> dict[str, Any]:
    """Project immutable recorded provenance through an explicit review policy.

    Only incidental Git status and the two study-approved OS build strings are
    normalized. Training runtime/runner bytes must be independently approved;
    neither a shared version label nor matching the current file grants trust.
    """

    validate_official_study_id(official_study_id)
    if (
        manifest.get("stage") != STAGE
        or manifest.get("version") != FULL_TRAINING_VERSION
        or manifest.get("official_study_id") != official_study_id
    ):
        raise FullTrainingError("candidate provenance stage/version/study is invalid")
    sources = manifest.get("e10_source_sha256")
    if (
        not isinstance(sources, Mapping)
        or set(sources) != set(E10_SOURCE_PATHS)
        or not all(_valid_sha256(value) for value in sources.values())
        or sources.get("src/training/full_training.py")
        not in provenance_policy.APPROVED_RUNTIME_SHA256
        or sources.get("scripts/run_stage_e10_full_training.py")
        != provenance_policy.APPROVED_RUNNER_SHA256
    ):
        raise FullTrainingError("unapproved E.10 training source revision")

    recorded_git = manifest.get("source_git_identity")
    if (
        not isinstance(recorded_git, Mapping)
        or recorded_git.get("head") != FROZEN_START_COMMIT
        or recorded_git.get("origin_main") != FROZEN_START_COMMIT
        or not isinstance(recorded_git.get("tags_at_head"), list)
        or not all(isinstance(tag, str) for tag in recorded_git["tags_at_head"])
        or FROZEN_START_TAG not in recorded_git["tags_at_head"]
        or type(recorded_git.get("dirty")) is not bool
        or not isinstance(recorded_git.get("status_short"), list)
        or not all(isinstance(line, str) for line in recorded_git["status_short"])
    ):
        raise FullTrainingError("candidate stable Git provenance is invalid")
    git = dict(recorded_git)
    git.pop("status_short")

    recorded_environment = manifest.get("environment_identity")
    required_environment = {
        "python", "python_implementation", "torch", "torch_geometric",
        "cuda_runtime", "cuda_available", "gpu", "gpu_total_memory_bytes",
        "platform",
    }
    if (
        not isinstance(recorded_environment, Mapping)
        or not required_environment.issubset(recorded_environment)
        or recorded_environment.get("platform")
        not in provenance_policy.AUDITED_WINDOWS_PLATFORMS
    ):
        raise FullTrainingError("unapproved or incomplete environment/platform provenance")
    environment = dict(recorded_environment)
    if official_study_id == provenance_policy.AUDITED_STUDY:
        environment["platform"] = list(provenance_policy.AUDITED_WINDOWS_PLATFORMS)

    return {
        "stage": manifest["stage"],
        "version": manifest["version"],
        "official_study_id": official_study_id,
        "source_git_identity": git,
        "environment_identity": environment,
        "stage_c_provenance": manifest["stage_c_provenance"],
        "upstream_evidence_sha256": manifest["upstream_evidence_sha256"],
        "training_semantics": provenance_policy.TRAINING_SEMANTICS_ID,
    }


def _candidate_provenance_identity(
    project_root: Path,
    official_study_id: str,
    candidate: FullTrainingCandidate,
) -> str:
    manifest = _read_json(
        candidate_paths(project_root, official_study_id, candidate).manifest
    )
    return d2.canonical_sha256(
        _provenance_compatibility_payload(manifest, official_study_id)
    )


def select_learning_rates(
    records: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    """Select one seed-2025 LR per family/task using validation only."""

    expected = enumerate_lr_selection_candidates()
    by_id = {str(record.get("candidate_id")): record for record in records}
    if set(by_id) != {candidate.candidate_id for candidate in expected}:
        raise FullTrainingError("learning-rate selection record set is incomplete")
    selected: dict[tuple[str, str], Mapping[str, Any]] = {}
    for family in MODEL_FAMILIES:
        for task in TASKS:
            candidates = [
                by_id[candidate.candidate_id]
                for candidate in expected
                if candidate.model_family == family and candidate.task == task
            ]
            ranked = sorted(
                candidates,
                key=lambda record: (
                    -float(record["best_validation_metric"]),
                    LEARNING_RATES.index(float(record["configuration"]["learning_rate"])),
                ),
            )
            selected[(family, task)] = ranked[0]
    return selected


def enumerate_additional_final_seed_candidates(
    selected: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[FullTrainingCandidate, ...]:
    if set(selected) != {(family, task) for family in MODEL_FAMILIES for task in TASKS}:
        raise FullTrainingError("selected learning-rate map is incomplete")
    return tuple(
        FullTrainingCandidate(
            family,
            task,
            float(selected[(family, task)]["configuration"]["learning_rate"]),
            seed,
            "final_seed",
        )
        for family in MODEL_FAMILIES
        for task in TASKS
        for seed in FINAL_SEEDS
        if seed != SELECTION_SEED
    )


def _selection_report(
    official_study_id: str,
    selected: Mapping[tuple[str, str], Mapping[str, Any]],
    selection_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    for family in MODEL_FAMILIES:
        for task in TASKS:
            candidates = [
                {
                    "candidate_id": record["candidate_id"],
                    "learning_rate": record["configuration"]["learning_rate"],
                    "best_validation_metric": record["best_validation_metric"],
                    "best_epoch": record["best_epoch"],
                    "best_validation_loss": record["best_validation_loss"],
                    "selection_eligible": True,
                    "selected": record["candidate_id"]
                    == selected[(family, task)]["candidate_id"],
                }
                for record in selection_records
                if record["configuration"]["model_family"] == family
                and record["configuration"]["task"] == task
            ]
            comparisons.append(
                {
                    "model_family": family,
                    "task": task,
                    "validation_selection_name": e7.SELECTION_NAMES[task],
                    "tie_break": LR_SELECTION_RULE,
                    "candidates": candidates,
                    "selected_learning_rate": selected[(family, task)][
                        "configuration"
                    ]["learning_rate"],
                    "selected_candidate_id": selected[(family, task)]["candidate_id"],
                }
            )
    report = {
        "stage": STAGE,
        "version": FULL_TRAINING_VERSION,
        "official_study_id": validate_official_study_id(official_study_id),
        "run_class": RUN_CLASS_OFFICIAL,
        "execution_authority": EXECUTION_AUTHORITY,
        "selection_seed": SELECTION_SEED,
        "rule": LR_SELECTION_RULE,
        "test_accessed": False,
        "comparisons": comparisons,
    }
    report["report_sha256"] = d2.canonical_sha256(report)
    return report


def _selection_plan(
    official_study_id: str,
    selected: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    plan: dict[str, Any] = {
        "stage": STAGE,
        "version": FULL_TRAINING_VERSION,
        "official_study_id": validate_official_study_id(official_study_id),
        "run_class": RUN_CLASS_OFFICIAL,
        "execution_authority": EXECUTION_AUTHORITY,
        "selection_status": "FROZEN_FOR_OFFICIAL_FINAL_SEEDS",
        "selection_seed": SELECTION_SEED,
        "test_accessed": False,
        "selected": [
            {
                "model_family": family,
                "task": task,
                "learning_rate": selected[(family, task)]["configuration"][
                    "learning_rate"
                ],
                "source_candidate_id": selected[(family, task)]["candidate_id"],
                "source_result_sha256": selected[(family, task)]["result_sha256"],
            }
            for family in MODEL_FAMILIES
            for task in TASKS
        ],
    }
    plan["plan_sha256"] = d2.canonical_sha256(plan)
    return plan


def aggregate_lr_selection(
    project_root: Path, official_study_id: str
) -> Mapping[str, Any]:
    """Select LRs from exactly 27 COMPLETE official results; never train."""

    root = Path(project_root).resolve()
    candidates = enumerate_lr_selection_candidates()
    records = load_completed_results(root, official_study_id, candidates)
    selected = select_learning_rates(records)
    report = _selection_report(official_study_id, selected, records)
    plan = _selection_plan(official_study_id, selected)
    paths = report_paths(root, official_study_id)
    _write_json_exclusive(paths.lr_selection_report, report)
    _write_json_exclusive(paths.selected_learning_rates, plan)
    return {
        "official_study_id": official_study_id,
        "lr_selection_report": str(paths.lr_selection_report),
        "selected_learning_rates": str(paths.selected_learning_rates),
        "candidate_results_read": len(records),
        "training_executed": False,
        "test_accessed": False,
    }


def load_selection_plan(
    project_root: Path,
    official_study_id: str,
    selection_plan_path: Path | None = None,
) -> dict[tuple[str, str], float]:
    root = Path(project_root).resolve()
    expected = report_paths(root, official_study_id).selected_learning_rates.resolve()
    path = expected if selection_plan_path is None else Path(selection_plan_path).resolve()
    if path != expected:
        raise FullTrainingError(
            "selection plan must be the canonical file for this official study"
        )
    plan = _read_json(path)
    unhashed = dict(plan)
    stored_hash = unhashed.pop("plan_sha256", None)
    if stored_hash != d2.canonical_sha256(unhashed):
        raise FullTrainingError("selected-learning-rate plan hash is invalid")
    if (
        plan.get("official_study_id") != official_study_id
        or plan.get("run_class") != RUN_CLASS_OFFICIAL
        or plan.get("execution_authority") != EXECUTION_AUTHORITY
        or plan.get("selection_status") != "FROZEN_FOR_OFFICIAL_FINAL_SEEDS"
        or plan.get("selection_seed") != SELECTION_SEED
        or plan.get("test_accessed") is not False
    ):
        raise FullTrainingError("selected-learning-rate plan identity is invalid")
    entries = plan.get("selected")
    if not isinstance(entries, list) or len(entries) != len(MODEL_FAMILIES) * len(TASKS):
        raise FullTrainingError("selected-learning-rate plan is incomplete")
    selected: dict[tuple[str, str], float] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise FullTrainingError("selected-learning-rate entry is malformed")
        key = (str(entry.get("model_family")), str(entry.get("task")))
        try:
            lr = float(entry.get("learning_rate"))
        except (TypeError, ValueError) as exc:
            raise FullTrainingError("selected learning rate is malformed") from exc
        if key in selected or key[0] not in MODEL_FAMILIES or key[1] not in TASKS:
            raise FullTrainingError("selected-learning-rate identity is invalid")
        if lr not in LEARNING_RATES:
            raise FullTrainingError("selected learning rate is outside the frozen matrix")
        source = FullTrainingCandidate(key[0], key[1], lr, SELECTION_SEED, "lr_selection")
        if entry.get("source_candidate_id") != source.candidate_id or not _valid_sha256(
            entry.get("source_result_sha256")
        ):
            raise FullTrainingError("selected learning rate lacks valid source provenance")
        selected[key] = lr
    expected_keys = {(family, task) for family in MODEL_FAMILIES for task in TASKS}
    if set(selected) != expected_keys:
        raise FullTrainingError("selected-learning-rate plan is incomplete")
    return selected


def enumerate_final_seed_candidates_from_plan(
    project_root: Path,
    official_study_id: str,
    selection_plan_path: Path | None = None,
) -> tuple[FullTrainingCandidate, ...]:
    selected = load_selection_plan(
        project_root, official_study_id, selection_plan_path
    )
    return tuple(
        FullTrainingCandidate(family, task, selected[(family, task)], seed, "final_seed")
        for family in MODEL_FAMILIES
        for task in TASKS
        for seed in FINAL_SEEDS
        if seed != SELECTION_SEED
    )


def resolve_single_job(
    project_root: Path,
    official_study_id: str,
    *,
    phase: str,
    experiment: str,
    family: str,
    learning_rate: float,
    seed: int,
    selection_plan_path: Path | None = None,
) -> FullTrainingCandidate:
    candidate = FullTrainingCandidate(
        family, experiment, learning_rate, seed, phase
    )
    if phase == "final_seed":
        selected = load_selection_plan(
            project_root, official_study_id, selection_plan_path
        )
        if selected[(family, experiment)] != learning_rate:
            raise FullTrainingError(
                "final-seed job LR does not match the frozen selection plan"
            )
    return candidate


def dry_run_jobs(
    project_root: Path,
    official_study_id: str,
    *,
    phase: str,
    selection_plan_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Describe exact jobs and paths without preflight, data, model, or optimizer work."""

    root = Path(project_root).resolve()
    validate_official_study_id(official_study_id)
    if phase == "lr_selection":
        candidates = enumerate_lr_selection_candidates()
    elif phase == "final_seed":
        candidates = enumerate_final_seed_candidates_from_plan(
            root, official_study_id, selection_plan_path
        )
    else:
        raise ValueError("phase must be lr_selection or final_seed")
    jobs: list[dict[str, Any]] = []
    for candidate in candidates:
        paths = candidate_paths(root, official_study_id, candidate)
        jobs.append(
            {
                "official_study_id": official_study_id,
                "phase": candidate.phase,
                "experiment": candidate.task,
                "family": candidate.model_family,
                "learning_rate": candidate.learning_rate,
                "seed": candidate.seed,
                "batch_size": BATCH_SIZE,
                "candidate_id": candidate.candidate_id,
                "checkpoint": str(paths.checkpoint),
                "log_directory": str(paths.log.parent),
                "result": str(paths.result),
            }
        )
    return jobs


def build_report(
    *,
    project_root: Path,
    run_id: str,
    preflight_result: Mapping[str, Any],
    postflight_result: Mapping[str, Any],
    selection_records: Sequence[Mapping[str, Any]],
    additional_final_records: Sequence[Mapping[str, Any]],
    verification: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected = select_learning_rates(selection_records)
    final_records: list[Mapping[str, Any]] = []
    additional_by_key = {
        (
            str(record["configuration"]["model_family"]),
            str(record["configuration"]["task"]),
            int(record["configuration"]["seed"]),
        ): record
        for record in additional_final_records
    }
    for family in MODEL_FAMILIES:
        for task in TASKS:
            final_records.append(selected[(family, task)])
            for seed in FINAL_SEEDS:
                if seed == SELECTION_SEED:
                    continue
                final_records.append(additional_by_key[(family, task, seed)])
    selection_report = _selection_report(run_id, selected, selection_records)
    report: dict[str, Any] = {
        "stage": STAGE,
        "version": FULL_TRAINING_VERSION,
        "scope": "full train and validation only",
        "official_study_id": validate_official_study_id(run_id),
        "run_class": RUN_CLASS_OFFICIAL,
        "execution_authority": EXECUTION_AUTHORITY,
        "overall_status": "PASS",
        "frozen_start": {
            "commit": FROZEN_START_COMMIT,
            "tag": FROZEN_START_TAG,
            "origin_main": FROZEN_START_COMMIT,
        },
        "protocol": {
            "families": list(MODEL_FAMILIES),
            "tasks": list(TASKS),
            "learning_rates": list(LEARNING_RATES),
            "selection_seed": SELECTION_SEED,
            "final_seeds": list(FINAL_SEEDS),
            "batch_size": BATCH_SIZE,
            "maximum_epochs": MAX_EPOCHS,
            "minimum_epochs": e7.MIN_EPOCHS,
            "patience": e7.PATIENCE,
            "min_delta": e7.MIN_DELTA,
            "lr_selection_rule": LR_SELECTION_RULE,
            "selection_seed_winner_reused_as_final_seed": True,
            "unique_full_runs": len(selection_records) + len(additional_final_records),
            "lr_selection_runs": len(selection_records),
            "additional_final_seed_runs": len(additional_final_records),
        },
        "preflight": dict(preflight_result),
        "postflight": dict(postflight_result),
        "lr_selection": selection_report,
        "completed_runs": [*selection_records, *additional_final_records],
        "final_validation_selected_checkpoints": final_records,
        "validation_only_family_comparison": [
            {
                "task": task,
                "families": [
                    {
                        "model_family": family,
                        "selected_learning_rate": selected[(family, task)][
                            "configuration"
                        ]["learning_rate"],
                        "seed_validation_metrics": [
                            {
                                "seed": record["configuration"]["seed"],
                                "best_validation_metric": record[
                                    "best_validation_metric"
                                ],
                            }
                            for record in final_records
                            if record["configuration"]["model_family"] == family
                            and record["configuration"]["task"] == task
                        ],
                    }
                    for family in MODEL_FAMILIES
                ],
                "interpretation": "validation performance only; no final scientific claim",
            }
            for task in TASKS
        ],
        "e3_threshold_selection": {
            "applicable": False,
            "reason": "the frozen E.7-E.9 contract selects E3 by raw-logit PR-AUC and defines no E.10 decision threshold",
        },
        "test_access_audit": {
            "predictive_test_arrays_opened": [],
            "predictive_test_array_access": False,
            "test_metrics_recorded": False,
            "result": "PASS",
        },
        "failed_or_retried_runs": [],
        "pruning_performed": False,
        "verification": dict(verification),
        "limitations": [
            "All reported predictive values are validation-only.",
            "No test predictions or metrics were produced.",
            "No E3 decision threshold was selected because the frozen contract defines none.",
            "Resource throughput is end-to-end train-plus-validation epoch throughput on this machine.",
            "CUDA determinism remains best-effort under the frozen PyG contract.",
            "No pruning or E.11 evaluation was performed.",
        ],
    }
    unhashed = dict(report)
    report["report_sha256"] = d2.canonical_sha256(unhashed)
    return report, selection_report


def write_reports(
    project_root: Path,
    run_id: str,
    report: Mapping[str, Any],
    selection_report: Mapping[str, Any],
) -> tuple[Path, Path, Path]:
    paths = report_paths(project_root, run_id)
    report_path = paths.final_report
    sidecar_path = paths.final_report_sha256
    selection_path = paths.lr_selection_report
    if selection_path.is_file():
        if _read_json(selection_path) != selection_report:
            raise FullTrainingError("existing LR-selection report does not match final report")
    else:
        _write_json_exclusive(selection_path, selection_report)
    _write_json_exclusive(report_path, report)
    physical = _sha256(report_path)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with sidecar_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(f"{physical}  {report_path.name}\n")
    except FileExistsError as exc:
        raise FullTrainingError(f"E.10 report sidecar already exists: {sidecar_path}") from exc
    return report_path, sidecar_path, selection_path


def aggregate_final_report(
    project_root: Path,
    official_study_id: str,
    selection_plan_path: Path | None = None,
) -> Mapping[str, Any]:
    """Build the final validation-only report from exact COMPLETE artifacts."""

    root = Path(project_root).resolve()
    plan_lrs = load_selection_plan(root, official_study_id, selection_plan_path)
    selection_records = load_completed_results(
        root, official_study_id, enumerate_lr_selection_candidates()
    )
    independently_selected = select_learning_rates(selection_records)
    plan_path = report_paths(root, official_study_id).selected_learning_rates
    plan = _read_json(plan_path)
    plan_entries = {
        (str(entry["model_family"]), str(entry["task"])): entry
        for entry in plan["selected"]
    }
    for key, lr in plan_lrs.items():
        selected_record = independently_selected[key]
        if float(selected_record["configuration"]["learning_rate"]) != lr:
            raise FullTrainingError("selection plan no longer matches official results")
        entry = plan_entries[key]
        if (
            entry.get("source_candidate_id") != selected_record.get("candidate_id")
            or entry.get("source_result_sha256")
            != selected_record.get("result_sha256")
        ):
            raise FullTrainingError(
                "selection-plan provenance no longer matches official results"
            )
    final_candidates = enumerate_final_seed_candidates_from_plan(
        root, official_study_id, selection_plan_path
    )
    additional_records = load_completed_results(
        root, official_study_id, final_candidates
    )
    if _candidate_provenance_identity(
        root, official_study_id, enumerate_lr_selection_candidates()[0]
    ) != _candidate_provenance_identity(
        root, official_study_id, final_candidates[0]
    ):
        raise FullTrainingError(
            "LR-selection and final-seed results have different provenance"
        )
    aggregation_evidence = {
        "mode": "result_only",
        "training_executed": False,
        "predictive_arrays_accessed": False,
        "official_results_validated": len(selection_records) + len(additional_records),
    }
    verification = {
        "result_artifact_validation": {
            "status": "PASS",
            "selection_results": len(selection_records),
            "additional_final_results": len(additional_records),
        },
        "training_during_aggregation": {"status": "NOT_RUN"},
    }
    report, selection_report = build_report(
        project_root=root,
        run_id=official_study_id,
        preflight_result=aggregation_evidence,
        postflight_result=aggregation_evidence,
        selection_records=selection_records,
        additional_final_records=additional_records,
        verification=verification,
    )
    report_path, sidecar_path, selection_path = write_reports(
        root, official_study_id, report, selection_report
    )
    return {
        "official_study_id": official_study_id,
        "overall_status": report["overall_status"],
        "report_path": str(report_path),
        "sidecar_path": str(sidecar_path),
        "lr_selection_report_path": str(selection_path),
        "unique_full_runs": report["protocol"]["unique_full_runs"],
        "training_executed": False,
        "test_predictive_array_access": False,
    }


__all__ = [
    "BATCH_SIZE",
    "FINAL_SEEDS",
    "FULL_TRAINING_VERSION",
    "FullTrainingCandidate",
    "FullTrainingError",
    "LEARNING_RATES",
    "LR_SELECTION_RULE",
    "MAX_EPOCHS",
    "MODEL_FAMILIES",
    "SELECTION_SEED",
    "TASKS",
    "aggregate_final_report",
    "aggregate_lr_selection",
    "build_report",
    "candidate_paths",
    "enumerate_additional_final_seed_candidates",
    "enumerate_final_seed_candidates_from_plan",
    "enumerate_lr_selection_candidates",
    "execute_candidate",
    "dry_run_jobs",
    "load_completed_results",
    "load_selection_plan",
    "new_run_id",
    "predictive_test_lockout",
    "preflight",
    "report_paths",
    "resolve_single_job",
    "select_learning_rates",
    "source_git_identity",
    "validate_candidate_result",
    "validate_official_study_id",
    "validate_run_id",
    "write_reports",
]
