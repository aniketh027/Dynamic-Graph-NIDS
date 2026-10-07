"""E.11 owner-only, one-time evaluation of the frozen E.10 checkpoints.

Preparation reads compact metadata and checkpoints only. Predictive test files
are inaccessible until an immutable study-wide consumption marker is reserved.
Frozen E.7 training and E.9 adapters are never modified or relabelled.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
from importlib.metadata import version as package_version
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import tempfile
from typing import Any, Iterator, Mapping

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch

from src.graph import adapter as d4, batching as d6, pyg_adapter as e2
from src.graph import schema as d2
from src.preprocessing.labels import LABEL_TO_INT, LABEL_VOCABULARY
from src.preprocessing import splitting
from src.training import full_training as e10, harness as e7, baselines


VERSION = "stage-e11-official-test-v1"
STUDY = "e10-official-20261001-v1"
# Independently reconciled completed evidence, not a live-file trust decision.
E10_REPORT_SHA256 = "8ff3f89c4aac3b2948936ea794e710059a7fe441c161b8c20cae0b8e51f6639c"
POLICY_SHA256 = "889e1265815a22acfa4d10d110c368f2106b799fbc5cc74612cf496da6376141"
STAGE_C_REPORT_SHA256 = "d1d33012e0135533585952aa8234154e03870ead0a78aa48edae393a602b3718"
RUN_PATTERN = re.compile(r"^e11-official-[0-9]{8}-v[1-9][0-9]*$")
SOURCE_PATHS = (
    "src/evaluation/official_test.py", "scripts/run_stage_e11_test_evaluation.py",
    "tests/test_stage_e11_test_evaluation.py", "docs/stage_e11_test_evaluation.md",
    "src/graph/adapter.py", "src/graph/batching.py", "src/graph/pyg_adapter.py",
    "src/graph/schema.py", "src/graph/taxonomy.py", "src/graph/validation.py",
    "src/graph/experiment_validation.py", "src/graph/full_split_integration.py",
    "src/preprocessing/labels.py", "src/preprocessing/splitting.py",
    "src/training/harness.py", "src/training/baselines.py",
    "src/training/baseline_integration.py", "src/training/full_training.py",
    "src/training/e10_provenance_policy.py",
)
_GUARDS: list[tuple[Path, bool]] = []
_HOOK_INSTALLED = False


class EvaluationError(RuntimeError):
    """A provenance, isolation, coverage, or one-time evaluation failure."""


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _audit_hook(event: str, args: tuple[Any, ...]) -> None:
    if not _GUARDS:
        return
    paths: list[object] = []
    writing = False
    if event == "open":
        paths = [args[0]]
        mode, flags = args[1], args[2]
        writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
            isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        )
    elif event in {"os.remove", "os.rmdir", "os.mkdir", "os.chmod", "os.utime"}:
        paths, writing = [args[0]], True
    elif event in {"os.rename", "os.link", "os.symlink"}:
        paths, writing = list(args[:2]), True
    else:
        return
    for value in paths:
        if not isinstance(value, (str, bytes, os.PathLike)):
            continue
        path = Path(os.fsdecode(value)).resolve()
        for root, allow_test in tuple(_GUARDS):
            if not _inside(path, root):
                continue
            if writing and path.name == "test_consumption.json" and _inside(path, root / "logs/stage_e11"):
                exclusive_create = event == "open" and isinstance(args[2], int) and bool(args[2] & os.O_EXCL) and bool(args[2] & os.O_CREAT)
                if not exclusive_create:
                    raise EvaluationError("official test consumption marker is immutable")
            creating_parent = event == "os.mkdir" and path in (root / "logs", root / "results")
            if writing and not creating_parent and not any(_inside(path, root / area / "stage_e11") for area in ("logs", "results")):
                raise EvaluationError(f"E.11 cannot write outside its namespace: {path}")
            if event == "open" and path.suffix == ".npy" and _inside(path, root / "data"):
                approved = {root / "data/processed/preprocessed" / directory / name
                            for directory in d4.EXPERIMENT_DIRS.values()
                            for name in ("X_test.npy", "y_test.npy")}
                if writing or not allow_test or path not in approved:
                    raise EvaluationError(f"predictive array access is locked: {path}")


@contextmanager
def file_guard(root: Path, *, allow_test: bool = False) -> Iterator[None]:
    """Process-wide guard, also covering reads/writes from worker threads.

This is an accidental-execution boundary, not a sandbox against a caller with
arbitrary Python code execution or external filesystem mutation privileges.
"""
    global _HOOK_INSTALLED
    if not _HOOK_INSTALLED:
        sys.addaudithook(_audit_hook)
        _HOOK_INSTALLED = True
    item = (Path(root).resolve(), allow_test)
    _GUARDS.append(item)
    try:
        yield
    finally:
        _GUARDS.remove(item)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EvaluationError(f"expected JSON object: {path}")
    return value


def _canonical_check(value: Mapping[str, Any], key: str) -> None:
    unhashed = dict(value)
    expected = unhashed.pop(key, None)
    if expected != d2.canonical_sha256(unhashed):
        raise EvaluationError(f"invalid canonical {key}")


def _candidate(record: Mapping[str, Any]) -> e10.FullTrainingCandidate:
    config = record["configuration"]
    return e10.FullTrainingCandidate(config["model_family"], config["task"],
                                     config["learning_rate"], config["seed"], config["phase"])


def _load_bound_checkpoint(root: Path, record: Mapping[str, Any]) -> torch.nn.Module:
    candidate = _candidate(record)
    path = e10.candidate_paths(root, STUDY, candidate).checkpoint
    if e10._sha256(path) != record["artifacts"]["checkpoint_sha256"]:
        raise EvaluationError("checkpoint bytes changed")
    model = e7.create_model(candidate.task, candidate.model_family)
    # E.7 constructs an AdamW solely to validate serialized optimizer binding.
    # No optimizer state is restored and no optimizer step is executed.
    metadata = e7.load_checkpoint(path, model=model, expected_task=candidate.task,
                                  expected_config=candidate.training_config, map_location="cpu")
    if (dict(metadata) != record["checkpoint_provenance"]
            or metadata["epoch"] != record["best_epoch"]
            or metadata["selection_value"] != record["best_validation_metric"]
            or record.get("checkpoint_provenance_verified") is not True):
        raise EvaluationError("checkpoint provenance differs from E.10 result")
    if e10._sha256(path) != record["artifacts"]["checkpoint_sha256"]:
        raise EvaluationError("checkpoint changed while loading")
    model.requires_grad_(False)
    model.eval()
    return model


def _counts(metadata: Mapping[str, Any], task: str, split: str) -> np.ndarray:
    prefix = "binary" if task == "E3" else "class"
    distribution = metadata[f"{split}_{prefix}_distribution"]
    names = ("Benign", "Attack") if task == "E3" else LABEL_VOCABULARY
    if not isinstance(distribution, dict) or set(distribution) - set(names):
        raise EvaluationError("invalid split support names")
    values = [distribution.get(name, 0) for name in names]
    if any(type(v) is not int or v < 0 for v in values):
        raise EvaluationError("invalid split support counts")
    result = np.array(values, dtype=np.int64)
    if int(result.sum()) != metadata[f"n_{split}"]:
        raise EvaluationError("split support does not match metadata size")
    return result


def reporting_policy(metadata: Mapping[str, Any], task: str) -> dict[str, Any]:
    if task not in e10.TASKS or metadata.get("experiment") != d4.EXPERIMENT_METADATA_IDENTITIES[task]:
        raise EvaluationError("cross-experiment split metadata")
    preprocessing = metadata.get("preprocessing", {})
    if preprocessing.get("fit_split") != "train" or preprocessing.get("validation_or_test_used_for_fit") is not False:
        raise EvaluationError("preprocessing is not frozen train-only state")
    train, val, test = (_counts(metadata, task, split) for split in ("train", "val", "test"))
    if task == "E3":
        codes = [0, 1]
    elif task == "E1":
        codes = np.flatnonzero((train > 0) & (test > 0)).tolist()
        available = metadata.get("class_availability_matrix", {}).get("classes_with_possible_supervised_evaluation")
        if available != [LABEL_VOCABULARY[c] for c in codes]:
            raise EvaluationError("E1 frozen availability disagrees with split counts")
    else:
        eligibility = metadata.get("class_eligibility", {})
        thresholds = {"min_train": splitting.ELIGIBILITY_MIN_TRAIN,
                      "min_val": splitting.ELIGIBILITY_MIN_VAL, "min_test": splitting.ELIGIBILITY_MIN_TEST}
        codes = np.flatnonzero((train >= thresholds["min_train"]) &
                              (val >= thresholds["min_val"]) & (test >= thresholds["min_test"])).tolist()
        if eligibility.get("thresholds") != thresholds or eligibility.get("eligible_class_codes") != codes:
            raise EvaluationError("E2 frozen reporting eligibility is inconsistent")
        for code, name in enumerate(LABEL_VOCABULARY):
            row = eligibility.get("per_class", {}).get(name, {})
            if any(row.get(key) != int(counts[code]) for key, counts in
                   (("n_train", train), ("n_val", val), ("n_test", test))) or row.get("eligible") is not (code in codes):
                raise EvaluationError("E2 per-class eligibility counts are inconsistent")
    if not codes:
        raise EvaluationError("no supported reporting classes")
    return {"task": task, "train_support": train.tolist(), "validation_support": val.tolist(),
            "test_support": test.tolist(), "reporting_class_codes": codes,
            "unseen_test_class_codes": np.flatnonzero((train == 0) & (test > 0)).tolist(),
            "test_order": "sequential", "output_logits": 1 if task == "E3" else 15,
            "binary_decision_logit_threshold": 0.0 if task == "E3" else None}


def audit_e10(root: Path, study: str = STUDY) -> dict[str, Any]:
    """Read-only preparation. No predictive arrays, training, or LR selection.

Verify the existing frozen LR plan; never choose or change a learning rate.
"""
    root = Path(root).resolve()
    if study != STUDY:
        raise EvaluationError("this reviewed E.11 contract is scoped to the audited E.10 study")
    with file_guard(root):
        paths = e10.report_paths(root, study)
        if e10._sha256(root / "src/training/e10_provenance_policy.py") != POLICY_SHA256:
            raise EvaluationError("reviewed provenance policy changed")
        if e10._sha256(root / "src/training/full_training.py") != e10.provenance_policy.CORRECTED_RUNTIME_SHA256:
            raise EvaluationError("reviewed E.10 runtime changed")
        if e10._sha256(root / "scripts/run_stage_e10_full_training.py") != e10.provenance_policy.APPROVED_RUNNER_SHA256:
            raise EvaluationError("reviewed E.10 runner changed")
        if e10._sha256(paths.final_report) != E10_REPORT_SHA256:
            raise EvaluationError("completed E.10 report changed")
        report = _read(paths.final_report)
        _canonical_check(report, "report_sha256")
        if paths.final_report_sha256.read_text().split()[0] != E10_REPORT_SHA256:
            raise EvaluationError("E.10 physical report sidecar mismatch")
        plan_lrs = e10.load_selection_plan(root, study)
        plan = _read(paths.selected_learning_rates)
        selection = _read(paths.lr_selection_report)
        _canonical_check(selection, "report_sha256")
        a = e10.load_completed_results(root, study, e10.enumerate_lr_selection_candidates())
        c = e10.load_completed_results(root, study, e10.enumerate_final_seed_candidates_from_plan(root, study))
        records = a + c
        if len(records) != 45 or len({r["candidate_id"] for r in records}) != 45 or report["completed_runs"] != records:
            raise EvaluationError("E.10 completed run set is not exactly 27 plus 18 unique results")
        if report["overall_status"] != "PASS" or report["lr_selection"] != selection:
            raise EvaluationError("E.10 final aggregation differs from frozen selection evidence")
        if report["test_access_audit"] != {"predictive_test_arrays_opened": [],
                "predictive_test_array_access": False, "test_metrics_recorded": False, "result": "PASS"}:
            raise EvaluationError("E.10 recorded test access")
        identities = {e10._candidate_provenance_identity(root, study, _candidate(r)) for r in records}
        if len(identities) != 1:
            raise EvaluationError("cross-phase provenance mismatch")
        by_id = {r["candidate_id"]: r for r in records}
        final = []
        for entry in plan["selected"]:
            family, task = entry["model_family"], entry["task"]
            for seed in e10.FINAL_SEEDS:
                candidate = e10.FullTrainingCandidate(family, task, plan_lrs[(family, task)], seed,
                                                      "lr_selection" if seed == 2025 else "final_seed")
                r = by_id[candidate.candidate_id]
                if seed == 2025 and (entry["source_candidate_id"] != r["candidate_id"] or
                                    entry["source_result_sha256"] != r["result_sha256"]):
                    raise EvaluationError("frozen LR source result mismatch")
                final.append(r)
        if report["final_validation_selected_checkpoints"] != final:
            raise EvaluationError("final selected checkpoints do not match the frozen plan and three seeds")
        artifact_hashes: dict[str, str] = {}
        for r in records:
            _load_bound_checkpoint(root, r)
            candidate = _candidate(r)
            run_paths = e10.candidate_paths(root, study, candidate)
            for path in (run_paths.checkpoint, run_paths.manifest, run_paths.status, run_paths.result,
                         run_paths.log, run_paths.training_history, run_paths.validation_history):
                artifact_hashes[path.relative_to(root).as_posix()] = e10._sha256(path)
        for path in (paths.final_report, paths.final_report_sha256, paths.selected_learning_rates, paths.lr_selection_report):
            artifact_hashes[path.relative_to(root).as_posix()] = e10._sha256(path)
        manifest = _read(e10.candidate_paths(root, study, _candidate(records[0])).manifest)
        if e10._evidence_hashes(root) != manifest["upstream_evidence_sha256"]:
            raise EvaluationError("frozen upstream evidence changed")
        e9 = e10._validate_frozen_e9_evidence(root)
        for relative, digest in {**e9["frozen_upstream_source_sha256"], **e9["e9_source_sha256"],
                                 **records[0]["checkpoint_provenance"]["frozen_source_sha256"]}.items():
            if e10._sha256(root / relative) != digest:
                raise EvaluationError(f"frozen execution source changed: {relative}")
        stage_c_path = root / "results/stage_c_full_validation.json"
        if e10._sha256(stage_c_path) != STAGE_C_REPORT_SHA256:
            raise EvaluationError("Stage C frozen integrity evidence changed")
        stage_c = _read(stage_c_path)
        if stage_c.get("status") != "PASS":
            raise EvaluationError("Stage C evidence is not PASS")
        metadata_hashes = {"results/stage_c_full_validation.json": STAGE_C_REPORT_SHA256,
                           **manifest["upstream_evidence_sha256"]}
        for relative, expected in stage_c["artefact_sha256"].items():
            if not relative.endswith(".npy"):
                path = root / "data/processed/preprocessed" / relative
                if e10._sha256(path) != expected:
                    raise EvaluationError(f"frozen Stage C metadata changed: {relative}")
                metadata_hashes[path.relative_to(root).as_posix()] = expected
        splits = {}
        for task, directory in d4.EXPERIMENT_DIRS.items():
            relative = f"{directory}/split_metadata.json"
            path = root / "data/processed/preprocessed" / relative
            digest = e10._sha256(path)
            if digest != manifest["stage_c_provenance"]["split_artifact_sha256"][relative] or digest != stage_c["artefact_sha256"][relative]:
                raise EvaluationError("frozen split metadata changed")
            metadata = _read(path)
            splits[task] = {"metadata_sha256": digest, "policy": reporting_policy(metadata, task),
                            "predictive_artifact_sha256": {f"{directory}/{name}": stage_c["artefact_sha256"][f"{directory}/{name}"]
                                                          for name in ("X_test.npy", "y_test.npy")}}
        return {"stage": "E.11", "version": VERSION, "official_study_id": study,
                "e10_audit_status": "PASS", "phase_a_runs": 27, "phase_c_runs": 18,
                "unique_full_runs": 45, "checkpoint_provenance_verified": 45,
                "e10_recorded_test_predictive_access": False, "preparation_test_predictive_access": False,
                "training_executed": False, "lr_selection_executed": False, "pruning_performed": False,
                "e10_report_canonical_sha256": report["report_sha256"],
                "e10_report_physical_sha256": E10_REPORT_SHA256,
                "selection_plan_physical_sha256": e10._sha256(paths.selected_learning_rates),
                "reviewed_policy_sha256": POLICY_SHA256, "stage_c_report_sha256": STAGE_C_REPORT_SHA256,
                "e10_compatibility_identity": next(iter(identities)),
                "e10_artifact_sha256": artifact_hashes, "frozen_metadata_sha256": metadata_hashes,
                "training_environment": manifest["environment_identity"], "splits": splits,
                "evaluation_records": final}


def _rates(confusion: np.ndarray, codes: list[int]) -> dict[str, Any]:
    support = confusion.sum(axis=1)
    tp = confusion.diagonal().astype(np.float64)
    predicted = confusion.sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros_like(tp), where=(support + predicted) > 0)
    total = int(support.sum())
    selected_support = int(support[codes].sum())
    return {"class_codes": codes, "macro_precision": float(precision[codes].mean()),
            "macro_recall": float(recall[codes].mean()), "macro_f1": float(f1[codes].mean()),
            "accuracy": float(tp.sum() / total) if total else None,
            "weighted_f1": float(np.dot(f1[codes], support[codes]) / selected_support) if selected_support else None,
            "per_class": [{"code": i, "precision": float(precision[i]), "recall": float(recall[i]),
                           "f1": float(f1[i]), "support": int(support[i])} for i in range(len(support))]}


def multiclass_metrics(confusion: np.ndarray, policy: Mapping[str, Any]) -> dict[str, Any]:
    if confusion.shape != (15, 15) or confusion.dtype.kind not in "iu" or np.any(confusion < 0):
        raise EvaluationError("invalid multiclass confusion matrix")
    if confusion.sum(axis=1).tolist() != policy["test_support"]:
        raise EvaluationError("observed test supports differ from frozen metadata")
    codes = policy["reporting_class_codes"]
    primary = confusion.copy()
    if policy["task"] == "E2":
        # Mask only true-label rows. Predictions still use all fifteen logits.
        primary[[i for i in range(15) if i not in codes], :] = 0
    supported = np.flatnonzero(confusion.sum(axis=1) > 0).tolist()
    descriptive = _rates(confusion, supported)
    for row in descriptive["per_class"]:
        code = row["code"]
        row["label"] = LABEL_VOCABULARY[code]
        row["train_support"] = policy["train_support"][code]
        row["validation_support"] = policy["validation_support"][code]
        row["reporting_eligible"] = code in codes
        row["status"] = ("absent from test" if row["support"] == 0 else
                         "unseen during training; descriptive only" if policy["train_support"][code] == 0 else
                         "primary reporting class" if code in codes else "ineligible; descriptive only")
    return {"primary_name": "support_restricted_macro_f1" if policy["task"] == "E1" else "eligible_class_macro_f1",
            "primary": _rates(primary, codes), "all_test_descriptive": descriptive,
            "confusion_matrix": confusion.tolist(), "confusion_label_order": list(LABEL_VOCABULARY),
            "primary_true_row_filter": "eligible_classes" if policy["task"] == "E2" else "all_test_rows",
            "absent_corpus_classes": [{"label": "Heartbleed", "status": "not evaluated - absent from corpus"}]}


def binary_metrics(targets: np.ndarray, scores: np.ndarray, policy: Mapping[str, Any]) -> dict[str, Any]:
    if targets.ndim != 1 or scores.shape != targets.shape or not len(targets) or not np.isfinite(scores).all():
        raise EvaluationError("invalid binary scores/targets")
    if not np.isin(targets, [0, 1]).all():
        raise EvaluationError("invalid binary target domain")
    counts = np.bincount(targets.astype(np.int64), minlength=2)
    if counts.tolist() != policy["test_support"]:
        raise EvaluationError("observed binary supports differ from frozen metadata")
    predictions = scores >= 0.0
    confusion = np.bincount(targets.astype(np.int64) * 2 + predictions.astype(np.int64), minlength=4).reshape(2, 2)
    tn, fp, fn, tp = (int(v) for v in confusion.ravel())
    both = bool(np.all(counts > 0))
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {"primary_name": "pr_auc_average_precision",
            "pr_auc_average_precision": float(average_precision_score(targets, scores)) if both else None,
            "roc_auc": float(roc_auc_score(targets, scores)) if both else None,
            "ranking_metrics_undefined_reason": None if both else "both target classes are required",
            "logit_threshold": 0.0, "probability_threshold": 0.5, "threshold_tuned": False,
            "attack_prevalence": float(counts[1] / len(targets)),
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "specificity": tn / (tn + fp) if tn + fp else None,
            "false_positive_rate": fp / (tn + fp) if tn + fp else None,
            "balanced_accuracy": (tp / (tp + fn) + tn / (tn + fp)) / 2 if both else None,
            "mcc": (tp * tn - fp * fn) / denominator if denominator else 0.0,
            "accuracy": (tp + tn) / len(targets), "support": counts.tolist(),
            "confusion_matrix": confusion.tolist(), "confusion_label_order": ["Benign", "Attack"]}


def iter_test_batches(root: Path, task: str) -> Iterator[Any]:
    root = Path(root).resolve()
    if not any(guard_root == root and allowed for guard_root, allowed in _GUARDS):
        raise EvaluationError("test traversal requires the owner evaluation reservation context")
    ledger = root / "logs/stage_e11" / STUDY / "test_consumption.json"
    reservation = _read(ledger)
    if reservation.get("e10_study") != STUDY or reservation.get("state") != "CONSUMED_OR_RESERVED":
        raise EvaluationError("invalid official test reservation")
    dataset = d4.LazyGraphDataset(task, "test", processed_root=root / "data/processed/preprocessed",
                                d1_manifest_path=root / "results/stage_d1/feature_taxonomy.json",
                                d2_manifest_path=root / "results/stage_d2/graph_schema.json",
                                d3_report_path=root / "results/stage_d3/structural_validation.json")
    if not dataset.is_lazy_read_only:
        raise EvaluationError("test dataset must remain lazy and read-only")
    for indices in e7.iter_index_batches(len(dataset), e10.BATCH_SIZE, shuffle=False):
        yield e2.logical_batch_to_heterodata(d6.build_logical_batch(dataset, indices))


def _inference_logits(model: torch.nn.Module, data: Any, task: str, family: str) -> torch.Tensor:
    if family == "htgnn":
        return model(data)
    # The frozen training adapter rejects test batches. Invoke its frozen dense
    # model directly with the same graph-major feature/time order from E.2.
    batch_size = data["sample"].y.numel()
    values = data["feature"].x
    if tuple(values.shape) != (batch_size * 78, 5) or values.dtype != torch.float32:
        raise EvaluationError("invalid E.2 baseline inference features")
    if e7.model_contract(model).family != family or e7.model_contract(model).task != task:
        raise EvaluationError("inference model family/task mismatch")
    expected_ids = torch.arange(78, device=values.device).repeat(batch_size)
    expected_batch = torch.arange(batch_size, device=values.device).repeat_interleave(78)
    if not torch.equal(data["feature"].feature_id, expected_ids) or not torch.equal(data["feature"].batch, expected_batch):
        raise EvaluationError("baseline feature order or ownership differs from E.2")
    inputs = values[:, 4].reshape(batch_size, 78) if family == baselines.FINAL_FLOW else values.reshape(batch_size, 78, 5).transpose(1, 2).contiguous()
    return model.model(inputs)


def _validate_inference_outputs(task: str, logits: torch.Tensor, targets: torch.Tensor,
                                batch_size: int) -> None:
    """Validate the frozen E.2 target dtype and E.5/E.9 raw output contract.

    E3 is float32[B] for both logits and labels. E1/E2 labels are long[B].
    Do not reshape, broadcast, cast or activate a model's output to hide a
    contract violation; the frozen E3 forward already squeezes its head axis.
    """
    if task not in e10.TASKS:
        raise EvaluationError("invalid inference task")
    expected_shape = (batch_size,) if task == "E3" else (batch_size, 15)
    if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != expected_shape:
        raise EvaluationError(f"{task} logits must have shape {expected_shape}")
    if logits.dtype != torch.float32 or not torch.isfinite(logits).all():
        raise EvaluationError(f"{task} logits must be finite float32")
    target_dtype = torch.float32 if task == "E3" else torch.long
    if not isinstance(targets, torch.Tensor) or tuple(targets.shape) != (batch_size,) or targets.dtype != target_dtype:
        raise EvaluationError(f"{task} targets must have shape [{batch_size}] and dtype {target_dtype}")


def evaluate_checkpoint(root: Path, record: Mapping[str, Any], policy: Mapping[str, Any],
                        scratch: Path, device: torch.device) -> dict[str, Any]:
    candidate = _candidate(record)
    model = _load_bound_checkpoint(root, record).to(device)
    n = sum(policy["test_support"])
    confusion = np.zeros((15, 15), dtype=np.int64)
    position = 0
    order_digest = hashlib.sha256()
    with tempfile.TemporaryDirectory(prefix="scores-", dir=scratch) as temporary:
        scores = targets_buffer = None
        if candidate.task == "E3":
            scores = np.memmap(Path(temporary) / "scores.bin", dtype=np.float64, mode="w+", shape=(n,))
            targets_buffer = np.memmap(Path(temporary) / "targets.bin", dtype=np.int8, mode="w+", shape=(n,))
        try:
            state_before = _state_digest(model)
            with torch.inference_mode():
                for data in iter_test_batches(root, candidate.task):
                    audit = getattr(data, "e2_audit", {})
                    indices = audit.get("ordered_source_sample_indices")
                    if audit.get("experiment") != candidate.task or audit.get("split") != "test" or not isinstance(indices, list) or not indices:
                        raise EvaluationError("cross-task/split or missing batch provenance")
                    if any(type(index) is not int for index in indices) or indices != list(range(position, position + len(indices))) or len(indices) > e10.BATCH_SIZE or position + len(indices) > n:
                        raise EvaluationError("test traversal is incomplete, repeated, or non-sequential")
                    order_digest.update(np.asarray(indices, dtype="<i8").tobytes())
                    data = data.to(device)
                    logits = _inference_logits(model, data, candidate.task, candidate.model_family)
                    targets = data["sample"].y
                    _validate_inference_outputs(candidate.task, logits, targets, len(indices))
                    y = targets.cpu().numpy()
                    if not np.isin(y, range(2 if candidate.task == "E3" else 15)).all():
                        raise EvaluationError("target domain violation")
                    if candidate.task == "E3":
                        assert scores is not None and targets_buffer is not None
                        scores[position:position + len(indices)] = logits.cpu().numpy()
                        targets_buffer[position:position + len(indices)] = y
                    else:
                        predictions = logits.argmax(dim=1).cpu().numpy()
                        confusion += np.bincount(y.astype(np.int64) * 15 + predictions, minlength=225).reshape(15, 15)
                    position += len(indices)
            if position != n or _state_digest(model) != state_before:
                raise EvaluationError("test coverage incomplete or model state changed during inference")
            metrics = binary_metrics(targets_buffer, scores, policy) if candidate.task == "E3" else multiclass_metrics(confusion, policy)
            return {"candidate_id": candidate.candidate_id, "task": candidate.task, "model_family": candidate.model_family,
                    "seed": candidate.seed, "learning_rate": candidate.learning_rate, "examples": position,
                    "order_sha256": order_digest.hexdigest(), "model_state_unchanged": True,
                    "checkpoint_sha256": record["artifacts"]["checkpoint_sha256"],
                    "checkpoint_provenance": record["checkpoint_provenance"],
                    "e10_result_canonical_sha256": record["result_sha256"], "metrics": metrics}
        finally:
            # Close newly-created score-only maps before Windows temp cleanup.
            for array in (scores, targets_buffer):
                if array is not None:
                    array.flush()
                    array._mmap.close()


def _state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    payload = d2.canonical_json_bytes(value) + b"\n"
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _status(directory: Path, state: str, **details: Any) -> None:
    path = directory / "status.json"
    temporary = directory / "status.next.json"
    _write_exclusive(temporary, {"state": state, **details})
    temporary.replace(path)


def summarize_seeds(results: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups = []
    for family in e10.MODEL_FAMILIES:
        for task in e10.TASKS:
            rows = [r for r in results if r["model_family"] == family and r["task"] == task]
            if sorted(r["seed"] for r in rows) != list(e10.FINAL_SEEDS):
                raise EvaluationError("seed summary requires exactly 2025, 2026, 2027")
            values = [r["metrics"]["pr_auc_average_precision"] if task == "E3" else r["metrics"]["primary"]["macro_f1"] for r in rows]
            valid = all(v is not None for v in values)
            if task == "E3":
                scalar_keys = ("pr_auc_average_precision", "roc_auc", "accuracy", "precision", "recall", "f1",
                               "specificity", "false_positive_rate", "balanced_accuracy", "mcc")
                summaries = {key: [r["metrics"].get(key) for r in rows] for key in scalar_keys}
            else:
                summaries = {f"{scope}.{key}": [r["metrics"].get(scope, {}).get(key) for r in rows]
                             for scope in ("primary", "all_test_descriptive")
                             for key in ("macro_precision", "macro_recall", "macro_f1", "accuracy", "weighted_f1")}
            metric_summary = {key: {"mean": statistics.mean(v) if all(x is not None for x in v) else None,
                                    "sample_std_ddof1": statistics.stdev(v) if all(x is not None for x in v) else None}
                              for key, v in summaries.items()}
            groups.append({"model_family": family, "task": task, "seeds": list(e10.FINAL_SEEDS),
                           "primary_mean": statistics.mean(values) if valid else None,
                           "primary_sample_std_ddof1": statistics.stdev(values) if valid else None,
                           "metrics_mean_sample_std": metric_summary,
                           "seed_metrics": [{"seed": r["seed"], "metrics": r["metrics"]} for r in rows],
                           "interpretation": "descriptive seed variation; no ensemble or confidence interval"})
    return groups


def run_owner_evaluation(root: Path, run_id: str, *, study: str = STUDY,
                         owner_confirmed_manual_test_evaluation: bool = False,
                         device: str = "cuda") -> dict[str, Any]:
    if owner_confirmed_manual_test_evaluation is not True:
        raise EvaluationError("explicit owner manual test evaluation acknowledgement is required")
    if not isinstance(run_id, str) or not RUN_PATTERN.fullmatch(run_id):
        raise EvaluationError("invalid E.11 run identity")
    root = Path(root).resolve()
    logs = root / "logs/stage_e11" / study
    outputs = root / "results/stage_e11" / study / run_id
    ledger = logs / "test_consumption.json"
    if ledger.exists():
        raise EvaluationError("official test already consumed or reserved; automatic retry is forbidden")
    # Everything that can be checked without test arrays is checked first.
    preparation = audit_e10(root, study)
    target_device = e7.resolve_device(device)
    source_hashes = {p: e10._sha256(root / p) for p in SOURCE_PATHS}
    environment = e10.environment_identity()
    trained_environment = preparation["training_environment"]
    for key in ("python", "python_implementation", "torch", "torch_geometric", "cuda_runtime"):
        if environment.get(key) != trained_environment.get(key):
            raise EvaluationError(f"evaluation environment differs from E.10: {key}")
    with file_guard(root):
        for directory in (logs, outputs, logs / run_id):
            if directory.resolve() != directory:
                raise EvaluationError("E.11 output namespace cannot contain links")
        logs.mkdir(parents=True, exist_ok=True)
        if outputs.exists() or (logs / run_id).exists():
            raise EvaluationError("E.11 requires fresh output directories")
        # O_EXCL serializes concurrent executions across *all* E.11 run IDs.
        _write_exclusive(ledger, {"stage": "E.11", "version": VERSION, "e10_study": study,
                                 "run_id": run_id, "state": "CONSUMED_OR_RESERVED",
                                 "e10_report_sha256": E10_REPORT_SHA256, "source_sha256": source_hashes})
        outputs.mkdir(parents=True)
        execution = logs / run_id
        execution.mkdir()
        manifest = {"stage": "E.11", "version": VERSION, "run_id": run_id,
                    "execution_authority": "owner_manual_only", "environment": environment,
                    "metric_library_versions": {"numpy": package_version("numpy"), "scikit_learn": package_version("scikit-learn")},
                    "source_sha256": source_hashes, "preparation": preparation,
                    "batch_size": 256, "test_order": "sequential", "training_executed": False,
                    "lr_selection_executed": False, "pruning_performed": False,
                    "threshold_tuned": False, "class_weighting_changed": False}
        manifest["manifest_sha256"] = d2.canonical_sha256(manifest)
        _write_exclusive(execution / "manifest.json", manifest)
        _status(execution, "RUNNING", completed_checkpoints=0)
    results = []
    try:
        with file_guard(root, allow_test=True):
            test_hashes = {relative: digest for split in preparation["splits"].values()
                           for relative, digest in split["predictive_artifact_sha256"].items()}
            # Owner-only verification: hashes are file-byte hashes, never inferred
            # from stored evidence alone. Hashing is read-only, before inference.
            for relative, expected in test_hashes.items():
                if e10._sha256(root / "data/processed/preprocessed" / relative) != expected:
                    raise EvaluationError("real test artifact differs from frozen Stage C evidence")
            for record in preparation["evaluation_records"]:
                result = evaluate_checkpoint(root, record, preparation["splits"][record["configuration"]["task"]]["policy"], execution, target_device)
                result["result_sha256"] = d2.canonical_sha256(result)
                _write_exclusive(outputs / f"{result['candidate_id']}.json", result)
                results.append(result)
                _status(execution, "RUNNING", completed_checkpoints=len(results))
                print(f"E.11 completed {len(results)}/27: {result['candidate_id']}", flush=True)
            for relative, expected in test_hashes.items():
                if e10._sha256(root / "data/processed/preprocessed" / relative) != expected:
                    raise EvaluationError("test artifact changed during evaluation")
            for relative, expected in preparation["e10_artifact_sha256"].items():
                if e10._sha256(root / relative) != expected:
                    raise EvaluationError("E.10 artifact changed during evaluation")
            for relative, expected in preparation["frozen_metadata_sha256"].items():
                if e10._sha256(root / relative) != expected:
                    raise EvaluationError("frozen metadata changed during evaluation")
            if {p: e10._sha256(root / p) for p in SOURCE_PATHS} != source_hashes:
                raise EvaluationError("execution source changed during evaluation")
            report = {"stage": "E.11", "version": VERSION, "run_id": run_id, "official_study_id": study,
                      "overall_status": "PASS", "evaluated_checkpoints": 27,
                      "manifest_canonical_sha256": manifest["manifest_sha256"],
                      "test_artifact_sha256": test_hashes, "results": results,
                      "seed_summary": summarize_seeds(results), "training_executed": False,
                      "lr_selection_executed": False, "pruning_performed": False,
                      "threshold_tuned": False, "e10_artifacts_unchanged": True}
            report["report_sha256"] = d2.canonical_sha256(report)
            report_path = outputs / "stage_e11_report.json"
            _write_exclusive(report_path, report)
            with (outputs / "stage_e11_report.sha256").open("x", encoding="ascii") as handle:
                handle.write(f"{e10._sha256(report_path)}  stage_e11_report.json\n")
            _status(execution, "COMPLETE", completed_checkpoints=27)
            return report
    except BaseException as exc:
        with file_guard(root):
            _status(execution, "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED",
                    completed_checkpoints=len(results), error_type=type(exc).__name__, error=str(exc))
        raise
