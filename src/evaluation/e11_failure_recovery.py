"""Reviewed continuation of the single E.11 target-dtype failure.

The original consumed marker, FAILED status, manifest and six results remain
immutable. This is not a general resume interface or an original-run retry.
Preparation is read-only; only the owner can launch the remaining 21 checkpoints.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.evaluation import official_test as e11
from src.training import full_training as e10, harness as e7


FAILED_RUN = "e11-official-20261006-v1"
RECOVERY_ID = "e11-recovery-20261006-v1"
CORRECTED_EVALUATOR_SHA256 = "ce41f85f285a0557ed9898c5c45deb6ba534e42804dab39e3cea521cf4c61da3"
FAILED_FILES = {
    "logs/stage_e11/e10-official-20261001-v1/test_consumption.json":
        "f1b528868e61c9650acb75692ae933bab1552aedb8b7fbffb6c53a25b58b0a0a",
    "logs/stage_e11/e10-official-20261001-v1/e11-official-20261006-v1/manifest.json":
        "0c6fa1e7fe67e36afb62cf6b128c3aed7a2f4bfdf16015d1f739b344b301675c",
    "logs/stage_e11/e10-official-20261001-v1/e11-official-20261006-v1/status.json":
        "5185ea83496865214c572c0c7a62efedffffd604bec4e7ee9888fafa6d886822",
    "results/stage_e11/e10-official-20261001-v1/e11-official-20261006-v1/e10-htgnn-e1-lr1e-3-s2025-b256-c2fd73b62871.json":
        "44f3bed7240d909a630d2425aab5726d20c764cd159ed0f5339fd0dc5e26ba4d",
    "results/stage_e11/e10-official-20261001-v1/e11-official-20261006-v1/e10-htgnn-e1-lr1e-3-s2026-b256-07fde048fbfd.json":
        "95686ffca2582c4022368c2421c0ff8231793f0426d6bde70318f06bd7ee2aef",
    "results/stage_e11/e10-official-20261001-v1/e11-official-20261006-v1/e10-htgnn-e1-lr1e-3-s2027-b256-79771d0a8e2c.json":
        "5e0f0e7b014e9fe4371d608d59d34b4ce840aaff7d75d16880ad64f5d8c03ca8",
    "results/stage_e11/e10-official-20261001-v1/e11-official-20261006-v1/e10-htgnn-e2-lr1e-3-s2025-b256-4c0dbe73d0e4.json":
        "9be44707b105403e784a9fb55cfa4397dfeb22db53d9d6c39f57e7ac36520b50",
    "results/stage_e11/e10-official-20261001-v1/e11-official-20261006-v1/e10-htgnn-e2-lr1e-3-s2026-b256-0a1122c0ee42.json":
        "6890e8c2a2c34fd9c4996810cc90ad2a7ecef1616cf410f6e1950fdca656baf1",
    "results/stage_e11/e10-official-20261001-v1/e11-official-20261006-v1/e10-htgnn-e2-lr1e-3-s2027-b256-3df7694884f0.json":
        "2b6091a02d9fc5baaa7b0d27e8d66ff0a5c165a298ce7ee378be4ffa28e5bacd",
}
RECOVERY_SOURCES = (
    *e11.SOURCE_PATHS, "src/evaluation/e11_failure_recovery.py",
    "scripts/recover_stage_e11_test_evaluation.py",
    "tests/test_stage_e11_failure_recovery.py", "docs/stage_e11_failure_recovery.md",
)
_REVIEWED_CHANGED_PATHS = frozenset(("src/evaluation/official_test.py",
                                   "tests/test_stage_e11_test_evaluation.py"))


def _check_hashes(root: Path, hashes: Mapping[str, str]) -> None:
    for relative, expected in hashes.items():
        if not (root / relative).is_file() or e10._sha256(root / relative) != expected:
            raise e11.EvaluationError(f"reviewed recovery input changed: {relative}")


def audit_failure(root: Path) -> dict[str, Any]:
    """Validate the exact reviewed failure and completed prefix without arrays."""
    root = Path(root).resolve()
    with e11.file_guard(root):
        _check_hashes(root, FAILED_FILES)
        if e10._sha256(root / "src/evaluation/official_test.py") != CORRECTED_EVALUATOR_SHA256:
            raise e11.EvaluationError("recovery requires the reviewed dtype-only evaluator correction")
        original_logs = root / "logs/stage_e11" / e11.STUDY / FAILED_RUN
        original_outputs = root / "results/stage_e11" / e11.STUDY / FAILED_RUN
        actual_files = {p.relative_to(root).as_posix() for directory in (original_logs, original_outputs)
                        for p in directory.rglob("*") if p.is_file()}
        expected_files = set(FAILED_FILES) - {f"logs/stage_e11/{e11.STUDY}/test_consumption.json"}
        if actual_files != expected_files or any(p.is_dir() for p in original_logs.iterdir()):
            raise e11.EvaluationError("reviewed failure has extra/missing artifacts or partial scratch state")
        marker = e11._read(root / "logs/stage_e11" / e11.STUDY / "test_consumption.json")
        manifest = e11._read(original_logs / "manifest.json")
        status = e11._read(original_logs / "status.json")
        e11._canonical_check(manifest, "manifest_sha256")
        if (marker.get("stage") != "E.11" or marker.get("e10_study") != e11.STUDY
                or marker.get("run_id") != FAILED_RUN or marker.get("state") != "CONSUMED_OR_RESERVED"
                or marker.get("source_sha256") != manifest.get("source_sha256")
                or marker.get("e10_report_sha256") != e11.E10_REPORT_SHA256):
            raise e11.EvaluationError("original consumed marker does not bind the reviewed failure")
        if status != {"completed_checkpoints": 6, "error": "invalid frozen output shape or nonfinite logits",
                      "error_type": "EvaluationError", "state": "FAILED"}:
            raise e11.EvaluationError("this recovery is limited to the audited six-checkpoint dtype failure")
        # Frozen inference/metric source is still identical; only the evaluator's
        # dtype validator and its synthetic regression tests are changed.
        _check_hashes(root, {p: h for p, h in manifest["source_sha256"].items()
                             if p not in _REVIEWED_CHANGED_PATHS})
        preparation = e11.audit_e10(root)
        if preparation != manifest["preparation"]:
            raise e11.EvaluationError("E.10/split authority differs from the original evaluation")
        completed = []
        records = preparation["evaluation_records"]
        for record in records[:6]:
            path = original_outputs / (record["candidate_id"] + ".json")
            result = e11._read(path)
            e11._canonical_check(result, "result_sha256")
            config = record["configuration"]
            policy = preparation["splits"][config["task"]]["policy"]
            if (result.get("candidate_id") != record["candidate_id"]
                    or result.get("checkpoint_sha256") != record["artifacts"]["checkpoint_sha256"]
                    or result.get("checkpoint_provenance") != record["checkpoint_provenance"]
                    or result.get("e10_result_canonical_sha256") != record["result_sha256"]
                    or result.get("model_state_unchanged") is not True
                    or any(result.get(k) != config[k] for k in ("task", "model_family", "seed", "learning_rate"))
                    or result.get("examples") != sum(policy["test_support"])):
                raise e11.EvaluationError("completed result checkpoint or coverage binding is invalid")
            digest = hashlib.sha256()
            for start in range(0, result["examples"], 4096):
                digest.update(np.arange(start, min(start + 4096, result["examples"]), dtype="<i8").tobytes())
            if result.get("order_sha256") != digest.hexdigest():
                raise e11.EvaluationError("completed result traversal hash is invalid")
            # Verify metrics from existing compact confusion data, never rerun a
            # model or open predictive arrays to validate completed test results.
            measured = e11.multiclass_metrics(np.asarray(result["metrics"]["confusion_matrix"], dtype=np.int64), policy)
            if result["metrics"] != measured:
                raise e11.EvaluationError("completed multiclass metrics disagree with saved confusion")
            completed.append(result)
        failed = records[6]
        if failed["candidate_id"] != "e10-htgnn-e3-lr1e-3-s2025-b256-b0ed07a45fa6":
            raise e11.EvaluationError("next checkpoint differs from the reviewed failure")
        return {"audit_status": "PASS", "failed_run_id": FAILED_RUN, "recovery_id": RECOVERY_ID,
                "failed_candidate_id": failed["candidate_id"], "reused_checkpoints": 6,
                "remaining_checkpoints": 21, "start_checkpoint": 7,
                "test_predictive_arrays_accessed": False, "original_file_sha256": dict(FAILED_FILES),
                "original_manifest_canonical_sha256": manifest["manifest_sha256"],
                "original_environment": manifest["environment"],
                "original_metric_library_versions": manifest["metric_library_versions"],
                "preparation": preparation, "completed_results": completed}


def run_owner_recovery(root: Path, *, owner_confirmed_reviewed_failure_recovery: bool = False) -> dict[str, Any]:
    """Continue only the missing suffix, preserving original consumption state.

    Requires separate explicit owner acknowledgement, pins the failed attempt
    and six completed results, and reserves this one reviewed continuation with
    exclusive creation. Original evaluate remains blocked by the original marker.
    """
    if owner_confirmed_reviewed_failure_recovery is not True:
        raise e11.EvaluationError("explicit owner acknowledgement of reviewed failure recovery is required")
    root = Path(root).resolve()
    recovery_logs = root / "logs/stage_e11" / e11.STUDY / "recovery" / RECOVERY_ID
    recovery_outputs = root / "results/stage_e11" / e11.STUDY / "recovery" / RECOVERY_ID
    if recovery_logs.exists() or recovery_outputs.exists():
        raise e11.EvaluationError("reviewed recovery already reserved; automatic recovery retry is forbidden")
    audit = audit_failure(root)
    preparation = audit["preparation"]
    environment = e10.environment_identity()
    if environment != audit["original_environment"]:
        raise e11.EvaluationError("recovery environment differs from the original owner execution")
    libraries = {"numpy": e11.package_version("numpy"), "scikit_learn": e11.package_version("scikit-learn")}
    if libraries != audit["original_metric_library_versions"]:
        raise e11.EvaluationError("recovery metric library versions changed")
    device = e7.resolve_device("cuda")
    sources = {p: e10._sha256(root / p) for p in RECOVERY_SOURCES}
    with e11.file_guard(root):
        if recovery_logs.resolve() != recovery_logs or recovery_outputs.resolve() != recovery_outputs:
            raise e11.EvaluationError("recovery namespace cannot contain links")
        recovery_logs.parent.mkdir(parents=True, exist_ok=True)
        # Atomic mkdir reserves the *continuation*, not a new test consumption.
        # If the process fails after reservation, another automatic attempt fails.
        recovery_logs.mkdir(exist_ok=False)
        recovery_manifest = {"stage": "E.11", "recovery_id": RECOVERY_ID, "failed_run_id": FAILED_RUN,
                             "reason": "correct frozen E3 float32 target validation; no methodology change",
                             "owner_confirmed_reviewed_failure_recovery": True,
                             "original_marker_preserved": True, "source_sha256": sources,
                             "environment": environment, "metric_library_versions": libraries,
                             "audit": audit, "start_checkpoint": 7, "test_order": "sequential",
                             "training_executed": False, "pruning_performed": False,
                             "threshold_tuned": False, "lr_selection_executed": False}
        recovery_manifest["manifest_sha256"] = e11.d2.canonical_sha256(recovery_manifest)
        e11._write_exclusive(recovery_logs / "recovery_authorization.json", recovery_manifest)
        recovery_outputs.mkdir(parents=True, exist_ok=False)
        e11._status(recovery_logs, "RUNNING", reused_checkpoints=6, newly_completed_checkpoints=0)
    results = list(audit["completed_results"])
    try:
        with e11.file_guard(root, allow_test=True):
            test_hashes = {p: h for split in preparation["splits"].values()
                           for p, h in split["predictive_artifact_sha256"].items()}
            _check_hashes(root, {f"data/processed/preprocessed/{p}": h for p, h in test_hashes.items()})
            for record in preparation["evaluation_records"][6:]:
                policy = preparation["splits"][record["configuration"]["task"]]["policy"]
                result = e11.evaluate_checkpoint(root, record, policy, recovery_logs, device)
                result["result_sha256"] = e11.d2.canonical_sha256(result)
                e11._write_exclusive(recovery_outputs / (result["candidate_id"] + ".json"), result)
                results.append(result)
                e11._status(recovery_logs, "RUNNING", reused_checkpoints=6, newly_completed_checkpoints=len(results) - 6)
                print(f"E.11 reviewed continuation completed {len(results)}/27: {result['candidate_id']}", flush=True)
            _check_hashes(root, {f"data/processed/preprocessed/{p}": h for p, h in test_hashes.items()})
            _check_hashes(root, FAILED_FILES)
            _check_hashes(root, preparation["e10_artifact_sha256"])
            _check_hashes(root, preparation["frozen_metadata_sha256"])
            _check_hashes(root, sources)
            report = {"stage": "E.11", "version": e11.VERSION, "overall_status": "PASS",
                      "official_study_id": e11.STUDY, "failed_run_id": FAILED_RUN, "recovery_id": RECOVERY_ID,
                      "original_failure_preserved": True, "original_marker_preserved": True,
                      "reused_checkpoints": 6, "newly_evaluated_checkpoints": 21, "evaluated_checkpoints": 27,
                      "original_file_sha256": dict(FAILED_FILES), "test_artifact_sha256": test_hashes,
                      "recovery_manifest_canonical_sha256": recovery_manifest["manifest_sha256"],
                      "results": results, "seed_summary": e11.summarize_seeds(results),
                      "training_executed": False, "pruning_performed": False,
                      "threshold_tuned": False, "lr_selection_executed": False}
            report["report_sha256"] = e11.d2.canonical_sha256(report)
            report_path = recovery_outputs / "stage_e11_report.json"
            e11._write_exclusive(report_path, report)
            with (recovery_outputs / "stage_e11_report.sha256").open("x", encoding="ascii") as handle:
                handle.write(f"{e10._sha256(report_path)}  stage_e11_report.json\n")
            e11._status(recovery_logs, "COMPLETE", reused_checkpoints=6, newly_completed_checkpoints=21)
            return report
    except BaseException as exc:
        with e11.file_guard(root):
            e11._status(recovery_logs, "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED",
                        reused_checkpoints=6, newly_completed_checkpoints=len(results) - 6,
                        error_type=type(exc).__name__, error=str(exc))
        raise
