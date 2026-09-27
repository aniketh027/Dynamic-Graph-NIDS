"""Validate Stage E.8 and write compact deterministic contract evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph import adapter as d4  # noqa: E402
from src.graph import schema as d2  # noqa: E402
from src.training import harness as e7  # noqa: E402
from src.training import optimization_pilot as e8  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_pytest(project_root: Path, args: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *args],
        cwd=project_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    passed = re.findall(r"(\d+) passed", completed.stdout)
    skipped = re.findall(r"(\d+) skipped", completed.stdout)
    return {
        "command": "python -m pytest -q " + " ".join(args),
        "exit_code": completed.returncode,
        "passed": int(passed[-1]) if passed else 0,
        "skipped": int(skipped[-1]) if skipped else 0,
        "status": "PASS" if completed.returncode == 0 else "FAIL",
    }


def _build_report(
    *,
    project_root: Path,
    observations_path: Path,
    observations: Mapping[str, Any],
    verification: Mapping[str, Any],
) -> dict[str, Any]:
    frozen_e7_hashes = e8.require_frozen_contracts(project_root)
    e8.validate_observations(observations)
    integrity_result = d4.verify_frozen_integrity(project_root)
    integrity = {
        "stage_c": "PASS" if integrity_result["stage_c_sha256_matches"] else "FAIL",
        "stage_c_matched": integrity_result["stage_c_artifact_count"],
        "stage_c_expected": integrity_result["stage_c_expected_count"],
        "raw_csv": "PASS" if integrity_result["raw_csv_sha256_matches"] else "FAIL",
        "raw_csv_matched": integrity_result["raw_csv_count"],
        "raw_csv_expected": integrity_result["raw_csv_expected_count"],
    }
    source_hashes = {
        "src/training/optimization_pilot.py": _sha256(
            project_root / "src/training/optimization_pilot.py"
        ),
        "scripts/run_stage_e8_optimization_pilot.py": _sha256(
            project_root / "scripts/run_stage_e8_optimization_pilot.py"
        ),
        "scripts/validate_stage_e8.py": _sha256(
            project_root / "scripts/validate_stage_e8.py"
        ),
        "tests/test_optimization_pilot.py": _sha256(
            project_root / "tests/test_optimization_pilot.py"
        ),
    }
    candidates = e8.enumerate_candidates()
    report: dict[str, Any] = {
        "stage": "E.8",
        "pilot_version": e8.PILOT_VERSION,
        "scope": "bounded train/validation learning-rate optimization pilot",
        "pilot_bounds": {
            "tasks": list(e8.PILOT_TASKS),
            "learning_rates": list(e8.PILOT_LEARNING_RATES),
            "loss_mode": e8.PILOT_LOSS_MODE,
            "seed": e8.PILOT_SEED,
            "batch_size": e8.PILOT_BATCH_SIZE,
            "max_epochs": e8.PILOT_MAX_EPOCHS,
            "max_train_batches_per_epoch": e8.PILOT_MAX_TRAIN_BATCHES,
            "max_validation_batches_per_epoch": (
                e8.PILOT_MAX_VALIDATION_BATCHES
            ),
            "production_e7_max_epochs_unchanged": e7.MAX_EPOCHS,
        },
        "candidate_definitions": [candidate.contract_dict() | {
            "candidate_id": candidate.candidate_id,
            "candidate_identity_sha256": candidate.identity_sha256,
        } for candidate in candidates],
        "candidate_fairness": {
            "same_seed_within_task": True,
            "same_initialization_contract_within_task": True,
            "same_train_membership_order_within_task": True,
            "same_validation_membership_order_within_task": True,
            "same_batch_and_epoch_budgets_within_task": True,
            "only_learning_rate_varies_within_task": True,
        },
        "data_boundary": {
            "allowed_partitions": list(e8.ALLOWED_DATA_PARTITIONS),
            "test_accessed": False,
            "test_metrics_recorded": False,
            "enforcement": (
                "E.8 exposes no split parameter and delegates only to frozen E.7 "
                "run_training, whose production path constructs train and validation."
            ),
        },
        "validation_objectives": dict(e8.EXPECTED_SELECTION_NAMES),
        "optimization": {
            "optimizer": "AdamW",
            "weight_decay": e7.WEIGHT_DECAY,
            "scheduler": None,
            "weighted_loss_ablations_included": False,
        },
        "interpretation": {
            "observational_ranking_rule": e8.OBSERVATIONAL_RANKING_RULE,
            "production_configuration_finalized": False,
            "multi_seed_robustness_established": False,
            "full_training_result": False,
            "locked_test_result": False,
        },
        "run_isolation": {
            "fresh_paths_required": True,
            "checkpoint_log_alias_rejected": True,
            "candidate_paths_unique": True,
            "existing_candidate_result_rejected": True,
            "aggregate_requires_all_complete_identity_matching_results": True,
            "model_weights_committed": False,
        },
        "completed_candidate_identities": (
            [candidate.candidate_id for candidate in candidates]
            if observations.get("overall_status") == "PASS"
            else []
        ),
        "observations": {
            "path": observations_path.relative_to(project_root).as_posix(),
            "physical_sha256": _sha256(observations_path),
            "run_id": observations.get("run_id"),
            "overall_status": observations.get("overall_status"),
            "content_class": "observational",
        },
        "frozen_e7_source_sha256": frozen_e7_hashes,
        "stage_e8_source_sha256": source_hashes,
        "frozen_parameter_counts": dict(e7.EXPECTED_PARAMETERS),
        "integrity": integrity,
        "verification": dict(verification),
    }
    verification_passed = (
        verification.get("status") == "PASS"
        and all(
            item.get("status") == "PASS" and int(item.get("passed", 0)) > 0
            for key, item in verification.items()
            if key != "status" and isinstance(item, Mapping)
        )
    )
    report["overall_status"] = (
        "PASS"
        if observations.get("overall_status") == "PASS"
        and integrity["stage_c"] == "PASS"
        and integrity["raw_csv"] == "PASS"
        and verification_passed
        else "FAIL"
    )
    report["report_sha256"] = d2.canonical_sha256(report)
    validate_report(report)
    return report


def validate_report(report: Mapping[str, Any]) -> None:
    errors: list[str] = []
    if report.get("stage") != "E.8" or report.get("pilot_version") != e8.PILOT_VERSION:
        errors.append("stage or pilot version mismatch")
    if report.get("frozen_parameter_counts") != e7.EXPECTED_PARAMETERS:
        errors.append("frozen parameter counts changed")
    if len(report.get("candidate_definitions", [])) != 9:
        errors.append("candidate definitions are incomplete")
    if len(report.get("completed_candidate_identities", [])) != 9:
        errors.append("completed candidate identities are incomplete")
    data_boundary = report.get("data_boundary", {})
    if (
        data_boundary.get("allowed_partitions") != ["train", "validation"]
        or data_boundary.get("test_accessed") is not False
        or data_boundary.get("test_metrics_recorded") is not False
    ):
        errors.append("test exclusion contract failed")
    interpretation = report.get("interpretation", {})
    if interpretation.get("production_configuration_finalized") is not False:
        errors.append("pilot incorrectly finalized a configuration")
    if report.get("observations", {}).get("overall_status") != "PASS":
        errors.append("pilot observations did not pass")
    integrity = report.get("integrity", {})
    if integrity.get("stage_c") != "PASS" or integrity.get("raw_csv") != "PASS":
        errors.append("frozen data integrity failed")
    verification = report.get("verification", {})
    if verification.get("status") != "PASS":
        errors.append("verification did not pass")
    for key in ("focused_e8", "frozen_e2_e7_regression", "complete_project_suite"):
        item = verification.get(key, {})
        if item.get("status") != "PASS" or int(item.get("passed", 0)) <= 0:
            errors.append(f"{key} did not pass")
    if report.get("overall_status") != "PASS":
        errors.append("overall status is not PASS")
    supplied = report.get("report_sha256")
    unhashed = dict(report)
    unhashed.pop("report_sha256", None)
    if supplied != d2.canonical_sha256(unhashed):
        errors.append("canonical report hash mismatch")
    if errors:
        raise e8.OptimizationPilotError("; ".join(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--observations",
        type=Path,
        default=Path("results/stage_e8/htgnn_optimization_pilot_observations.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/stage_e8/htgnn_optimization_pilot_contract.json"),
    )
    parser.add_argument("--run-tests", action="store_true")
    args = parser.parse_args()
    if not args.run_tests:
        parser.error("--run-tests is required when writing canonical E.8 evidence")
    project_root = args.project_root.resolve()
    observations_path = (
        args.observations
        if args.observations.is_absolute()
        else project_root / args.observations
    )
    output = args.output if args.output.is_absolute() else project_root / args.output
    observations = json.loads(observations_path.read_text(encoding="utf-8"))
    e8.validate_observations(observations)
    verification: dict[str, Any] = {
        "focused_e8": _run_pytest(
            project_root, ["tests/test_optimization_pilot.py"]
        ),
        "frozen_e2_e7_regression": _run_pytest(
            project_root,
            [
                "tests/test_pyg_adapter.py",
                "tests/test_e2_provenance_cache.py",
                "tests/test_feature_encoder.py",
                "tests/test_hierarchical_encoder.py",
                "tests/test_htgnn_classifier.py",
                "tests/test_resource_pilot.py",
                "tests/test_training_harness.py",
            ],
        ),
        "complete_project_suite": _run_pytest(project_root, []),
    }
    verification["status"] = (
        "PASS"
        if all(item["status"] == "PASS" for item in verification.values())
        else "FAIL"
    )
    report = _build_report(
        project_root=project_root,
        observations_path=observations_path,
        observations=observations,
        verification=verification,
    )
    payload = d2.canonical_json_bytes(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    physical_hash = hashlib.sha256(payload).hexdigest()
    output.with_suffix(".sha256").write_text(
        f"{physical_hash}  {output.name}\n", encoding="ascii"
    )
    print(f"overall_status={report['overall_status']}")
    print(f"report_sha256={report['report_sha256']}")
    print(f"physical_sha256={physical_hash}")
    print(f"output={output}")
    return 0 if report["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
