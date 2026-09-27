"""Deterministic compact evidence helpers for Stage E.7."""

from __future__ import annotations

import json
from typing import Any, Mapping

from src.graph import schema as d2
from src.preprocessing import splitting as stage_c_splitting
from src.training import harness as e7


class TrainingEvidenceError(ValueError):
    """Raised when Stage E.7 evidence is incomplete or inconsistent."""


def build_report(
    *,
    frozen_source_sha256: Mapping[str, str],
    correction_source_sha256: Mapping[str, str],
    smoke_validation: list[Mapping[str, Any]],
    integrity: Mapping[str, Any],
    verification: Mapping[str, Any],
) -> dict[str, Any]:
    e2_smoke = next(
        (item for item in smoke_validation if item.get("task") == "E2"), {}
    )
    report: dict[str, Any] = {
        "stage": "E.7",
        "harness_version": e7.HARNESS_VERSION,
        "scope": "production training harness; no full study or test evaluation",
        "tasks": {
            "E1": {
                "loss": "unweighted CrossEntropyLoss on raw [B,15] logits",
                "validation_selection": e7.SELECTION_NAMES["E1"],
            },
            "E2": {
                "loss": "unweighted CrossEntropyLoss on raw [B,15] logits",
                "validation_selection": e7.SELECTION_NAMES["E2"],
            },
            "E3": {
                "loss": "unweighted BCEWithLogitsLoss on raw [B] logits",
                "validation_selection": e7.SELECTION_NAMES["E3"],
            },
        },
        "optimizer": {
            "name": "AdamW",
            "weight_decay": e7.WEIGHT_DECAY,
            "learning_rate_candidates_supported": list(e7.LEARNING_RATE_CANDIDATES),
            "scheduler": None,
        },
        "early_stopping": {
            "max_epochs": e7.MAX_EPOCHS,
            "minimum_epochs": e7.MIN_EPOCHS,
            "patience": e7.PATIENCE,
            "min_delta": e7.MIN_DELTA,
            "selection_partition": "validation only",
            "checkpoint_policy": "best qualifying validation checkpoint",
        },
        "seeds": {
            "approved": list(e7.APPROVED_SEEDS),
            "components": ["python", "numpy", "torch_cpu", "torch_cuda"],
            "cuda_determinism": "best effort; bitwise equality is not claimed",
        },
        "data_order": {
            "production_path": "D4 -> D6 -> E.2",
            "train": "seeded bounded affine permutation",
            "validation": "sequential",
            "test": "not accepted by E.7 training/selection",
            "full_split_feature_or_graph_materialization": False,
        },
        "e2_validation_selection_policy": {
            "authority": "authoritative E2 train and validation target memmaps",
            "thresholds": {
                "min_train": stage_c_splitting.ELIGIBILITY_MIN_TRAIN,
                "min_val": stage_c_splitting.ELIGIBILITY_MIN_VAL,
            },
            "validation_selection_class_codes": e2_smoke.get(
                "validation_selection_class_codes"
            ),
            "test_labels_consulted": False,
            "test_support_consulted": False,
            "test_membership_consulted": False,
            "test_metadata_consulted": False,
            "final_reporting_eligibility_consulted": False,
        },
        "e2_final_reporting_policy": {
            "separate_from_validation_selection": True,
            "thresholds": {
                "min_train": stage_c_splitting.ELIGIBILITY_MIN_TRAIN,
                "min_val": stage_c_splitting.ELIGIBILITY_MIN_VAL,
                "min_test": stage_c_splitting.ELIGIBILITY_MIN_TEST,
            },
            "use": "later locked-test/final reporting only",
        },
        "checkpoint": {
            "schema_version": e7.CHECKPOINT_SCHEMA_VERSION,
            "state_dict_only": True,
            "contains_optimizer_state": True,
            "contains_task_seed_lr_epoch_validation_and_frozen_identifiers": True,
            "task_and_architecture_mismatch_rejected": True,
            "model_weights_committed": False,
        },
        "logging": {
            "ownership": "one JSONL file per training run",
            "existing_paths_rejected": True,
            "empty_precreated_paths_rejected": True,
            "exclusive_atomic_reservation": True,
            "checkpoint_log_alias_rejected": True,
            "validated_before_training": True,
            "resume_or_cross_run_append_supported": False,
        },
        "weighted_ablations": "not implemented in E.7",
        "parameter_counts": dict(e7.EXPECTED_PARAMETERS),
        "frozen_source_sha256": dict(frozen_source_sha256),
        "correction_source_sha256": dict(correction_source_sha256),
        "bounded_real_smoke": smoke_validation,
        "integrity": dict(integrity),
        "verification": dict(verification),
    }
    smoke_passed = (
        len(smoke_validation) == 3
        and {item.get("task") for item in smoke_validation} == set(e7.TASKS)
        and all(item.get("status") == "PASS" for item in smoke_validation)
    )
    verification_passed = all(
        item.get("status") == "PASS" and int(item.get("passed", 0)) > 0
        for key, item in verification.items()
        if key != "status" and isinstance(item, Mapping)
    ) and verification.get("status") == "PASS"
    integrity_passed = (
        integrity.get("stage_c") == "PASS"
        and integrity.get("raw_csv") == "PASS"
    )
    report["overall_status"] = (
        "PASS" if smoke_passed and verification_passed and integrity_passed else "FAIL"
    )
    report["report_sha256"] = d2.canonical_sha256(report)
    validate_report(report)
    return report


def validate_report(report: Mapping[str, Any]) -> None:
    errors: list[str] = []
    if report.get("stage") != "E.7":
        errors.append("stage must be E.7")
    if report.get("harness_version") != e7.HARNESS_VERSION:
        errors.append("harness version mismatch")
    if report.get("parameter_counts") != e7.EXPECTED_PARAMETERS:
        errors.append("frozen parameter counts changed")
    correction_hashes = report.get("correction_source_sha256", {})
    required_correction_sources = {
        "src/preprocessing/splitting.py",
        "src/training/harness.py",
        "src/training/validation.py",
        "tests/test_preprocessing_splitting.py",
        "tests/test_training_harness.py",
    }
    if (
        not isinstance(correction_hashes, Mapping)
        or set(correction_hashes) != required_correction_sources
        or not all(
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
            for value in correction_hashes.values()
        )
    ):
        errors.append("correction source identity is incomplete")
    if report.get("optimizer", {}).get("weight_decay") != e7.WEIGHT_DECAY:
        errors.append("optimizer weight decay changed")
    stopping = report.get("early_stopping", {})
    if (
        stopping.get("max_epochs") != e7.MAX_EPOCHS
        or stopping.get("minimum_epochs") != e7.MIN_EPOCHS
        or stopping.get("patience") != e7.PATIENCE
        or stopping.get("min_delta") != e7.MIN_DELTA
    ):
        errors.append("early stopping contract changed")
    smoke = report.get("bounded_real_smoke")
    if (
        not isinstance(smoke, list)
        or {item.get("task") for item in smoke if isinstance(item, Mapping)}
        != set(e7.TASKS)
        or not all(
            isinstance(item, Mapping) and item.get("status") == "PASS"
            for item in smoke
        )
    ):
        errors.append("bounded real smoke coverage is incomplete")
    if report.get("checkpoint", {}).get("model_weights_committed") is not False:
        errors.append("checkpoint policy permits committed model weights")
    logging = report.get("logging", {})
    if (
        logging.get("ownership") != "one JSONL file per training run"
        or logging.get("existing_paths_rejected") is not True
        or logging.get("empty_precreated_paths_rejected") is not True
        or logging.get("exclusive_atomic_reservation") is not True
        or logging.get("checkpoint_log_alias_rejected") is not True
        or logging.get("validated_before_training") is not True
        or logging.get("resume_or_cross_run_append_supported") is not False
    ):
        errors.append("single-run JSONL ownership contract changed")
    e2_selection = report.get("e2_validation_selection_policy", {})
    if (
        e2_selection.get("thresholds") != {"min_train": 100, "min_val": 30}
        or not isinstance(
            e2_selection.get("validation_selection_class_codes"), list
        )
        or not e2_selection.get("validation_selection_class_codes")
        or any(
            e2_selection.get(key) is not False
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
    e2_reporting = report.get("e2_final_reporting_policy", {})
    if (
        e2_reporting.get("separate_from_validation_selection") is not True
        or e2_reporting.get("thresholds")
        != {"min_train": 100, "min_val": 30, "min_test": 30}
    ):
        errors.append("E2 final-reporting policy is not separate")
    verification = report.get("verification", {})
    if not isinstance(verification, Mapping) or verification.get("status") != "PASS":
        errors.append("verification status is not PASS")
    else:
        for name in (
            "focused_e7",
            "frozen_e2_e6_regression",
            "complete_project_suite",
        ):
            item = verification.get(name, {})
            if item.get("status") != "PASS" or int(item.get("passed", 0)) <= 0:
                errors.append(f"{name} was not run successfully")
    if report.get("overall_status") != "PASS":
        errors.append("overall status is not PASS")
    supplied = report.get("report_sha256")
    unhashed = dict(report)
    unhashed.pop("report_sha256", None)
    if supplied != d2.canonical_sha256(unhashed):
        errors.append("canonical report hash mismatch")
    if errors:
        raise TrainingEvidenceError(errors)


def report_bytes(report: Mapping[str, Any]) -> bytes:
    validate_report(report)
    return d2.canonical_json_bytes(report)
