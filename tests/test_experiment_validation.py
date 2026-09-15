"""Stage D5 real-provenance and adversarial boundary tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.graph import schema as d2
from src.graph.experiment_validation import (
    APPROVED_D4_REPORT_SHA256,
    EXPECTED_SPLIT_LENGTHS,
    VALIDATION_VERSION,
    ExperimentBoundaryError,
    build_validation_report,
    report_bytes,
    validate_report,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "results" / "stage_d5" / "experiment_boundary_validation.json"
EXPECTED_REPORT_SHA256 = "1ddfa6caaaf33a04d8b4a973b39b3b815be9ef11904bd0b6e5bf91a95f898b16"


@pytest.fixture()
def report() -> dict[str, object]:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def _record(report: dict[str, object], experiment: str, split: str) -> dict[str, object]:
    return next(
        item
        for item in report["experiment_split_registry"]
        if item["experiment"] == experiment and item["split"] == split
    )


def _rehash(report: dict[str, object]) -> None:
    report.pop("report_sha256", None)
    report["report_sha256"] = d2.canonical_sha256(report)


def _assert_rejected(report: dict[str, object], message: str) -> None:
    _rehash(report)
    with pytest.raises(ExperimentBoundaryError) as caught:
        validate_report(report)
    assert message in str(caught.value)


def test_real_report_validates_all_nine_pairs_and_hashes(report) -> None:
    validate_report(report)
    assert report["validation_version"] == VALIDATION_VERSION
    assert report["report_sha256"] == EXPECTED_REPORT_SHA256
    assert len(report["experiment_split_registry"]) == 9
    assert report["overall_status"] == "PASS"
    assert report["provenance"]["d4_report_sha256"] == APPROVED_D4_REPORT_SHA256


def test_real_split_lengths_shapes_dtypes_and_tasks(report) -> None:
    for experiment, splits in EXPECTED_SPLIT_LENGTHS.items():
        for split, length in splits.items():
            record = _record(report, experiment, split)
            assert record["expected_length"] == record["metadata_length"] == record["actual_length"] == length
            assert record["x_shape"] == [length, 5, 78]
            assert record["y_shape"] == [length]
            assert record["x_dtype"] == "float32"
            assert record["y_dtype"] == "int16"
            assert record["target_task_type"] == ("binary" if experiment == "E3" else "multiclass")
            assert record["preprocessing_state_sha256"] == report["preprocessing_provenance"][experiment]["state_sha256"]
            assert record["preprocessing_fit_experiment"] == report["preprocessing_provenance"][experiment]["fit_experiment"]
            assert record["preprocessing_fit_split"] == "train"


def test_real_e1_e3_x_hashes_and_full_target_conversion(report) -> None:
    for item in report["e1_e3_x_identity"]["splits"]:
        assert item["e1_x_sha256"] == item["e3_x_sha256"]
        assert item["sha256_equal"] is True
    assert report["e1_e3_x_identity"]["all_split_sha256_equal"] is True
    assert report["e1_to_e3_target_conversion"] == {
        "method": "complete chunked read-only y comparison; E3 == (E1 != Benign code 0)",
        "targets_compared": 3_119_247,
        "all_targets_match": True,
    }


def test_real_target_vocabularies_and_binary_mapping(report) -> None:
    vocabulary = report["target_vocabulary_validation"]
    assert vocabulary["E1_actual_codes"] == list(range(15))
    assert vocabulary["E2_actual_codes"] == list(range(15))
    assert vocabulary["E3_actual_codes"] == [0, 1]
    mapping = vocabulary["binary_mapping"]
    assert mapping["label_to_binary"]["Benign"] == 0
    assert len(mapping["attack_labels"]) == 14
    assert all(mapping["label_to_binary"][label] == 1 for label in mapping["attack_labels"])


def test_real_preprocessing_provenance_is_train_only_and_independent(report) -> None:
    preprocessing = report["preprocessing_provenance"]
    assert preprocessing["E1"]["fit_experiment"] == "E1"
    assert preprocessing["E1"]["fit_split"] == "train"
    assert preprocessing["E2"]["fit_experiment"] == "E2"
    assert preprocessing["E2"]["fit_split"] == "train"
    assert preprocessing["E3"]["fit_experiment"] == "E1"
    assert preprocessing["E3"]["fit_split"] == "train"
    assert preprocessing["E2"]["state_sha256"] != preprocessing["E1"]["state_sha256"]
    assert preprocessing["E3"]["state_sha256"] == preprocessing["E1"]["state_sha256"]
    assert report["e2_independence"]["all_exp2_x_artifacts_independent"] is True
    assert report["e2_class_eligibility_policy"]["policy_is_metadata_and_reporting_only"] is True
    assert report["e2_class_eligibility_policy"]["ineligible_present_codes_retained_in_targets"] is True


def test_real_splits_do_not_alias_and_graph_contract_is_shared(report) -> None:
    assert report["cross_split_aliasing"]["no_x_or_y_split_aliasing"] is True
    assert all(
        item["x_artifacts_distinct"] and item["y_artifacts_distinct"]
        for item in report["cross_split_aliasing"]["experiments"]
    )
    assert all(report["graph_contract_sharing"].values())
    assert report["leakage_boundary"]["stage_d_learned_quantities"] is False
    assert report["leakage_boundary"]["validation_or_test_used_to_fit_stage_d_quantity"] is False


def test_report_regenerates_byte_identically_without_rehashing_large_x(report) -> None:
    first = build_validation_report(ROOT, integrity_result=report["stage_c_integrity"])
    second = build_validation_report(ROOT, integrity_result=report["stage_c_integrity"])
    assert first == second == report
    assert report_bytes(first) == report_bytes(second) == REPORT_PATH.read_bytes()


@pytest.mark.parametrize(
    ("experiment", "wrong_directory"),
    [("E1", "exp2"), ("E2", "exp1"), ("E3", "exp2")],
)
def test_wrong_experiment_x_mapping_is_rejected(report, experiment: str, wrong_directory: str) -> None:
    record = _record(report, experiment, "train")
    record["x_artifact"] = f"data/processed/preprocessed/{wrong_directory}/X_train.npy"
    _assert_rejected(report, f"{experiment}/train mapped to wrong X artifact")


@pytest.mark.parametrize(
    ("split", "wrong_stem"),
    [("train", "val"), ("validation", "test")],
)
def test_cross_split_mapping_is_rejected(report, split: str, wrong_stem: str) -> None:
    record = _record(report, "E1", split)
    record["x_artifact"] = f"data/processed/preprocessed/exp1/X_{wrong_stem}.npy"
    _assert_rejected(report, f"E1/{split} mapped to wrong X artifact")


def test_x_y_length_mismatch_is_rejected(report) -> None:
    _record(report, "E2", "test")["y_shape"][0] -= 1
    _assert_rejected(report, "E2/test y shape mismatch")


def test_wrong_expected_split_count_is_rejected(report) -> None:
    _record(report, "E1", "validation")["expected_length"] += 1
    _assert_rejected(report, "E1/validation has wrong expected split count")


def test_e1_e3_x_hash_mismatch_is_rejected(report) -> None:
    _record(report, "E3", "train")["x_sha256"] = "0" * 64
    _assert_rejected(report, "E1/E3 X hash mismatch for train")


def test_nonbinary_e3_target_vocabulary_is_rejected(report) -> None:
    report["target_vocabulary_validation"]["E3_actual_codes"] = [0, 1, 2]
    _assert_rejected(report, "E3 target contains a value other than 0/1")


def test_incorrect_benign_mapping_is_rejected(report) -> None:
    report["target_vocabulary_validation"]["binary_mapping"]["label_to_binary"]["Benign"] = 1
    _assert_rejected(report, "incorrect Benign binary mapping")


def test_omitted_attack_label_is_rejected(report) -> None:
    mapping = report["target_vocabulary_validation"]["binary_mapping"]
    omitted = mapping["attack_labels"].pop()
    mapping["label_to_binary"].pop(omitted)
    _assert_rejected(report, "binary mapping omits or mis-maps an attack label")


def test_e2_referencing_e1_preprocessing_is_rejected(report) -> None:
    e1 = report["preprocessing_provenance"]["E1"]
    e2 = report["preprocessing_provenance"]["E2"]
    e2["fit_experiment"] = "E1"
    e2["state_sha256"] = e1["state_sha256"]
    _assert_rejected(report, "E2 preprocessing provenance is incorrect")


def test_corrupted_experiment_id_is_rejected(report) -> None:
    _record(report, "E1", "train")["experiment"] = "E4"
    _assert_rejected(report, "corrupted experiment ID")


def test_corrupted_split_id_is_rejected(report) -> None:
    _record(report, "E1", "train")["split"] = "val"
    _assert_rejected(report, "corrupted split ID")


@pytest.mark.parametrize(
    "field",
    [
        "d1_taxonomy_sha256",
        "d2_schema_sha256",
        "d3_report_sha256",
        "d4_report_sha256",
    ],
)
def test_corrupted_stage_d_provenance_hash_is_rejected(report, field: str) -> None:
    report["provenance"][field] = "0" * 64
    _assert_rejected(report, "D1/D2/D3/D4 provenance hash mismatch")


def test_graph_schema_identity_differing_across_experiments_is_rejected(report) -> None:
    _record(report, "E2", "train")["graph_provenance"]["d2_schema_sha256"] = "0" * 64
    _assert_rejected(report, "graph schema identity differs for E2/train")


def test_e1_to_e3_conversion_failure_is_rejected(report) -> None:
    report["e1_to_e3_target_conversion"]["all_targets_match"] = False
    _assert_rejected(report, "E1 to E3 binary target conversion failed")


def test_cross_split_file_alias_is_rejected(report) -> None:
    report["cross_split_aliasing"]["no_x_or_y_split_aliasing"] = False
    _assert_rejected(report, "cross-split artifact aliasing detected")


def test_corrupted_report_hash_is_rejected(report) -> None:
    report["report_sha256"] = "0" * 64
    with pytest.raises(ExperimentBoundaryError, match="D5 report SHA-256"):
        validate_report(report)
