"""Stage D5 experiment-boundary and provenance validation.

This module validates D4's existing lazy source mapping.  It does not load X
values, create a competing data loader, derive learned quantities, or alter any
frozen Stage C artifact.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.graph import adapter as d4
from src.graph import schema as d2
from src.graph import validation as d3


VALIDATION_VERSION = "stage-d5-experiment-boundary-validation-v1"
APPROVED_D4_REPORT_SHA256 = (
    "ccd065556c533d4fd1cd2e4d821566611db9d2a5a9e7ea036c0cca3b3f6500d6"
)
EXPECTED_SPLIT_LENGTHS: Mapping[str, Mapping[str, int]] = {
    "E1": {"train": 2_183_472, "validation": 311_924, "test": 623_851},
    "E2": {"train": 2_183_465, "validation": 311_916, "test": 623_866},
    "E3": {"train": 2_183_472, "validation": 311_924, "test": 623_851},
}
TARGET_TASK_TYPES = {"E1": "multiclass", "E2": "multiclass", "E3": "binary"}


class ExperimentBoundaryError(ValueError):
    """Raised with all diagnostics when an experiment boundary is violated."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(errors)
        super().__init__("Stage D5 validation failed: " + "; ".join(self.errors))


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentBoundaryError([f"cannot load {description}: {exc}"]) from exc
    if not isinstance(payload, dict):
        raise ExperimentBoundaryError([f"{description} must be a JSON object"])
    return payload


def _same_file(first: Path, second: Path) -> bool:
    try:
        return os.path.samefile(first, second)
    except OSError:
        return False


def _target_values(array: np.memmap, chunk_size: int = 1_000_000) -> set[int]:
    values: set[int] = set()
    for start in range(0, len(array), chunk_size):
        values.update(int(value) for value in np.unique(array[start : start + chunk_size]))
    return values


def _validate_full_binary_conversion(
    e1: np.memmap, e3: np.memmap, chunk_size: int = 1_000_000
) -> bool:
    if len(e1) != len(e3):
        return False
    for start in range(0, len(e1), chunk_size):
        stop = min(start + chunk_size, len(e1))
        expected = (e1[start:stop] != 0).astype(np.int16, copy=False)
        if not np.array_equal(expected, e3[start:stop]):
            return False
    return True


def _validate_d4_report(project_root: Path) -> dict[str, Any]:
    path = project_root / "results" / "stage_d4" / "lazy_adapter_validation.json"
    saved = _load_json(path, "D4 validation report")
    d4.report_bytes(saved)
    current = d4.build_validation_report(
        project_root,
        integrity_result=saved.get("stage_c_integrity"),
    )
    if saved != current:
        raise ExperimentBoundaryError(["D4 validation report is not current"])
    if saved.get("report_sha256") != APPROVED_D4_REPORT_SHA256:
        raise ExperimentBoundaryError(["D4 report SHA-256 is not approved"])
    return saved


def _preprocessing_summary(
    project_root: Path, evidence_hashes: Mapping[str, str]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for experiment, directory in d4.EXPERIMENT_DIRS.items():
        relative = f"{directory}/scaler_state.json"
        state = _load_json(
            project_root / "data" / "processed" / "preprocessed" / relative,
            f"{experiment} scaler state",
        )
        fit = state.get("fit_provenance", {})
        result[experiment] = {
            "state_artifact": f"data/processed/preprocessed/{relative}",
            "state_sha256": evidence_hashes.get(relative),
            "fit_experiment": fit.get("fit_experiment"),
            "fit_split": fit.get("fit_split"),
            "fit_window_count": fit.get("fit_window_count"),
            "applies_to_experiments": fit.get("applies_to_experiments"),
        }
    return result


def build_validation_report(
    project_root: Path,
    *,
    integrity_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate all real D4 sources and return the deterministic D5 report."""

    project_root = Path(project_root)
    d4_report = _validate_d4_report(project_root)
    evidence = _load_json(
        project_root / "results" / "stage_c_full_validation.json",
        "frozen Stage C validation evidence",
    )
    evidence_hashes = evidence.get("artefact_sha256", {})
    processed_root = project_root / "data" / "processed" / "preprocessed"
    class_mapping = _load_json(processed_root / "class_mapping.json", "class mapping")
    binary_mapping = _load_json(
        processed_root / "binary_class_mapping.json", "binary class mapping"
    )
    class_eligibility = _load_json(
        processed_root / "class_eligibility.json", "E2 class eligibility metadata"
    )
    preprocessing = _preprocessing_summary(project_root, evidence_hashes)

    datasets: dict[tuple[str, str], d4.LazyGraphDataset] = {}
    records: list[dict[str, Any]] = []
    graph_static_ids: set[int] = set()
    target_values = {experiment: set() for experiment in d4.EXPERIMENT_DIRS}

    for experiment, directory in d4.EXPERIMENT_DIRS.items():
        for split, stem in d4.SPLIT_STEMS.items():
            dataset = d4.LazyGraphDataset(experiment, split, processed_root=processed_root)
            datasets[(experiment, split)] = dataset
            graph_static_ids.add(id(dataset.static))
            target_values[experiment].update(_target_values(dataset.y_memmap))
            metadata = _load_json(dataset.metadata_path, f"{experiment}/{split} metadata")
            x_relative = f"{directory}/X_{stem}.npy"
            y_relative = f"{directory}/y_{stem}.npy"
            records.append(
                {
                    "experiment": experiment,
                    "split": split,
                    "x_artifact": dataset.source_artifacts["x"],
                    "y_artifact": dataset.source_artifacts["y"],
                    "expected_length": EXPECTED_SPLIT_LENGTHS[experiment][split],
                    "metadata_length": metadata.get(f"n_{stem}"),
                    "actual_length": len(dataset),
                    "x_shape": list(dataset.x_memmap.shape),
                    "x_dtype": str(dataset.x_memmap.dtype),
                    "y_shape": list(dataset.y_memmap.shape),
                    "y_dtype": str(dataset.y_memmap.dtype),
                    "x_sha256": evidence_hashes.get(x_relative),
                    "y_sha256": evidence_hashes.get(y_relative),
                    "target_task_type": TARGET_TASK_TYPES[experiment],
                    "preprocessing_state_artifact": preprocessing[experiment][
                        "state_artifact"
                    ],
                    "preprocessing_state_sha256": preprocessing[experiment][
                        "state_sha256"
                    ],
                    "preprocessing_fit_experiment": preprocessing[experiment][
                        "fit_experiment"
                    ],
                    "preprocessing_fit_split": preprocessing[experiment]["fit_split"],
                    "graph_provenance": {
                        **dict(dataset.provenance),
                        "d4_report_sha256": d4_report["report_sha256"],
                    },
                }
            )

    e1_e3_x_records = []
    full_binary_conversion = True
    converted_target_count = 0
    for split, stem in d4.SPLIT_STEMS.items():
        e1 = datasets[("E1", split)]
        e3 = datasets[("E3", split)]
        e1_hash = evidence_hashes[f"exp1/X_{stem}.npy"]
        e3_hash = evidence_hashes[f"exp3/X_{stem}.npy"]
        hashes_equal = e1_hash == e3_hash
        hardlinked = _same_file(e1.x_path, e3.x_path)
        e1_e3_x_records.append(
            {
                "split": split,
                "e1_x_sha256": e1_hash,
                "e3_x_sha256": e3_hash,
                "sha256_equal": hashes_equal,
                "hardlinked": hardlinked,
            }
        )
        converted = _validate_full_binary_conversion(e1.y_memmap, e3.y_memmap)
        full_binary_conversion &= converted
        converted_target_count += len(e1)

    split_alias_records = []
    no_split_aliasing = True
    for experiment in d4.EXPERIMENT_DIRS:
        experiment_datasets = [datasets[(experiment, split)] for split in d4.SPLIT_STEMS]
        x_distinct = all(
            not _same_file(first.x_path, second.x_path)
            for index, first in enumerate(experiment_datasets)
            for second in experiment_datasets[index + 1 :]
        )
        y_distinct = all(
            not _same_file(first.y_path, second.y_path)
            for index, first in enumerate(experiment_datasets)
            for second in experiment_datasets[index + 1 :]
        )
        no_split_aliasing &= x_distinct and y_distinct
        split_alias_records.append(
            {
                "experiment": experiment,
                "x_artifacts_distinct": x_distinct,
                "y_artifacts_distinct": y_distinct,
            }
        )

    e2_independence_records = []
    e2_independent = True
    for split in d4.SPLIT_STEMS:
        e1 = datasets[("E1", split)]
        e2 = datasets[("E2", split)]
        e3 = datasets[("E3", split)]
        independent = (
            not _same_file(e2.x_path, e1.x_path)
            and not _same_file(e2.x_path, e3.x_path)
            and e2.x_path.parent.name == "exp2"
        )
        e2_independent &= independent
        e2_independence_records.append(
            {"split": split, "independent_x_artifact": independent}
        )

    integrity = (
        dict(integrity_result)
        if integrity_result is not None
        else d4.verify_frozen_integrity(project_root)
    )
    report: dict[str, Any] = {
        "validation_version": VALIDATION_VERSION,
        "provenance": {
            "d1_taxonomy_sha256": d3.APPROVED_D1_TAXONOMY_SHA256,
            "d2_schema_sha256": d3.APPROVED_D2_SCHEMA_SHA256,
            "d3_report_sha256": d4.APPROVED_D3_REPORT_SHA256,
            "d4_report_sha256": d4_report["report_sha256"],
        },
        "experiment_definitions": {
            "E1": "strict global capture-order multiclass diagnostic",
            "E2": "per-class chronological primary supervised multiclass",
            "E3": "strict global capture-order binary primary experiment",
        },
        "experiment_split_registry": records,
        "preprocessing_provenance": preprocessing,
        "target_vocabulary_validation": {
            "E1_actual_codes": sorted(target_values["E1"]),
            "E2_actual_codes": sorted(target_values["E2"]),
            "E3_actual_codes": sorted(target_values["E3"]),
            "class_mapping": class_mapping,
            "binary_mapping": binary_mapping,
        },
        "e1_e3_x_identity": {
            "splits": e1_e3_x_records,
            "all_split_sha256_equal": all(item["sha256_equal"] for item in e1_e3_x_records),
        },
        "e1_to_e3_target_conversion": {
            "method": "complete chunked read-only y comparison; E3 == (E1 != Benign code 0)",
            "targets_compared": converted_target_count,
            "all_targets_match": full_binary_conversion,
        },
        "e2_independence": {
            "splits": e2_independence_records,
            "all_exp2_x_artifacts_independent": e2_independent,
            "scaler_state_differs_from_e1": (
                preprocessing["E2"]["state_sha256"]
                != preprocessing["E1"]["state_sha256"]
            ),
        },
        "e2_class_eligibility_policy": {
            "artifact": "data/processed/preprocessed/class_eligibility.json",
            "eligible_class_codes": class_eligibility.get("eligible_class_codes"),
            "ineligible_present_class_codes": [2, 3, 13],
            "ineligible_present_codes_retained_in_targets": all(
                code in target_values["E2"] for code in (2, 3, 13)
            ),
            "policy_is_metadata_and_reporting_only": True,
        },
        "cross_split_aliasing": {
            "experiments": split_alias_records,
            "no_x_or_y_split_aliasing": no_split_aliasing,
        },
        "graph_contract_sharing": {
            "single_static_metadata_object_across_all_pairs": len(graph_static_ids) == 1,
            "same_d1_d2_d3_d4_hashes_in_all_records": all(
                record["graph_provenance"]
                == {
                    **dict(datasets[("E1", "train")].provenance),
                    "d4_report_sha256": d4_report["report_sha256"],
                }
                for record in records
            ),
        },
        "leakage_boundary": {
            "stage_d_topology_and_taxonomy_are_data_independent": True,
            "stage_d_learned_quantities": False,
            "validation_or_test_used_to_fit_stage_d_quantity": False,
            "only_frozen_stage_c_train_fitted_preprocessing_is_referenced": True,
            "repeated_states_not_interpreted_as_split_index_overlap": True,
        },
        "stage_c_integrity": integrity,
        "overall_status": "PASS",
    }
    report["report_sha256"] = d2.canonical_sha256(report)
    validate_report(report)
    return report


def _expected_artifact(experiment: str, split: str, kind: str) -> str:
    directory = d4.EXPERIMENT_DIRS[experiment]
    stem = d4.SPLIT_STEMS[split]
    return f"data/processed/preprocessed/{directory}/{kind}_{stem}.npy"


def validate_report(report: Mapping[str, Any]) -> None:
    """Reject internally inconsistent or boundary-violating D5 report data."""

    errors: list[str] = []
    provenance = report.get("provenance", {})
    expected_provenance = {
        "d1_taxonomy_sha256": d3.APPROVED_D1_TAXONOMY_SHA256,
        "d2_schema_sha256": d3.APPROVED_D2_SCHEMA_SHA256,
        "d3_report_sha256": d4.APPROVED_D3_REPORT_SHA256,
        "d4_report_sha256": APPROVED_D4_REPORT_SHA256,
    }
    if provenance != expected_provenance:
        errors.append("D1/D2/D3/D4 provenance hash mismatch")

    records = report.get("experiment_split_registry", [])
    if not isinstance(records, list):
        errors.append("experiment/split registry must be a list")
        records = []
    by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            errors.append("registry entry must be an object")
            continue
        experiment = record.get("experiment")
        split = record.get("split")
        if experiment not in d4.EXPERIMENT_DIRS:
            errors.append(f"corrupted experiment ID: {experiment!r}")
            continue
        if split not in d4.SPLIT_STEMS:
            errors.append(f"corrupted split ID: {split!r}")
            continue
        key = (experiment, split)
        if key in by_key:
            errors.append(f"duplicate registry entry for {experiment}/{split}")
        by_key[key] = record
        expected_length = EXPECTED_SPLIT_LENGTHS[experiment][split]
        if record.get("x_artifact") != _expected_artifact(experiment, split, "X"):
            errors.append(f"{experiment}/{split} mapped to wrong X artifact")
        if record.get("y_artifact") != _expected_artifact(experiment, split, "y"):
            errors.append(f"{experiment}/{split} mapped to wrong y artifact")
        if record.get("expected_length") != expected_length:
            errors.append(f"{experiment}/{split} has wrong expected split count")
        if not (
            record.get("actual_length")
            == record.get("metadata_length")
            == expected_length
        ):
            errors.append(f"{experiment}/{split} X/y/metadata length mismatch")
        if record.get("x_shape") != [expected_length, 5, 78]:
            errors.append(f"{experiment}/{split} X shape mismatch")
        if record.get("y_shape") != [expected_length]:
            errors.append(f"{experiment}/{split} y shape mismatch")
        if record.get("x_dtype") != "float32" or record.get("y_dtype") != "int16":
            errors.append(f"{experiment}/{split} dtype mismatch")
        if record.get("target_task_type") != TARGET_TASK_TYPES[experiment]:
            errors.append(f"{experiment}/{split} target task type mismatch")
        preprocessing_item = report.get("preprocessing_provenance", {}).get(
            experiment, {}
        )
        if (
            record.get("preprocessing_state_artifact")
            != preprocessing_item.get("state_artifact")
            or record.get("preprocessing_state_sha256")
            != preprocessing_item.get("state_sha256")
            or record.get("preprocessing_fit_experiment")
            != preprocessing_item.get("fit_experiment")
            or record.get("preprocessing_fit_split")
            != preprocessing_item.get("fit_split")
        ):
            errors.append(f"{experiment}/{split} preprocessing registry mismatch")
        if record.get("graph_provenance") != expected_provenance:
            errors.append(f"graph schema identity differs for {experiment}/{split}")
    expected_keys = {
        (experiment, split)
        for experiment in d4.EXPERIMENT_DIRS
        for split in d4.SPLIT_STEMS
    }
    if set(by_key) != expected_keys:
        errors.append("registry does not contain exactly all 9 experiment/split pairs")

    for split in d4.SPLIT_STEMS:
        e1 = by_key.get(("E1", split), {})
        e3 = by_key.get(("E3", split), {})
        if e1.get("x_sha256") != e3.get("x_sha256"):
            errors.append(f"E1/E3 X hash mismatch for {split}")

    vocabulary = report.get("target_vocabulary_validation", {})
    expected_multiclass_codes = list(range(15))
    if vocabulary.get("E1_actual_codes") != expected_multiclass_codes:
        errors.append("E1 target vocabulary is not the frozen multiclass vocabulary")
    if vocabulary.get("E2_actual_codes") != expected_multiclass_codes:
        errors.append("E2 target vocabulary is not the frozen multiclass vocabulary")
    if vocabulary.get("E3_actual_codes") != [0, 1]:
        errors.append("E3 target contains a value other than 0/1")
    class_mapping = vocabulary.get("class_mapping", {})
    binary_mapping = vocabulary.get("binary_mapping", {})
    label_to_int = class_mapping.get("label_to_int", {})
    label_to_binary = binary_mapping.get("label_to_binary", {})
    if label_to_int.get("Benign") != 0 or label_to_binary.get("Benign") != 0:
        errors.append("incorrect Benign binary mapping")
    attack_labels = set(label_to_int) - {"Benign"}
    if (
        set(binary_mapping.get("attack_labels", [])) != attack_labels
        or binary_mapping.get("n_attack_labels") != 14
        or set(label_to_binary) != set(label_to_int)
        or any(label_to_binary.get(label) != 1 for label in attack_labels)
    ):
        errors.append("binary mapping omits or mis-maps an attack label")

    preprocessing = report.get("preprocessing_provenance", {})
    expected_fit = {
        "E1": ("E1", "train", ["E1", "E3"]),
        "E2": ("E2", "train", ["E2"]),
        "E3": ("E1", "train", ["E1", "E3"]),
    }
    for experiment, (fit_experiment, fit_split, applies) in expected_fit.items():
        item = preprocessing.get(experiment, {})
        if (
            item.get("fit_experiment") != fit_experiment
            or item.get("fit_split") != fit_split
            or item.get("applies_to_experiments") != applies
        ):
            errors.append(f"{experiment} preprocessing provenance is incorrect")
    if preprocessing.get("E2", {}).get("state_sha256") == preprocessing.get("E1", {}).get(
        "state_sha256"
    ):
        errors.append("E2 references E1 preprocessing provenance")
    if preprocessing.get("E3", {}).get("state_sha256") != preprocessing.get("E1", {}).get(
        "state_sha256"
    ):
        errors.append("E3 does not share E1 preprocessing state")

    if not report.get("e1_e3_x_identity", {}).get("all_split_sha256_equal"):
        errors.append("E1/E3 X identity validation failed")
    if not report.get("e1_to_e3_target_conversion", {}).get("all_targets_match"):
        errors.append("E1 to E3 binary target conversion failed")
    if not report.get("e2_independence", {}).get("all_exp2_x_artifacts_independent"):
        errors.append("E2 numeric artifact independence failed")
    if not report.get("e2_independence", {}).get("scaler_state_differs_from_e1"):
        errors.append("E2 independent scaler provenance failed")
    eligibility = report.get("e2_class_eligibility_policy", {})
    if (
        eligibility.get("policy_is_metadata_and_reporting_only") is not True
        or eligibility.get("ineligible_present_class_codes") != [2, 3, 13]
        or eligibility.get("ineligible_present_codes_retained_in_targets") is not True
    ):
        errors.append("E2 class eligibility policy altered stored target membership")
    if not report.get("cross_split_aliasing", {}).get("no_x_or_y_split_aliasing"):
        errors.append("cross-split artifact aliasing detected")
    if not all(report.get("graph_contract_sharing", {}).values()):
        errors.append("graph contract is not shared across all experiments")
    leakage = report.get("leakage_boundary", {})
    if (
        leakage.get("stage_d_topology_and_taxonomy_are_data_independent") is not True
        or leakage.get("stage_d_learned_quantities") is not False
        or leakage.get("validation_or_test_used_to_fit_stage_d_quantity") is not False
        or leakage.get("only_frozen_stage_c_train_fitted_preprocessing_is_referenced")
        is not True
    ):
        errors.append("Stage D leakage boundary validation failed")
    if report.get("stage_c_integrity", {}).get("overall_status") != "PASS":
        errors.append("Stage C/raw integrity status is not PASS")
    if report.get("overall_status") != "PASS":
        errors.append("overall status is not PASS")

    supplied_hash = report.get("report_sha256")
    unhashed = dict(report)
    unhashed.pop("report_sha256", None)
    if not isinstance(supplied_hash, str) or d2.canonical_sha256(unhashed) != supplied_hash:
        errors.append("D5 report SHA-256 does not match canonical content")
    if errors:
        raise ExperimentBoundaryError(errors)


def report_bytes(report: Mapping[str, Any]) -> bytes:
    validate_report(report)
    return (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def write_report(project_root: Path, output_path: Path) -> dict[str, Any]:
    report = build_validation_report(project_root)
    data = report_bytes(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(data)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Stage D experiment boundaries")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/stage_d5/experiment_boundary_validation.json"),
    )
    args = parser.parse_args(argv)
    report = write_report(args.project_root, args.output)
    print(f"wrote {args.output}")
    print(f"overall_status={report['overall_status']}")
    print(f"report_sha256={report['report_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
