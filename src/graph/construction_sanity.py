"""Stage D7 bounded end-to-end graph-input construction sanity validation.

The runner reads only five deterministic samples per frozen experiment/split,
reuses D4 and D6 directly, and constructs no framework or model objects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.graph import adapter as d4
from src.graph import batching as d6
from src.graph import experiment_validation as d5
from src.graph import schema as d2
from src.graph import taxonomy as d1
from src.graph import validation as d3


SANITY_VERSION = "stage-d7-bounded-construction-sanity-v1"
APPROVED_D6_REPORT_SHA256 = (
    "5e9808efba9754dc40af9503a7ef7fc70cca8204e2ca24e72eac1620376e07ac"
)
APPROVED_PROVENANCE: Mapping[str, str] = {
    "d1_taxonomy_sha256": d3.APPROVED_D1_TAXONOMY_SHA256,
    "d2_schema_sha256": d3.APPROVED_D2_SCHEMA_SHA256,
    "d3_report_sha256": d4.APPROVED_D3_REPORT_SHA256,
    "d4_report_sha256": d5.APPROVED_D4_REPORT_SHA256,
    "d5_report_sha256": d6.APPROVED_D5_REPORT_SHA256,
    "d6_report_sha256": APPROVED_D6_REPORT_SHA256,
}


class ConstructionSanityError(ValueError):
    """Raised when a D7 logical input violates the approved contract."""


@dataclass(frozen=True, slots=True)
class SanityRunResult:
    """Internal deterministic result of one bounded construction pass."""

    records: tuple[Mapping[str, Any], ...]
    topology_identity: str
    e1_e3_features_equal: bool
    e1_e3_targets_convert: bool
    e2_provenance_valid: bool
    no_model_input_leakage: bool


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def deterministic_sanity_indices(length: int) -> tuple[int, ...]:
    """Select first, quartiles, midpoint, and last without randomness."""

    if isinstance(length, bool) or not isinstance(length, int):
        raise TypeError("dataset length must be an integer")
    if length < 5:
        raise ConstructionSanityError("dataset length must support five unique indices")
    indices = (0, length // 4, length // 2, (3 * length) // 4, length - 1)
    if len(set(indices)) != 5:
        raise ConstructionSanityError("deterministic selection produced duplicate indices")
    return indices


def _topology_payload(static: d4.StaticGraphMetadata) -> dict[str, Any]:
    return {
        "feature_ids": list(static.feature_ids),
        "feature_names": list(static.feature_names),
        "feature_category_ids": list(static.feature_category_ids),
        "category_ids": list(static.category_ids),
        "sample_node_id": static.sample_node_id,
        "feature_to_category_sources": list(static.feature_to_category_sources),
        "feature_to_category_destinations": list(
            static.feature_to_category_destinations
        ),
        "category_to_sample_sources": list(static.category_to_sample_sources),
        "category_to_sample_destinations": list(
            static.category_to_sample_destinations
        ),
        "d1_taxonomy_sha256": static.d1_taxonomy_sha256,
        "d2_schema_sha256": static.d2_schema_sha256,
    }


def topology_identity(static: d4.StaticGraphMetadata) -> str:
    return d2.canonical_sha256(_topology_payload(static))


def _update_array_hash(
    digest: Any, name: str, array: np.ndarray, dtype: str
) -> None:
    payload = np.ascontiguousarray(array, dtype=np.dtype(dtype)).tobytes(order="C")
    name_bytes = name.encode("utf-8")
    digest.update(len(name_bytes).to_bytes(4, "big"))
    digest.update(name_bytes)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def logical_sample_fingerprint(sample: d4.LogicalGraphSample) -> str:
    """Hash canonical metadata followed by exact little-endian logical values."""

    metadata = {
        "fingerprint_version": "d7-logical-sample-sha256-v1",
        "experiment": sample.experiment,
        "split": sample.split,
        "sample_index": sample.sample_index,
        "target": int(sample.target),
        "d1_taxonomy_sha256": sample.static.d1_taxonomy_sha256,
        "d2_schema_sha256": sample.static.d2_schema_sha256,
        "topology_identity": topology_identity(sample.static),
        "feature_x_shape": list(sample.feature_x.shape),
        "feature_x_dtype": str(sample.feature_x.dtype),
    }
    digest = hashlib.sha256(d2.canonical_json_bytes(metadata))
    _update_array_hash(digest, "feature_x", sample.feature_x, "<f4")
    _update_array_hash(digest, "target", np.asarray([sample.target]), "<i2")
    return digest.hexdigest()


def logical_batch_fingerprint(batch: d6.LogicalGraphBatch) -> str:
    """Hash ordered D6 values, topology, ownership, and D1-D6 provenance."""

    metadata = {
        "fingerprint_version": "d7-logical-batch-sha256-v1",
        "experiment": batch.experiment,
        "split": batch.split,
        "sample_indices": batch.sample_indices.tolist(),
        "feature_x_shape": list(batch.feature_x.shape),
        "targets_shape": list(batch.targets.shape),
        "provenance": dict(APPROVED_PROVENANCE),
        "topology_identity": topology_identity(batch.static),
    }
    digest = hashlib.sha256(d2.canonical_json_bytes(metadata))
    arrays = (
        ("feature_x", batch.feature_x, "<f4"),
        ("targets", batch.targets, "<i2"),
        ("sample_indices", batch.sample_indices, "<i8"),
        ("feature_batch", batch.feature_batch, "<i8"),
        ("category_batch", batch.category_batch, "<i8"),
        ("sample_batch", batch.sample_batch, "<i8"),
        ("feature_to_category_sources", batch.feature_to_category_sources, "<i8"),
        (
            "feature_to_category_destinations",
            batch.feature_to_category_destinations,
            "<i8",
        ),
        ("category_to_sample_sources", batch.category_to_sample_sources, "<i8"),
        (
            "category_to_sample_destinations",
            batch.category_to_sample_destinations,
            "<i8",
        ),
    )
    for name, array, dtype in arrays:
        _update_array_hash(digest, name, array, dtype)
    return digest.hexdigest()


def validate_stage_provenance(provenance: Mapping[str, str]) -> None:
    if dict(provenance) != dict(APPROVED_PROVENANCE):
        differing = [
            key
            for key, value in APPROVED_PROVENANCE.items()
            if provenance.get(key) != value
        ]
        raise ConstructionSanityError(
            "corrupted Stage D provenance: " + ", ".join(differing)
        )


def validate_sample_contract(
    dataset: d4.LazyGraphDataset,
    source_index: int,
    sample: d4.LogicalGraphSample,
) -> None:
    """Independently validate one D4 sample against its exact Stage C row."""

    if sample.experiment != dataset.experiment:
        raise ConstructionSanityError("sample experiment mismatch")
    if sample.split != dataset.split:
        raise ConstructionSanityError("sample split mismatch")
    if sample.sample_index != source_index:
        raise ConstructionSanityError("sample index mismatch")
    if sample.static is not dataset.static or dataset.static is not d4.STATIC_GRAPH_METADATA:
        raise ConstructionSanityError("sample topology identity mismatch")
    if sample.feature_x.shape != (78, 5):
        raise ConstructionSanityError("wrong feature shape")
    if sample.feature_x.dtype != np.dtype(np.float32):
        raise ConstructionSanityError("wrong feature dtype")
    if not isinstance(sample.target, np.int16):
        raise ConstructionSanityError("wrong target dtype")

    expected = dataset.x_memmap[source_index].T
    if np.array_equal(sample.feature_x, expected[::-1, :], equal_nan=True):
        raise ConstructionSanityError("feature order mismatch")
    if np.array_equal(sample.feature_x, expected[:, ::-1], equal_nan=True):
        raise ConstructionSanityError("temporal order reversal")
    if not np.array_equal(sample.feature_x, expected, equal_nan=True):
        raise ConstructionSanityError("feature value mismatch")
    if sample.target != dataset.y_memmap[source_index]:
        raise ConstructionSanityError("target mismatch")

    static = sample.static
    if (
        static.feature_ids != tuple(range(78))
        or static.feature_names != d1.MODEL_FEATURE_COLUMNS
        or static.feature_category_ids != d1.FEATURE_INDEX_TO_CATEGORY_ID
        or static.category_ids != tuple(range(8))
        or static.sample_node_id != 0
    ):
        raise ConstructionSanityError("feature identity or category alignment mismatch")
    if d2.NODE_COUNTS != {"feature": 78, "category": 8, "sample": 1}:
        raise ConstructionSanityError("wrong graph node contract")
    if d2.TOTAL_NODE_COUNT != 87 or d2.TOTAL_EDGE_COUNT != 86:
        raise ConstructionSanityError("wrong graph node or edge total")
    if dict(d1.EXPECTED_CATEGORY_COUNTS) != {
        "Context": 2,
        "Time": 23,
        "Packets and size": 24,
        "Rates and ratios": 5,
        "Flags": 12,
        "Bulk": 6,
        "Subflow": 4,
        "Window": 2,
    }:
        raise ConstructionSanityError("category membership counts changed")


def validate_batch_contract(
    dataset: d4.LazyGraphDataset,
    requested_indices: Sequence[int],
    batch: d6.LogicalGraphBatch,
) -> None:
    """Validate exact D4/D6 equivalence, offsets, counts, and isolation."""

    indices = tuple(requested_indices)
    size = len(indices)
    if batch.experiment != dataset.experiment:
        raise ConstructionSanityError("batch experiment mismatch")
    if batch.split != dataset.split:
        raise ConstructionSanityError("batch split mismatch")
    if batch.source_dataset_length != len(dataset):
        raise ConstructionSanityError("batch source length mismatch")
    if batch.sample_indices.tolist() != list(indices):
        raise ConstructionSanityError("batch requested index order mismatch")
    if batch.feature_x.shape != (size, 78, 5) or batch.feature_x.dtype != np.float32:
        raise ConstructionSanityError("batch feature shape or dtype mismatch")
    if batch.targets.shape != (size,) or batch.targets.dtype != np.int16:
        raise ConstructionSanityError("batch target shape or dtype mismatch")
    batch_arrays = (
        batch.feature_x,
        batch.targets,
        batch.sample_indices,
        batch.feature_batch,
        batch.category_batch,
        batch.sample_batch,
        batch.feature_to_category_sources,
        batch.feature_to_category_destinations,
        batch.category_to_sample_sources,
        batch.category_to_sample_destinations,
    )
    if dataset.x_memmap.flags.writeable or dataset.y_memmap.flags.writeable:
        raise ConstructionSanityError("source memory map is writable")
    if any(array.flags.writeable for array in batch_arrays):
        raise ConstructionSanityError("batch array is writable")
    if batch.feature_to_category_sources.size != size * 78:
        raise ConstructionSanityError("wrong feature-to-category edge count")
    if batch.category_to_sample_sources.size != size * 8:
        raise ConstructionSanityError("wrong category-to-sample edge count")
    if batch.total_edge_count != size * 86:
        raise ConstructionSanityError("wrong total edge count")

    feature_owner = batch.feature_to_category_sources // 78
    feature_destination_owner = batch.feature_to_category_destinations // 8
    category_owner = batch.category_to_sample_sources // 8
    if not np.array_equal(feature_owner, feature_destination_owner):
        raise ConstructionSanityError("cross-sample feature-to-category edge")
    if not np.array_equal(category_owner, batch.category_to_sample_destinations):
        raise ConstructionSanityError("cross-sample category-to-sample edge")
    if not np.array_equal(batch.feature_batch, np.repeat(np.arange(size), 78)):
        raise ConstructionSanityError("feature ownership mismatch")
    if not np.array_equal(batch.category_batch, np.repeat(np.arange(size), 8)):
        raise ConstructionSanityError("category ownership mismatch")
    if not np.array_equal(batch.sample_batch, np.arange(size)):
        raise ConstructionSanityError("sample ownership mismatch")

    for position, source_index in enumerate(indices):
        sample = dataset[source_index]
        validate_sample_contract(dataset, source_index, sample)
        if not np.array_equal(batch.feature_x[position], sample.feature_x, equal_nan=True):
            raise ConstructionSanityError("batch feature value mismatch")
        if batch.targets[position] != sample.target:
            raise ConstructionSanityError("batch target mismatch")


def validate_repeatability(first: SanityRunResult, second: SanityRunResult) -> None:
    if first != second:
        raise ConstructionSanityError("nondeterministic fingerprint or sanity result")


def _load_approved_d6_report(project_root: Path) -> dict[str, Any]:
    path = project_root / "results" / "stage_d6" / "template_batching_validation.json"
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        d6.report_bytes(report)
    except Exception as exc:
        raise ConstructionSanityError(f"D6 validation report is invalid: {exc}") from exc
    if report.get("report_sha256") != APPROVED_D6_REPORT_SHA256:
        raise ConstructionSanityError("D6 validation report hash is not approved")
    provenance = {**report.get("provenance", {}), "d6_report_sha256": report["report_sha256"]}
    validate_stage_provenance(provenance)
    return report


def _same_file(first: Path, second: Path) -> bool:
    try:
        return os.path.samefile(first, second)
    except OSError:
        return False


def _execute_once(project_root: Path) -> SanityRunResult:
    datasets: dict[tuple[str, str], d4.LazyGraphDataset] = {}
    records: list[Mapping[str, Any]] = []
    observed_topologies: set[str] = set()
    no_leakage = True
    processed_root = project_root / "data" / "processed" / "preprocessed"

    for experiment in d4.EXPERIMENT_DIRS:
        for split in d4.SPLIT_STEMS:
            dataset = d4.LazyGraphDataset(
                experiment, split, processed_root=processed_root
            )
            datasets[(experiment, split)] = dataset
            indices = deterministic_sanity_indices(len(dataset))
            samples = [dataset[index] for index in indices]
            sample_fingerprints: list[str] = []
            for index, sample in zip(indices, samples, strict=True):
                validate_sample_contract(dataset, index, sample)
                sample_fingerprints.append(logical_sample_fingerprint(sample))
                observed_topologies.add(topology_identity(sample.static))

            groupings = ((indices[0],), (indices[1], indices[2]), indices)
            batch_records: list[dict[str, Any]] = []
            for grouping in groupings:
                batch = d6.build_logical_batch(dataset, grouping)
                validate_batch_contract(dataset, grouping, batch)
                repeated = d6.build_logical_batch(dataset, grouping)
                repeated_fingerprint = logical_batch_fingerprint(repeated)
                fingerprint = logical_batch_fingerprint(batch)
                if fingerprint != repeated_fingerprint:
                    raise ConstructionSanityError("repeated batch fingerprint mismatch")
                observed_topologies.add(topology_identity(batch.static))
                no_leakage &= batch.feature_x.shape[1:] == (78, 5) and all(
                    np.array_equal(
                        batch.feature_x[position], dataset[index].feature_x, equal_nan=True
                    )
                    for position, index in enumerate(grouping)
                )
                batch_records.append(
                    {
                        "ordered_indices": list(grouping),
                        "batch_size": len(grouping),
                        "feature_x_shape": list(batch.feature_x.shape),
                        "feature_x_dtype": str(batch.feature_x.dtype),
                        "targets_shape": list(batch.targets.shape),
                        "targets_dtype": str(batch.targets.dtype),
                        "feature_nodes": batch.feature_node_count,
                        "category_nodes": batch.category_node_count,
                        "sample_nodes": batch.sample_node_count,
                        "feature_to_category_edges": int(
                            batch.feature_to_category_sources.size
                        ),
                        "category_to_sample_edges": int(
                            batch.category_to_sample_sources.size
                        ),
                        "total_edges": batch.total_edge_count,
                        "batch_fingerprint_sha256": fingerprint,
                    }
                )
            records.append(
                {
                    "experiment": experiment,
                    "split": split,
                    "dataset_length": len(dataset),
                    "source_artifacts": dict(dataset.source_artifacts),
                    "selected_indices": list(indices),
                    "sample_feature_shape": [78, 5],
                    "sample_feature_dtype": "float32",
                    "sample_target_dtype": "int16",
                    "sample_fingerprints_sha256": sample_fingerprints,
                    "batch_groupings": batch_records,
                    "source_read_only": dataset.is_lazy_read_only,
                }
            )

    e1_e3_features_equal = True
    e1_e3_targets_convert = True
    for split in d4.SPLIT_STEMS:
        e1 = datasets[("E1", split)]
        e3 = datasets[("E3", split)]
        indices = deterministic_sanity_indices(len(e1))
        e1_batch = d6.build_logical_batch(e1, indices)
        e3_batch = d6.build_logical_batch(e3, indices)
        e1_e3_features_equal &= np.array_equal(
            e1_batch.feature_x, e3_batch.feature_x, equal_nan=True
        )
        e1_e3_targets_convert &= np.array_equal(
            (e1_batch.targets != 0).astype(np.int16, copy=False), e3_batch.targets
        )

    e2_provenance_valid = all(
        datasets[("E2", split)].x_path.parent.name == "exp2"
        and datasets[("E2", split)].source_artifacts["x"].startswith(
            "data/processed/preprocessed/exp2/"
        )
        and not _same_file(
            datasets[("E2", split)].x_path, datasets[("E1", split)].x_path
        )
        and not _same_file(
            datasets[("E2", split)].x_path, datasets[("E3", split)].x_path
        )
        for split in d4.SPLIT_STEMS
    )
    if len(observed_topologies) != 1:
        raise ConstructionSanityError("topology identity varied across sanity inputs")
    return SanityRunResult(
        records=tuple(records),
        topology_identity=next(iter(observed_topologies)),
        e1_e3_features_equal=bool(e1_e3_features_equal),
        e1_e3_targets_convert=bool(e1_e3_targets_convert),
        e2_provenance_valid=bool(e2_provenance_valid),
        no_model_input_leakage=bool(no_leakage),
    )


def _source_mtimes(project_root: Path) -> dict[str, int]:
    roots = (
        project_root / "data" / "processed" / "preprocessed",
        project_root / "data" / "raw" / "csv",
    )
    return {
        path.relative_to(project_root).as_posix(): path.stat().st_mtime_ns
        for root in roots
        for path in root.rglob("*")
        if path.is_file()
    }


def run_construction_sanity(
    project_root: Path,
    *,
    integrity_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run two bounded real construction passes and return a D7 report."""

    project_root = Path(project_root)
    _load_approved_d6_report(project_root)
    before_mtimes = _source_mtimes(project_root)
    first = _execute_once(project_root)
    second = _execute_once(project_root)
    validate_repeatability(first, second)
    after_mtimes = _source_mtimes(project_root)
    mtimes_unchanged = before_mtimes == after_mtimes
    integrity = (
        dict(integrity_result)
        if integrity_result is not None
        else d4.verify_frozen_integrity(project_root)
    )
    source_read_only = all(record["source_read_only"] for record in first.records)
    overall = all(
        (
            len(first.records) == 9,
            first.e1_e3_features_equal,
            first.e1_e3_targets_convert,
            first.e2_provenance_valid,
            first.no_model_input_leakage,
            source_read_only,
            mtimes_unchanged,
            integrity.get("overall_status") == "PASS",
        )
    )
    report: dict[str, Any] = {
        "sanity_version": SANITY_VERSION,
        "provenance": dict(APPROVED_PROVENANCE),
        "selection_rule": "[0, N//4, N//2, (3*N)//4, N-1]",
        "selected_sample_count_per_run": 45,
        "construction_passes": 2,
        "tested_batch_sizes": [1, 2, 5],
        "maximum_batch_size": 5,
        "maximum_batch_feature_payload_bytes": 5 * 78 * 5 * 4,
        "experiment_split_results": list(first.records),
        "graph_contract": {
            "node_counts_per_sample": {"feature": 78, "category": 8, "sample": 1},
            "total_nodes_per_sample": 87,
            "edges_per_sample": 86,
            "hierarchy": "feature -> category -> sample",
            "category_membership_counts": dict(d1.EXPECTED_CATEGORY_COUNTS),
            "status": "PASS",
        },
        "topology_invariance": {
            "topology_identity_sha256": first.topology_identity,
            "single_identity_across_all_inputs": True,
            "independent_of_experiment_split_index_target_and_values": True,
        },
        "sample_value_and_target_preservation_status": True,
        "batch_equivalence_status": True,
        "no_cross_sample_edge_status": True,
        "e1_e3_feature_equality_status": first.e1_e3_features_equal,
        "e1_to_e3_target_conversion_status": first.e1_e3_targets_convert,
        "e2_independent_source_provenance_status": first.e2_provenance_valid,
        "no_model_input_leakage_status": first.no_model_input_leakage,
        "repeatability": {
            "selected_indices_identical": True,
            "sample_fingerprints_identical": True,
            "batch_fingerprints_identical": True,
            "topology_identity_identical": True,
            "overall_status": "PASS",
        },
        "source_immutability": {
            "memory_maps_read_only": source_read_only,
            "batch_arrays_read_only": True,
            "source_mtimes_unchanged": mtimes_unchanged,
            "writes_performed": False,
        },
        "boundedness": {
            "whole_split_preloaded": False,
            "whole_corpus_graphs_materialized": False,
            "individual_graph_files_serialized": False,
        },
        "stage_c_integrity": integrity,
        "raw_data_integrity": {
            "raw_csv_count": integrity.get("raw_csv_count"),
            "raw_csv_expected_count": integrity.get("raw_csv_expected_count"),
            "raw_csv_file_set_matches": integrity.get("raw_csv_file_set_matches"),
            "raw_csv_sha256_matches": integrity.get("raw_csv_sha256_matches"),
        },
        "overall_status": "PASS" if overall else "FAIL",
    }
    report["report_sha256"] = d2.canonical_sha256(report)
    validate_report(report)
    return report


def validate_report(report: Mapping[str, Any]) -> None:
    errors: list[str] = []
    try:
        validate_stage_provenance(report.get("provenance", {}))
    except ConstructionSanityError as exc:
        errors.append(str(exc))
    records = report.get("experiment_split_results", [])
    keys = {
        (record.get("experiment"), record.get("split"))
        for record in records
        if isinstance(record, Mapping)
    }
    expected_keys = {
        (experiment, split)
        for experiment in d4.EXPERIMENT_DIRS
        for split in d4.SPLIT_STEMS
    }
    if keys != expected_keys or len(records) != 9:
        errors.append("report does not contain exactly all nine experiment/split pairs")
    required_true = (
        "sample_value_and_target_preservation_status",
        "batch_equivalence_status",
        "no_cross_sample_edge_status",
        "e1_e3_feature_equality_status",
        "e1_to_e3_target_conversion_status",
        "e2_independent_source_provenance_status",
        "no_model_input_leakage_status",
    )
    for field in required_true:
        if report.get(field) is not True:
            errors.append(f"{field} is not true")
    if report.get("repeatability", {}).get("overall_status") != "PASS":
        errors.append("repeatability status is not PASS")
    if report.get("stage_c_integrity", {}).get("overall_status") != "PASS":
        errors.append("Stage C/raw integrity status is not PASS")
    if report.get("overall_status") != "PASS":
        errors.append("overall status is not PASS")
    supplied = report.get("report_sha256")
    unhashed = dict(report)
    unhashed.pop("report_sha256", None)
    if not isinstance(supplied, str) or d2.canonical_sha256(unhashed) != supplied:
        errors.append("D7 report SHA-256 does not match canonical content")
    if errors:
        raise ConstructionSanityError("Stage D7 validation failed: " + "; ".join(errors))


def report_bytes(report: Mapping[str, Any]) -> bytes:
    validate_report(report)
    return (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def write_report(project_root: Path, output_path: Path) -> dict[str, Any]:
    report = run_construction_sanity(project_root)
    data = report_bytes(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(data)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage D7 construction sanity")
    parser.add_argument("--project-root", type=Path, default=_project_root())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/stage_d7/construction_sanity_validation.json"),
    )
    args = parser.parse_args(argv)
    report = write_report(args.project_root, args.output)
    print(f"wrote {args.output}")
    print(f"overall_status={report['overall_status']}")
    print(f"report_sha256={report['report_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
