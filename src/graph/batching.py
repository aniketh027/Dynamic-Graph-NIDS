"""Stage D6 bounded, framework-independent batching of D4 logical samples.

Only the requested samples are assembled.  Frozen Stage C memory maps stay
read-only, while every returned numeric and structural batch array is also
marked read-only.  Flattened node IDs are batch-local conversion metadata and
do not replace the canonical D2 type-local schema.
"""

from __future__ import annotations

import argparse
import json
import operator
import os
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.graph import adapter as d4
from src.graph import experiment_validation as d5
from src.graph import schema as d2


BATCHING_VERSION = "stage-d6-template-batching-v1"
APPROVED_D5_REPORT_SHA256 = (
    "1ddfa6caaaf33a04d8b4a973b39b3b815be9ef11904bd0b6e5bf91a95f898b16"
)
FEATURE_COUNT = 78
CATEGORY_COUNT = 8
TIME_COUNT = 5
EDGES_PER_GRAPH = 86


class BatchContractError(ValueError):
    """Raised when a requested batch or its source violates the D6 contract."""


@dataclass(frozen=True, slots=True)
class LogicalGraphBatch:
    """One bounded logical batch drawn from exactly one D4 dataset."""

    feature_x: np.ndarray
    targets: np.ndarray
    sample_indices: np.ndarray
    experiment: str
    split: str
    source_dataset_length: int
    provenance: Mapping[str, str]
    static: d4.StaticGraphMetadata
    feature_batch: np.ndarray
    category_batch: np.ndarray
    sample_batch: np.ndarray
    feature_to_category_sources: np.ndarray
    feature_to_category_destinations: np.ndarray
    category_to_sample_sources: np.ndarray
    category_to_sample_destinations: np.ndarray

    @property
    def batch_size(self) -> int:
        return int(self.sample_indices.shape[0])

    @property
    def feature_node_count(self) -> int:
        return self.batch_size * FEATURE_COUNT

    @property
    def category_node_count(self) -> int:
        return self.batch_size * CATEGORY_COUNT

    @property
    def sample_node_count(self) -> int:
        return self.batch_size

    @property
    def total_edge_count(self) -> int:
        return int(
            self.feature_to_category_sources.size
            + self.category_to_sample_sources.size
        )


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_approved_d5_report(path: Path) -> dict[str, Any]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BatchContractError(f"cannot load D5 validation report: {exc}") from exc
    try:
        d5.validate_report(report)
    except Exception as exc:
        raise BatchContractError(f"D5 validation report is invalid: {exc}") from exc
    if report.get("report_sha256") != APPROVED_D5_REPORT_SHA256:
        raise BatchContractError("D5 validation report hash is not approved")
    return report


def _validate_dataset(
    dataset: d4.LazyGraphDataset,
    d5_report_path: Path | None,
    allow_noncanonical_fixture: bool,
) -> Mapping[str, str]:
    if not isinstance(dataset, d4.LazyGraphDataset):
        raise TypeError("dataset must be a validated LazyGraphDataset instance")
    if not dataset.is_lazy_read_only:
        raise BatchContractError("source dataset is not a read-only D4 dataset")
    if dataset.static is not d4.STATIC_GRAPH_METADATA:
        raise BatchContractError("source dataset does not use the approved shared topology")

    report = _load_approved_d5_report(
        Path(d5_report_path)
        if d5_report_path is not None
        else _project_root()
        / "results"
        / "stage_d5"
        / "experiment_boundary_validation.json"
    )
    expected_graph = {
        **dict(dataset.provenance),
        "d4_report_sha256": d5.APPROVED_D4_REPORT_SHA256,
    }
    record = next(
        (
            item
            for item in report["experiment_split_registry"]
            if item["experiment"] == dataset.experiment
            and item["split"] == dataset.split
        ),
        None,
    )
    if record is None:
        raise BatchContractError("dataset experiment/split is absent from D5 registry")
    if record.get("graph_provenance") != expected_graph:
        raise BatchContractError("dataset provenance disagrees with D5 registry")
    if (
        record.get("x_artifact") != dataset.source_artifacts.get("x")
        or record.get("y_artifact") != dataset.source_artifacts.get("y")
    ):
        raise BatchContractError("dataset artifact mapping disagrees with D5 registry")
    if not allow_noncanonical_fixture:
        canonical_root = (
            _project_root() / "data" / "processed" / "preprocessed"
        ).resolve()
        if dataset._processed_root.resolve() != canonical_root:  # noqa: SLF001
            raise BatchContractError("dataset does not use the D5-approved processed root")
        if len(dataset) != record.get("actual_length"):
            raise BatchContractError("dataset length disagrees with D5 registry")
    return MappingProxyType(
        {
            **expected_graph,
            "d5_report_sha256": APPROVED_D5_REPORT_SHA256,
        }
    )


def _normalize_indices(indices: Iterable[int], dataset_length: int) -> tuple[int, ...]:
    try:
        raw = tuple(indices)
    except TypeError as exc:
        raise TypeError("indices must be an iterable of integers") from exc
    if not raw:
        raise BatchContractError("empty index collections are not supported")

    normalized: list[int] = []
    for position, value in enumerate(raw):
        if isinstance(value, bool):
            raise TypeError(f"index at position {position} must be an integer, not bool")
        try:
            index = operator.index(value)
        except TypeError as exc:
            raise TypeError(f"index at position {position} must be an integer") from exc
        if index < 0:
            raise IndexError(f"negative index at position {position} is not supported")
        if index >= dataset_length:
            raise IndexError(
                f"index {index} at position {position} is out of range for length {dataset_length}"
            )
        normalized.append(index)
    if len(set(normalized)) != len(normalized):
        raise BatchContractError("duplicate sample indices are not supported")
    return tuple(normalized)


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def build_logical_batch(
    dataset: d4.LazyGraphDataset,
    indices: Iterable[int],
    *,
    d5_report_path: Path | None = None,
    _allow_noncanonical_fixture: bool = False,
) -> LogicalGraphBatch:
    """Assemble requested D4 samples in exact caller order into one D6 batch."""

    provenance = _validate_dataset(
        dataset, d5_report_path, _allow_noncanonical_fixture
    )
    source_indices = _normalize_indices(indices, len(dataset))
    batch_size = len(source_indices)

    feature_x = np.empty((batch_size, FEATURE_COUNT, TIME_COUNT), dtype=np.float32)
    targets = np.empty((batch_size,), dtype=np.int16)
    for batch_position, source_index in enumerate(source_indices):
        sample = dataset[source_index]
        if (
            sample.experiment != dataset.experiment
            or sample.split != dataset.split
            or sample.sample_index != source_index
            or sample.static is not dataset.static
        ):
            raise BatchContractError("D4 sample identity disagrees with its source dataset")
        feature_x[batch_position] = sample.feature_x
        targets[batch_position] = sample.target

    graph_ids = np.arange(batch_size, dtype=np.int64)
    feature_batch = np.repeat(graph_ids, FEATURE_COUNT)
    category_batch = np.repeat(graph_ids, CATEGORY_COUNT)
    sample_batch = graph_ids.copy()

    feature_offsets = np.repeat(graph_ids * FEATURE_COUNT, FEATURE_COUNT)
    category_offsets_for_features = np.repeat(graph_ids * CATEGORY_COUNT, FEATURE_COUNT)
    feature_to_category_sources = (
        np.tile(np.asarray(dataset.static.feature_to_category_sources, dtype=np.int64), batch_size)
        + feature_offsets
    )
    feature_to_category_destinations = (
        np.tile(
            np.asarray(dataset.static.feature_to_category_destinations, dtype=np.int64),
            batch_size,
        )
        + category_offsets_for_features
    )

    category_offsets = np.repeat(graph_ids * CATEGORY_COUNT, CATEGORY_COUNT)
    category_to_sample_sources = (
        np.tile(np.asarray(dataset.static.category_to_sample_sources, dtype=np.int64), batch_size)
        + category_offsets
    )
    category_to_sample_destinations = np.repeat(graph_ids, CATEGORY_COUNT)

    arrays = (
        feature_x,
        targets,
        np.asarray(source_indices, dtype=np.int64),
        feature_batch,
        category_batch,
        sample_batch,
        feature_to_category_sources,
        feature_to_category_destinations,
        category_to_sample_sources,
        category_to_sample_destinations,
    )
    for array in arrays:
        _readonly(array)

    batch = LogicalGraphBatch(
        feature_x=arrays[0],
        targets=arrays[1],
        sample_indices=arrays[2],
        experiment=dataset.experiment,
        split=dataset.split,
        source_dataset_length=len(dataset),
        provenance=provenance,
        static=dataset.static,
        feature_batch=arrays[3],
        category_batch=arrays[4],
        sample_batch=arrays[5],
        feature_to_category_sources=arrays[6],
        feature_to_category_destinations=arrays[7],
        category_to_sample_sources=arrays[8],
        category_to_sample_destinations=arrays[9],
    )
    _validate_batched_topology(batch)
    return batch


def _validate_batched_topology(batch: LogicalGraphBatch) -> None:
    """Assert edge counts, ownership, and sample isolation for one batch."""

    size = batch.batch_size
    if batch.feature_to_category_sources.size != size * FEATURE_COUNT:
        raise BatchContractError("feature-to-category edge count is not B*78")
    if batch.category_to_sample_sources.size != size * CATEGORY_COUNT:
        raise BatchContractError("category-to-sample edge count is not B*8")
    if batch.total_edge_count != size * EDGES_PER_GRAPH:
        raise BatchContractError("total structural edge count is not B*86")
    ftc_source_owner = batch.feature_to_category_sources // FEATURE_COUNT
    ftc_destination_owner = batch.feature_to_category_destinations // CATEGORY_COUNT
    cts_source_owner = batch.category_to_sample_sources // CATEGORY_COUNT
    if not np.array_equal(ftc_source_owner, ftc_destination_owner):
        raise BatchContractError("cross-sample feature-to-category edge detected")
    if not np.array_equal(cts_source_owner, batch.category_to_sample_destinations):
        raise BatchContractError("cross-sample category-to-sample edge detected")
    if not np.array_equal(batch.feature_batch, ftc_source_owner):
        raise BatchContractError("feature ownership vector is inconsistent")
    if not np.array_equal(batch.category_batch, cts_source_owner):
        raise BatchContractError("category ownership vector is inconsistent")
    if not np.array_equal(batch.sample_batch, np.arange(size, dtype=np.int64)):
        raise BatchContractError("sample ownership vector is inconsistent")


def _same_file(first: Path, second: Path) -> bool:
    try:
        return os.path.samefile(first, second)
    except OSError:
        return False


def build_validation_report(
    project_root: Path,
    *,
    integrity_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the deterministic real-artifact D6 validation report."""

    project_root = Path(project_root)
    d5_path = project_root / "results" / "stage_d5" / "experiment_boundary_validation.json"
    d5_report = _load_approved_d5_report(d5_path)
    datasets: dict[tuple[str, str], d4.LazyGraphDataset] = {}
    records: list[dict[str, Any]] = []
    all_equivalent = True
    all_ownership = True
    all_isolated = True
    all_immutable = True

    for experiment in d4.EXPERIMENT_DIRS:
        for split in d4.SPLIT_STEMS:
            dataset = d4.LazyGraphDataset(experiment, split)
            datasets[(experiment, split)] = dataset
            indices = [0, len(dataset) // 2, len(dataset) - 1]
            batch = build_logical_batch(dataset, indices, d5_report_path=d5_path)
            equivalent = all(
                np.array_equal(batch.feature_x[position], dataset[index].feature_x, equal_nan=True)
                and batch.targets[position] == dataset[index].target
                and batch.static.feature_category_ids
                == dataset[index].feature_category_ids
                for position, index in enumerate(indices)
            )
            ownership = (
                np.array_equal(batch.feature_batch, np.repeat(np.arange(3), FEATURE_COUNT))
                and np.array_equal(batch.category_batch, np.repeat(np.arange(3), CATEGORY_COUNT))
                and np.array_equal(batch.sample_batch, np.arange(3))
            )
            isolated = (
                np.array_equal(
                    batch.feature_to_category_sources // FEATURE_COUNT,
                    batch.feature_to_category_destinations // CATEGORY_COUNT,
                )
                and np.array_equal(
                    batch.category_to_sample_sources // CATEGORY_COUNT,
                    batch.category_to_sample_destinations,
                )
            )
            immutable = (
                dataset.is_lazy_read_only
                and not batch.feature_x.flags.writeable
                and not batch.targets.flags.writeable
                and all(
                    not value.flags.writeable
                    for value in (
                        batch.sample_indices,
                        batch.feature_batch,
                        batch.category_batch,
                        batch.sample_batch,
                        batch.feature_to_category_sources,
                        batch.feature_to_category_destinations,
                        batch.category_to_sample_sources,
                        batch.category_to_sample_destinations,
                    )
                )
            )
            all_equivalent &= equivalent
            all_ownership &= ownership
            all_isolated &= isolated
            all_immutable &= immutable
            records.append(
                {
                    "experiment": experiment,
                    "split": split,
                    "source_dataset_length": len(dataset),
                    "tested_batch_size": batch.batch_size,
                    "tested_source_indices": indices,
                    "feature_x_shape": list(batch.feature_x.shape),
                    "feature_x_dtype": str(batch.feature_x.dtype),
                    "targets_shape": list(batch.targets.shape),
                    "targets_dtype": str(batch.targets.dtype),
                    "node_counts": {
                        "feature": batch.feature_node_count,
                        "category": batch.category_node_count,
                        "sample": batch.sample_node_count,
                    },
                    "edge_counts": {
                        "feature_to_category": int(batch.feature_to_category_sources.size),
                        "category_to_sample": int(batch.category_to_sample_sources.size),
                        "total": batch.total_edge_count,
                    },
                    "ownership_vectors_valid": ownership,
                    "no_cross_sample_edges": isolated,
                    "individual_batch_equivalent": equivalent,
                    "source_and_batch_read_only": immutable,
                }
            )

    e1_e3_equal = all(
        np.array_equal(
            build_logical_batch(
                datasets[("E1", split)],
                [0, len(datasets[("E1", split)]) // 2, len(datasets[("E1", split)]) - 1],
                d5_report_path=d5_path,
            ).feature_x,
            build_logical_batch(
                datasets[("E3", split)],
                [0, len(datasets[("E3", split)]) // 2, len(datasets[("E3", split)]) - 1],
                d5_report_path=d5_path,
            ).feature_x,
            equal_nan=True,
        )
        for split in d4.SPLIT_STEMS
    )
    e2_independent = all(
        datasets[("E2", split)].x_path.parent.name == "exp2"
        and not _same_file(
            datasets[("E2", split)].x_path, datasets[("E1", split)].x_path
        )
        and not _same_file(
            datasets[("E2", split)].x_path, datasets[("E3", split)].x_path
        )
        for split in d4.SPLIT_STEMS
    )
    size_one = build_logical_batch(
        datasets[("E1", "train")], [0], d5_report_path=d5_path
    )
    size_four_indices = [10, 3, 50, 7]
    size_four = build_logical_batch(
        datasets[("E1", "train")], size_four_indices, d5_report_path=d5_path
    )
    additional_sizes_valid = (
        size_one.feature_x.shape == (1, FEATURE_COUNT, TIME_COUNT)
        and size_one.total_edge_count == EDGES_PER_GRAPH
        and size_four.sample_indices.tolist() == size_four_indices
        and size_four.feature_x.shape == (4, FEATURE_COUNT, TIME_COUNT)
        and size_four.total_edge_count == 4 * EDGES_PER_GRAPH
    )
    integrity = (
        dict(integrity_result)
        if integrity_result is not None
        else d4.verify_frozen_integrity(project_root)
    )
    overall = all(
        (
            len(records) == 9,
            all_equivalent,
            all_ownership,
            all_isolated,
            all_immutable,
            e1_e3_equal,
            e2_independent,
            additional_sizes_valid,
            integrity.get("overall_status") == "PASS",
        )
    )
    report: dict[str, Any] = {
        "batching_version": BATCHING_VERSION,
        "provenance": {
            **d5_report["provenance"],
            "d5_report_sha256": APPROVED_D5_REPORT_SHA256,
        },
        "batching_contract": {
            "framework_independent": True,
            "one_experiment_and_split_per_batch": True,
            "requested_order_preserved": True,
            "feature_x_shape": "(B,78,5)",
            "feature_x_dtype": "float32",
            "targets_shape": "(B,)",
            "targets_dtype": "int16",
            "batch_scale_allocation_only": True,
            "static_d2_template_reused": True,
        },
        "index_policy": {
            "empty_indices": "rejected",
            "duplicate_indices": "rejected",
            "negative_indices": "rejected",
            "out_of_range_indices": "rejected",
            "non_integer_indices": "rejected",
            "sorting": False,
            "internal_shuffling": False,
        },
        "tested_batch_sizes": [1, 3, 4],
        "additional_batch_size_validation": {
            "size_one_pass": size_one.batch_size == 1,
            "size_four_noncontiguous_order": size_four_indices,
            "size_four_pass": additional_sizes_valid,
        },
        "experiment_split_batches": records,
        "ownership_vector_validation": all_ownership,
        "no_cross_sample_edge_status": all_isolated,
        "batch_equivalence_status": all_equivalent,
        "e1_e3_feature_batch_identity_status": e1_e3_equal,
        "e2_independent_source_status": e2_independent,
        "source_immutability_status": all_immutable,
        "stage_c_integrity": integrity,
        "overall_status": "PASS" if overall else "FAIL",
    }
    report["report_sha256"] = d2.canonical_sha256(report)
    if not overall:
        raise BatchContractError("D6 real-artifact validation did not pass")
    return report


def report_bytes(report: Mapping[str, Any]) -> bytes:
    supplied = report.get("report_sha256")
    unhashed = dict(report)
    unhashed.pop("report_sha256", None)
    if not isinstance(supplied, str) or d2.canonical_sha256(unhashed) != supplied:
        raise BatchContractError("D6 report SHA-256 does not match canonical content")
    if report.get("overall_status") != "PASS":
        raise BatchContractError("D6 report overall status is not PASS")
    return (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def write_report(project_root: Path, output_path: Path) -> dict[str, Any]:
    report = build_validation_report(project_root)
    data = report_bytes(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(data)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Stage D6 logical batching")
    parser.add_argument("--project-root", type=Path, default=_project_root())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/stage_d6/template_batching_validation.json"),
    )
    args = parser.parse_args(argv)
    report = write_report(args.project_root, args.output)
    print(f"wrote {args.output}")
    print(f"overall_status={report['overall_status']}")
    print(f"report_sha256={report['report_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
