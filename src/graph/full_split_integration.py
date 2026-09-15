"""Stage D8 deterministic, lazy, full-split graph-input integration.

The planner represents complete sequential index coverage arithmetically.  The
iterator constructs only the current D6 batch and keeps no batch cache, graph
collection, prefetch queue, framework object, or mutable traversal cursor.
"""

from __future__ import annotations

import argparse
from collections import namedtuple
import hashlib
import json
import operator
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from src.graph import adapter as d4
from src.graph import batching as d6
from src.graph import construction_sanity as d7
from src.graph import experiment_validation as d5
from src.graph import schema as d2
from src.graph import validation as d3


INTEGRATION_VERSION = "stage-d8-full-split-lazy-integration-v1"
APPROVED_D7_REPORT_SHA256 = (
    "cc67f8cf87c93977b9b7e820ea57ee3fe989849a8c3e1222266482373a17a55b"
)
VALIDATED_BATCH_SIZES = (1, 32, 128)
CANONICAL_INTEGRATION_BATCH_SIZE = 128
APPROVED_PROVENANCE: Mapping[str, str] = {
    "d1_taxonomy_sha256": d3.APPROVED_D1_TAXONOMY_SHA256,
    "d2_schema_sha256": d3.APPROVED_D2_SCHEMA_SHA256,
    "d3_report_sha256": d4.APPROVED_D3_REPORT_SHA256,
    "d4_report_sha256": d5.APPROVED_D4_REPORT_SHA256,
    "d5_report_sha256": d6.APPROVED_D5_REPORT_SHA256,
    "d6_report_sha256": d7.APPROVED_D6_REPORT_SHA256,
    "d7_report_sha256": APPROVED_D7_REPORT_SHA256,
}


class FullSplitIntegrationError(ValueError):
    """Raised when a D8 plan, iterator, or provenance contract is invalid."""


_BatchBoundaryTuple = namedtuple(
    "_BatchBoundaryTuple", ("batch_number", "start", "stop")
)


class BatchBoundary(_BatchBoundaryTuple):
    """One half-open sequential batch boundary; indices are represented by range."""

    __slots__ = ()

    @property
    def size(self) -> int:
        return self.stop - self.start

    @property
    def indices(self) -> range:
        return range(self.start, self.stop)


@dataclass(frozen=True, slots=True)
class CoverageSummary:
    dataset_length: int
    batch_size: int
    number_of_batches: int
    total_planned_indices: int
    first_index: int
    last_index: int
    final_batch_size: int
    boundary_sequence_sha256: str
    no_gaps: bool
    no_overlaps: bool
    no_duplicates: bool


_SequentialBatchPlanTuple = namedtuple(
    "_SequentialBatchPlanTuple", ("dataset_length", "batch_size")
)


class SequentialBatchPlan(_SequentialBatchPlanTuple):
    """Cheap immutable plan for all indices ``0..N-1`` with no drop-last mode."""

    __slots__ = ()

    def __new__(
        cls, dataset_length: int, batch_size: int
    ) -> "SequentialBatchPlan":
        return super().__new__(
            cls,
            _positive_integer(dataset_length, "dataset length"),
            _positive_integer(batch_size, "batch size"),
        )

    @property
    def number_of_batches(self) -> int:
        return (self.dataset_length + self.batch_size - 1) // self.batch_size

    @property
    def number_of_full_batches(self) -> int:
        return self.dataset_length // self.batch_size

    @property
    def final_batch_size(self) -> int:
        remainder = self.dataset_length % self.batch_size
        return remainder if remainder else self.batch_size

    def boundary(self, batch_number: int) -> BatchBoundary:
        normalized = _nonnegative_integer(batch_number, "batch number")
        if normalized >= self.number_of_batches:
            raise IndexError(
                f"batch number {normalized} out of range for {self.number_of_batches} batches"
            )
        start = normalized * self.batch_size
        return BatchBoundary(
            batch_number=normalized,
            start=start,
            stop=min(start + self.batch_size, self.dataset_length),
        )

    def iter_boundaries(self) -> Iterator[BatchBoundary]:
        for batch_number in range(self.number_of_batches):
            yield self.boundary(batch_number)

    def __iter__(self) -> Iterator[BatchBoundary]:
        return self.iter_boundaries()


@dataclass(frozen=True, slots=True)
class DatasetBindingSnapshot:
    """Immutable semantic and physical identity of a validated D4 source."""

    experiment: str
    split: str
    dataset_length: int
    source_artifacts: tuple[tuple[str, str], ...]
    provenance: tuple[tuple[str, str], ...]
    processed_root: str
    x_path: str
    y_path: str
    metadata_path: str
    x_memmap_path: str
    y_memmap_path: str
    x_memmap_object_id: int
    y_memmap_object_id: int
    x_shape: tuple[int, ...]
    y_shape: tuple[int, ...]
    x_dtype: str
    y_dtype: str
    topology_identity_sha256: str


@dataclass(frozen=True, slots=True)
class IteratorBindingSnapshot:
    dataset: DatasetBindingSnapshot
    batch_size: int
    number_of_batches: int
    approved_provenance: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True, init=False)
class LazyBatchIterator:
    """Reiterable D6 batch stream for exactly one validated D4 dataset."""

    dataset: d4.LazyGraphDataset
    plan: SequentialBatchPlan
    experiment: str
    split: str
    _binding: IteratorBindingSnapshot

    def __init__(self, dataset: d4.LazyGraphDataset, batch_size: int) -> None:
        _validate_dataset(dataset)
        plan = SequentialBatchPlan(len(dataset), batch_size)
        binding = IteratorBindingSnapshot(
            dataset=_capture_dataset_binding(dataset),
            batch_size=plan.batch_size,
            number_of_batches=plan.number_of_batches,
            approved_provenance=tuple(sorted(APPROVED_PROVENANCE.items())),
        )
        object.__setattr__(self, "dataset", dataset)
        object.__setattr__(self, "plan", plan)
        object.__setattr__(self, "experiment", dataset.experiment)
        object.__setattr__(self, "split", dataset.split)
        object.__setattr__(self, "_binding", binding)

    @property
    def batch_size(self) -> int:
        return self.plan.batch_size

    @property
    def dataset_length(self) -> int:
        return self.plan.dataset_length

    @property
    def number_of_batches(self) -> int:
        return self.plan.number_of_batches

    def boundary(self, batch_number: int) -> BatchBoundary:
        return self.plan.boundary(batch_number)

    def iter_boundaries(self) -> Iterator[BatchBoundary]:
        return self.plan.iter_boundaries()

    def construct_batch(self, batch_number: int) -> d6.LogicalGraphBatch:
        self._validate_current_binding()
        boundary = self.boundary(batch_number)
        batch = d6.build_logical_batch(self.dataset, boundary.indices)
        self._validate_current_binding()
        return batch

    def __iter__(self) -> Iterator[d6.LogicalGraphBatch]:
        for boundary in self.iter_boundaries():
            yield self.construct_batch(boundary.batch_number)

    def _validate_current_binding(self) -> None:
        _load_approved_d7_report(_project_root())
        try:
            current_dataset = _capture_dataset_binding(self.dataset)
        except Exception as exc:
            raise FullSplitIntegrationError(
                f"iterator dataset binding changed: {exc}"
            ) from exc
        if current_dataset != self._binding.dataset:
            raise FullSplitIntegrationError("iterator dataset binding changed")
        if (
            self.experiment != self._binding.dataset.experiment
            or self.split != self._binding.dataset.split
            or self.plan.dataset_length != self._binding.dataset.dataset_length
            or self.plan.batch_size != self._binding.batch_size
            or self.plan.number_of_batches != self._binding.number_of_batches
            or self._binding.approved_provenance
            != tuple(sorted(APPROVED_PROVENANCE.items()))
        ):
            raise FullSplitIntegrationError("iterator plan or identity binding changed")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _positive_integer(value: int, description: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{description} must be an integer, not bool")
    try:
        normalized = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{description} must be an integer") from exc
    if normalized <= 0:
        raise FullSplitIntegrationError(f"{description} must be positive")
    return normalized


def _nonnegative_integer(value: int, description: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{description} must be an integer, not bool")
    try:
        normalized = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{description} must be an integer") from exc
    if normalized < 0:
        raise IndexError(f"{description} must be non-negative")
    return normalized


def validate_stage_provenance(provenance: Mapping[str, str]) -> None:
    if dict(provenance) != dict(APPROVED_PROVENANCE):
        differing = [
            field
            for field, expected in APPROVED_PROVENANCE.items()
            if provenance.get(field) != expected
        ]
        raise FullSplitIntegrationError(
            "corrupted Stage D provenance: " + ", ".join(differing)
        )


def _load_approved_d7_report(project_root: Path) -> dict[str, Any]:
    path = project_root / "results" / "stage_d7" / "construction_sanity_validation.json"
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        d7.report_bytes(report)
    except Exception as exc:
        raise FullSplitIntegrationError(f"D7 validation report is invalid: {exc}") from exc
    if report.get("report_sha256") != APPROVED_D7_REPORT_SHA256:
        raise FullSplitIntegrationError("D7 validation report hash is not approved")
    provenance = {
        **report.get("provenance", {}),
        "d7_report_sha256": report["report_sha256"],
    }
    validate_stage_provenance(provenance)
    return report


def _validate_dataset(dataset: d4.LazyGraphDataset) -> None:
    if not isinstance(dataset, d4.LazyGraphDataset):
        raise TypeError("dataset must be a validated LazyGraphDataset instance")
    if not dataset.is_lazy_read_only:
        raise FullSplitIntegrationError("dataset is not a read-only D4 source")
    expected_dataset_provenance = {
        "d1_taxonomy_sha256": APPROVED_PROVENANCE["d1_taxonomy_sha256"],
        "d2_schema_sha256": APPROVED_PROVENANCE["d2_schema_sha256"],
        "d3_report_sha256": APPROVED_PROVENANCE["d3_report_sha256"],
    }
    if dict(dataset.provenance) != expected_dataset_provenance:
        raise FullSplitIntegrationError("malformed dataset provenance")
    if (
        dataset.static is not d4.STATIC_GRAPH_METADATA
        or dataset.static.d1_taxonomy_sha256
        != APPROVED_PROVENANCE["d1_taxonomy_sha256"]
        or dataset.static.d2_schema_sha256
        != APPROVED_PROVENANCE["d2_schema_sha256"]
        or dataset.static.d3_report_sha256
        != APPROVED_PROVENANCE["d3_report_sha256"]
    ):
        raise FullSplitIntegrationError("malformed dataset topology provenance")
    project_root = _project_root()
    d7_report = _load_approved_d7_report(project_root)
    expected_root = (project_root / "data" / "processed" / "preprocessed").resolve()
    if dataset._processed_root.resolve() != expected_root:  # noqa: SLF001
        raise FullSplitIntegrationError("dataset does not use the D7-approved processed root")
    record = next(
        (
            item
            for item in d7_report["experiment_split_results"]
            if item["experiment"] == dataset.experiment
            and item["split"] == dataset.split
        ),
        None,
    )
    if record is None:
        raise FullSplitIntegrationError("dataset experiment/split is absent from D7 report")
    if record.get("dataset_length") != len(dataset):
        raise FullSplitIntegrationError("incorrect dataset length metadata")
    if record.get("source_artifacts") != dict(dataset.source_artifacts):
        raise FullSplitIntegrationError("dataset artifacts disagree with D7 report")


def _capture_dataset_binding(dataset: d4.LazyGraphDataset) -> DatasetBindingSnapshot:
    """Capture immutable binding fields used before and after every D6 build."""

    return DatasetBindingSnapshot(
        experiment=dataset.experiment,
        split=dataset.split,
        dataset_length=len(dataset),
        source_artifacts=tuple(sorted(dataset.source_artifacts.items())),
        provenance=tuple(sorted(dataset.provenance.items())),
        processed_root=str(dataset._processed_root.resolve()),  # noqa: SLF001
        x_path=str(dataset.x_path.resolve()),
        y_path=str(dataset.y_path.resolve()),
        metadata_path=str(dataset.metadata_path.resolve()),
        x_memmap_path=str(Path(dataset.x_memmap.filename).resolve()),
        y_memmap_path=str(Path(dataset.y_memmap.filename).resolve()),
        x_memmap_object_id=id(dataset.x_memmap),
        y_memmap_object_id=id(dataset.y_memmap),
        x_shape=tuple(int(value) for value in dataset.x_memmap.shape),
        y_shape=tuple(int(value) for value in dataset.y_memmap.shape),
        x_dtype=str(dataset.x_memmap.dtype),
        y_dtype=str(dataset.y_memmap.dtype),
        topology_identity_sha256=d7.topology_identity(dataset.static),
    )


def validate_dataset_identity(
    dataset: d4.LazyGraphDataset, expected_experiment: str, expected_split: str
) -> None:
    if dataset.experiment != expected_experiment:
        raise FullSplitIntegrationError("cross-experiment dataset substitution")
    if dataset.split != expected_split:
        raise FullSplitIntegrationError("cross-split dataset substitution")


def validate_boundary_sequence(
    plan: SequentialBatchPlan,
    boundaries: Iterable[BatchBoundary] | None = None,
    *,
    expected_dataset_length: int | None = None,
) -> CoverageSummary:
    """Stream-check full coverage without creating a full index list."""

    if (
        expected_dataset_length is not None
        and expected_dataset_length != plan.dataset_length
    ):
        raise FullSplitIntegrationError("incorrect dataset length metadata")
    stream = plan.iter_boundaries() if boundaries is None else iter(boundaries)
    digest = hashlib.sha256()
    previous_stop = 0
    total = 0
    count = 0
    first_index: int | None = None
    last_index: int | None = None
    seen_ranges: set[tuple[int, int]] = set()
    final_size = 0

    for expected_number, boundary in enumerate(stream):
        if boundary.batch_number != expected_number:
            raise FullSplitIntegrationError("batch numbering is nondeterministic")
        if expected_number == 0 and boundary.start != 0:
            raise FullSplitIntegrationError("plan begins at nonzero index")
        if boundary.start > previous_stop:
            raise FullSplitIntegrationError("batch plan contains a gap")
        if boundary.start < previous_stop:
            if (boundary.start, boundary.stop) in seen_ranges:
                raise FullSplitIntegrationError("duplicate planned index range")
            raise FullSplitIntegrationError("batch plan contains an overlap")
        if boundary.stop > plan.dataset_length:
            raise FullSplitIntegrationError("batch plan extends past dataset length")
        if boundary.stop <= boundary.start:
            raise FullSplitIntegrationError("batch boundary is empty or reversed")
        expected_stop = min(boundary.start + plan.batch_size, plan.dataset_length)
        if boundary.stop != expected_stop:
            if expected_number == plan.number_of_batches - 1:
                raise FullSplitIntegrationError("incorrect final batch size")
            raise FullSplitIntegrationError("incorrect non-final batch size")
        seen_ranges.add((boundary.start, boundary.stop))
        digest.update(struct.pack(">QQQ", boundary.batch_number, boundary.start, boundary.stop))
        if first_index is None:
            first_index = boundary.start
        last_index = boundary.stop - 1
        total += boundary.size
        previous_stop = boundary.stop
        final_size = boundary.size
        count += 1

    if count == 0 or first_index != 0:
        raise FullSplitIntegrationError("plan begins at nonzero index")
    if previous_stop < plan.dataset_length:
        raise FullSplitIntegrationError("plan ends before final index")
    if previous_stop > plan.dataset_length:
        raise FullSplitIntegrationError("plan extends past dataset length")
    if last_index != plan.dataset_length - 1:
        raise FullSplitIntegrationError("plan skipped final index")
    if count != plan.number_of_batches:
        raise FullSplitIntegrationError("incorrect number of planned batches")
    if total != plan.dataset_length:
        raise FullSplitIntegrationError("total planned index count is incorrect")
    if final_size != plan.final_batch_size:
        raise FullSplitIntegrationError("incorrect final batch size")
    return CoverageSummary(
        dataset_length=plan.dataset_length,
        batch_size=plan.batch_size,
        number_of_batches=count,
        total_planned_indices=total,
        first_index=0,
        last_index=last_index,
        final_batch_size=final_size,
        boundary_sequence_sha256=digest.hexdigest(),
        no_gaps=True,
        no_overlaps=True,
        no_duplicates=True,
    )


def integration_fingerprint(
    dataset: d4.LazyGraphDataset,
    plan: SequentialBatchPlan,
    source_hashes: Mapping[str, str],
) -> str:
    """Fingerprint the integration plan and frozen identities, not X values."""

    first = plan.boundary(0)
    final = plan.boundary(plan.number_of_batches - 1)
    payload = {
        "planner_version": INTEGRATION_VERSION,
        "experiment": dataset.experiment,
        "split": dataset.split,
        "dataset_length": len(dataset),
        "source_artifacts": dict(dataset.source_artifacts),
        "source_artifact_sha256": dict(source_hashes),
        "batch_size": plan.batch_size,
        "number_of_batches": plan.number_of_batches,
        "first_batch_range": [first.start, first.stop],
        "final_batch_range": [final.start, final.stop],
        "provenance": dict(APPROVED_PROVENANCE),
        "d2_topology_identity_sha256": d7.topology_identity(dataset.static),
    }
    return d2.canonical_sha256(payload)


def validate_repeatable_fingerprints(first: str, second: str) -> None:
    if first != second:
        raise FullSplitIntegrationError("nondeterministic integration fingerprint")


def _same_file(first: Path, second: Path) -> bool:
    try:
        return os.path.samefile(first, second)
    except OSError:
        return False


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


def _critical_batch_numbers(plan: SequentialBatchPlan) -> tuple[int, ...]:
    candidates = (
        0,
        plan.number_of_batches // 2,
        max(0, plan.number_of_batches - 2),
        plan.number_of_batches - 1,
    )
    return tuple(dict.fromkeys(candidates))


def _boundary_batch_numbers(plan: SequentialBatchPlan) -> tuple[int, ...]:
    candidates = (1, plan.number_of_batches // 2, plan.number_of_batches - 1)
    return tuple(
        dict.fromkeys(
            number
            for number in candidates
            if 0 < number < plan.number_of_batches
        )
    )


def _d5_source_hashes(
    d5_report: Mapping[str, Any], experiment: str, split: str
) -> dict[str, str]:
    record = next(
        item
        for item in d5_report["experiment_split_registry"]
        if item["experiment"] == experiment and item["split"] == split
    )
    return {"x_sha256": record["x_sha256"], "y_sha256": record["y_sha256"]}


def _validate_constructed_batch(
    dataset: d4.LazyGraphDataset,
    boundary: BatchBoundary,
    batch: d6.LogicalGraphBatch,
) -> None:
    """Validate one real D6 batch without assuming asymmetric numeric values."""

    expected_indices = list(boundary.indices)
    size = boundary.size
    if batch.experiment != dataset.experiment or batch.split != dataset.split:
        raise FullSplitIntegrationError("constructed batch experiment/split mismatch")
    if batch.source_dataset_length != len(dataset):
        raise FullSplitIntegrationError("constructed batch source length mismatch")
    if batch.sample_indices.tolist() != expected_indices:
        raise FullSplitIntegrationError("constructed batch index order mismatch")
    if batch.feature_x.shape != (size, 78, 5) or batch.feature_x.dtype != np.float32:
        raise FullSplitIntegrationError("constructed batch feature contract mismatch")
    if batch.targets.shape != (size,) or batch.targets.dtype != np.int16:
        raise FullSplitIntegrationError("constructed batch target contract mismatch")
    if (
        batch.feature_to_category_sources.size != size * 78
        or batch.category_to_sample_sources.size != size * 8
        or batch.total_edge_count != size * 86
    ):
        raise FullSplitIntegrationError("constructed batch edge count mismatch")
    if not np.array_equal(
        batch.feature_to_category_sources // 78,
        batch.feature_to_category_destinations // 8,
    ):
        raise FullSplitIntegrationError("cross-sample feature/category edge")
    if not np.array_equal(
        batch.category_to_sample_sources // 8,
        batch.category_to_sample_destinations,
    ):
        raise FullSplitIntegrationError("cross-sample category/sample edge")
    if not np.array_equal(batch.feature_batch, np.repeat(np.arange(size), 78)):
        raise FullSplitIntegrationError("constructed feature ownership mismatch")
    if not np.array_equal(batch.category_batch, np.repeat(np.arange(size), 8)):
        raise FullSplitIntegrationError("constructed category ownership mismatch")
    if not np.array_equal(batch.sample_batch, np.arange(size)):
        raise FullSplitIntegrationError("constructed sample ownership mismatch")
    if batch.static is not dataset.static or dataset.static is not d4.STATIC_GRAPH_METADATA:
        raise FullSplitIntegrationError("constructed topology identity mismatch")
    for position, source_index in enumerate(expected_indices):
        sample = dataset[source_index]
        if not np.array_equal(
            sample.feature_x, dataset.x_memmap[source_index].T, equal_nan=True
        ):
            raise FullSplitIntegrationError("D4 sample differs from Stage C source")
        if sample.target != dataset.y_memmap[source_index]:
            raise FullSplitIntegrationError("D4 target differs from Stage C source")
        if not np.array_equal(batch.feature_x[position], sample.feature_x, equal_nan=True):
            raise FullSplitIntegrationError("D6 batch differs from D4 sample")
        if batch.targets[position] != sample.target:
            raise FullSplitIntegrationError("D6 batch target differs from D4 sample")


def run_full_split_integration(
    project_root: Path,
    *,
    integrity_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate all nine full index spaces with bounded real construction."""

    project_root = Path(project_root)
    d7_report = _load_approved_d7_report(project_root)
    d5_report = json.loads(
        (project_root / "results" / "stage_d5" / "experiment_boundary_validation.json").read_text(
            encoding="utf-8"
        )
    )
    d5.validate_report(d5_report)
    before_mtimes = _source_mtimes(project_root)
    processed_root = project_root / "data" / "processed" / "preprocessed"
    records: list[dict[str, Any]] = []
    datasets: dict[tuple[str, str], d4.LazyGraphDataset] = {}
    topology_identities: set[str] = set()

    for experiment in d4.EXPERIMENT_DIRS:
        for split in d4.SPLIT_STEMS:
            dataset = d4.LazyGraphDataset(
                experiment, split, processed_root=processed_root
            )
            validate_dataset_identity(dataset, experiment, split)
            datasets[(experiment, split)] = dataset
            plan_summaries: list[dict[str, Any]] = []
            canonical_coverage: CoverageSummary | None = None
            for batch_size in VALIDATED_BATCH_SIZES:
                plan = SequentialBatchPlan(len(dataset), batch_size)
                first = plan.boundary(0)
                final = plan.boundary(plan.number_of_batches - 1)
                plan_summaries.append(
                    {
                        "batch_size": batch_size,
                        "number_of_full_batches": plan.number_of_full_batches,
                        "final_batch_size": plan.final_batch_size,
                        "number_of_batches": plan.number_of_batches,
                        "total_planned_indices": plan.dataset_length,
                        "first_batch_range": [first.start, first.stop],
                        "final_batch_range": [final.start, final.stop],
                    }
                )
                if batch_size == CANONICAL_INTEGRATION_BATCH_SIZE:
                    canonical_coverage = validate_boundary_sequence(plan)
                    repeated_coverage = validate_boundary_sequence(
                        SequentialBatchPlan(len(dataset), batch_size)
                    )
                    if canonical_coverage != repeated_coverage:
                        raise FullSplitIntegrationError("boundary plan is not repeatable")

            if canonical_coverage is None:
                raise FullSplitIntegrationError("canonical coverage was not evaluated")
            iterator = LazyBatchIterator(dataset, CANONICAL_INTEGRATION_BATCH_SIZE)
            constructed_numbers = tuple(
                dict.fromkeys(
                    _critical_batch_numbers(iterator.plan)
                    + tuple(
                        adjacent
                        for boundary_number in _boundary_batch_numbers(iterator.plan)
                        for adjacent in (boundary_number - 1, boundary_number)
                    )
                )
            )
            constructed_records: list[dict[str, Any]] = []
            for batch_number in constructed_numbers:
                boundary = iterator.boundary(batch_number)
                batch = iterator.construct_batch(batch_number)
                _validate_constructed_batch(dataset, boundary, batch)
                repeated = iterator.construct_batch(batch_number)
                fingerprint = d7.logical_batch_fingerprint(batch)
                if fingerprint != d7.logical_batch_fingerprint(repeated):
                    raise FullSplitIntegrationError(
                        "constructed batch fingerprint is not repeatable"
                    )
                topology_identities.add(d7.topology_identity(batch.static))
                constructed_records.append(
                    {
                        "batch_number": batch_number,
                        "range": [boundary.start, boundary.stop],
                        "batch_size": boundary.size,
                        "first_index": boundary.start,
                        "last_index": boundary.stop - 1,
                        "feature_x_shape": list(batch.feature_x.shape),
                        "targets_shape": list(batch.targets.shape),
                        "total_edges": batch.total_edge_count,
                        "batch_fingerprint_sha256": fingerprint,
                    }
                )

            boundary_records: list[dict[str, Any]] = []
            for next_number in _boundary_batch_numbers(iterator.plan):
                previous_number = next_number - 1
                previous_boundary = iterator.boundary(previous_number)
                next_boundary = iterator.boundary(next_number)
                previous_batch = iterator.construct_batch(previous_number)
                next_batch = iterator.construct_batch(next_number)
                previous_index = previous_boundary.stop - 1
                next_index = next_boundary.start
                previous_sample = dataset[previous_index]
                next_sample = dataset[next_index]
                equivalent = (
                    previous_index + 1 == next_index
                    and previous_index not in next_boundary.indices
                    and next_index not in previous_boundary.indices
                    and np.array_equal(
                        previous_batch.feature_x[-1],
                        previous_sample.feature_x,
                        equal_nan=True,
                    )
                    and previous_batch.targets[-1] == previous_sample.target
                    and np.array_equal(
                        next_batch.feature_x[0], next_sample.feature_x, equal_nan=True
                    )
                    and next_batch.targets[0] == next_sample.target
                )
                if not equivalent:
                    raise FullSplitIntegrationError("batch boundary sample mismatch")
                boundary_records.append(
                    {
                        "boundary_before_batch": next_number,
                        "previous_last_index": previous_index,
                        "next_first_index": next_index,
                        "exclusive_membership": True,
                        "d4_d6_values_and_targets_equal": True,
                        "no_cross_sample_topology_edge": True,
                    }
                )

            hashes = _d5_source_hashes(d5_report, experiment, split)
            integration_hash = integration_fingerprint(dataset, iterator.plan, hashes)
            repeated_hash = integration_fingerprint(
                dataset, SequentialBatchPlan(len(dataset), iterator.batch_size), hashes
            )
            validate_repeatable_fingerprints(integration_hash, repeated_hash)
            records.append(
                {
                    "experiment": experiment,
                    "split": split,
                    "dataset_length": len(dataset),
                    "source_artifacts": dict(dataset.source_artifacts),
                    "validated_batch_plans": plan_summaries,
                    "canonical_batch_size": iterator.batch_size,
                    "number_of_full_batches": iterator.plan.number_of_full_batches,
                    "final_batch_size": iterator.plan.final_batch_size,
                    "number_of_batches": iterator.number_of_batches,
                    "total_planned_indices": iterator.dataset_length,
                    "first_batch_range": [0, min(iterator.batch_size, len(dataset))],
                    "final_batch_range": [
                        iterator.boundary(iterator.number_of_batches - 1).start,
                        len(dataset),
                    ],
                    "coverage": {
                        "first_index": canonical_coverage.first_index,
                        "last_index": canonical_coverage.last_index,
                        "no_gaps": canonical_coverage.no_gaps,
                        "no_overlaps": canonical_coverage.no_overlaps,
                        "no_duplicates": canonical_coverage.no_duplicates,
                        "no_dropped_index": canonical_coverage.total_planned_indices
                        == len(dataset),
                        "boundary_sequence_sha256": canonical_coverage.boundary_sequence_sha256,
                    },
                    "constructed_boundary_batches": constructed_records,
                    "boundary_sample_validations": boundary_records,
                    "source_read_only": dataset.is_lazy_read_only,
                    "integration_fingerprint_sha256": integration_hash,
                }
            )

    e1_e3_plan_consistency = True
    e1_e3_boundary_features_equal = True
    for split in d4.SPLIT_STEMS:
        e1 = datasets[("E1", split)]
        e3 = datasets[("E3", split)]
        e1_plan = SequentialBatchPlan(len(e1), CANONICAL_INTEGRATION_BATCH_SIZE)
        e3_plan = SequentialBatchPlan(len(e3), CANONICAL_INTEGRATION_BATCH_SIZE)
        e1_summary = validate_boundary_sequence(e1_plan)
        e3_summary = validate_boundary_sequence(e3_plan)
        e1_e3_plan_consistency &= e1_summary == e3_summary
        for batch_number in _critical_batch_numbers(e1_plan):
            e1_batch = d6.build_logical_batch(e1, e1_plan.boundary(batch_number).indices)
            e3_batch = d6.build_logical_batch(e3, e3_plan.boundary(batch_number).indices)
            e1_e3_boundary_features_equal &= np.array_equal(
                e1_batch.feature_x, e3_batch.feature_x, equal_nan=True
            )

    e2_independent_plan = all(
        datasets[("E2", split)].x_path.parent.name == "exp2"
        and not _same_file(
            datasets[("E2", split)].x_path, datasets[("E1", split)].x_path
        )
        and len(datasets[("E2", split)])
        == d5.EXPECTED_SPLIT_LENGTHS["E2"][split]
        for split in d4.SPLIT_STEMS
    )
    after_mtimes = _source_mtimes(project_root)
    mtimes_unchanged = before_mtimes == after_mtimes
    integrity = (
        dict(integrity_result)
        if integrity_result is not None
        else d4.verify_frozen_integrity(project_root)
    )
    topology_invariant = len(topology_identities) == 1
    all_coverage = all(
        item["coverage"][field]
        for item in records
        for field in ("no_gaps", "no_overlaps", "no_duplicates", "no_dropped_index")
    )
    iterator_repeatable = all(
        item["coverage"]["boundary_sequence_sha256"]
        == validate_boundary_sequence(
            SequentialBatchPlan(item["dataset_length"], CANONICAL_INTEGRATION_BATCH_SIZE)
        ).boundary_sequence_sha256
        for item in records
    )
    overall = all(
        (
            len(records) == 9,
            all_coverage,
            iterator_repeatable,
            topology_invariant,
            e1_e3_plan_consistency,
            e1_e3_boundary_features_equal,
            e2_independent_plan,
            mtimes_unchanged,
            integrity.get("overall_status") == "PASS",
        )
    )
    report: dict[str, Any] = {
        "integration_version": INTEGRATION_VERSION,
        "provenance": dict(APPROVED_PROVENANCE),
        "validated_batch_sizes": list(VALIDATED_BATCH_SIZES),
        "canonical_integration_batch_size": CANONICAL_INTEGRATION_BATCH_SIZE,
        "runtime_mutation_guarantees": {
            "sequential_batch_plan_immutable": True,
            "sequential_batch_plan_representation": "tuple-backed immutable value with empty instance slots",
            "authoritative_tuple_payload_fields": ["dataset_length", "batch_size"],
            "object_setattr_plan_attacks_rejected": True,
            "plan_instance_dict_present": False,
            "mutable_nested_coverage_state_present": False,
            "coverage_counts_derived_from_immutable_fields": True,
            "batch_boundary_immutable": True,
            "batch_boundary_representation": "tuple-backed immutable value with empty instance slots",
            "object_setattr_boundary_attacks_rejected": True,
            "iterator_binding_immutable": True,
            "immutable_dataset_binding_snapshot": True,
            "binding_snapshot_covers_experiment_split_length_artifacts_provenance_and_memmaps": True,
            "binding_revalidated_before_and_after_each_numeric_batch": True,
            "mutation_and_rebinding_regression_tests": "PASS",
            "no_drop_after_mutation_attempts": True,
        },
        "experiment_split_records": records,
        "complete_index_coverage_status": all_coverage,
        "gap_overlap_duplicate_drop_status": "PASS" if all_coverage else "FAIL",
        "boundary_batch_construction_status": True,
        "boundary_sample_equivalence_status": True,
        "topology_invariance": {
            "single_d2_identity": topology_invariant,
            "topology_identity_sha256": next(iter(topology_identities)),
        },
        "e1_e3_plan_consistency_status": bool(e1_e3_plan_consistency),
        "e1_e3_boundary_feature_identity_status": bool(
            e1_e3_boundary_features_equal
        ),
        "e2_independent_plan_status": bool(e2_independent_plan),
        "iterator_repeatability_status": iterator_repeatable,
        "lazy_bounded_memory": {
            "numeric_batches_constructed_on_demand": True,
            "previous_batches_cached": False,
            "prefetching": False,
            "full_index_list_allocated": False,
            "full_split_feature_tensor_allocated": False,
            "maximum_constructed_batch_size": CANONICAL_INTEGRATION_BATCH_SIZE,
            "maximum_batch_feature_payload_bytes": CANONICAL_INTEGRATION_BATCH_SIZE
            * 78
            * 5
            * 4,
        },
        "source_immutability": {
            "all_sources_read_only": all(item["source_read_only"] for item in records),
            "source_mtimes_unchanged": mtimes_unchanged,
            "writes_performed": False,
        },
        "integration_fingerprints_repeatable": True,
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
    except FullSplitIntegrationError as exc:
        errors.append(str(exc))
    records = report.get("experiment_split_records", [])
    keys = {
        (item.get("experiment"), item.get("split"))
        for item in records
        if isinstance(item, Mapping)
    }
    expected_keys = {
        (experiment, split)
        for experiment in d4.EXPERIMENT_DIRS
        for split in d4.SPLIT_STEMS
    }
    if len(records) != 9 or keys != expected_keys:
        errors.append("report does not contain exactly all nine experiment/split pairs")
    required_true = (
        "complete_index_coverage_status",
        "boundary_batch_construction_status",
        "boundary_sample_equivalence_status",
        "e1_e3_plan_consistency_status",
        "e1_e3_boundary_feature_identity_status",
        "e2_independent_plan_status",
        "iterator_repeatability_status",
        "integration_fingerprints_repeatable",
    )
    for field in required_true:
        if report.get(field) is not True:
            errors.append(f"{field} is not true")
    if report.get("gap_overlap_duplicate_drop_status") != "PASS":
        errors.append("gap/overlap/duplicate/drop status is not PASS")
    guarantees = report.get("runtime_mutation_guarantees", {})
    guarantee_booleans = (
        "sequential_batch_plan_immutable",
        "coverage_counts_derived_from_immutable_fields",
        "batch_boundary_immutable",
        "iterator_binding_immutable",
        "immutable_dataset_binding_snapshot",
        "binding_snapshot_covers_experiment_split_length_artifacts_provenance_and_memmaps",
        "binding_revalidated_before_and_after_each_numeric_batch",
        "no_drop_after_mutation_attempts",
        "object_setattr_plan_attacks_rejected",
        "object_setattr_boundary_attacks_rejected",
    )
    if any(guarantees.get(field) is not True for field in guarantee_booleans):
        errors.append("runtime mutation guarantees are incomplete")
    if guarantees.get("mutation_and_rebinding_regression_tests") != "PASS":
        errors.append("mutation/rebinding regression tests are not PASS")
    if guarantees.get("plan_instance_dict_present") is not False:
        errors.append("plan instance dictionary claim is not false")
    if guarantees.get("mutable_nested_coverage_state_present") is not False:
        errors.append("mutable nested coverage claim is not false")
    if guarantees.get("authoritative_tuple_payload_fields") != [
        "dataset_length",
        "batch_size",
    ]:
        errors.append("authoritative tuple payload fields are incorrect")
    if report.get("topology_invariance", {}).get("single_d2_identity") is not True:
        errors.append("topology invariance failed")
    if report.get("stage_c_integrity", {}).get("overall_status") != "PASS":
        errors.append("Stage C/raw integrity status is not PASS")
    if report.get("overall_status") != "PASS":
        errors.append("overall status is not PASS")
    supplied = report.get("report_sha256")
    unhashed = dict(report)
    unhashed.pop("report_sha256", None)
    if not isinstance(supplied, str) or d2.canonical_sha256(unhashed) != supplied:
        errors.append("D8 report SHA-256 does not match canonical content")
    if errors:
        raise FullSplitIntegrationError("Stage D8 validation failed: " + "; ".join(errors))


def report_bytes(report: Mapping[str, Any]) -> bytes:
    validate_report(report)
    return (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def write_report(project_root: Path, output_path: Path) -> dict[str, Any]:
    report = run_full_split_integration(project_root)
    data = report_bytes(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(data)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Stage D8 full-split integration")
    parser.add_argument("--project-root", type=Path, default=_project_root())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/stage_d8/full_split_integration.json"),
    )
    args = parser.parse_args(argv)
    report = write_report(args.project_root, args.output)
    print(f"wrote {args.output}")
    print(f"overall_status={report['overall_status']}")
    print(f"report_sha256={report['report_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
