"""Stage D8 planner, iterator, mutation, and real integration tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from src.graph import adapter as d4
from src.graph import full_split_integration as d8


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "results" / "stage_d8" / "full_split_integration.json"


def test_plan_arithmetic_for_partial_final_batch() -> None:
    plan = d8.SequentialBatchPlan(10, 4)
    assert plan.dataset_length == 10
    assert plan.batch_size == 4
    assert plan.number_of_full_batches == 2
    assert plan.number_of_batches == 3
    assert plan.final_batch_size == 2
    assert list(plan.iter_boundaries()) == [
        d8.BatchBoundary(0, 0, 4),
        d8.BatchBoundary(1, 4, 8),
        d8.BatchBoundary(2, 8, 10),
    ]


def test_plan_arithmetic_for_divisible_split() -> None:
    plan = d8.SequentialBatchPlan(12, 4)
    assert plan.number_of_full_batches == 3
    assert plan.number_of_batches == 3
    assert plan.final_batch_size == 4
    assert plan.boundary(2) == d8.BatchBoundary(2, 8, 12)


@pytest.mark.parametrize(
    ("batch_size", "error", "message"),
    [
        (0, d8.FullSplitIntegrationError, "positive"),
        (-1, d8.FullSplitIntegrationError, "positive"),
        (True, TypeError, "bool"),
        (1.5, TypeError, "integer"),
    ],
)
def test_invalid_batch_sizes_are_rejected(
    batch_size: object, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        d8.SequentialBatchPlan(10, batch_size)  # type: ignore[arg-type]


def test_boundary_lookup_rejects_invalid_numbers() -> None:
    plan = d8.SequentialBatchPlan(10, 4)
    with pytest.raises(IndexError, match="non-negative"):
        plan.boundary(-1)
    with pytest.raises(IndexError, match="out of range"):
        plan.boundary(3)
    with pytest.raises(TypeError, match="bool"):
        plan.boundary(True)


def test_complete_streaming_coverage_summary() -> None:
    plan = d8.SequentialBatchPlan(1_000_003, 128)
    summary = d8.validate_boundary_sequence(plan)
    assert summary.first_index == 0
    assert summary.last_index == 1_000_002
    assert summary.total_planned_indices == 1_000_003
    assert summary.number_of_batches == (1_000_003 + 127) // 128
    assert summary.final_batch_size == 1_000_003 % 128
    assert summary.no_gaps and summary.no_overlaps and summary.no_duplicates


def test_incorrect_dataset_length_metadata_is_rejected() -> None:
    plan = d8.SequentialBatchPlan(10, 4)
    with pytest.raises(d8.FullSplitIntegrationError, match="dataset length metadata"):
        d8.validate_boundary_sequence(plan, expected_dataset_length=11)


@pytest.mark.parametrize("field", tuple(d8.APPROVED_PROVENANCE))
def test_corrupted_d1_through_d7_provenance_is_rejected(field: str) -> None:
    provenance = {**d8.APPROVED_PROVENANCE, field: "0" * 64}
    with pytest.raises(d8.FullSplitIntegrationError, match=field):
        d8.validate_stage_provenance(provenance)


def test_gap_in_plan_is_rejected() -> None:
    plan = d8.SequentialBatchPlan(10, 4)
    boundaries = [d8.BatchBoundary(0, 0, 4), d8.BatchBoundary(1, 5, 9)]
    with pytest.raises(d8.FullSplitIntegrationError, match="gap"):
        d8.validate_boundary_sequence(plan, boundaries)


def test_overlap_in_plan_is_rejected() -> None:
    plan = d8.SequentialBatchPlan(10, 4)
    boundaries = [d8.BatchBoundary(0, 0, 4), d8.BatchBoundary(1, 3, 7)]
    with pytest.raises(d8.FullSplitIntegrationError, match="overlap"):
        d8.validate_boundary_sequence(plan, boundaries)


def test_duplicate_planned_range_is_rejected() -> None:
    plan = d8.SequentialBatchPlan(10, 4)
    boundaries = [d8.BatchBoundary(0, 0, 4), d8.BatchBoundary(1, 0, 4)]
    with pytest.raises(d8.FullSplitIntegrationError, match="duplicate"):
        d8.validate_boundary_sequence(plan, boundaries)


def test_skipped_final_index_is_rejected() -> None:
    plan = d8.SequentialBatchPlan(10, 4)
    boundaries = [d8.BatchBoundary(0, 0, 4), d8.BatchBoundary(1, 4, 8)]
    with pytest.raises(d8.FullSplitIntegrationError, match="ends before"):
        d8.validate_boundary_sequence(plan, boundaries)


def test_incorrect_final_batch_size_is_rejected() -> None:
    plan = d8.SequentialBatchPlan(10, 4)
    boundaries = [
        d8.BatchBoundary(0, 0, 4),
        d8.BatchBoundary(1, 4, 8),
        d8.BatchBoundary(2, 8, 9),
    ]
    with pytest.raises(d8.FullSplitIntegrationError, match="final batch size"):
        d8.validate_boundary_sequence(plan, boundaries)


def test_plan_beginning_at_nonzero_is_rejected() -> None:
    plan = d8.SequentialBatchPlan(10, 4)
    with pytest.raises(d8.FullSplitIntegrationError, match="begins at nonzero"):
        d8.validate_boundary_sequence(plan, [d8.BatchBoundary(0, 1, 4)])


def test_plan_ending_before_dataset_is_rejected() -> None:
    plan = d8.SequentialBatchPlan(12, 4)
    boundaries = [d8.BatchBoundary(0, 0, 4), d8.BatchBoundary(1, 4, 8)]
    with pytest.raises(d8.FullSplitIntegrationError, match="ends before"):
        d8.validate_boundary_sequence(plan, boundaries)


def test_plan_extending_past_dataset_is_rejected() -> None:
    plan = d8.SequentialBatchPlan(10, 4)
    boundaries = [
        d8.BatchBoundary(0, 0, 4),
        d8.BatchBoundary(1, 4, 8),
        d8.BatchBoundary(2, 8, 11),
    ]
    with pytest.raises(d8.FullSplitIntegrationError, match="extends past"):
        d8.validate_boundary_sequence(plan, boundaries)


def test_cross_experiment_dataset_substitution_is_rejected() -> None:
    dataset = SimpleNamespace(experiment="E2", split="train")
    with pytest.raises(d8.FullSplitIntegrationError, match="cross-experiment"):
        d8.validate_dataset_identity(dataset, "E1", "train")  # type: ignore[arg-type]


def test_cross_split_dataset_substitution_is_rejected() -> None:
    dataset = SimpleNamespace(experiment="E1", split="test")
    with pytest.raises(d8.FullSplitIntegrationError, match="cross-split"):
        d8.validate_dataset_identity(dataset, "E1", "train")  # type: ignore[arg-type]


def test_nondeterministic_integration_fingerprint_is_rejected() -> None:
    with pytest.raises(d8.FullSplitIntegrationError, match="nondeterministic"):
        d8.validate_repeatable_fingerprints("a" * 64, "b" * 64)


def test_iterator_restarts_at_zero_and_constructs_lazily(monkeypatch) -> None:
    dataset = d4.LazyGraphDataset("E1", "validation")
    calls: list[tuple[int, ...]] = []
    original = d8.d6.build_logical_batch

    def recording_builder(source, indices):
        materialized = tuple(indices)
        calls.append(materialized)
        return original(source, materialized)

    monkeypatch.setattr(d8.d6, "build_logical_batch", recording_builder)
    iterator = d8.LazyBatchIterator(dataset, 32)
    assert calls == []
    first = next(iter(iterator))
    second_first = next(iter(iterator))
    assert calls == [tuple(range(32)), tuple(range(32))]
    assert first.sample_indices.tolist() == second_first.sample_indices.tolist() == list(range(32))
    assert np.array_equal(first.feature_x, second_first.feature_x, equal_nan=True)


def test_iterator_is_bound_to_one_experiment_and_split() -> None:
    dataset = d4.LazyGraphDataset("E2", "test")
    iterator = d8.LazyBatchIterator(dataset, 128)
    assert iterator.dataset is dataset
    assert iterator.experiment == "E2"
    assert iterator.split == "test"
    assert iterator.dataset_length == len(dataset)


def test_noncanonical_dataset_is_rejected(tmp_path: Path) -> None:
    directory = tmp_path / "exp1"
    directory.mkdir(parents=True)
    np.save(directory / "X_train.npy", np.zeros((5, 5, 78), np.float32))
    np.save(directory / "y_train.npy", np.zeros(5, np.int16))
    (directory / "split_metadata.json").write_text(
        json.dumps({"experiment": "1_global_chronological", "n_train": 5}),
        encoding="utf-8",
    )
    dataset = d4.LazyGraphDataset("E1", "train", processed_root=tmp_path)
    with pytest.raises(d8.FullSplitIntegrationError, match="approved processed root"):
        d8.LazyBatchIterator(dataset, 2)


def test_mutated_live_dataset_provenance_is_rejected() -> None:
    dataset = d4.LazyGraphDataset("E1", "validation")
    dataset.provenance = MappingProxyType(
        {**dict(dataset.provenance), "d1_taxonomy_sha256": "0" * 64}
    )
    with pytest.raises(d8.FullSplitIntegrationError, match="dataset provenance"):
        d8.LazyBatchIterator(dataset, 32)


def test_plan_coverage_state_is_immutable_after_validation() -> None:
    plan = d8.SequentialBatchPlan(10, 4)
    before = d8.validate_boundary_sequence(plan)
    attempts = (
        ("dataset_length", 8),
        ("batch_size", 2),
        ("number_of_batches", 2),
        ("experiment", "E2"),
        ("split", "test"),
    )
    for field, value in attempts:
        with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
            setattr(plan, field, value)
    after = d8.validate_boundary_sequence(plan)
    assert after == before
    assert after.first_index == 0
    assert after.last_index == 9
    assert after.total_planned_indices == 10
    assert after.number_of_batches == 3
    assert after.final_batch_size == 2


def test_batch_boundary_is_immutable_and_cannot_change_plan() -> None:
    plan = d8.SequentialBatchPlan(10, 4)
    boundary = plan.boundary(1)
    for field, value in (("batch_number", 7), ("start", 0), ("stop", 10)):
        with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
            setattr(boundary, field, value)
    assert plan.boundary(1) == d8.BatchBoundary(1, 4, 8)
    assert d8.validate_boundary_sequence(plan).total_planned_indices == 10


def test_iterator_direct_dataset_and_identity_rebinding_is_rejected() -> None:
    e1 = d4.LazyGraphDataset("E1", "validation")
    e2 = d4.LazyGraphDataset("E2", "validation")
    iterator = d8.LazyBatchIterator(e1, 128)
    for field, value in (
        ("dataset", e2),
        ("experiment", "E2"),
        ("split", "test"),
    ):
        with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
            setattr(iterator, field, value)
    batch = iterator.construct_batch(0)
    assert batch.experiment == "E1"
    assert batch.split == "validation"


def test_iterator_plan_rebinding_is_rejected() -> None:
    dataset = d4.LazyGraphDataset("E1", "validation")
    iterator = d8.LazyBatchIterator(dataset, 128)
    replacement = d8.SequentialBatchPlan(len(dataset), 32)
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        iterator.plan = replacement
    assert iterator.batch_size == 128
    assert iterator.plan.dataset_length == len(dataset)


def test_runtime_snapshot_rejects_bypassed_iterator_dataset_rebind() -> None:
    e1 = d4.LazyGraphDataset("E1", "validation")
    e2 = d4.LazyGraphDataset("E2", "validation")
    iterator = d8.LazyBatchIterator(e1, 128)
    object.__setattr__(iterator, "dataset", e2)
    with pytest.raises(d8.FullSplitIntegrationError, match="binding changed"):
        iterator.construct_batch(0)


def test_runtime_snapshot_rejects_underlying_numeric_source_rebind() -> None:
    dataset = d4.LazyGraphDataset("E1", "validation")
    replacement = d4.LazyGraphDataset("E2", "validation")
    iterator = d8.LazyBatchIterator(dataset, 128)
    dataset._x = replacement.x_memmap  # noqa: SLF001 - deliberate adversarial mutation
    dataset._y = replacement.y_memmap  # noqa: SLF001 - deliberate adversarial mutation
    with pytest.raises(d8.FullSplitIntegrationError, match="binding changed"):
        iterator.construct_batch(0)


def test_no_drop_guarantee_survives_all_rejected_plan_mutations() -> None:
    plan = d8.SequentialBatchPlan(1_003, 128)
    for field, value in (
        ("dataset_length", 1_000),
        ("batch_size", 100),
        ("number_of_batches", 1),
    ):
        with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
            setattr(plan, field, value)
    summary = d8.validate_boundary_sequence(plan)
    boundaries = tuple(plan.iter_boundaries())
    assert boundaries[0].start == 0
    assert boundaries[-1].stop == 1_003
    assert all(left.stop == right.start for left, right in zip(boundaries, boundaries[1:]))
    assert summary.total_planned_indices == 1_003
    assert summary.number_of_batches == 8
    assert summary.final_batch_size == 107


def test_low_level_dataset_length_mutation_is_rejected() -> None:
    plan = d8.SequentialBatchPlan(1_003, 128)
    with pytest.raises((AttributeError, TypeError)):
        object.__setattr__(plan, "dataset_length", 1_000)
    assert (plan.dataset_length, plan.batch_size) == (1_003, 128)
    assert (plan.number_of_batches, plan.final_batch_size) == (8, 107)
    summary = d8.validate_boundary_sequence(plan)
    assert (summary.first_index, summary.last_index) == (0, 1_002)
    assert summary.total_planned_indices == 1_003


def test_low_level_batch_size_mutation_is_rejected() -> None:
    plan = d8.SequentialBatchPlan(1_003, 128)
    with pytest.raises((AttributeError, TypeError)):
        object.__setattr__(plan, "batch_size", 100)
    assert (plan.dataset_length, plan.batch_size) == (1_003, 128)
    assert (plan.number_of_batches, plan.final_batch_size) == (8, 107)
    summary = d8.validate_boundary_sequence(plan)
    assert summary.total_planned_indices == 1_003
    assert summary.no_gaps and summary.no_overlaps and summary.no_duplicates


def test_tuple_backed_plan_has_no_replaceable_or_nested_authority() -> None:
    plan = d8.SequentialBatchPlan(1_003, 128)
    assert isinstance(plan, tuple)
    authoritative_payload = (plan[0], plan[1])
    assert authoritative_payload == (1_003, 128)
    assert not hasattr(plan, "__dict__")
    for field, value in (
        ("_dataset_length", 1_000),
        ("_batch_size", 100),
        ("number_of_batches", 1),
        ("final_batch_size", 1),
        ("__dict__", {"dataset_length": 1_000}),
    ):
        with pytest.raises((AttributeError, TypeError)):
            object.__setattr__(plan, field, value)
    with pytest.raises(TypeError):
        plan[0] = 1_000  # type: ignore[index]
    assert all(
        not isinstance(value, (dict, list, set, bytearray))
        for value in authoritative_payload
    )
    summary = d8.validate_boundary_sequence(plan)
    assert (summary.last_index, summary.total_planned_indices) == (1_002, 1_003)


def test_tuple_backed_boundary_rejects_all_mutation_paths() -> None:
    plan = d8.SequentialBatchPlan(1_003, 128)
    boundary = plan.boundary(1)
    assert isinstance(boundary, tuple)
    assert tuple(boundary) == (1, 128, 256)
    assert not hasattr(boundary, "__dict__")
    for field, value in (
        ("batch_number", 7),
        ("start", 0),
        ("stop", 1_003),
        ("_start", 0),
        ("__dict__", {"stop": 1_003}),
    ):
        with pytest.raises((AttributeError, TypeError)):
            object.__setattr__(boundary, field, value)
    with pytest.raises(TypeError):
        boundary[1] = 0  # type: ignore[index]
    assert plan.boundary(1) == d8.BatchBoundary(1, 128, 256)
    assert d8.validate_boundary_sequence(plan).total_planned_indices == 1_003


def test_real_report_is_complete_and_deterministic() -> None:
    saved = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    regenerated = d8.run_full_split_integration(
        ROOT, integrity_result=saved["stage_c_integrity"]
    )
    assert regenerated == saved
    assert d8.report_bytes(regenerated) == REPORT_PATH.read_bytes()
    assert len(saved["experiment_split_records"]) == 9
    assert saved["validated_batch_sizes"] == [1, 32, 128]
    assert saved["canonical_integration_batch_size"] == 128
    assert saved["overall_status"] == "PASS"


def test_report_rejects_corrupted_hash() -> None:
    saved = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    saved["report_sha256"] = "0" * 64
    with pytest.raises(d8.FullSplitIntegrationError, match="SHA-256"):
        d8.validate_report(saved)
