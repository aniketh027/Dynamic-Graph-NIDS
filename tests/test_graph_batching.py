"""Stage D6 synthetic, real-artifact, and adversarial batching tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from src.graph import adapter as d4
from src.graph import batching as d6


ROOT = Path(__file__).resolve().parents[1]
D6_REPORT = ROOT / "results" / "stage_d6" / "template_batching_validation.json"


def _fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture()
def synthetic_dataset(tmp_path: Path) -> d4.LazyGraphDataset:
    directory = tmp_path / "exp1"
    directory.mkdir(parents=True)
    x = np.arange(8 * 5 * 78, dtype=np.float32).reshape(8, 5, 78)
    y = np.arange(8, dtype=np.int16)
    np.save(directory / "X_train.npy", x, allow_pickle=False)
    np.save(directory / "y_train.npy", y, allow_pickle=False)
    (directory / "split_metadata.json").write_text(
        json.dumps({"experiment": "1_global_chronological", "n_train": 8}),
        encoding="utf-8",
    )
    return d4.LazyGraphDataset("E1", "train", processed_root=tmp_path)


def _fixture_batch(
    dataset: d4.LazyGraphDataset, indices: object, **kwargs: object
) -> d6.LogicalGraphBatch:
    return d6.build_logical_batch(
        dataset,
        indices,  # type: ignore[arg-type]
        _allow_noncanonical_fixture=True,
        **kwargs,  # type: ignore[arg-type]
    )


def test_batch_size_one_contract(synthetic_dataset: d4.LazyGraphDataset) -> None:
    batch = _fixture_batch(synthetic_dataset, [3])
    assert batch.batch_size == 1
    assert batch.feature_x.shape == (1, 78, 5)
    assert batch.targets.shape == (1,)
    assert batch.sample_node_count == 1


def test_multiple_noncontiguous_indices_preserve_requested_order(
    synthetic_dataset: d4.LazyGraphDataset,
) -> None:
    indices = [6, 1, 4, 0]
    batch = _fixture_batch(synthetic_dataset, indices)
    assert batch.sample_indices.tolist() == indices
    for position, index in enumerate(indices):
        assert np.array_equal(batch.feature_x[position], synthetic_dataset[index].feature_x)
        assert batch.targets[position] == synthetic_dataset[index].target


def test_contiguous_indices_are_supported(synthetic_dataset: d4.LazyGraphDataset) -> None:
    batch = _fixture_batch(synthetic_dataset, [2, 3, 4])
    assert batch.sample_indices.tolist() == [2, 3, 4]


def test_feature_and_target_values_dtypes_and_shapes(
    synthetic_dataset: d4.LazyGraphDataset,
) -> None:
    batch = _fixture_batch(synthetic_dataset, [7, 2])
    assert batch.feature_x.shape == (2, 78, 5)
    assert batch.feature_x.dtype == np.dtype(np.float32)
    assert batch.targets.shape == (2,)
    assert batch.targets.dtype == np.dtype(np.int16)
    assert np.array_equal(batch.feature_x[0], synthetic_dataset.x_memmap[7].T)
    assert np.array_equal(batch.feature_x[1], synthetic_dataset.x_memmap[2].T)
    assert batch.targets.tolist() == [7, 2]


def test_repeated_calls_are_deterministic(synthetic_dataset: d4.LazyGraphDataset) -> None:
    first = _fixture_batch(synthetic_dataset, [5, 0, 7])
    second = _fixture_batch(synthetic_dataset, [5, 0, 7])
    for field in (
        "feature_x",
        "targets",
        "sample_indices",
        "feature_batch",
        "category_batch",
        "sample_batch",
        "feature_to_category_sources",
        "feature_to_category_destinations",
        "category_to_sample_sources",
        "category_to_sample_destinations",
    ):
        assert np.array_equal(getattr(first, field), getattr(second, field))


def test_ownership_vectors(synthetic_dataset: d4.LazyGraphDataset) -> None:
    batch = _fixture_batch(synthetic_dataset, [4, 1, 7])
    assert np.array_equal(batch.feature_batch, np.repeat(np.arange(3), 78))
    assert np.array_equal(batch.category_batch, np.repeat(np.arange(3), 8))
    assert np.array_equal(batch.sample_batch, np.arange(3))
    assert batch.feature_batch.dtype == np.int64
    assert batch.category_batch.dtype == np.int64
    assert batch.sample_batch.dtype == np.int64


def test_feature_and_category_offsets_follow_templates(
    synthetic_dataset: d4.LazyGraphDataset,
) -> None:
    batch = _fixture_batch(synthetic_dataset, [0, 2, 5])
    for graph_id in range(3):
        feature_slice = slice(graph_id * 78, (graph_id + 1) * 78)
        assert np.array_equal(
            batch.feature_to_category_sources[feature_slice],
            np.asarray(batch.static.feature_to_category_sources) + graph_id * 78,
        )
        assert np.array_equal(
            batch.feature_to_category_destinations[feature_slice],
            np.asarray(batch.static.feature_to_category_destinations) + graph_id * 8,
        )
        category_slice = slice(graph_id * 8, (graph_id + 1) * 8)
        assert np.array_equal(
            batch.category_to_sample_sources[category_slice],
            np.asarray(batch.static.category_to_sample_sources) + graph_id * 8,
        )
        assert np.array_equal(
            batch.category_to_sample_destinations[category_slice],
            np.full(8, graph_id),
        )


@pytest.mark.parametrize("size", [1, 2, 4])
def test_exact_edge_and_node_counts(
    synthetic_dataset: d4.LazyGraphDataset, size: int
) -> None:
    batch = _fixture_batch(synthetic_dataset, range(size))
    assert batch.feature_node_count == size * 78
    assert batch.category_node_count == size * 8
    assert batch.sample_node_count == size
    assert batch.feature_to_category_sources.size == size * 78
    assert batch.category_to_sample_sources.size == size * 8
    assert batch.total_edge_count == size * 86


def test_no_cross_sample_edges(synthetic_dataset: d4.LazyGraphDataset) -> None:
    batch = _fixture_batch(synthetic_dataset, [7, 0, 3, 2])
    assert np.array_equal(
        batch.feature_to_category_sources // 78,
        batch.feature_to_category_destinations // 8,
    )
    assert np.array_equal(
        batch.category_to_sample_sources // 8,
        batch.category_to_sample_destinations,
    )


def test_each_feature_retains_approved_category(
    synthetic_dataset: d4.LazyGraphDataset,
) -> None:
    batch = _fixture_batch(synthetic_dataset, [5, 1])
    approved = np.asarray(batch.static.feature_category_ids)
    for graph_id in range(2):
        edge_slice = slice(graph_id * 78, (graph_id + 1) * 78)
        local_destinations = batch.feature_to_category_destinations[edge_slice] - graph_id * 8
        assert np.array_equal(local_destinations, approved)


def test_all_batch_arrays_are_read_only_and_source_is_unchanged(
    synthetic_dataset: d4.LazyGraphDataset,
) -> None:
    before = (
        _fingerprint(synthetic_dataset.x_path),
        _fingerprint(synthetic_dataset.y_path),
    )
    batch = _fixture_batch(synthetic_dataset, [4, 0, 6])
    arrays = (
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
    assert all(not array.flags.writeable for array in arrays)
    with pytest.raises(ValueError):
        batch.feature_x[0, 0, 0] = -1
    assert (_fingerprint(synthetic_dataset.x_path), _fingerprint(synthetic_dataset.y_path)) == before


@pytest.mark.parametrize(
    ("indices", "error", "message"),
    [
        ([-1], IndexError, "negative"),
        ([8], IndexError, "out of range"),
        ([1.5], TypeError, "integer"),
        ([True], TypeError, "bool"),
        ([2, 2], d6.BatchContractError, "duplicate"),
        ([], d6.BatchContractError, "empty"),
    ],
)
def test_invalid_index_collections_are_rejected(
    synthetic_dataset: d4.LazyGraphDataset,
    indices: list[object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        _fixture_batch(synthetic_dataset, indices)


def test_noniterable_indices_are_rejected(synthetic_dataset: d4.LazyGraphDataset) -> None:
    with pytest.raises(TypeError, match="iterable"):
        _fixture_batch(synthetic_dataset, 3)


def test_non_dataset_object_is_rejected() -> None:
    with pytest.raises(TypeError, match="LazyGraphDataset"):
        d6.build_logical_batch(object(), [0])  # type: ignore[arg-type]


def test_noncanonical_dataset_is_rejected_by_production_api(
    synthetic_dataset: d4.LazyGraphDataset,
) -> None:
    with pytest.raises(d6.BatchContractError, match="D5-approved processed root"):
        d6.build_logical_batch(synthetic_dataset, [0])


def test_corrupted_dataset_provenance_is_rejected(
    synthetic_dataset: d4.LazyGraphDataset,
) -> None:
    synthetic_dataset.provenance = MappingProxyType(
        {**dict(synthetic_dataset.provenance), "d2_schema_sha256": "0" * 64}
    )
    with pytest.raises(d6.BatchContractError, match="provenance"):
        _fixture_batch(synthetic_dataset, [0])


def test_corrupted_d5_report_is_rejected(
    synthetic_dataset: d4.LazyGraphDataset, tmp_path: Path
) -> None:
    report = json.loads(
        (ROOT / "results" / "stage_d5" / "experiment_boundary_validation.json").read_text(
            encoding="utf-8"
        )
    )
    report["report_sha256"] = "0" * 64
    path = tmp_path / "d5.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(d6.BatchContractError, match="D5 validation report is invalid"):
        _fixture_batch(synthetic_dataset, [0], d5_report_path=path)


@pytest.mark.parametrize(
    ("experiment", "split"),
    [
        (experiment, split)
        for experiment in d4.EXPERIMENT_DIRS
        for split in d4.SPLIT_STEMS
    ],
)
def test_real_nine_pairs_exact_batch_equivalence(
    experiment: str, split: str
) -> None:
    dataset = d4.LazyGraphDataset(experiment, split)
    indices = [0, len(dataset) // 2, len(dataset) - 1]
    batch = d6.build_logical_batch(dataset, indices)
    assert batch.experiment == experiment
    assert batch.split == split
    assert batch.source_dataset_length == len(dataset)
    assert batch.sample_indices.tolist() == indices
    assert batch.feature_x.shape == (3, 78, 5)
    assert batch.targets.shape == (3,)
    assert batch.total_edge_count == 3 * 86
    for position, index in enumerate(indices):
        sample = dataset[index]
        assert np.array_equal(batch.feature_x[position], sample.feature_x, equal_nan=True)
        assert batch.targets[position] == sample.target
        assert batch.static.feature_category_ids == sample.feature_category_ids
    assert np.array_equal(
        batch.feature_to_category_sources // 78,
        batch.feature_to_category_destinations // 8,
    )
    assert np.array_equal(
        batch.category_to_sample_sources // 8,
        batch.category_to_sample_destinations,
    )


@pytest.mark.parametrize("split", tuple(d4.SPLIT_STEMS))
def test_real_e1_e3_feature_batches_are_identical(split: str) -> None:
    e1 = d4.LazyGraphDataset("E1", split)
    e3 = d4.LazyGraphDataset("E3", split)
    indices = [0, len(e1) // 2, len(e1) - 1]
    assert np.array_equal(
        d6.build_logical_batch(e1, indices).feature_x,
        d6.build_logical_batch(e3, indices).feature_x,
        equal_nan=True,
    )


@pytest.mark.parametrize("split", tuple(d4.SPLIT_STEMS))
def test_real_e2_batch_uses_independent_exp2_source(split: str) -> None:
    e1 = d4.LazyGraphDataset("E1", split)
    e2 = d4.LazyGraphDataset("E2", split)
    batch = d6.build_logical_batch(e2, [0, len(e2) // 2, len(e2) - 1])
    assert batch.experiment == "E2"
    assert e2.x_path.parent.name == "exp2"
    assert e2.x_path != e1.x_path
    assert batch.static is e2.static


def test_saved_d6_report_is_deterministic_and_complete() -> None:
    saved = json.loads(D6_REPORT.read_text(encoding="utf-8"))
    regenerated = d6.build_validation_report(
        ROOT, integrity_result=saved["stage_c_integrity"]
    )
    assert regenerated == saved
    assert d6.report_bytes(regenerated) == D6_REPORT.read_bytes()
    assert len(saved["experiment_split_batches"]) == 9
    assert saved["overall_status"] == "PASS"
    assert saved["provenance"]["d5_report_sha256"] == d6.APPROVED_D5_REPORT_SHA256


def test_corrupted_d6_report_hash_is_rejected() -> None:
    saved = json.loads(D6_REPORT.read_text(encoding="utf-8"))
    saved["report_sha256"] = "0" * 64
    with pytest.raises(d6.BatchContractError, match="SHA-256"):
        d6.report_bytes(saved)
