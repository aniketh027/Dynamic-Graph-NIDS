"""Stage D4 synthetic and real read-only lazy-adapter tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.graph import schema as d2
from src.graph import taxonomy as d1
from src.graph.adapter import (
    ADAPTER_VERSION,
    EXPERIMENT_DIRS,
    EXPERIMENT_METADATA_IDENTITIES,
    SPLIT_STEMS,
    STATIC_GRAPH_METADATA,
    AdapterContractError,
    LazyGraphDataset,
    build_validation_report,
    report_bytes,
)


ROOT = Path(__file__).resolve().parents[1]
D1_PATH = ROOT / "results" / "stage_d1" / "feature_taxonomy.json"
D2_PATH = ROOT / "results" / "stage_d2" / "graph_schema.json"
D3_PATH = ROOT / "results" / "stage_d3" / "structural_validation.json"
D4_PATH = ROOT / "results" / "stage_d4" / "lazy_adapter_validation.json"
PROCESSED_ROOT = ROOT / "data" / "processed" / "preprocessed"
EXPECTED_REPORT_SHA256 = "ccd065556c533d4fd1cd2e4d821566611db9d2a5a9e7ea036c0cca3b3f6500d6"


def _write_fixture(
    tmp_path: Path,
    *,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    metadata_count: int | None = None,
) -> tuple[Path, np.ndarray, np.ndarray]:
    x = (
        np.arange(3 * 5 * 78, dtype=np.float32).reshape(3, 5, 78)
        if x is None
        else x
    )
    y = np.array([14, 3, 0], dtype=np.int16) if y is None else y
    experiment_dir = tmp_path / "exp1"
    experiment_dir.mkdir(parents=True)
    np.save(experiment_dir / "X_train.npy", x, allow_pickle=False)
    np.save(experiment_dir / "y_train.npy", y, allow_pickle=False)
    (experiment_dir / "split_metadata.json").write_text(
        json.dumps(
            {
                "experiment": EXPERIMENT_METADATA_IDENTITIES["E1"],
                "n_train": len(x) if metadata_count is None else metadata_count,
            }
        ),
        encoding="utf-8",
    )
    return tmp_path, x, y


def _fingerprint(path: Path) -> tuple[int, int, str]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest()


def test_synthetic_exact_transpose_feature_and_time_identity(tmp_path: Path) -> None:
    processed, x, _ = _write_fixture(tmp_path)
    dataset = LazyGraphDataset("E1", "train", processed_root=processed)
    sample = dataset[1]
    assert sample.feature_x.shape == (78, 5)
    assert np.array_equal(sample.feature_x, x[1].T, equal_nan=True)
    for feature_id in range(78):
        assert np.array_equal(sample.feature_x[feature_id], x[1, :, feature_id])
    for time_id in range(5):
        assert np.array_equal(sample.feature_x[:, time_id], x[1, time_id, :])


def test_synthetic_target_and_dtypes_are_preserved(tmp_path: Path) -> None:
    processed, _, y = _write_fixture(tmp_path)
    sample = LazyGraphDataset("E1", "train", processed_root=processed)[1]
    assert sample.feature_x.dtype == np.dtype(np.float32)
    assert isinstance(sample.target, np.int16)
    assert sample.target.dtype == np.dtype(np.int16)
    assert sample.target == y[1]


def test_static_topology_and_category_metadata_are_shared_and_aligned(tmp_path: Path) -> None:
    processed, _, _ = _write_fixture(tmp_path)
    dataset = LazyGraphDataset("E1", "train", processed_root=processed)
    first = dataset[0]
    second = dataset[2]
    assert first.static is second.static is dataset.static is STATIC_GRAPH_METADATA
    assert first.feature_ids == tuple(range(78))
    assert first.feature_names == d1.MODEL_FEATURE_COLUMNS
    assert first.feature_category_ids is d1.FEATURE_INDEX_TO_CATEGORY_ID
    assert first.category_ids == tuple(range(8))
    assert first.sample_node_id == 0
    assert first.static.feature_to_category_destinations is d2.FEATURE_TO_CATEGORY_DESTINATION_INDICES
    assert len(first.static.feature_to_category_sources) == 78
    assert len(first.static.category_to_sample_sources) == 8


def test_target_is_not_inserted_into_predictive_feature_array(tmp_path: Path) -> None:
    processed, _, _ = _write_fixture(tmp_path)
    dataset = LazyGraphDataset("E1", "train", processed_root=processed)
    sample = dataset[0]
    assert sample.feature_x.shape == (78, 5)
    assert sample.feature_x.size == 78 * 5
    assert np.shares_memory(sample.feature_x, dataset.x_memmap)
    assert not np.shares_memory(sample.feature_x, dataset.y_memmap)


def test_initialization_and_indexing_remain_lazy_and_read_only(tmp_path: Path) -> None:
    processed, _, _ = _write_fixture(tmp_path)
    dataset = LazyGraphDataset("E1", "train", processed_root=processed)
    assert isinstance(dataset.x_memmap, np.memmap)
    assert isinstance(dataset.y_memmap, np.memmap)
    assert dataset.x_memmap.mode == dataset.y_memmap.mode == "r"
    assert dataset.is_lazy_read_only
    sample = dataset[0]
    assert sample.feature_x.flags.writeable is False
    assert sample.feature_x.nbytes == 78 * 5 * np.dtype(np.float32).itemsize


@pytest.mark.parametrize(
    ("x", "message"),
    [
        (np.zeros((3, 78, 5), dtype=np.float32), "X shape must be"),
        (np.zeros((3, 5, 78), dtype=np.float64), "X dtype must be float32"),
    ],
)
def test_invalid_x_contract_is_rejected(tmp_path: Path, x: np.ndarray, message: str) -> None:
    processed, _, _ = _write_fixture(tmp_path, x=x)
    with pytest.raises(AdapterContractError, match=message):
        LazyGraphDataset("E1", "train", processed_root=processed)


@pytest.mark.parametrize(
    ("y", "message"),
    [
        (np.zeros((3, 1), dtype=np.int16), "y shape must be"),
        (np.zeros(3, dtype=np.int32), "y dtype must be int16"),
        (np.zeros(2, dtype=np.int16), "X/y length mismatch"),
    ],
)
def test_invalid_y_contract_is_rejected(tmp_path: Path, y: np.ndarray, message: str) -> None:
    processed, _, _ = _write_fixture(tmp_path, y=y)
    with pytest.raises(AdapterContractError, match=message):
        LazyGraphDataset("E1", "train", processed_root=processed)


def test_metadata_count_mismatch_is_rejected(tmp_path: Path) -> None:
    processed, _, _ = _write_fixture(tmp_path, metadata_count=999)
    with pytest.raises(AdapterContractError, match="split metadata count"):
        LazyGraphDataset("E1", "train", processed_root=processed)


def test_index_contract_rejects_negative_end_and_noninteger(tmp_path: Path) -> None:
    processed, _, _ = _write_fixture(tmp_path)
    dataset = LazyGraphDataset("E1", "train", processed_root=processed)
    with pytest.raises(IndexError, match="negative"):
        dataset[-1]
    with pytest.raises(IndexError, match="out of range"):
        dataset[len(dataset)]
    with pytest.raises(IndexError, match="out of range"):
        dataset[len(dataset) + 10]
    with pytest.raises(TypeError, match="integer"):
        dataset[1.5]
    with pytest.raises(TypeError, match="bool"):
        dataset[True]


def test_unknown_experiment_and_split_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(AdapterContractError, match="experiment must"):
        LazyGraphDataset("E4", "train", processed_root=tmp_path)
    with pytest.raises(AdapterContractError, match="split must"):
        LazyGraphDataset("E1", "val", processed_root=tmp_path)


def test_source_arrays_remain_byte_identical_after_indexing(tmp_path: Path) -> None:
    processed, _, _ = _write_fixture(tmp_path)
    x_path = processed / "exp1" / "X_train.npy"
    y_path = processed / "exp1" / "y_train.npy"
    before = (_fingerprint(x_path), _fingerprint(y_path))
    dataset = LazyGraphDataset("E1", "train", processed_root=processed)
    _ = dataset[0]
    _ = dataset[len(dataset) - 1]
    assert (_fingerprint(x_path), _fingerprint(y_path)) == before


@pytest.mark.parametrize("artifact", ["d1", "d2", "d3"])
def test_corrupted_graph_provenance_is_rejected(tmp_path: Path, artifact: str) -> None:
    processed, _, _ = _write_fixture(tmp_path / "arrays")
    paths = {
        "d1": tmp_path / "d1.json",
        "d2": tmp_path / "d2.json",
        "d3": tmp_path / "d3.json",
    }
    originals = {"d1": D1_PATH, "d2": D2_PATH, "d3": D3_PATH}
    for name, path in paths.items():
        payload = json.loads(originals[name].read_text(encoding="utf-8"))
        if name == artifact:
            hash_field = {"d1": "taxonomy_sha256", "d2": "schema_sha256", "d3": "report_sha256"}[name]
            payload[hash_field] = "0" * 64
        path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AdapterContractError, match="graph provenance validation failed"):
        LazyGraphDataset(
            "E1",
            "train",
            processed_root=processed,
            d1_manifest_path=paths["d1"],
            d2_manifest_path=paths["d2"],
            d3_report_path=paths["d3"],
        )


@pytest.mark.parametrize(
    ("experiment", "split"),
    [(experiment, split) for experiment in EXPERIMENT_DIRS for split in SPLIT_STEMS],
)
def test_real_split_boundaries_and_interior_preserve_values(
    experiment: str, split: str
) -> None:
    dataset = LazyGraphDataset(experiment, split)
    assert dataset.is_lazy_read_only
    for index in (0, len(dataset) // 2, len(dataset) - 1):
        sample = dataset[index]
        source = dataset.x_memmap[index]
        assert sample.sample_index == index
        assert sample.experiment == experiment
        assert sample.split == split
        assert sample.feature_x.shape == (78, 5)
        assert sample.feature_x.dtype == np.dtype(np.float32)
        assert np.array_equal(sample.feature_x, source.T, equal_nan=True)
        assert np.array_equal(sample.feature_x[0], source[:, 0], equal_nan=True)
        assert np.array_equal(sample.feature_x[77], source[:, 77], equal_nan=True)
        assert np.array_equal(sample.feature_x[:, 0], source[0, :], equal_nan=True)
        assert np.array_equal(sample.feature_x[:, 4], source[4, :], equal_nan=True)
        assert isinstance(sample.target, np.int16)
        assert sample.target == dataset.y_memmap[index]
        assert sample.static is STATIC_GRAPH_METADATA


@pytest.mark.parametrize("split", tuple(SPLIT_STEMS))
def test_real_e1_e3_selected_feature_values_match(split: str) -> None:
    e1 = LazyGraphDataset("E1", split)
    e3 = LazyGraphDataset("E3", split)
    assert len(e1) == len(e3)
    for index in (0, len(e1) // 2, len(e1) - 1):
        assert np.array_equal(e1[index].feature_x, e3[index].feature_x, equal_nan=True)


@pytest.mark.parametrize("split", tuple(SPLIT_STEMS))
def test_real_e2_uses_its_own_numeric_artifacts(split: str) -> None:
    e1 = LazyGraphDataset("E1", split)
    e2 = LazyGraphDataset("E2", split)
    assert e2.x_path.parent.name == "exp2"
    assert e2.y_path.parent.name == "exp2"
    assert e2.x_path != e1.x_path
    assert e2.y_path != e1.y_path
    assert e2.static is e1.static is STATIC_GRAPH_METADATA


def test_checked_d4_report_is_deterministic_and_complete() -> None:
    saved = json.loads(D4_PATH.read_text(encoding="utf-8"))
    regenerated = build_validation_report(
        ROOT,
        integrity_result=saved["stage_c_integrity"],
    )
    assert saved["adapter_version"] == ADAPTER_VERSION
    assert len(saved["experiment_splits"]) == 9
    assert saved["report_sha256"] == EXPECTED_REPORT_SHA256
    assert regenerated == saved
    assert report_bytes(regenerated) == D4_PATH.read_bytes()
    assert saved["overall_status"] == "PASS"
