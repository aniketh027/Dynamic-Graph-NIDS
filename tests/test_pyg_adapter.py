"""Stage E.2 synthetic, adversarial, real-data, and CUDA adapter tests."""

from __future__ import annotations

import ast
from dataclasses import replace
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
import torch

from src.graph import adapter as d4
from src.graph import batching as d6
from src.graph import pyg_adapter as e2


ROOT = Path(__file__).resolve().parents[1]
E2_REPORT = ROOT / "results" / "stage_e2" / "heterodata_adapter_validation.json"


def _synthetic_dataset(tmp_path: Path, experiment: str = "E1") -> d4.LazyGraphDataset:
    directory_name = {"E1": "exp1", "E2": "exp2", "E3": "exp3"}[experiment]
    experiment_name = d4.EXPERIMENT_METADATA_IDENTITIES[experiment]
    directory = tmp_path / directory_name
    directory.mkdir(parents=True)
    values = np.arange(8 * 5 * 78, dtype=np.float32).reshape(8, 5, 78)
    targets = np.arange(8, dtype=np.int16) if experiment != "E3" else np.arange(8, dtype=np.int16) % 2
    np.save(directory / "X_train.npy", values, allow_pickle=False)
    np.save(directory / "y_train.npy", targets, allow_pickle=False)
    (directory / "split_metadata.json").write_text(
        json.dumps({"experiment": experiment_name, "n_train": 8}), encoding="utf-8"
    )
    return d4.LazyGraphDataset(experiment, "train", processed_root=tmp_path)


def _batch(tmp_path: Path, indices: list[int], experiment: str = "E1") -> d6.LogicalGraphBatch:
    return d6.build_logical_batch(
        _synthetic_dataset(tmp_path, experiment),
        indices,
        _allow_noncanonical_fixture=True,
    )


@pytest.mark.parametrize("indices", [[3], [6, 1, 4]])
def test_direct_batched_conversion_contract(tmp_path: Path, indices: list[int]) -> None:
    batch = _batch(tmp_path, indices)
    data = e2.logical_batch_to_heterodata(batch)
    size = len(indices)

    assert type(data).__name__ == "HeteroData"
    assert tuple(data.node_types) == e2.NODE_TYPES
    assert tuple(data.edge_types) == e2.EDGE_TYPES
    assert data["feature"].num_nodes == size * 78
    assert data["category"].num_nodes == size * 8
    assert data["sample"].num_nodes == size
    assert data["feature"].x.shape == (size * 78, 5)
    assert data["feature"].x.dtype == torch.float32
    assert torch.equal(
        data["feature"].x.reshape(size, 78, 5),
        torch.tensor(np.array(batch.feature_x, copy=True)),
    )
    assert data["feature"].feature_id.dtype == torch.long
    assert data["feature"].feature_id.tolist() == list(range(78)) * size
    assert data["category"].category_id.dtype == torch.long
    assert data["category"].category_id.tolist() == list(range(8)) * size
    assert data["feature"].batch.tolist() == np.repeat(np.arange(size), 78).tolist()
    assert data["category"].batch.tolist() == np.repeat(np.arange(size), 8).tolist()
    assert data["sample"].batch.tolist() == list(range(size))
    assert data["feature"].ptr.tolist() == list(range(0, (size + 1) * 78, 78))
    assert data["category"].ptr.tolist() == list(range(0, (size + 1) * 8, 8))
    assert data["sample"].ptr.tolist() == list(range(size + 1))
    assert data.validate()


def test_feature_and_temporal_order_are_exact(tmp_path: Path) -> None:
    batch = _batch(tmp_path, [7, 2])
    data = e2.logical_batch_to_heterodata(batch)
    restored = data["feature"].x.reshape(2, 78, 5)
    for graph in range(2):
        for feature in (0, 31, 77):
            assert torch.equal(
                restored[graph, feature],
                torch.tensor(batch.feature_x[graph, feature].copy()),
            )


@pytest.mark.parametrize(
    ("experiment", "expected_dtype"),
    [("E1", torch.long), ("E2", torch.long), ("E3", torch.float32)],
)
def test_target_policy_preserves_values(
    tmp_path: Path, experiment: str, expected_dtype: torch.dtype
) -> None:
    batch = _batch(tmp_path, [0, 3, 5], experiment)
    data = e2.logical_batch_to_heterodata(batch)
    assert data["sample"].y.dtype == expected_dtype
    assert data["sample"].y.tolist() == batch.targets.astype(
        np.float32 if experiment == "E3" else np.int64
    ).tolist()
    assert set(data["sample"].y.tolist()).issubset({0.0, 1.0}) if experiment == "E3" else True


def test_topology_is_direct_exact_and_sample_isolated(tmp_path: Path) -> None:
    batch = _batch(tmp_path, [6, 1, 4])
    data = e2.logical_batch_to_heterodata(batch)
    ftc = data[e2.FEATURE_RELATION].edge_index
    cts = data[e2.CATEGORY_RELATION].edge_index
    assert ftc.shape == (2, 3 * 78)
    assert cts.shape == (2, 3 * 8)
    assert ftc.shape[1] + cts.shape[1] == 3 * 86
    assert torch.equal(ftc[0], torch.tensor(batch.feature_to_category_sources.copy()))
    assert torch.equal(ftc[1], torch.tensor(batch.feature_to_category_destinations.copy()))
    assert torch.equal(cts[0], torch.tensor(batch.category_to_sample_sources.copy()))
    assert torch.equal(cts[1], torch.tensor(batch.category_to_sample_destinations.copy()))
    assert torch.equal(ftc[0] // 78, ftc[1] // 8)
    assert torch.equal(cts[0] // 8, cts[1])
    assert int(ftc[0].min()) == 0 and int(ftc[0].max()) == 3 * 78 - 1
    assert int(ftc[1].min()) >= 0 and int(ftc[1].max()) < 3 * 8
    assert int(cts[0].min()) == 0 and int(cts[0].max()) == 3 * 8 - 1
    assert int(cts[1].min()) == 0 and int(cts[1].max()) == 2


def test_predictive_stores_exclude_audit_fields(tmp_path: Path) -> None:
    batch = _batch(tmp_path, [2, 5])
    data = e2.logical_batch_to_heterodata(batch)
    assert set(data["feature"].keys()) == {"x", "feature_id", "batch", "ptr", "num_nodes"}
    assert set(data["category"].keys()) == {"category_id", "batch", "ptr", "num_nodes"}
    assert set(data["sample"].keys()) == {"y", "batch", "ptr", "num_nodes"}
    assert data.e2_audit["experiment"] == batch.experiment
    assert data.e2_audit["split"] == batch.split
    assert data.e2_audit["ordered_source_sample_indices"] == [2, 5]
    assert data.e2_audit["source_dataset_length"] == batch.source_dataset_length
    assert data.e2_audit["provenance"]["e1_environment_report_sha256"] == e2.APPROVED_E1_REPORT_SHA256


def test_all_returned_numeric_tensors_own_independent_storage(tmp_path: Path) -> None:
    batch = _batch(tmp_path, [1, 6])
    snapshots = {
        field: np.array(getattr(batch, field), copy=True)
        for field in (
            "feature_x", "targets", "feature_batch", "category_batch", "sample_batch",
            "feature_to_category_sources", "feature_to_category_destinations",
            "category_to_sample_sources", "category_to_sample_destinations",
        )
    }
    data = e2.logical_batch_to_heterodata(batch)
    for value in data.to_dict().values():
        if isinstance(value, dict):
            for tensor in value.values():
                if isinstance(tensor, torch.Tensor) and tensor.numel():
                    tensor.reshape(-1)[0] = tensor.reshape(-1)[0] + 1
    for field, expected in snapshots.items():
        assert np.array_equal(getattr(batch, field), expected)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("feature_x", np.zeros((2, 77, 5), dtype=np.float32), "feature_x has shape"),
        ("feature_x", np.zeros((2, 78, 5), dtype=np.float64), "feature_x has dtype"),
        ("targets", np.zeros((2,), dtype=np.int64), "targets has dtype"),
        ("feature_batch", np.zeros((156,), dtype=np.int64), "malformed feature ownership"),
        ("category_batch", np.zeros((16,), dtype=np.int64), "malformed category ownership"),
        ("sample_batch", np.zeros((2,), dtype=np.int64), "malformed sample ownership"),
        ("feature_to_category_sources", np.full((156,), 156, dtype=np.int64), "edge endpoint"),
        ("category_to_sample_destinations", np.full((16,), 2, dtype=np.int64), "edge endpoint"),
    ],
)
def test_malformed_batch_is_rejected(
    tmp_path: Path, field: str, value: np.ndarray, message: str
) -> None:
    batch = _batch(tmp_path, [1, 3])
    value.setflags(write=False)
    with pytest.raises(e2.HeteroDataAdapterError, match=message):
        e2.logical_batch_to_heterodata(replace(batch, **{field: value}))


def test_corrupted_provenance_and_static_identity_are_rejected(tmp_path: Path) -> None:
    batch = _batch(tmp_path, [1])
    corrupt = MappingProxyType({**dict(batch.provenance), "d2_schema_sha256": "0" * 64})
    with pytest.raises(e2.HeteroDataAdapterError, match="provenance"):
        e2.logical_batch_to_heterodata(replace(batch, provenance=corrupt))
    copied_static = replace(batch.static)
    with pytest.raises(e2.HeteroDataAdapterError, match="approved Stage D topology"):
        e2.logical_batch_to_heterodata(replace(batch, static=copied_static))


def test_writable_impostor_and_in_range_topology_tampering_are_rejected(
    tmp_path: Path,
) -> None:
    batch = _batch(tmp_path, [1, 3])
    writable = np.array(batch.feature_x, copy=True)
    with pytest.raises(e2.HeteroDataAdapterError, match="read-only D6"):
        e2.logical_batch_to_heterodata(replace(batch, feature_x=writable))

    destinations = np.array(batch.feature_to_category_destinations, copy=True)
    destinations[0] = (destinations[0] + 1) % 8
    destinations.setflags(write=False)
    with pytest.raises(e2.HeteroDataAdapterError, match="approved Stage D edges"):
        e2.logical_batch_to_heterodata(
            replace(batch, feature_to_category_destinations=destinations)
        )


def test_e3_nonbinary_target_is_rejected(tmp_path: Path) -> None:
    batch = _batch(tmp_path, [1], "E3")
    targets = np.asarray([2], dtype=np.int16)
    targets.setflags(write=False)
    with pytest.raises(e2.HeteroDataAdapterError, match="binary"):
        e2.logical_batch_to_heterodata(replace(batch, targets=targets))


@pytest.mark.parametrize(
    ("experiment", "split"),
    [(experiment, split) for experiment in d4.EXPERIMENT_DIRS for split in d4.SPLIT_STEMS],
)
def test_real_nine_pairs_exact_d6_equivalence(experiment: str, split: str) -> None:
    dataset = d4.LazyGraphDataset(experiment, split)
    indices = [0, len(dataset) // 2, len(dataset) - 1]
    batch = d6.build_logical_batch(dataset, indices)
    data = e2.logical_batch_to_heterodata(batch)
    assert torch.equal(
        data["feature"].x.reshape(3, 78, 5),
        torch.tensor(np.array(batch.feature_x, copy=True)),
    )
    expected_y = torch.tensor(
        np.array(batch.targets, copy=True),
        dtype=torch.float32 if experiment == "E3" else torch.long,
    )
    assert torch.equal(data["sample"].y, expected_y)
    assert torch.equal(
        data[e2.FEATURE_RELATION].edge_index,
        torch.tensor(np.stack((batch.feature_to_category_sources, batch.feature_to_category_destinations))),
    )
    assert torch.equal(
        data[e2.CATEGORY_RELATION].edge_index,
        torch.tensor(np.stack((batch.category_to_sample_sources, batch.category_to_sample_destinations))),
    )
    assert torch.equal(data["feature"].batch, torch.tensor(batch.feature_batch.copy()))
    assert torch.equal(data["category"].batch, torch.tensor(batch.category_batch.copy()))
    assert torch.equal(data["sample"].batch, torch.tensor(batch.sample_batch.copy()))
    assert data.validate()


@pytest.mark.parametrize("split", tuple(d4.SPLIT_STEMS))
def test_real_e1_e3_converted_features_are_identical(split: str) -> None:
    e1_dataset = d4.LazyGraphDataset("E1", split)
    e3_dataset = d4.LazyGraphDataset("E3", split)
    indices = [0, len(e1_dataset) // 2, len(e1_dataset) - 1]
    e1_data = e2.logical_batch_to_heterodata(d6.build_logical_batch(e1_dataset, indices))
    e3_data = e2.logical_batch_to_heterodata(d6.build_logical_batch(e3_dataset, indices))
    assert torch.equal(e1_data["feature"].x, e3_data["feature"].x)


@pytest.mark.parametrize("split", tuple(d4.SPLIT_STEMS))
def test_real_e2_conversion_is_independent(split: str) -> None:
    e1_dataset = d4.LazyGraphDataset("E1", split)
    e2_dataset = d4.LazyGraphDataset("E2", split)
    indices = [0, len(e2_dataset) // 2, len(e2_dataset) - 1]
    data = e2.logical_batch_to_heterodata(d6.build_logical_batch(e2_dataset, indices))
    assert data.e2_audit["experiment"] == "E2"
    assert e2_dataset.x_path.parent.name == "exp2"
    assert e2_dataset.x_path != e1_dataset.x_path


@pytest.mark.skipif(not torch.cuda.is_available(), reason="E.1 requires CUDA")
def test_bounded_heterodata_cuda_transfer() -> None:
    dataset = d4.LazyGraphDataset("E1", "train")
    cpu = e2.logical_batch_to_heterodata(d6.build_logical_batch(dataset, [0]))
    cuda = cpu.to("cuda")
    for node_type in e2.NODE_TYPES:
        for key, value in cuda[node_type].items():
            if isinstance(value, torch.Tensor):
                assert value.is_cuda, (node_type, key)
    for edge_type in e2.EDGE_TYPES:
        assert cuda[edge_type].edge_index.is_cuda
    torch.cuda.synchronize()


def test_production_adapter_has_zero_learned_parameters_or_model_layers() -> None:
    source_path = ROOT / "src" / "graph" / "pyg_adapter.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    forbidden_names = {"Module", "Parameter", "Embedding", "Linear", "Conv1d", "GATv2Conv"}
    assert not any(
        isinstance(node, (ast.ClassDef, ast.Call))
        and (
            (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in forbidden_names)
            or (isinstance(node, ast.ClassDef) and any(isinstance(base, ast.Name) and base.id == "Module" for base in node.bases))
        )
        for node in ast.walk(tree)
    )


def test_saved_e2_report_is_canonical_complete_and_passing() -> None:
    report = json.loads(E2_REPORT.read_text(encoding="utf-8"))
    supplied_hash = report.pop("report_sha256")
    assert e2.report_bytes({**report, "report_sha256": supplied_hash}) == E2_REPORT.read_bytes()
    assert d6.d2.canonical_sha256(report) == supplied_hash
    assert len(report["real_nine_pair_batches"]) == 9
    assert all(item["d6_equivalence"] for item in report["real_nine_pair_batches"])
    assert all(item["heterodata_validation"] for item in report["real_nine_pair_batches"])
    assert report["learned_parameter_count"] == 0
    assert report["overall_status"] == "PASS"
