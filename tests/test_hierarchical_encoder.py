from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATv2Conv

from src.graph import adapter as d4
from src.graph import hierarchical_encoder as e4
from src.graph import pyg_adapter as e2
from src.graph import schema as d2


ROOT = Path(__file__).resolve().parents[1]
E4_REPORT = ROOT / "results" / "stage_e4" / "hierarchical_encoder_validation.json"
E4_REPORT_SHA = E4_REPORT.with_suffix(".sha256")


def _synthetic_data(batch_size: int, *, device: str = "cpu") -> HeteroData:
    data = HeteroData()
    data["feature"].num_nodes = batch_size * 78
    data["category"].category_id = torch.arange(8, device=device).repeat(batch_size)
    data["category"].num_nodes = batch_size * 8
    data["sample"].num_nodes = batch_size
    data["sample"].y = torch.arange(batch_size, device=device)
    feature_offsets = torch.arange(batch_size, device=device).repeat_interleave(78)
    category_offsets = torch.arange(batch_size, device=device).repeat_interleave(8)
    feature_sources = torch.arange(batch_size * 78, device=device)
    feature_destinations = (
        torch.tensor(
            d4.STATIC_GRAPH_METADATA.feature_to_category_destinations,
            dtype=torch.long,
            device=device,
        ).repeat(batch_size)
        + feature_offsets * 8
    )
    category_sources = torch.arange(batch_size * 8, device=device)
    category_destinations = category_offsets
    data[e2.FEATURE_RELATION].edge_index = torch.stack(
        (feature_sources, feature_destinations)
    )
    data[e2.CATEGORY_RELATION].edge_index = torch.stack(
        (category_sources, category_destinations)
    )
    data.e2_audit = {"experiment": "synthetic", "split": "test"}
    return data


def _assert_destination_normalization(
    edge_index: torch.Tensor, alpha: torch.Tensor, destinations: int
) -> None:
    assert alpha.shape[1] == 4
    assert torch.isfinite(alpha).all()
    for destination in range(destinations):
        incoming = alpha[edge_index[1] == destination]
        assert incoming.shape[0] > 0
        assert torch.allclose(
            incoming.sum(dim=0),
            torch.ones(4, device=alpha.device),
            atol=1e-6,
            rtol=1e-6,
        )


def _destination_head_sums(
    edge_index: torch.Tensor, alpha: torch.Tensor, destinations: int
) -> torch.Tensor:
    return torch.stack(
        [
            alpha[edge_index[1] == destination].sum(dim=0)
            for destination in range(destinations)
        ]
    )


def test_exact_initialization_modules_and_parameter_count() -> None:
    model = e4.HierarchicalGraphEncoder()
    assert tuple(model.category_embedding.weight.shape) == (8, 32)
    assert tuple(model.sample_token.shape) == (32,)
    assert [name for name, _ in model.named_parameters() if name == "sample_token"] == [
        "sample_token"
    ]
    assert not torch.equal(
        model.category_embedding.weight[0], model.category_embedding.weight[1]
    )
    assert model.feature_to_category is not model.category_to_sample
    first_parameters = dict(model.feature_to_category.named_parameters())
    second_parameters = dict(model.category_to_sample.named_parameters())
    assert first_parameters.keys() == second_parameters.keys()
    assert all(
        first_parameters[name] is not second_parameters[name]
        for name in first_parameters
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == 4768
    assert sum(
        parameter.numel() for parameter in model.feature_to_category.parameters()
    ) == 2176
    assert sum(
        parameter.numel() for parameter in model.category_to_sample.parameters()
    ) == 2176


def test_topology_validation_reuses_frozen_schema_authority() -> None:
    assert (
        e4.frozen_schema.FEATURE_TO_CATEGORY_SOURCE_INDICES
        is d2.FEATURE_TO_CATEGORY_SOURCE_INDICES
    )
    assert (
        e4.frozen_schema.FEATURE_TO_CATEGORY_DESTINATION_INDICES
        is d2.FEATURE_TO_CATEGORY_DESTINATION_INDICES
    )
    assert (
        e4.frozen_schema.CATEGORY_TO_SAMPLE_SOURCE_INDICES
        is d2.CATEGORY_TO_SAMPLE_SOURCE_INDICES
    )
    assert (
        e4.frozen_schema.CATEGORY_TO_SAMPLE_DESTINATION_INDICES
        is d2.CATEGORY_TO_SAMPLE_DESTINATION_INDICES
    )


@pytest.mark.parametrize("name", ["feature_to_category", "category_to_sample"])
def test_exact_gatv2_configuration_without_internal_residual_or_loops(name: str) -> None:
    layer = getattr(e4.HierarchicalGraphEncoder(), name)
    assert isinstance(layer, GATv2Conv)
    assert layer.in_channels == (32, 32)
    assert layer.out_channels == 8
    assert layer.heads == 4
    assert layer.concat is True
    assert layer.dropout == 0.10
    assert layer.add_self_loops is False
    assert layer.share_weights is False
    assert layer.edge_dim is None
    assert layer.res is None
    assert layer.lin_l is not layer.lin_r


@pytest.mark.parametrize("batch_size", [1, 3])
def test_forward_shapes_and_attention_contract(batch_size: int) -> None:
    torch.manual_seed(20260921)
    data = _synthetic_data(batch_size)
    feature_embeddings = torch.randn(batch_size * 78, 32)
    model = e4.HierarchicalGraphEncoder().eval()
    category_shapes: list[tuple[int, ...]] = []
    hook = model.feature_to_category_activation.register_forward_hook(
        lambda _module, _inputs, output: category_shapes.append(tuple(output.shape))
    )
    with torch.no_grad():
        normal = model(data, feature_embeddings)
        sample_embeddings, attention = model(
            data, feature_embeddings, return_attention=True
        )
    hook.remove()
    assert normal.shape == (batch_size, 32)
    assert sample_embeddings.shape == (batch_size, 32)
    assert torch.equal(normal, sample_embeddings)
    assert category_shapes == [(batch_size * 8, 32), (batch_size * 8, 32)]
    assert tuple(attention) == (e2.FEATURE_RELATION, e2.CATEGORY_RELATION)
    feature_edges, feature_alpha = attention[e2.FEATURE_RELATION]
    category_edges, category_alpha = attention[e2.CATEGORY_RELATION]
    assert torch.equal(feature_edges, data[e2.FEATURE_RELATION].edge_index)
    assert torch.equal(category_edges, data[e2.CATEGORY_RELATION].edge_index)
    assert feature_alpha.shape == (batch_size * 78, 4)
    assert category_alpha.shape == (batch_size * 8, 4)
    assert torch.isfinite(normal).all()
    assert torch.isfinite(feature_alpha).all()
    assert torch.isfinite(category_alpha).all()


def test_attention_normalizes_per_category_neighborhood_and_per_graph_sample() -> None:
    torch.manual_seed(20260921)
    batch_size = 2
    data = _synthetic_data(batch_size)
    model = e4.HierarchicalGraphEncoder().eval()
    with torch.no_grad():
        _, attention = model(
            data, torch.randn(batch_size * 78, 32), return_attention=True
        )
    feature_edges, feature_alpha = attention[e2.FEATURE_RELATION]
    category_edges, category_alpha = attention[e2.CATEGORY_RELATION]
    _assert_destination_normalization(feature_edges, feature_alpha, batch_size * 8)
    _assert_destination_normalization(category_edges, category_alpha, batch_size)
    assert torch.allclose(
        feature_alpha.sum(dim=0),
        torch.full((4,), float(batch_size * 8)),
        atol=1e-5,
        rtol=1e-5,
    )
    assert not torch.allclose(feature_alpha.sum(dim=0), torch.ones(4))
    assert torch.allclose(
        category_alpha.sum(dim=0),
        torch.full((4,), float(batch_size)),
        atol=1e-6,
        rtol=1e-6,
    )


def test_attention_eval_and_train_dropout_semantics_for_installed_pyg() -> None:
    torch.manual_seed(20260921)
    data = _synthetic_data(2)
    values = torch.randn(156, 32)
    model = e4.HierarchicalGraphEncoder()
    model.eval()
    with torch.no_grad():
        _, eval_attention = model(data, values, return_attention=True)

    model.train()
    torch.manual_seed(20260921)
    with torch.no_grad():
        _, train_attention = model(data, values, return_attention=True)

    for relation, destination_count in (
        (e2.FEATURE_RELATION, 16),
        (e2.CATEGORY_RELATION, 2),
    ):
        eval_edges, eval_alpha = eval_attention[relation]
        train_edges, train_alpha = train_attention[relation]
        assert torch.equal(eval_edges, train_edges)
        assert torch.allclose(
            _destination_head_sums(eval_edges, eval_alpha, destination_count),
            torch.ones(destination_count, 4),
            atol=1e-6,
            rtol=1e-6,
        )
        train_sums = _destination_head_sums(
            train_edges, train_alpha, destination_count
        )
        assert torch.any(train_alpha == 0)
        assert not torch.allclose(
            train_sums,
            torch.ones(destination_count, 4),
            atol=1e-6,
            rtol=1e-6,
        )

    eval_feature_alpha = eval_attention[e2.FEATURE_RELATION][1]
    train_feature_alpha = train_attention[e2.FEATURE_RELATION][1]
    retained = train_feature_alpha != 0
    assert torch.allclose(
        train_feature_alpha[retained],
        eval_feature_alpha[retained] / (1.0 - e4.DROPOUT_PROBABILITY),
        atol=1e-6,
        rtol=1e-5,
    )


def test_graph_block_operation_order_and_explicit_target_residuals() -> None:
    torch.manual_seed(20260921)
    data = _synthetic_data(2)
    values = torch.randn(156, 32)
    model = e4.HierarchicalGraphEncoder().eval()
    events: list[str] = []
    captured: dict[str, torch.Tensor] = {}

    def record(name: str):
        def hook(_module, inputs, output):
            events.append(name)
            if isinstance(inputs[0], torch.Tensor):
                captured[f"{name}_input"] = inputs[0].detach().clone()
            captured[f"{name}_output"] = output.detach().clone()

        return hook

    handles = [
        model.feature_to_category.register_forward_hook(record("ftc_gat")),
        model.feature_to_category_dropout.register_forward_hook(record("ftc_dropout")),
        model.feature_to_category_norm.register_forward_hook(record("ftc_norm")),
        model.feature_to_category_activation.register_forward_hook(record("ftc_gelu")),
        model.category_to_sample.register_forward_hook(record("cts_gat")),
        model.category_to_sample_dropout.register_forward_hook(record("cts_dropout")),
        model.category_to_sample_norm.register_forward_hook(record("cts_norm")),
        model.category_to_sample_activation.register_forward_hook(record("cts_gelu")),
    ]
    with torch.no_grad():
        model(data, values)
    for handle in handles:
        handle.remove()
    assert events == [
        "ftc_gat",
        "ftc_dropout",
        "ftc_norm",
        "ftc_gelu",
        "cts_gat",
        "cts_dropout",
        "cts_norm",
        "cts_gelu",
    ]
    category_targets = model.category_embedding(data["category"].category_id)
    sample_targets = model.sample_token.unsqueeze(0).expand(2, -1)
    assert torch.equal(captured["ftc_dropout_input"], captured["ftc_gat_output"])
    assert torch.equal(
        captured["ftc_norm_input"], captured["ftc_dropout_output"] + category_targets
    )
    assert torch.equal(captured["ftc_gelu_input"], captured["ftc_norm_output"])
    assert torch.equal(captured["cts_dropout_input"], captured["cts_gat_output"])
    assert torch.equal(
        captured["cts_norm_input"], captured["cts_dropout_output"] + sample_targets
    )
    assert torch.equal(captured["cts_gelu_input"], captured["cts_norm_output"])


def test_eval_is_deterministic_and_batches_are_directly_isolated() -> None:
    torch.manual_seed(20260921)
    data = _synthetic_data(3)
    values = torch.randn(234, 32)
    changed = values.clone()
    changed[:78] += 1000.0
    model = e4.HierarchicalGraphEncoder().eval()
    with torch.no_grad():
        first = model(data, values)
        repeated = model(data, values)
        perturbed = model(data, changed)
    assert torch.equal(first, repeated)
    assert not torch.equal(first[0], perturbed[0])
    assert torch.equal(first[1:], perturbed[1:])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_finite_forward_backward_on_supported_devices(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.manual_seed(20260921)
    data = _synthetic_data(2, device=device)
    values = torch.randn(156, 32, device=device, requires_grad=True)
    model = e4.HierarchicalGraphEncoder().to(device)
    output, attention = model(data, values, return_attention=True)
    output.square().mean().backward()
    gradients = [values.grad] + [parameter.grad for parameter in model.parameters()]
    assert output.shape == (2, 32)
    assert torch.isfinite(output).all()
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)
    assert attention[e2.FEATURE_RELATION][1].shape == (156, 4)
    assert attention[e2.CATEGORY_RELATION][1].shape == (16, 4)


def test_forward_does_not_mutate_heterodata_or_feature_embeddings() -> None:
    data = _synthetic_data(2)
    values = torch.randn(156, 32)
    before = {
        "values": values.clone(),
        "category_id": data["category"].category_id.clone(),
        "feature_edges": data[e2.FEATURE_RELATION].edge_index.clone(),
        "category_edges": data[e2.CATEGORY_RELATION].edge_index.clone(),
        "target": data["sample"].y.clone(),
        "audit": dict(data.e2_audit),
    }
    e4.HierarchicalGraphEncoder().eval()(data, values, return_attention=True)
    assert torch.equal(values, before["values"])
    assert torch.equal(data["category"].category_id, before["category_id"])
    assert torch.equal(data[e2.FEATURE_RELATION].edge_index, before["feature_edges"])
    assert torch.equal(data[e2.CATEGORY_RELATION].edge_index, before["category_edges"])
    assert torch.equal(data["sample"].y, before["target"])
    assert data.e2_audit == before["audit"]


def test_targets_and_audit_metadata_cannot_change_sample_embeddings() -> None:
    torch.manual_seed(20260921)
    data = _synthetic_data(2)
    values = torch.randn(156, 32)
    model = e4.HierarchicalGraphEncoder().eval()
    with torch.no_grad():
        original = model(data, values)
        data["sample"].y = torch.tensor([999, -999])
        data.e2_audit = {
            "experiment": "E3",
            "split": "different",
            "paths": ["forbidden"],
            "provenance": {"hash": "different"},
        }
        changed_metadata = model(data, values)
        del data["sample"].y
        del data.e2_audit
        absent_metadata = model(data, values)
    assert torch.equal(original, changed_metadata)
    assert torch.equal(original, absent_metadata)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("extra_feature_edge", "shape"),
        ("cross_feature_edge", "frozen graph-major E.2 topology"),
        ("cross_category_edge", "frozen graph-major E.2 topology"),
        ("wrong_edge_dtype", "dtype"),
        ("wrong_category_ids", "0..7"),
        ("wrong_feature_count", "multiple of 78"),
    ],
)
def test_malformed_topology_or_embeddings_are_rejected(
    mutation: str, message: str
) -> None:
    data = _synthetic_data(2)
    values = torch.randn(156, 32)
    if mutation == "extra_feature_edge":
        edges = data[e2.FEATURE_RELATION].edge_index
        data[e2.FEATURE_RELATION].edge_index = torch.cat((edges, edges[:, :1]), dim=1)
    elif mutation == "cross_feature_edge":
        data[e2.FEATURE_RELATION].edge_index[1, 0] = 8
    elif mutation == "cross_category_edge":
        data[e2.CATEGORY_RELATION].edge_index[1, 0] = 1
    elif mutation == "wrong_edge_dtype":
        data[e2.FEATURE_RELATION].edge_index = data[
            e2.FEATURE_RELATION
        ].edge_index.to(torch.int32)
    elif mutation == "wrong_category_ids":
        data["category"].category_id[0] = 1
    elif mutation == "wrong_feature_count":
        values = values[:-1]
    with pytest.raises(e4.HierarchicalEncoderContractError, match=message):
        e4.HierarchicalGraphEncoder()(data, values)


def test_same_graph_feature_destination_redirect_is_rejected() -> None:
    data = _synthetic_data(2)
    edges = data[e2.FEATURE_RELATION].edge_index
    original = int(edges[1, 0])
    edges[1, 0] = (original + 1) % 8
    assert edges.shape == (2, 156)
    assert edges.dtype == torch.long
    assert int(edges[1, 0]) // 8 == int(edges[0, 0]) // 78
    with pytest.raises(
        e4.HierarchicalEncoderContractError,
        match="feature-to-category.*frozen graph-major E.2 topology",
    ):
        e4.HierarchicalGraphEncoder()(
            data, torch.randn(156, 32)
        )


def test_feature_destination_swap_preserving_counts_is_rejected() -> None:
    data = _synthetic_data(1)
    destinations = data[e2.FEATURE_RELATION].edge_index[1]
    left = 0
    right = next(
        index
        for index in range(1, destinations.numel())
        if destinations[index] != destinations[left]
    )
    counts_before = torch.bincount(destinations, minlength=8)
    swapped = destinations[[right, left]].clone()
    destinations[[left, right]] = swapped
    assert torch.equal(torch.bincount(destinations, minlength=8), counts_before)
    with pytest.raises(
        e4.HierarchicalEncoderContractError,
        match="feature-to-category.*frozen graph-major E.2 topology",
    ):
        e4.HierarchicalGraphEncoder()(data, torch.randn(78, 32))


def test_category_to_sample_valid_destination_redirect_is_rejected() -> None:
    data = _synthetic_data(2)
    edges = data[e2.CATEGORY_RELATION].edge_index
    edges[1, 0] = 1
    assert 0 <= int(edges[1, 0]) < 2
    assert edges.shape == (2, 16)
    with pytest.raises(
        e4.HierarchicalEncoderContractError,
        match="category-to-sample.*frozen graph-major E.2 topology",
    ):
        e4.HierarchicalGraphEncoder()(data, torch.randn(156, 32))


def test_reordered_otherwise_valid_feature_edges_are_rejected() -> None:
    data = _synthetic_data(1)
    edges = data[e2.FEATURE_RELATION].edge_index
    data[e2.FEATURE_RELATION].edge_index = edges[:, torch.arange(77, -1, -1)]
    assert torch.equal(
        torch.sort(data[e2.FEATURE_RELATION].edge_index[0]).values,
        torch.arange(78),
    )
    with pytest.raises(
        e4.HierarchicalEncoderContractError,
        match="feature-to-category.*frozen graph-major E.2 topology",
    ):
        e4.HierarchicalGraphEncoder()(data, torch.randn(78, 32))


def test_untouched_frozen_e2_topology_passes_exact_equality_validation() -> None:
    data = _synthetic_data(3)
    output = e4.HierarchicalGraphEncoder().eval()(data, torch.randn(234, 32))
    assert output.shape == (3, 32)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    "mutation",
    [
        "same_graph_redirect",
        "destination_swap",
        "category_redirect",
        "edge_reorder",
    ],
)
def test_topology_rejection_occurs_before_either_gatv2_module(
    mutation: str,
) -> None:
    data = _synthetic_data(2)
    if mutation == "same_graph_redirect":
        data[e2.FEATURE_RELATION].edge_index[1, 0] = 1
    elif mutation == "destination_swap":
        destinations = data[e2.FEATURE_RELATION].edge_index[1]
        right = next(
            index
            for index in range(1, destinations.numel())
            if destinations[index] != destinations[0]
        )
        destinations[[0, right]] = destinations[[right, 0]].clone()
    elif mutation == "category_redirect":
        data[e2.CATEGORY_RELATION].edge_index[1, 0] = 1
    elif mutation == "edge_reorder":
        edges = data[e2.FEATURE_RELATION].edge_index
        data[e2.FEATURE_RELATION].edge_index = edges.flip(1)

    model = e4.HierarchicalGraphEncoder()
    calls: list[str] = []
    handles = [
        model.feature_to_category.register_forward_hook(
            lambda _module, _inputs, _output: calls.append("feature_to_category")
        ),
        model.category_to_sample.register_forward_hook(
            lambda _module, _inputs, _output: calls.append("category_to_sample")
        ),
    ]
    with pytest.raises(e4.HierarchicalEncoderContractError):
        model(data, torch.randn(156, 32))
    for handle in handles:
        handle.remove()
    assert calls == []


def test_source_has_no_feature_recomputation_raw_bypass_or_downstream_components() -> None:
    source = (ROOT / "src" / "graph" / "hierarchical_encoder.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    names = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    } | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert "FeatureNodeEncoder" not in names
    assert "CausalTemporalEncoder" not in names
    assert not (
        {
            "Conv1d",
            "Linear",
            "CrossEntropyLoss",
            "BCEWithLogitsLoss",
            "Optimizer",
        }
        & names
    )
    assert "sample'].y" not in source
    assert 'sample\"].y' not in source
    assert "e2_audit" not in source
    assert "src.graph.adapter" not in source
    assert "src.graph.batching" not in source
    assert '["feature"].x' not in source
    assert '["sample"].x' not in source


def test_saved_e4_report_is_canonical_complete_and_passing() -> None:
    report_bytes = E4_REPORT.read_bytes()
    report = json.loads(report_bytes)
    supplied_hash = report.pop("report_sha256")
    assert d2.canonical_sha256(report) == supplied_hash
    assert (
        d2.canonical_json_bytes({**report, "report_sha256": supplied_hash})
        == report_bytes
    )
    assert report["overall_status"] == "PASS"
    assert report["parameter_count"]["measured"] == 4768
    assert report["parameter_count"]["difference"] == 0
    assert report["parameter_count"]["combined_e3_e4_measured"] == 6688
    assert report["multi_graph_direct_isolation"] is True
    assert report["target_and_metadata_independence"] is True
    topology = report["exact_e2_topology_validation"]
    assert topology["untouched_frozen_e2_topology_accepted"] is True
    assert topology["all_adversarial_cases_rejected"] is True
    assert topology["all_rejected_before_message_passing"] is True
    assert set(topology["cases"]) == {
        "same_graph_feature_redirect",
        "feature_destination_swap",
        "category_to_sample_redirect",
        "edge_reorder",
    }
    attention_semantics = report["attention_semantics"][
        "pyg_2_8_0_post1_eval_train_behavior"
    ]
    assert attention_semantics["installed_pyg_version"] == "2.8.0.post1"
    assert attention_semantics[
        "returned_coefficients_after_internal_attention_dropout"
    ] is True
    assert attention_semantics["train_values_are_post_dropout_scaled"] is True
    assert report["attention_semantics"][
        "future_pruning_attention_collection_mode"
    ] == "eval"
    assert len(report["real_nine_pair_batches"]) == 9
    source_bytes = (ROOT / report["implementation"]["source_path"]).read_bytes()
    assert hashlib.sha256(source_bytes).hexdigest() == report["implementation"][
        "source_sha256"
    ]
    physical_hash = hashlib.sha256(report_bytes).hexdigest()
    assert E4_REPORT_SHA.read_text(encoding="ascii") == (
        f"{physical_hash}  hierarchical_encoder_validation.json\n"
    )
