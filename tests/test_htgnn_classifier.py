"""Stage E.5 complete-model contract, leakage, device, and integration tests."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch_geometric.data import HeteroData

from src.graph import adapter as d4
from src.graph import batching as d6
from src.graph import hierarchical_encoder as e4
from src.graph import htgnn_classifier as e5
from src.graph import pyg_adapter as e2
from src.graph import schema as d2


ROOT = Path(__file__).resolve().parents[1]
E5_REPORT = ROOT / "results" / "stage_e5" / "htgnn_classifier_validation.json"
E5_REPORT_SHA = E5_REPORT.with_suffix(".sha256")
E5_DOC = ROOT / "docs" / "stage_e5_htgnn_classifier.md"
HANDOFF = ROOT / "PROJECT_HANDOFF.md"
CUDA_ISOLATION_RTOL = 1e-5
CUDA_ISOLATION_ATOL = 1e-6
ISOLATION_MIN_PERTURBED_ABS_CHANGE = 1e-3
ISOLATION_ADVERSARIAL_SHIFT = 1e-3


def _synthetic_data(batch_size: int, *, device: str = "cpu") -> HeteroData:
    data = HeteroData()
    data["feature"].x = torch.randn(batch_size * 78, 5, device=device)
    data["feature"].feature_id = torch.arange(
        78, dtype=torch.long, device=device
    ).repeat(batch_size)
    data["feature"].batch = torch.arange(
        batch_size, dtype=torch.long, device=device
    ).repeat_interleave(78)
    data["feature"].ptr = torch.arange(
        0, (batch_size + 1) * 78, 78, dtype=torch.long, device=device
    )
    data["feature"].num_nodes = batch_size * 78
    data["category"].category_id = torch.arange(
        8, dtype=torch.long, device=device
    ).repeat(batch_size)
    data["category"].batch = torch.arange(
        batch_size, dtype=torch.long, device=device
    ).repeat_interleave(8)
    data["category"].ptr = torch.arange(
        0, (batch_size + 1) * 8, 8, dtype=torch.long, device=device
    )
    data["category"].num_nodes = batch_size * 8
    data["sample"].batch = torch.arange(
        batch_size, dtype=torch.long, device=device
    )
    data["sample"].ptr = torch.arange(
        batch_size + 1, dtype=torch.long, device=device
    )
    data["sample"].num_nodes = batch_size
    data["sample"].y = torch.arange(batch_size, device=device)
    data["sample"].source_sample_indices = torch.arange(
        100, 100 + batch_size, dtype=torch.long, device=device
    )

    graph_ids = torch.arange(batch_size, dtype=torch.long, device=device)
    feature_sources = torch.arange(
        batch_size * 78, dtype=torch.long, device=device
    )
    feature_destinations = (
        torch.tensor(
            d4.STATIC_GRAPH_METADATA.feature_to_category_destinations,
            dtype=torch.long,
            device=device,
        ).repeat(batch_size)
        + graph_ids.repeat_interleave(78) * 8
    )
    category_sources = torch.arange(
        batch_size * 8, dtype=torch.long, device=device
    )
    category_destinations = graph_ids.repeat_interleave(8)
    data[e2.FEATURE_RELATION].edge_index = torch.stack(
        (feature_sources, feature_destinations)
    )
    data[e2.CATEGORY_RELATION].edge_index = torch.stack(
        (category_sources, category_destinations)
    )
    data.e2_audit = {
        "experiment": "synthetic",
        "split": "test",
        "ordered_source_sample_indices": list(range(batch_size)),
        "paths": ["not-a-model-input"],
        "provenance": {"hash": "synthetic"},
    }
    return data


class _MalformedFifteenOutputHead(nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values[:, :15]


@pytest.mark.parametrize(
    ("task", "out_features", "parameter_count"),
    [("E1", 15, 7183), ("E2", 15, 7183), ("E3", 1, 6721)],
)
def test_exact_composition_head_and_parameter_contract(
    task: str, out_features: int, parameter_count: int
) -> None:
    model = e5.HTGNNClassifier(task)
    assert model.task == task
    assert isinstance(model.head, nn.Linear)
    assert model.head.in_features == 32
    assert model.head.out_features == out_features
    assert sum(parameter.numel() for parameter in model.feature_encoder.parameters()) == 1920
    assert sum(parameter.numel() for parameter in model.graph_encoder.parameters()) == 4768
    assert sum(parameter.numel() for parameter in model.head.parameters()) == (
        495 if task in {"E1", "E2"} else 33
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == parameter_count
    with pytest.raises(AttributeError):
        model.task = "E3"  # type: ignore[misc]
    assert model.head.out_features == out_features


@pytest.mark.parametrize("task", ["", "e1", "E4", "train", 1, None])
def test_invalid_task_identity_is_rejected(task: object) -> None:
    expected = TypeError if not isinstance(task, str) else ValueError
    with pytest.raises(expected):
        e5.HTGNNClassifier(task)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("task", "replacement"),
    [
        ("E1", nn.Linear(32, 1)),
        ("E3", nn.Linear(32, 15)),
        ("E1", nn.Linear(16, 15)),
        ("E1", _MalformedFifteenOutputHead()),
    ],
)
def test_normal_head_replacement_is_rejected(
    task: str, replacement: nn.Module
) -> None:
    model = e5.HTGNNClassifier(task)
    original = model.head
    with pytest.raises(e5.TaskHeadContractError, match="replacement is forbidden"):
        model.head = replacement
    assert model.head is original


@pytest.mark.parametrize("task", ["E1", "E2", "E3"])
def test_task_identity_comes_from_concrete_type_and_ignores_instance_injection(
    task: str,
) -> None:
    model = e5.HTGNNClassifier(task)
    with pytest.raises(AttributeError, match="construction-time task contract"):
        model.task = "E3"  # type: ignore[misc]
    with pytest.raises(AttributeError, match="construction-time task contract"):
        model._task = "E3"  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        object.__setattr__(model, "task", "E3")
    object.__setattr__(model, "_task", "E3")
    object.__setattr__(model, "_task_id", "E3")
    object.__setattr__(model, "_expected_output_width", 1)
    object.__setattr__(model, "shadow_task", "E3")
    model.__dict__["task"] = "E3"
    assert model.task == task
    assert model.head.out_features == (1 if task == "E3" else 15)


@pytest.mark.parametrize(
    ("task", "replacement", "message"),
    [
        ("E1", nn.Linear(32, 1), "shadowed|out_features"),
        ("E3", nn.Linear(32, 15), "shadowed|out_features"),
        ("E1", nn.Linear(16, 15), "shadowed|in_features"),
        ("E1", _MalformedFifteenOutputHead(), "shadowed|exactly nn.Linear"),
    ],
)
def test_low_level_head_bypass_fails_before_either_encoder(
    task: str, replacement: nn.Module, message: str
) -> None:
    data = _synthetic_data(1)
    model = e5.HTGNNClassifier(task).eval()
    calls: list[str] = []
    handles = [
        model.feature_encoder.register_forward_hook(
            lambda _module, _inputs, _output: calls.append("E3")
        ),
        model.graph_encoder.register_forward_hook(
            lambda _module, _inputs, _output: calls.append("E4")
        ),
    ]
    object.__setattr__(model, "head", replacement)
    with pytest.raises(e5.TaskHeadContractError, match=message):
        model(data)
    for handle in handles:
        handle.remove()
    assert calls == []


def test_registered_module_bypass_is_runtime_validated() -> None:
    data = _synthetic_data(1)
    model = e5.HTGNNClassifier("E1").eval()
    model._modules["head"] = nn.Linear(32, 1)
    with pytest.raises(e5.TaskHeadContractError, match="out_features must be 15"):
        model(data)


@pytest.mark.parametrize("mutation", ["weight_shape", "bias_shape", "missing_bias"])
def test_head_parameter_structure_corruption_fails_before_encoders(
    mutation: str,
) -> None:
    data = _synthetic_data(1)
    model = e5.HTGNNClassifier("E1").eval()
    if mutation == "weight_shape":
        model.head.weight = nn.Parameter(torch.randn(14, 32))
    elif mutation == "bias_shape":
        model.head.bias = nn.Parameter(torch.randn(14))
    else:
        model.head.bias = None
    calls: list[str] = []
    handle = model.feature_encoder.register_forward_pre_hook(
        lambda _module, _inputs: calls.append("E3")
    )
    with pytest.raises(e5.TaskHeadContractError, match="weight|bias"):
        model(data)
    handle.remove()
    assert calls == []


def test_combined_task_and_head_corruption_cannot_turn_e1_into_e3() -> None:
    data = _synthetic_data(3)
    model = e5.HTGNNClassifier("E1").eval()
    object.__setattr__(model, "_task", "E3")
    object.__setattr__(model, "task_id", "E3")
    object.__setattr__(model, "head", nn.Linear(32, 1))
    assert model.task == "E1"
    with pytest.raises(e5.TaskHeadContractError):
        model(data)


@pytest.mark.parametrize(
    ("task", "head_weight_shape", "head_bias_shape"),
    [
        ("E1", (15, 32), (15,)),
        ("E2", (15, 32), (15,)),
        ("E3", (1, 32), (1,)),
    ],
)
def test_state_dict_contract_and_key_structure_are_stable(
    task: str,
    head_weight_shape: tuple[int, ...],
    head_bias_shape: tuple[int, ...],
) -> None:
    first = e5.HTGNNClassifier(task)
    second = e5.HTGNNClassifier(task)
    first_state = first.state_dict()
    second_state = second.state_dict()
    assert tuple(first_state) == tuple(second_state)
    assert any(key.startswith("feature_encoder.") for key in first_state)
    assert any(key.startswith("graph_encoder.") for key in first_state)
    assert tuple(first_state["head.weight"].shape) == head_weight_shape
    assert tuple(first_state["head.bias"].shape) == head_bias_shape


def test_task_contract_protection_does_not_freeze_numerical_parameters() -> None:
    torch.manual_seed(20260921)
    data = _synthetic_data(3)
    model = e5.HTGNNClassifier("E1")
    assert all(parameter.requires_grad for parameter in model.parameters())
    logits = model(data)
    logits.sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    before = model.head.weight.detach().clone()
    with torch.no_grad():
        model.head.weight.add_(0.01 * model.head.weight.grad)
    assert not torch.equal(before, model.head.weight)
    assert model.task == "E1"
    assert sum(parameter.numel() for parameter in model.parameters()) == 7183


@pytest.mark.parametrize(
    ("task", "batch_size", "shape"),
    [
        ("E1", 1, (1, 15)),
        ("E1", 3, (3, 15)),
        ("E2", 1, (1, 15)),
        ("E2", 3, (3, 15)),
        ("E3", 1, (1,)),
        ("E3", 3, (3,)),
    ],
)
def test_forward_shapes_finite_and_eval_deterministic(
    task: str, batch_size: int, shape: tuple[int, ...]
) -> None:
    torch.manual_seed(20260921)
    data = _synthetic_data(batch_size)
    model = e5.HTGNNClassifier(task).eval()
    with torch.no_grad():
        first = model(data)
        second = model(data)
    assert first.shape == shape
    assert torch.isfinite(first).all()
    assert torch.equal(first, second)


def test_classifier_input_is_exactly_final_e4_sample_embedding_only() -> None:
    torch.manual_seed(20260921)
    data = _synthetic_data(3)
    model = e5.HTGNNClassifier("E1").eval()
    captured: dict[str, torch.Tensor] = {}

    graph_hook = model.graph_encoder.register_forward_hook(
        lambda _module, _inputs, output: captured.__setitem__(
            "sample_embeddings", output.detach().clone()
        )
    )
    head_hook = model.head.register_forward_pre_hook(
        lambda _module, inputs: captured.__setitem__(
            "head_input", inputs[0].detach().clone()
        )
    )
    with torch.no_grad():
        logits = model(data)
    graph_hook.remove()
    head_hook.remove()
    assert captured["sample_embeddings"].shape == (3, 32)
    assert torch.equal(captured["head_input"], captured["sample_embeddings"])
    assert logits.shape == (3, 15)


def test_diagnostic_pre_head_path_cannot_bypass_e3_or_e4() -> None:
    data = _synthetic_data(1)
    model = e5.HTGNNClassifier("E1").eval()
    calls: list[str] = []
    handles = [
        model.feature_encoder.register_forward_hook(
            lambda _module, _inputs, _output: calls.append("E3")
        ),
        model.graph_encoder.register_forward_hook(
            lambda _module, _inputs, _output: calls.append("E4")
        ),
    ]
    with torch.no_grad():
        embedding = model._encode_sample_embeddings(data)
    for handle in handles:
        handle.remove()
    assert calls == ["E3", "E4"]
    assert embedding.shape == (1, 32)


def test_target_indices_and_all_audit_metadata_are_not_predictive_inputs() -> None:
    torch.manual_seed(20260921)
    data = _synthetic_data(3)
    model = e5.HTGNNClassifier("E2").eval()
    with torch.no_grad():
        original = model(data)
        data["sample"].y = torch.tensor([-999, 500, 999])
        data["sample"].source_sample_indices = torch.tensor([9, 8, 7])
        data.e2_audit = {
            "experiment": "E3",
            "split": "different",
            "ordered_source_sample_indices": [999, 998, 997],
            "paths": ["changed", "secret"],
            "provenance": {"all": "changed"},
        }
        changed = model(data)
        del data["sample"].y
        del data["sample"].source_sample_indices
        del data.e2_audit
        absent = model(data)
    assert torch.equal(original, changed)
    assert torch.equal(original, absent)


def test_attention_mode_returns_identical_logits_and_exact_e4_payload() -> None:
    torch.manual_seed(20260921)
    data = _synthetic_data(3)
    model = e5.HTGNNClassifier("E3").eval()
    captured: list[object] = []
    hook = model.graph_encoder.register_forward_hook(
        lambda _module, _inputs, output: captured.append(output)
    )
    with torch.no_grad():
        normal = model(data)
        logits, attention = model(data, return_attention=True)
    hook.remove()
    assert torch.equal(normal, logits)
    assert attention is captured[-1][1]  # exact E.4 payload object, not processed
    assert tuple(attention) == (e2.FEATURE_RELATION, e2.CATEGORY_RELATION)
    feature_edges, feature_alpha = attention[e2.FEATURE_RELATION]
    category_edges, category_alpha = attention[e2.CATEGORY_RELATION]
    assert torch.equal(feature_edges, data[e2.FEATURE_RELATION].edge_index)
    assert torch.equal(category_edges, data[e2.CATEGORY_RELATION].edge_index)
    assert feature_alpha.shape == (3 * 78, 4)
    assert category_alpha.shape == (3 * 8, 4)
    assert model.graph_encoder.feature_to_category.add_self_loops is False
    assert model.graph_encoder.category_to_sample.add_self_loops is False


@pytest.mark.parametrize(
    "mutation",
    [
        "feature_semantic_redirect",
        "feature_edge_reorder",
        "cross_graph_category_redirect",
    ],
)
def test_complete_model_rejects_topology_corruption_before_message_passing(
    mutation: str,
) -> None:
    data = _synthetic_data(3)
    if mutation == "feature_semantic_redirect":
        destinations = data[e2.FEATURE_RELATION].edge_index[1]
        destinations[0] = (int(destinations[0]) + 1) % 8
    elif mutation == "feature_edge_reorder":
        edges = data[e2.FEATURE_RELATION].edge_index
        data[e2.FEATURE_RELATION].edge_index = edges.flip(1)
    elif mutation == "cross_graph_category_redirect":
        data[e2.CATEGORY_RELATION].edge_index[1, 0] = 1

    model = e5.HTGNNClassifier("E1").eval()
    calls: list[str] = []
    handles = [
        model.graph_encoder.feature_to_category.register_forward_pre_hook(
            lambda _module, _inputs: calls.append("feature_to_category")
        ),
        model.graph_encoder.category_to_sample.register_forward_pre_hook(
            lambda _module, _inputs: calls.append("category_to_sample")
        ),
    ]
    with pytest.raises(e4.HierarchicalEncoderContractError):
        model(data)
    for handle in handles:
        handle.remove()
    assert calls == []


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_complete_model_multi_graph_logit_isolation(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.manual_seed(20260921)
    original = _synthetic_data(3, device=device)
    changed = original.clone()
    changed["feature"].x[:78] += 1000.0
    model = e5.HTGNNClassifier("E1").to(device).eval()
    with torch.no_grad():
        original_logits = model(original)
        changed_logits = model(changed)
    graph_zero_max_abs_change = torch.max(
        torch.abs(original_logits[0] - changed_logits[0])
    )
    assert float(graph_zero_max_abs_change) > ISOLATION_MIN_PERTURBED_ABS_CHANGE
    if device == "cpu":
        assert torch.equal(original_logits[1:], changed_logits[1:])
    else:
        torch.testing.assert_close(
            original_logits[1:],
            changed_logits[1:],
            rtol=CUDA_ISOLATION_RTOL,
            atol=CUDA_ISOLATION_ATOL,
        )
        with pytest.raises(AssertionError):
            torch.testing.assert_close(
                original_logits[1:],
                changed_logits[1:] + ISOLATION_ADVERSARIAL_SHIFT,
                rtol=CUDA_ISOLATION_RTOL,
                atol=CUDA_ISOLATION_ATOL,
            )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("task", ["E1", "E2", "E3"])
def test_finite_forward_backward_on_supported_devices(task: str, device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.manual_seed(20260921)
    data = _synthetic_data(3, device=device)
    model = e5.HTGNNClassifier(task).to(device)
    logits = model(data)
    logits.sum().backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    assert logits.device.type == device
    assert torch.isfinite(logits).all()
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)
    if device == "cuda":
        torch.cuda.synchronize()


def test_three_task_instances_share_no_parameters_or_automatic_weights() -> None:
    torch.manual_seed(20260921)
    models = {task: e5.HTGNNClassifier(task) for task in ("E1", "E2", "E3")}
    parameter_ids = {
        task: {id(parameter) for parameter in model.parameters()}
        for task, model in models.items()
    }
    assert parameter_ids["E1"].isdisjoint(parameter_ids["E2"])
    assert parameter_ids["E1"].isdisjoint(parameter_ids["E3"])
    assert parameter_ids["E2"].isdisjoint(parameter_ids["E3"])
    assert models["E1"].head.out_features == 15
    assert models["E2"].head.out_features == 15
    assert models["E3"].head.out_features == 1
    assert not torch.equal(
        models["E1"].feature_encoder.fusion.weight,
        models["E2"].feature_encoder.fusion.weight,
    )


def test_forward_leaves_source_heterodata_unmodified() -> None:
    torch.manual_seed(20260921)
    data = _synthetic_data(3)
    before = {
        "x": data["feature"].x.clone(),
        "feature_id": data["feature"].feature_id.clone(),
        "feature_batch": data["feature"].batch.clone(),
        "category_id": data["category"].category_id.clone(),
        "sample_y": data["sample"].y.clone(),
        "sample_indices": data["sample"].source_sample_indices.clone(),
        "feature_edges": data[e2.FEATURE_RELATION].edge_index.clone(),
        "category_edges": data[e2.CATEGORY_RELATION].edge_index.clone(),
        "audit": json.loads(json.dumps(data.e2_audit)),
    }
    e5.HTGNNClassifier("E1").eval()(data, return_attention=True)
    assert torch.equal(data["feature"].x, before["x"])
    assert torch.equal(data["feature"].feature_id, before["feature_id"])
    assert torch.equal(data["feature"].batch, before["feature_batch"])
    assert torch.equal(data["category"].category_id, before["category_id"])
    assert torch.equal(data["sample"].y, before["sample_y"])
    assert torch.equal(data["sample"].source_sample_indices, before["sample_indices"])
    assert torch.equal(data[e2.FEATURE_RELATION].edge_index, before["feature_edges"])
    assert torch.equal(data[e2.CATEGORY_RELATION].edge_index, before["category_edges"])
    assert data.e2_audit == before["audit"]


@pytest.mark.parametrize(
    ("experiment", "split"),
    [(experiment, split) for experiment in d4.EXPERIMENT_DIRS for split in d4.SPLIT_STEMS],
)
def test_real_bounded_complete_model_integration(experiment: str, split: str) -> None:
    torch.manual_seed(20260921)
    dataset = d4.LazyGraphDataset(experiment, split)
    indices = [0, len(dataset) // 2, len(dataset) - 1]
    batch = d6.build_logical_batch(dataset, indices)
    batch_before = np.array(batch.feature_x, copy=True)
    data = e2.logical_batch_to_heterodata(batch)
    x_before = data["feature"].x.clone()
    feature_edges_before = data[e2.FEATURE_RELATION].edge_index.clone()
    category_edges_before = data[e2.CATEGORY_RELATION].edge_index.clone()
    model = e5.HTGNNClassifier(experiment).eval()
    with torch.no_grad():
        logits, attention = model(data, return_attention=True)
    expected_shape = (3,) if experiment == "E3" else (3, 15)
    assert logits.shape == expected_shape
    assert torch.isfinite(logits).all()
    assert torch.equal(
        attention[e2.FEATURE_RELATION][0], feature_edges_before
    )
    assert torch.equal(
        attention[e2.CATEGORY_RELATION][0], category_edges_before
    )
    assert np.array_equal(batch.feature_x, batch_before)
    assert torch.equal(data["feature"].x, x_before)
    assert torch.equal(data[e2.FEATURE_RELATION].edge_index, feature_edges_before)
    assert torch.equal(data[e2.CATEGORY_RELATION].edge_index, category_edges_before)
    assert dataset.is_lazy_read_only


@pytest.mark.parametrize("split", tuple(d4.SPLIT_STEMS))
def test_e1_e3_pre_head_embeddings_match_under_same_encoder_state(split: str) -> None:
    torch.manual_seed(20260921)
    model = e5.HTGNNClassifier("E1").eval()
    e1_dataset = d4.LazyGraphDataset("E1", split)
    e3_dataset = d4.LazyGraphDataset("E3", split)
    indices = [0, len(e1_dataset) // 2, len(e1_dataset) - 1]
    e1_data = e2.logical_batch_to_heterodata(
        d6.build_logical_batch(e1_dataset, indices)
    )
    e3_data = e2.logical_batch_to_heterodata(
        d6.build_logical_batch(e3_dataset, indices)
    )
    with torch.no_grad():
        e1_feature_embeddings = model.feature_encoder(
            e1_data["feature"].x, e1_data["feature"].feature_id
        )
        e3_feature_embeddings = model.feature_encoder(
            e3_data["feature"].x, e3_data["feature"].feature_id
        )
        e1_sample_embeddings = model._encode_sample_embeddings(e1_data)
        e3_sample_embeddings = model._encode_sample_embeddings(e3_data)
    assert torch.equal(e1_feature_embeddings, e3_feature_embeddings)
    assert torch.equal(e1_sample_embeddings, e3_sample_embeddings)


def test_source_excludes_bypasses_activations_and_downstream_training() -> None:
    source = (ROOT / "src" / "graph" / "htgnn_classifier.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    names = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    } | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert not (
        {
            "softmax",
            "sigmoid",
            "CrossEntropyLoss",
            "BCEWithLogitsLoss",
            "AdamW",
            "Optimizer",
            "DataLoader",
        }
        & names
    )
    assert "['sample'].y" not in source and '["sample"].y' not in source
    assert "e2_audit" not in source
    assert "source_sample_indices" not in source
    assert "feature" not in {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert "contract, head = _validated_head(self)" in source
    assert source.count("head(sample_embeddings)") == 1
    assert "self.head.out_features" not in source


def test_reflective_python_tampering_is_documented_outside_supported_contract() -> None:
    """Document the boundary; do not require survival of arbitrary code execution."""

    module_contract = e5.__doc__ or ""
    stage_document = E5_DOC.read_text(encoding="utf-8")
    handoff = HANDOFF.read_text(encoding="utf-8")
    for text in (module_contract, stage_document, handoff):
        assert "supported public model API" in text
        assert "arbitrary" in text and "Python" in text
        assert "__class__" in text
        assert "_modules" in text
        assert "same-process" in text
    assert "not E.5 acceptance requirements" in stage_document
    assert "must not require resilience" in handoff
    assert "does not authorize E.6" in handoff


def test_saved_e5_report_is_canonical_complete_and_passing() -> None:
    report_bytes = E5_REPORT.read_bytes()
    report = json.loads(report_bytes)
    supplied_hash = report.pop("report_sha256")
    assert d2.canonical_sha256(report) == supplied_hash
    assert d2.canonical_json_bytes({**report, "report_sha256": supplied_hash}) == report_bytes
    assert report["overall_status"] == "PASS"
    assert report["parameter_counts"]["E1"]["measured"] == 7183
    assert report["parameter_counts"]["E2"]["measured"] == 7183
    assert report["parameter_counts"]["E3"]["measured"] == 6721
    task_contract = report["construction_time_task_and_head_contract"]
    assert task_contract["status"] == "PASS"
    assert "concrete task-specific model type" in task_contract[
        "authoritative_mechanism"
    ]
    assert all(
        item["assignment_rejected"]
        for item in task_contract["normal_head_replacement"].values()
    )
    assert all(
        item["forward_rejected"] and item["rejected_before_e3_or_e4"]
        for item in task_contract["low_level_head_corruption"].values()
    )
    assert task_contract["combined_e1_to_e3_attack"]["authoritative_task"] == "E1"
    assert task_contract["combined_e1_to_e3_attack"]["forward_rejected"] is True
    boundary = report["integrity_threat_model"]
    assert boundary["status"] == "PASS"
    assert boundary["required_scope"] == (
        "construction-time task/output immutability through supported public API"
    )
    assert boundary["arbitrary_same_process_python_execution_in_scope"] is False
    assert boundary["reflective_tampering_survival_is_acceptance_gate"] is False
    assert "__class__ reassignment" in boundary["outside_supported_api"]
    assert "direct nn.Module._modules rewriting" in boundary[
        "outside_supported_api"
    ]
    assert boundary["isolated_internal_head_detection_is_defense_in_depth"] is True
    assert boundary["stage_document_sha256"] == hashlib.sha256(
        E5_DOC.read_bytes()
    ).hexdigest()
    assert report["complete_model_topology_corruption"]["status"] == "PASS"
    assert report["multi_graph_logit_isolation"]["status"] == "PASS"
    isolation = report["multi_graph_logit_isolation"]
    assert isolation["comparison_dtype"] == "torch.float32"
    assert isolation["graph_zero_minimum_required_max_abs_change"] == 1e-3
    assert isolation["graph_zero_material_response_all"] is True
    assert isolation["untouched_graphs_compared_per_graph"] is True
    assert isolation["cpu"]["bitwise_identical"] is True
    assert isolation["cuda"]["rtol"] == 1e-5
    assert isolation["cuda"]["atol"] == 1e-6
    assert isolation["cuda"]["stress_replay_count"] == 20
    assert isolation["cuda"]["all_stress_replays_passed"] is True
    assert isolation["cuda"]["adversarial_shift"] == 1e-3
    assert isolation["cuda"]["adversarial_shift_rejected_all"] is True
    assert "raw_cuda_max_difference" not in isolation["cuda"]
    assert report["numerical_parameter_trainability"]["status"] == "PASS"
    assert report["readout_path"]["sample_embedding_only"] is True
    assert report["target_and_metadata_independence"] is True
    assert report["attention_mode_equivalence"] is True
    assert report["cpu_forward_backward"]["status"] == "PASS"
    assert report["cuda_forward_backward"]["status"] == "PASS"
    assert len(report["real_nine_pair_batches"]) == 9
    source_bytes = (ROOT / report["implementation"]["source_path"]).read_bytes()
    assert hashlib.sha256(source_bytes).hexdigest() == report["implementation"]["source_sha256"]
    physical_hash = hashlib.sha256(report_bytes).hexdigest()
    assert E5_REPORT_SHA.read_text(encoding="ascii") == (
        f"{physical_hash}  htgnn_classifier_validation.json\n"
    )
