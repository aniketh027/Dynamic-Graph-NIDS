"""Stage E.3 causal, adversarial, device, and real-integration tests."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.graph import adapter as d4
from src.graph import batching as d6
from src.graph import feature_encoder as e3
from src.graph import pyg_adapter as e2
from src.graph import schema as d2


ROOT = Path(__file__).resolve().parents[1]
E3_REPORT = ROOT / "results" / "stage_e3" / "feature_encoder_validation.json"
E3_REPORT_SHA = ROOT / "results" / "stage_e3" / "feature_encoder_validation.sha256"


def _set_asymmetric_positive_weights(module: e3.CausalTemporalEncoder) -> None:
    with torch.no_grad():
        module.conv1.weight.copy_(
            torch.tensor([1.0, 2.0, 4.0]).reshape(1, 1, 3).repeat(8, 1, 1)
        )
        module.conv1.bias.zero_()
        module.conv2.weight.copy_(
            torch.tensor([1.0, 2.0, 4.0]).reshape(1, 1, 3).repeat(16, 8, 1)
        )
        module.conv2.bias.zero_()


def test_temporal_architecture_uses_explicit_left_padding_and_retains_length() -> None:
    encoder = e3.CausalTemporalEncoder()
    values = torch.randn(7, 5)
    sequence = encoder.forward_sequence(values)
    assert encoder.conv1.in_channels == 1
    assert encoder.conv1.out_channels == 8
    assert encoder.conv1.kernel_size == (3,)
    assert encoder.conv1.padding == (0,)
    assert encoder.conv2.in_channels == 8
    assert encoder.conv2.out_channels == 16
    assert encoder.conv2.kernel_size == (3,)
    assert encoder.conv2.padding == (0,)
    assert e3.LEFT_PADDING == 2
    assert sequence.shape == (7, 16, 5)
    assert encoder(values).shape == (7, 16)
    assert torch.equal(encoder(values), sequence[:, :, 4])


def test_prefix_invariance_holds_at_every_temporal_position() -> None:
    torch.manual_seed(20260921)
    encoder = e3.CausalTemporalEncoder().eval()
    original = torch.randn(3, 5)
    for temporal_position in range(5):
        changed = original.clone()
        changed[:, temporal_position + 1 :] += 1000.0
        left = encoder.forward_sequence(original)[:, :, temporal_position]
        right = encoder.forward_sequence(changed)[:, :, temporal_position]
        assert torch.equal(left, right), temporal_position


def test_zero_gradient_from_each_output_position_to_future_inputs() -> None:
    torch.manual_seed(20260921)
    encoder = e3.CausalTemporalEncoder().eval()
    for temporal_position in range(5):
        values = torch.randn(2, 5, requires_grad=True)
        output = encoder.forward_sequence(values)[:, :, temporal_position].sum()
        gradient = torch.autograd.grad(output, values)[0]
        assert torch.count_nonzero(gradient[:, temporal_position + 1 :]) == 0


def test_position_five_receptive_field_reaches_position_one() -> None:
    encoder = e3.CausalTemporalEncoder().eval()
    _set_asymmetric_positive_weights(encoder)
    baseline = torch.zeros(1, 5)
    changed = baseline.clone()
    changed[0, 0] = 1.0
    assert not torch.equal(encoder(baseline), encoder(changed))


def test_position_five_change_cannot_alter_position_four_representation() -> None:
    encoder = e3.CausalTemporalEncoder().eval()
    original = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    changed = original.clone()
    changed[0, 4] = -999.0
    assert torch.equal(
        encoder.forward_sequence(original)[:, :, 3],
        encoder.forward_sequence(changed)[:, :, 3],
    )


def test_temporal_order_sensitivity_is_deterministic() -> None:
    encoder = e3.CausalTemporalEncoder().eval()
    _set_asymmetric_positive_weights(encoder)
    ordered = torch.tensor([[1.0, 2.0, 4.0, 8.0, 16.0]])
    reversed_order = ordered.flip(1)
    first = encoder(ordered)
    second = encoder(reversed_order)
    assert torch.equal(first, encoder(ordered))
    assert not torch.equal(first, second)


def test_feature_identity_distinguishes_equal_temporal_inputs() -> None:
    torch.manual_seed(20260921)
    encoder = e3.FeatureNodeEncoder().eval()
    values = torch.ones(2, 5)
    output = encoder(values, torch.tensor([0, 1], dtype=torch.long))
    assert output.shape == (2, 32)
    assert not torch.equal(output[0], output[1])


def test_eval_mode_is_deterministic_and_does_not_modify_inputs() -> None:
    torch.manual_seed(20260921)
    encoder = e3.FeatureNodeEncoder().eval()
    values = torch.randn(78, 5)
    ids = torch.arange(78)
    values_before = values.clone()
    ids_before = ids.clone()
    first = encoder(values, ids)
    second = encoder(values, ids)
    assert torch.equal(first, second)
    assert torch.equal(values, values_before)
    assert torch.equal(ids, ids_before)


def test_measured_parameter_count_reconciles_with_design_estimate() -> None:
    encoder = e3.FeatureNodeEncoder()
    assert sum(parameter.numel() for parameter in encoder.parameters()) == 1920
    assert sum(parameter.numel() for parameter in encoder.temporal_encoder.parameters()) == 432
    assert encoder.feature_embedding.weight.numel() == 624
    assert sum(parameter.numel() for parameter in encoder.fusion.parameters()) == 800
    assert sum(parameter.numel() for parameter in encoder.normalization.parameters()) == 64


def test_cpu_forward_backward() -> None:
    torch.manual_seed(20260921)
    encoder = e3.FeatureNodeEncoder()
    values = torch.randn(156, 5, requires_grad=True)
    ids = torch.arange(78).repeat(2)
    output = encoder(values, ids)
    assert output.shape == (156, 32)
    output.square().mean().backward()
    assert values.grad is not None and torch.isfinite(values.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in encoder.parameters()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="E.1 requires CUDA")
def test_cuda_forward_backward() -> None:
    torch.manual_seed(20260921)
    encoder = e3.FeatureNodeEncoder().to("cuda")
    values = torch.randn(156, 5, device="cuda", requires_grad=True)
    ids = torch.arange(78, device="cuda").repeat(2)
    output = encoder(values, ids)
    assert output.shape == (156, 32) and output.is_cuda
    output.square().mean().backward()
    assert values.grad is not None and torch.isfinite(values.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in encoder.parameters()
    )
    torch.cuda.synchronize()


@pytest.mark.parametrize(
    ("values", "ids", "message"),
    [
        (torch.zeros(2, 4), torch.arange(2), "shape"),
        (torch.zeros(2, 5, dtype=torch.float64), torch.arange(2), "float32"),
        (torch.zeros(2, 5), torch.arange(2, dtype=torch.int32), "torch.long"),
        (torch.zeros(2, 5), torch.arange(3), "matching"),
        (torch.zeros(2, 5), torch.tensor([0, 78]), "range"),
    ],
)
def test_malformed_inputs_are_rejected(
    values: torch.Tensor, ids: torch.Tensor, message: str
) -> None:
    with pytest.raises(e3.FeatureEncoderContractError, match=message):
        e3.FeatureNodeEncoder()(values, ids)


@pytest.mark.parametrize(
    ("experiment", "split"),
    [(experiment, split) for experiment in d4.EXPERIMENT_DIRS for split in d4.SPLIT_STEMS],
)
def test_real_bounded_d4_d6_e2_e3_integration(experiment: str, split: str) -> None:
    torch.manual_seed(20260921)
    dataset = d4.LazyGraphDataset(experiment, split)
    indices = [0, len(dataset) // 2, len(dataset) - 1]
    batch = d6.build_logical_batch(dataset, indices)
    batch_before = np.array(batch.feature_x, copy=True)
    data = e2.logical_batch_to_heterodata(batch)
    values_before = data["feature"].x.clone()
    ids_before = data["feature"].feature_id.clone()
    output = e3.FeatureNodeEncoder().eval()(
        data["feature"].x, data["feature"].feature_id
    )
    assert output.shape == (3 * 78, 32)
    assert torch.isfinite(output).all()
    assert np.array_equal(batch.feature_x, batch_before)
    assert torch.equal(data["feature"].x, values_before)
    assert torch.equal(data["feature"].feature_id, ids_before)
    assert dataset.is_lazy_read_only


@pytest.mark.parametrize("split", tuple(d4.SPLIT_STEMS))
def test_real_e1_e3_same_inputs_have_identical_embeddings(split: str) -> None:
    torch.manual_seed(20260921)
    encoder = e3.FeatureNodeEncoder().eval()
    e1_dataset = d4.LazyGraphDataset("E1", split)
    e3_dataset = d4.LazyGraphDataset("E3", split)
    indices = [0, len(e1_dataset) // 2, len(e1_dataset) - 1]
    e1_data = e2.logical_batch_to_heterodata(
        d6.build_logical_batch(e1_dataset, indices)
    )
    e3_data = e2.logical_batch_to_heterodata(
        d6.build_logical_batch(e3_dataset, indices)
    )
    assert torch.equal(e1_data["feature"].x, e3_data["feature"].x)
    assert torch.equal(
        encoder(e1_data["feature"].x, e1_data["feature"].feature_id),
        encoder(e3_data["feature"].x, e3_data["feature"].feature_id),
    )


def test_stage_e3_source_excludes_downstream_model_components() -> None:
    source = (ROOT / "src" / "graph" / "feature_encoder.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
    } | {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
    }
    assert not ({"GATv2Conv", "CrossEntropyLoss", "BCEWithLogitsLoss", "Optimizer"} & names)
    assert "category_embedding" not in source
    assert "sample_token" not in source


def test_saved_e3_report_is_canonical_complete_and_passing() -> None:
    report_bytes = E3_REPORT.read_bytes()
    report = json.loads(report_bytes)
    supplied_hash = report.pop("report_sha256")
    assert d2.canonical_sha256(report) == supplied_hash
    assert d2.canonical_json_bytes({**report, "report_sha256": supplied_hash}) == report_bytes
    assert report["overall_status"] == "PASS"
    assert report["parameter_count"]["measured"] == 1920
    assert len(report["real_nine_pair_batches"]) == 9
    source_bytes = (ROOT / report["implementation"]["source_path"]).read_bytes()
    assert hashlib.sha256(source_bytes).hexdigest() == report["implementation"]["source_sha256"]
    physical_hash = hashlib.sha256(report_bytes).hexdigest()
    assert E3_REPORT_SHA.read_text(encoding="ascii") == (
        f"{physical_hash}  feature_encoder_validation.json\n"
    )
