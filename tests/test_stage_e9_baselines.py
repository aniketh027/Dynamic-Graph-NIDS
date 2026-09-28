from __future__ import annotations

import inspect

import pytest
import torch
from torch import nn
from torch_geometric.data import HeteroData

from src.training import baseline_integration as integration
from src.training import baselines
from src.training import harness as e7


def _batch(task: str, split: str = "train", batch_size: int = 4) -> HeteroData:
    windows = torch.arange(
        batch_size * 5 * 78, dtype=torch.float32
    ).reshape(batch_size, 5, 78)
    data = HeteroData()
    data["feature"].x = windows.transpose(1, 2).reshape(batch_size * 78, 5)
    data["feature"].feature_id = torch.arange(78).repeat(batch_size)
    data["feature"].batch = torch.arange(batch_size).repeat_interleave(78)
    data["sample"].y = (
        torch.tensor([0, 1, 2, 3], dtype=torch.long)[:batch_size]
        if task in {"E1", "E2"}
        else torch.tensor([0, 1, 0, 1], dtype=torch.float32)[:batch_size]
    )
    data.e2_audit = {
        "experiment": task,
        "split": split,
        "ordered_source_sample_indices": list(range(batch_size)),
    }
    return data


def _devices() -> tuple[str, ...]:
    return ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)


@pytest.mark.parametrize("kind", baselines.BASELINE_KINDS)
@pytest.mark.parametrize("task", baselines.TASKS)
def test_exact_architecture_parameter_counts_and_shapes(kind: str, task: str):
    model = baselines.create_baseline(kind, task)
    expected_first_width = 78 if kind == baselines.FINAL_FLOW else 390
    assert list(model.hidden) == [
        model.hidden[0],
        model.hidden[1],
        model.hidden[2],
        model.hidden[3],
        model.hidden[4],
        model.hidden[5],
        model.hidden[6],
        model.hidden[7],
    ]
    assert type(model.hidden[0]) is nn.Linear
    assert (model.hidden[0].in_features, model.hidden[0].out_features) == (
        expected_first_width,
        64,
    )
    assert model.hidden[0].bias is not None
    assert type(model.hidden[1]) is nn.LayerNorm
    assert type(model.hidden[2]) is nn.GELU
    assert type(model.hidden[3]) is nn.Dropout and model.hidden[3].p == 0.10
    assert type(model.hidden[4]) is nn.Linear
    assert (model.hidden[4].in_features, model.hidden[4].out_features) == (64, 32)
    assert model.hidden[4].bias is not None
    assert type(model.hidden[5]) is nn.LayerNorm
    assert type(model.hidden[6]) is nn.GELU
    assert type(model.hidden[7]) is nn.Dropout and model.hidden[7].p == 0.10
    assert (model.head.in_features, model.head.out_features) == (
        32,
        1 if task == "E3" else 15,
    )
    assert baselines.parameter_count(model) == baselines.EXPECTED_PARAMETER_COUNTS[
        kind
    ][task]

    inputs = (
        torch.randn(3, 78)
        if kind == baselines.FINAL_FLOW
        else torch.randn(3, 5, 78)
    )
    assert tuple(model(inputs).shape) == ((3,) if task == "E3" else (3, 15))


@pytest.mark.parametrize("kind", baselines.BASELINE_KINDS)
@pytest.mark.parametrize("task", baselines.TASKS)
def test_forward_backward_is_finite_and_bounded_on_available_devices(
    kind: str, task: str
):
    for device in _devices():
        torch.manual_seed(2025)
        model = baselines.create_baseline(kind, task).to(device)
        inputs = (
            torch.randn(4, 78, device=device)
            if kind == baselines.FINAL_FLOW
            else torch.randn(4, 5, 78, device=device)
        )
        logits = model(inputs)
        loss = logits.square().mean()
        loss.backward()
        assert torch.isfinite(logits).all()
        assert torch.isfinite(loss)
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )


@pytest.mark.parametrize("kind", baselines.BASELINE_KINDS)
@pytest.mark.parametrize("task", baselines.TASKS)
def test_forward_returns_raw_logits(kind: str, task: str):
    model = baselines.create_baseline(kind, task).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        if task == "E3":
            model.head.bias.fill_(2.0)
        else:
            model.head.bias.copy_(torch.linspace(-2.0, 2.0, 15))
    inputs = (
        torch.zeros(2, 78)
        if kind == baselines.FINAL_FLOW
        else torch.zeros(2, 5, 78)
    )
    logits = model(inputs)
    if task == "E3":
        assert torch.equal(logits, torch.full((2,), 2.0))
    else:
        expected = torch.linspace(-2.0, 2.0, 15).repeat(2, 1)
        assert torch.equal(logits, expected)
        assert not torch.allclose(logits.sum(dim=1), torch.ones(2))


def test_final_flow_adapter_cannot_observe_temporal_positions_one_to_four():
    torch.manual_seed(2025)
    model = baselines.FinalFlowMLP("E2").eval()
    adapter = integration.BaselineTrainingAdapter(
        model, kind=baselines.FINAL_FLOW, task="E2"
    ).eval()
    first = _batch("E2")
    changed = first.clone()
    changed["feature"].x[:, :4] += 1_000_000.0
    original_source = first["feature"].x.clone()
    with torch.no_grad():
        left = adapter(first)
        right = adapter(changed)
    assert torch.equal(left, right)
    assert torch.equal(first["feature"].x, original_source)


def test_flattened_adapter_preserves_temporal_major_order_and_earlier_influence():
    data = _batch("E2", batch_size=1)
    source = data["feature"].x.clone()
    ordered = integration.ordered_window_inputs(data, task="E2")
    expected = torch.arange(390, dtype=torch.float32).reshape(1, 5, 78)
    assert torch.equal(ordered, expected)
    assert torch.equal(ordered.reshape(1, 390), torch.arange(390.0).reshape(1, 390))
    assert torch.equal(data["feature"].x, source)

    torch.manual_seed(2025)
    model = baselines.FlattenedFiveFlowMLP("E2").eval()
    earlier = ordered.clone().requires_grad_(True)
    model(earlier).sum().backward()
    assert earlier.grad is not None
    assert bool(torch.count_nonzero(earlier.grad[:, :4, :]))


@pytest.mark.parametrize("kind", baselines.BASELINE_KINDS)
@pytest.mark.parametrize("task", baselines.TASKS)
def test_state_dict_round_trip_and_task_instances_are_independent(kind: str, task: str):
    first = baselines.create_baseline(kind, task).eval()
    second = baselines.create_baseline(kind, task).eval()
    assert not any(
        left.data_ptr() == right.data_ptr()
        for left, right in zip(first.parameters(), second.parameters(), strict=True)
    )
    second.load_state_dict(first.state_dict(), strict=True)
    inputs = (
        torch.randn(2, 78)
        if kind == baselines.FINAL_FLOW
        else torch.randn(2, 5, 78)
    )
    with torch.no_grad():
        assert torch.equal(first(inputs), second(inputs))


def test_baseline_models_have_no_graph_attention_pruning_or_target_dependency():
    source = inspect.getsource(baselines)
    assert "torch_geometric" not in source
    assert "src.graph" not in source
    signatures = {
        inspect.signature(baselines.FinalFlowMLP.forward),
        inspect.signature(baselines.FlattenedFiveFlowMLP.forward),
    }
    assert {tuple(signature.parameters) for signature in signatures} == {
        ("self", "values")
    }
    for kind in baselines.BASELINE_KINDS:
        model = baselines.create_baseline(kind, "E2")
        module_names = {type(module).__name__.lower() for module in model.modules()}
        assert not any("gat" in name or "attention" in name for name in module_names)
        assert not hasattr(model, "return_attention")
        assert not hasattr(model, "prune")


def test_adapter_rejects_test_cross_task_and_corrupted_feature_order():
    with pytest.raises(integration.BaselineIntegrationError, match="rejects test"):
        integration.final_flow_inputs(_batch("E2", split="test"), task="E2")
    with pytest.raises(integration.BaselineIntegrationError, match="does not match"):
        integration.final_flow_inputs(_batch("E1"), task="E2")
    corrupted = _batch("E2")
    corrupted["feature"].feature_id[0] = 77
    with pytest.raises(integration.BaselineIntegrationError, match="feature-ID"):
        integration.ordered_window_inputs(corrupted, task="E2")


@pytest.mark.parametrize("kind", baselines.BASELINE_KINDS)
def test_targets_cannot_change_baseline_logits(kind: str):
    torch.manual_seed(2025)
    model = integration.BaselineTrainingAdapter(
        baselines.create_baseline(kind, "E2"), kind=kind, task="E2"
    ).eval()
    first = _batch("E2")
    changed = first.clone()
    changed["sample"].y = torch.tensor([14, 13, 12, 11], dtype=torch.long)
    with torch.no_grad():
        assert torch.equal(model(first), model(changed))


def test_baselines_reuse_e7_checkpoint_identity_and_log_ownership_helpers(tmp_path):
    model = integration.BaselineTrainingAdapter(
        baselines.FinalFlowMLP("E2"), kind=baselines.FINAL_FLOW, task="E2"
    )
    schema = e7.model_state_schema_sha256(model)
    replacement = integration.BaselineTrainingAdapter(
        baselines.FinalFlowMLP("E2"), kind=baselines.FINAL_FLOW, task="E2"
    )
    assert e7.model_state_schema_sha256(replacement) == schema

    checkpoint = tmp_path / "baseline.pt"
    log = tmp_path / "baseline.jsonl"
    resolved_checkpoint, resolved_log = e7._reserve_run_outputs(checkpoint, log)
    assert resolved_checkpoint == checkpoint.resolve()
    assert resolved_log == log.resolve() and log.is_file()
    with pytest.raises(e7.TrainingContractError, match="one log file"):
        e7._reserve_run_outputs(checkpoint, log)
    with pytest.raises(e7.TrainingContractError, match="different targets"):
        e7._reserve_run_outputs(checkpoint, checkpoint)


@pytest.mark.parametrize("kind", baselines.BASELINE_KINDS)
@pytest.mark.parametrize("task", baselines.TASKS)
def test_frozen_e7_epoch_helper_trains_each_baseline(kind: str, task: str):
    torch.manual_seed(2025)
    wrapped = integration.BaselineTrainingAdapter(
        baselines.create_baseline(kind, task), kind=kind, task=task
    )
    before = [parameter.detach().clone() for parameter in wrapped.parameters()]
    stats = e7.train_one_epoch(
        task=task,
        model=wrapped,
        batches=[_batch(task)],
        criterion=e7.create_loss(task),
        optimizer=e7.create_optimizer(wrapped, 3e-4),
        device=torch.device("cpu"),
    )
    assert stats.batches == 1 and stats.examples == 4
    assert torch.isfinite(torch.tensor(stats.loss))
    assert any(
        not torch.equal(left, right.detach())
        for left, right in zip(before, wrapped.parameters(), strict=True)
    )
