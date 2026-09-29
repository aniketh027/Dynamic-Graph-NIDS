from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from src.graph import adapter as d4
from src.graph import resource_pilot as e6
from src.training import baseline_integration
from src.training import baselines
from src.training import harness as e7


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "preprocessed"


def _config(task: str, *, batch_size: int = 4) -> e7.TrainingConfig:
    return e7.TrainingConfig(
        task=task,
        learning_rate=3e-4,
        seed=2025,
        batch_size=batch_size,
        max_epochs=1,
        max_train_batches=1,
        max_validation_batches=1,
    )


def _stats(task: str) -> e7.EpochStats:
    return e7.EpochStats(
        loss=0.5,
        batches=1,
        examples=4,
        order_sha256="0" * 64,
        target_codes=(0, 1),
        selection_value=0.25,
        selection_name=e7.SELECTION_NAMES[task],
        selection_support_complete=True,
        selection_class_codes=(0, 1) if task != "E3" else (),
    )


def _prime_optimizer(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 0.01)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _assert_nested_equal(left, right) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert left.dtype == right.dtype
        assert left.device == right.device
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def _save(
    path: Path, family: str, task: str, *, prime_optimizer: bool = False
) -> None:
    config = _config(task)
    model = e7.create_model(task, family)
    optimizer = e7.create_optimizer(model, config.learning_rate)
    if prime_optimizer:
        _prime_optimizer(model, optimizer)
    e7.save_checkpoint(
        path,
        config=config,
        model=model,
        optimizer=optimizer,
        epoch=1,
        validation=_stats(task),
        frozen_source_hashes=e6.FROZEN_SOURCE_SHA256,
    )


@pytest.mark.parametrize("accepted", [1, 128, 256])
def test_production_batch_limit_accepts_authorized_boundary(accepted: int):
    assert _config("E2", batch_size=accepted).batch_size == accepted
    batches = list(
        e7.iter_index_batches(
            512, accepted, shuffle=True, seed=2025, max_batches=1
        )
    )
    assert len(batches) == 1 and len(batches[0]) == accepted


@pytest.mark.parametrize("rejected", [257, 512])
def test_production_batch_limit_rejects_above_authorized_boundary(rejected: int):
    with pytest.raises(ValueError, match="between 1 and 256"):
        _config("E2", batch_size=rejected)
    with pytest.raises(ValueError, match="between 1 and 256"):
        list(e7.iter_index_batches(1024, rejected, shuffle=True, seed=2025))


@pytest.mark.parametrize("family", e7.MODEL_FAMILIES)
@pytest.mark.parametrize("task", e7.TASKS)
def test_model_contract_is_derived_from_actual_supported_instance(family: str, task: str):
    model = e7.create_model(task, family)
    contract = e7.model_contract(model)
    expected = (
        e7.EXPECTED_PARAMETERS[task]
        if family == e7.HTGNN_MODEL_FAMILY
        else baselines.EXPECTED_PARAMETER_COUNTS[family][task]
    )
    assert contract.family == family
    assert contract.task == task
    assert contract.parameter_count == expected
    assert contract.input_contract == e7.MODEL_INPUT_CONTRACTS[family]


def test_baseline_cannot_be_relabelled_by_caller_string():
    with pytest.raises(
        baseline_integration.BaselineIntegrationError,
        match="declared baseline family",
    ):
        baseline_integration.BaselineTrainingAdapter(
            baselines.FinalFlowMLP("E2"),
            kind=baselines.FLATTENED_FIVE_FLOW,
            task="E2",
        )


@pytest.mark.parametrize(
    ("saved_family", "saved_task", "loaded_family", "loaded_task"),
    [
        (baselines.FINAL_FLOW, "E1", baselines.FINAL_FLOW, "E2"),
        (baselines.FINAL_FLOW, "E2", baselines.FLATTENED_FIVE_FLOW, "E2"),
        (baselines.FLATTENED_FIVE_FLOW, "E2", baselines.FINAL_FLOW, "E2"),
        (baselines.FINAL_FLOW, "E2", e7.HTGNN_MODEL_FAMILY, "E2"),
        (e7.HTGNN_MODEL_FAMILY, "E2", baselines.FINAL_FLOW, "E2"),
    ],
)
def test_checkpoint_rejects_task_and_family_mismatch_before_weights(
    tmp_path: Path,
    saved_family: str,
    saved_task: str,
    loaded_family: str,
    loaded_task: str,
):
    checkpoint = tmp_path / "mismatch.pt"
    _save(checkpoint, saved_family, saved_task)
    with pytest.raises(e7.TrainingContractError, match="mismatch"):
        e7.load_checkpoint(
            checkpoint,
            model=e7.create_model(loaded_task, loaded_family),
            expected_task=loaded_task,
        )


@pytest.mark.parametrize("family", e7.MODEL_FAMILIES)
@pytest.mark.parametrize("task", e7.TASKS)
def test_matching_checkpoint_round_trip_binds_full_model_identity(
    tmp_path: Path, family: str, task: str
):
    checkpoint = tmp_path / f"{family}-{task}.pt"
    _save(checkpoint, family, task, prime_optimizer=True)
    payload = torch.load(checkpoint, weights_only=True)
    fresh = e7.create_model(task, family)
    optimizer = e7.create_optimizer(fresh, 1e-4)
    _prime_optimizer(fresh, optimizer)
    metadata = e7.load_checkpoint(
        checkpoint,
        model=fresh,
        expected_task=task,
        optimizer=optimizer,
        expected_config=_config(task),
    )
    identity = e7.model_contract(fresh)
    assert metadata["schema_version"] == e7.CHECKPOINT_SCHEMA_VERSION
    assert metadata["model_family"] == family
    assert metadata["model_contract_version"] == identity.architecture_version
    assert metadata["model_input_contract"] == identity.input_contract
    assert metadata["parameter_count"] == identity.parameter_count
    _assert_nested_equal(fresh.state_dict(), payload["model_state_dict"])
    _assert_nested_equal(optimizer.state_dict(), payload["optimizer_state_dict"])


@pytest.mark.parametrize(
    "case",
    [
        "missing_optimizer",
        "missing_model_key",
        "unexpected_model_key",
        "wrong_model_shape",
        "incompatible_optimizer_state",
        "task_mismatch",
        "family_mismatch",
        "architecture_mismatch",
        "input_contract_mismatch",
    ],
)
def test_rejected_checkpoint_load_is_atomic_for_model_and_optimizer(
    tmp_path: Path, case: str
):
    checkpoint = tmp_path / f"atomic-{case}.pt"
    _save(checkpoint, baselines.FINAL_FLOW, "E2", prime_optimizer=True)
    payload = torch.load(checkpoint, weights_only=True)
    model_keys = list(payload["model_state_dict"])
    if case == "missing_optimizer":
        payload.pop("optimizer_state_dict")
    elif case == "missing_model_key":
        payload["model_state_dict"].pop(model_keys[0])
    elif case == "unexpected_model_key":
        payload["model_state_dict"]["unexpected.weight"] = torch.zeros(1)
    elif case == "wrong_model_shape":
        value = payload["model_state_dict"][model_keys[0]]
        payload["model_state_dict"][model_keys[0]] = value.reshape(-1)[:1]
    elif case == "incompatible_optimizer_state":
        first_state = next(iter(payload["optimizer_state_dict"]["state"].values()))
        first_state["exp_avg"] = torch.zeros(1)
    elif case == "task_mismatch":
        payload["metadata"]["task"] = "E1"
    elif case == "family_mismatch":
        payload["metadata"]["model_family"] = baselines.FLATTENED_FIVE_FLOW
    elif case == "architecture_mismatch":
        payload["metadata"]["model_contract_version"] = "forged"
    elif case == "input_contract_mismatch":
        payload["metadata"]["model_input_contract"] = "forged"
    else:  # pragma: no cover - parametrization is closed above
        raise AssertionError(case)
    torch.save(payload, checkpoint)

    target = e7.create_model("E2", baselines.FINAL_FLOW)
    target_optimizer = e7.create_optimizer(target, 1e-4)
    _prime_optimizer(target, target_optimizer)
    model_before = copy.deepcopy(target.state_dict())
    optimizer_before = copy.deepcopy(target_optimizer.state_dict())
    with pytest.raises(e7.TrainingContractError):
        e7.load_checkpoint(
            checkpoint,
            model=target,
            expected_task="E2",
            optimizer=target_optimizer,
            expected_config=_config("E2"),
        )
    _assert_nested_equal(target.state_dict(), model_before)
    _assert_nested_equal(target_optimizer.state_dict(), optimizer_before)


def test_checkpoint_load_rolls_back_both_targets_on_post_preflight_failure(
    tmp_path: Path, monkeypatch
):
    checkpoint = tmp_path / "rollback.pt"
    _save(checkpoint, baselines.FINAL_FLOW, "E2", prime_optimizer=True)
    target = e7.create_model("E2", baselines.FINAL_FLOW)
    target_optimizer = e7.create_optimizer(target, 1e-4)
    _prime_optimizer(target, target_optimizer)
    model_before = copy.deepcopy(target.state_dict())
    optimizer_before = copy.deepcopy(target_optimizer.state_dict())
    optimizer_type = type(target_optimizer)
    original_load = optimizer_type.load_state_dict
    calls = 0

    def fail_only_on_target_load(self, state_dict):
        nonlocal calls
        calls += 1
        if calls == 2:
            self.param_groups[0]["lr"] = 9.0
            next(iter(self.param_groups[0]["params"])).data.add_(1.0)
            raise RuntimeError("synthetic failure after target mutation")
        return original_load(self, state_dict)

    monkeypatch.setattr(optimizer_type, "load_state_dict", fail_only_on_target_load)
    with pytest.raises(e7.TrainingContractError, match="target state was restored"):
        e7.load_checkpoint(
            checkpoint,
            model=target,
            expected_task="E2",
            optimizer=target_optimizer,
            expected_config=_config("E2"),
        )
    assert calls == 3
    _assert_nested_equal(target.state_dict(), model_before)
    _assert_nested_equal(target_optimizer.state_dict(), optimizer_before)


def test_legacy_valid_htgnn_checkpoint_remains_loadable(tmp_path: Path):
    checkpoint = tmp_path / "legacy-htgnn.pt"
    _save(checkpoint, e7.HTGNN_MODEL_FAMILY, "E2")
    payload = torch.load(checkpoint, weights_only=True)
    payload["metadata"]["schema_version"] = e7.LEGACY_HTGNN_CHECKPOINT_SCHEMA_VERSION
    for key in ("model_family", "model_contract_version", "model_input_contract"):
        payload["metadata"].pop(key)
    torch.save(payload, checkpoint)
    e7.load_checkpoint(
        checkpoint,
        model=e7.create_model("E2"),
        expected_task="E2",
        expected_config=_config("E2"),
    )


@pytest.mark.parametrize("family", baselines.BASELINE_KINDS)
@pytest.mark.parametrize("task", e7.TASKS)
def test_baseline_uses_shared_training_checkpoint_and_jsonl_path(
    tmp_path: Path, family: str, task: str
):
    config = _config(task)
    checkpoint = tmp_path / f"{family}-{task}.pt"
    log = tmp_path / f"{family}-{task}.jsonl"
    result = e7.run_training(
        config,
        project_root=PROJECT_ROOT,
        checkpoint_path=checkpoint,
        log_path=log,
        device="cpu",
        model_family=family,
    )
    records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert checkpoint.is_file()
    assert len(records) == 1
    assert records[0]["model_family"] == family
    assert result.model_family == family

    metadata, _, first = e7.evaluate_saved_validation_checkpoint(
        config,
        project_root=PROJECT_ROOT,
        checkpoint_path=checkpoint,
        device="cpu",
        model_family=family,
    )
    second = e7.create_model(task, family)
    e7.load_checkpoint(
        checkpoint,
        model=second,
        expected_task=task,
        expected_config=config,
    )
    batch = next(
        e7.iter_production_batches(
            d4.LazyGraphDataset(task, "validation", processed_root=PROCESSED_ROOT),
            task=task,
            split="validation",
            batch_size=4,
            epoch_seed=None,
            max_batches=1,
        )
    )
    first.eval()
    second.eval()
    with torch.no_grad():
        assert torch.allclose(first(batch), second(batch), rtol=0, atol=0)
    assert metadata["model_family"] == family


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real CUDA smoke requires CUDA")
def test_batch_256_real_production_iterator_and_htgnn_train_step():
    config = _config("E2", batch_size=256)
    dataset = d4.LazyGraphDataset("E2", "train", processed_root=PROCESSED_ROOT)
    batches = e7.iter_production_batches(
        dataset,
        task="E2",
        split="train",
        batch_size=config.batch_size,
        epoch_seed=config.seed,
        max_batches=1,
    )
    model = e7.create_model("E2").to("cuda")
    observed_shapes: list[tuple[int, ...]] = []
    hook = model.register_forward_hook(
        lambda _module, _inputs, output: observed_shapes.append(tuple(output.shape))
    )
    try:
        stats = e7.train_one_epoch(
            task="E2",
            model=model,
            batches=batches,
            criterion=e7.create_loss("E2").to("cuda"),
            optimizer=e7.create_optimizer(model, config.learning_rate),
            device=torch.device("cuda"),
        )
    finally:
        hook.remove()
    assert observed_shapes == [(256, 15)]
    assert stats.examples == 256 and stats.batches == 1
    assert torch.isfinite(torch.tensor(stats.loss))
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
