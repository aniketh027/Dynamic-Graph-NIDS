from __future__ import annotations

import json
from pathlib import Path
import random
import subprocess

import numpy as np
import pytest
import torch
from torch import nn
from torch_geometric.data import HeteroData

from src.graph import htgnn_classifier as e5
from src.graph import resource_pilot as e6
from src.training import harness as e7
from src.training import validation as evidence


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TinyModel(nn.Module):
    def __init__(self, task: str) -> None:
        super().__init__()
        self.task = task
        self.head = nn.Linear(2, 1 if task == "E3" else 15)
        self.forward_training_modes: list[bool] = []

    def forward(self, data: HeteroData) -> torch.Tensor:
        self.forward_training_modes.append(self.training)
        logits = self.head(data["sample"].features)
        return logits.squeeze(-1) if self.task == "E3" else logits


def make_batch(
    task: str,
    split: str,
    targets: list[int],
    *,
    offset: int = 0,
) -> HeteroData:
    data = HeteroData()
    data["sample"].features = torch.tensor(
        [[float(index + 1), float((index + 1) % 3)] for index in range(len(targets))],
        dtype=torch.float32,
    )
    target_dtype = torch.float32 if task == "E3" else torch.long
    data["sample"].y = torch.tensor(targets, dtype=target_dtype)
    data["sample"].num_nodes = len(targets)
    data.e2_audit = {
        "experiment": task,
        "split": split,
        "ordered_source_sample_indices": list(
            range(offset, offset + len(targets))
        ),
    }
    return data


def config(task: str = "E1", **overrides: object) -> e7.TrainingConfig:
    values: dict[str, object] = {
        "task": task,
        "learning_rate": 3e-4,
        "seed": 2025,
        "batch_size": 4,
        "max_epochs": 1,
        "max_train_batches": 1,
        "max_validation_batches": 1,
    }
    values.update(overrides)
    return e7.TrainingConfig(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("task", ["E1", "E2"])
def test_multiclass_losses_are_unweighted_cross_entropy(task):
    criterion = e7.create_loss(task)
    assert type(criterion) is nn.CrossEntropyLoss
    assert criterion.weight is None
    logits = torch.tensor([[2.0, 0.0] + [0.0] * 13])
    targets = torch.tensor([0], dtype=torch.long)
    assert torch.equal(
        e7.compute_loss(task, criterion, logits, targets), criterion(logits, targets)
    )
    assert not torch.equal(
        e7.compute_loss(task, criterion, logits, targets),
        criterion(torch.softmax(logits, dim=1), targets),
    )


def test_binary_loss_is_unweighted_bce_with_logits():
    criterion = e7.create_loss("E3")
    assert type(criterion) is nn.BCEWithLogitsLoss
    assert criterion.weight is None
    assert criterion.pos_weight is None
    logits = torch.tensor([2.0, -1.0])
    targets = torch.tensor([1.0, 0.0])
    assert torch.equal(
        e7.compute_loss("E3", criterion, logits, targets),
        criterion(logits, targets),
    )
    assert not torch.equal(
        e7.compute_loss("E3", criterion, logits, targets),
        criterion(torch.sigmoid(logits), targets),
    )


def test_optimizer_is_adamw_with_exact_configuration():
    model = TinyModel("E1")
    optimizer = e7.create_optimizer(model, 1e-3)
    assert type(optimizer) is torch.optim.AdamW
    assert optimizer.param_groups[0]["lr"] == 1e-3
    assert optimizer.param_groups[0]["weight_decay"] == 1e-4


@pytest.mark.parametrize("bad", [42, 2024, 2028, True])
def test_only_approved_seeds_are_accepted(bad):
    with pytest.raises((TypeError, ValueError)):
        config(seed=bad)


def test_seed_everything_repeats_python_numpy_and_torch():
    first = e7.seed_everything(2026)
    values1 = (random.random(), np.random.random(), torch.rand(2))
    second = e7.seed_everything(2026)
    values2 = (random.random(), np.random.random(), torch.rand(2))
    assert first == second
    assert values1[0] == values2[0]
    assert values1[1] == values2[1]
    assert torch.equal(values1[2], values2[2])


def test_sequential_validation_order_and_final_partial_batch():
    assert list(
        e7.iter_index_batches(10, 4, shuffle=False)
    ) == [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9)]


def test_seeded_train_order_is_repeatable_unique_and_not_materialized():
    first = list(e7.iter_index_batches(17, 4, shuffle=True, seed=2025))
    second = list(e7.iter_index_batches(17, 4, shuffle=True, seed=2025))
    different = list(e7.iter_index_batches(17, 4, shuffle=True, seed=2026))
    flat = [value for batch in first for value in batch]
    assert first == second
    assert first != different
    assert sorted(flat) == list(range(17))
    assert all(len(batch) <= 4 for batch in first)


def test_batch_limit_keeps_iteration_bounded():
    batches = list(
        e7.iter_index_batches(1_000_000, 32, shuffle=True, seed=2025, max_batches=2)
    )
    assert len(batches) == 2
    assert sum(map(len, batches)) == 64


def test_test_split_is_rejected_before_training():
    model = TinyModel("E1")
    with pytest.raises(e7.TrainingContractError, match="E1/train"):
        e7.train_one_epoch(
            task="E1",
            model=model,
            batches=[make_batch("E1", "test", [0, 1])],
            criterion=e7.create_loss("E1"),
            optimizer=e7.create_optimizer(model, 1e-3),
            device=torch.device("cpu"),
        )


def test_training_and_validation_switch_modes_and_update_parameters():
    model = TinyModel("E1")
    before = [parameter.detach().clone() for parameter in model.parameters()]
    train = e7.train_one_epoch(
        task="E1",
        model=model,
        batches=[make_batch("E1", "train", [0, 1, 0])],
        criterion=e7.create_loss("E1"),
        optimizer=e7.create_optimizer(model, 1e-3),
        device=torch.device("cpu"),
    )
    validation = e7.validate_one_epoch(
        task="E1",
        model=model,
        batches=[make_batch("E1", "validation", [0, 1, 0])],
        criterion=e7.create_loss("E1"),
        device=torch.device("cpu"),
        train_target_codes=train.target_codes,
    )
    assert train.batches == 1 and train.examples == 3
    assert validation.batches == 1 and validation.examples == 3
    assert model.forward_training_modes == [True, False]
    assert any(
        not torch.equal(left, right)
        for left, right in zip(before, model.parameters(), strict=True)
    )


def test_nonfinite_loss_fails_loudly():
    model = TinyModel("E3")
    batch = make_batch("E3", "train", [0, 1])
    batch["sample"].features[0, 0] = float("inf")
    with pytest.raises(e7.TrainingContractError, match="nonfinite train loss"):
        e7.train_one_epoch(
            task="E3",
            model=model,
            batches=[batch],
            criterion=e7.create_loss("E3"),
            optimizer=e7.create_optimizer(model, 1e-3),
            device=torch.device("cpu"),
        )


def test_nonfinite_gradient_fails_loudly():
    model = TinyModel("E3")
    model.head.weight.register_hook(lambda gradient: gradient * float("nan"))
    with pytest.raises(e7.TrainingContractError, match="nonfinite gradient"):
        e7.train_one_epoch(
            task="E3",
            model=model,
            batches=[make_batch("E3", "train", [0, 1])],
            criterion=e7.create_loss("E3"),
            optimizer=e7.create_optimizer(model, 1e-3),
            device=torch.device("cpu"),
        )


def test_early_stopping_obeys_minimum_epoch_patience_and_min_delta():
    state = e7.EarlyStopping(minimum_epochs=3, patience=1, min_delta=1e-4)
    assert state.update(1, 0.5) == (True, False)
    assert state.update(2, 0.50005) == (False, False)
    assert state.update(3, 0.50005) == (False, True)


def test_default_early_stopping_patience_is_five():
    state = e7.EarlyStopping()
    assert state.update(1, 0.5) == (True, False)
    for epoch in range(2, 6):
        assert state.update(epoch, 0.5) == (False, False)
    assert state.update(6, 0.5) == (False, True)


def test_best_epoch_is_validation_best_not_last_epoch():
    state = e7.EarlyStopping()
    assert state.update(1, 0.5)[0]
    assert state.update(2, 0.6)[0]
    assert not state.update(3, 0.55)[0]
    assert state.best_epoch == 2
    assert state.best_value == 0.6


def test_e1_macro_f1_is_restricted_to_train_validation_support():
    model = TinyModel("E1")
    stats = e7.validate_one_epoch(
        task="E1",
        model=model,
        batches=[make_batch("E1", "validation", [0, 1, 2])],
        criterion=e7.create_loss("E1"),
        device=torch.device("cpu"),
        train_target_codes=(0, 2, 4),
    )
    assert stats.selection_class_codes == (0, 2)


def test_e2_macro_f1_uses_only_supplied_frozen_eligible_codes():
    model = TinyModel("E2")
    stats = e7.validate_one_epoch(
        task="E2",
        model=model,
        batches=[make_batch("E2", "validation", [0, 1, 2])],
        criterion=e7.create_loss("E2"),
        device=torch.device("cpu"),
        train_target_codes=(),
        eligible_codes=(0, 1),
    )
    assert stats.selection_class_codes == (0, 1)


def test_e3_selection_is_pr_auc_average_precision():
    model = TinyModel("E3")
    with torch.no_grad():
        model.head.weight.copy_(torch.tensor([[1.0, 0.0]]))
        model.head.bias.zero_()
    stats = e7.validate_one_epoch(
        task="E3",
        model=model,
        batches=[make_batch("E3", "validation", [0, 1, 1])],
        criterion=e7.create_loss("E3"),
        device=torch.device("cpu"),
        train_target_codes=(),
    )
    assert stats.selection_name == "pr_auc_average_precision"
    assert stats.selection_support_complete is True
    assert 0.0 <= stats.selection_value <= 1.0


def checkpoint_stats(task: str) -> e7.EpochStats:
    return e7.EpochStats(
        loss=0.5,
        batches=1,
        examples=2,
        order_sha256="0" * 64,
        target_codes=(0, 1),
        selection_value=0.25,
        selection_name=e7.SELECTION_NAMES[task],
        selection_support_complete=True,
        selection_class_codes=(0, 1) if task != "E3" else (),
    )


def test_checkpoint_round_trip_uses_state_dicts(tmp_path):
    run_config = config("E1")
    model = e7.create_model("E1")
    optimizer = e7.create_optimizer(model, run_config.learning_rate)
    path = tmp_path / "checkpoint.pt"
    e7.save_checkpoint(
        path,
        config=run_config,
        model=model,
        optimizer=optimizer,
        epoch=1,
        validation=checkpoint_stats("E1"),
        frozen_source_hashes=e6.require_frozen_sources(PROJECT_ROOT),
    )
    fresh = e7.create_model("E1")
    metadata = e7.load_checkpoint(
        path,
        model=fresh,
        expected_task="E1",
        expected_config=run_config,
    )
    assert metadata["epoch"] == 1
    assert all(
        torch.equal(left, right)
        for left, right in zip(model.state_dict().values(), fresh.state_dict().values(), strict=True)
    )


@pytest.mark.parametrize(("saved", "requested"), [("E1", "E3"), ("E3", "E1"), ("E1", "E2")])
def test_cross_task_checkpoint_load_fails_loudly(tmp_path, saved, requested):
    saved_config = config(saved)
    model = e7.create_model(saved)
    path = tmp_path / "checkpoint.pt"
    e7.save_checkpoint(
        path,
        config=saved_config,
        model=model,
        optimizer=e7.create_optimizer(model, saved_config.learning_rate),
        epoch=1,
        validation=checkpoint_stats(saved),
        frozen_source_hashes=e6.require_frozen_sources(PROJECT_ROOT),
    )
    with pytest.raises(e7.TrainingContractError, match="mismatch"):
        e7.load_checkpoint(
            path,
            model=e7.create_model(requested),
            expected_task=requested,
        )


def test_checkpoint_architecture_metadata_mismatch_fails(tmp_path):
    run_config = config("E1")
    model = e7.create_model("E1")
    path = tmp_path / "checkpoint.pt"
    e7.save_checkpoint(
        path,
        config=run_config,
        model=model,
        optimizer=e7.create_optimizer(model, run_config.learning_rate),
        epoch=1,
        validation=checkpoint_stats("E1"),
        frozen_source_hashes=e6.require_frozen_sources(PROJECT_ROOT),
    )
    payload = torch.load(path, weights_only=True)
    payload["metadata"]["model_state_schema_sha256"] = "bad"
    torch.save(payload, path)
    with pytest.raises(e7.TrainingContractError, match="model_state_schema_sha256"):
        e7.load_checkpoint(path, model=e7.create_model("E1"), expected_task="E1")


def test_checkpoint_configuration_mismatch_fails(tmp_path):
    run_config = config("E1")
    model = e7.create_model("E1")
    path = tmp_path / "checkpoint.pt"
    e7.save_checkpoint(
        path,
        config=run_config,
        model=model,
        optimizer=e7.create_optimizer(model, run_config.learning_rate),
        epoch=1,
        validation=checkpoint_stats("E1"),
        frozen_source_hashes=e6.require_frozen_sources(PROJECT_ROOT),
    )
    with pytest.raises(e7.TrainingContractError, match="configuration"):
        e7.load_checkpoint(
            path,
            model=e7.create_model("E1"),
            expected_task="E1",
            expected_config=config("E1", learning_rate=1e-4),
        )


def test_no_model_weights_are_tracked_by_git():
    tracked = subprocess.run(
        ["git", "ls-files"],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    forbidden = (".pt", ".pth", ".ckpt", ".safetensors")
    assert not [path for path in tracked if path.lower().endswith(forbidden)]
    assert not [path for path in tracked if path.startswith("checkpoints/") and path != "checkpoints/.gitkeep"]


def test_frozen_e2_e5_sources_and_parameter_counts_remain_unchanged():
    assert e6.require_frozen_sources(PROJECT_ROOT) == e6.FROZEN_SOURCE_SHA256
    assert {
        task: sum(parameter.numel() for parameter in e7.create_model(task).parameters())
        for task in e7.TASKS
    } == e7.EXPECTED_PARAMETERS


def test_training_harness_has_no_pruning_dependency_or_attention_collection():
    source = Path(e7.__file__).read_text(encoding="utf-8")
    assert "from src.pruning" not in source
    assert "import src.pruning" not in source
    assert "return_attention=" not in source


def test_deterministic_evidence_validation_rejects_tampering():
    smoke = [{"task": task, "status": "PASS"} for task in e7.TASKS]
    verification = {
        "focused_e7": {"status": "PASS", "passed": 1},
        "frozen_e2_e6_regression": {"status": "PASS", "passed": 1},
        "complete_project_suite": {"status": "PASS", "passed": 1},
        "status": "PASS",
    }
    report = evidence.build_report(
        frozen_source_sha256=e6.FROZEN_SOURCE_SHA256,
        smoke_validation=smoke,
        integrity={"stage_c": "PASS", "raw_csv": "PASS"},
        verification=verification,
    )
    evidence.validate_report(report)
    corrupted = json.loads(json.dumps(report))
    corrupted["early_stopping"]["patience"] = 4
    with pytest.raises(evidence.TrainingEvidenceError):
        evidence.validate_report(corrupted)


@pytest.mark.parametrize("task", e7.TASKS)
def test_bounded_real_training_checkpoint_and_reload(task, tmp_path):
    run_config = config(task, batch_size=8)
    checkpoint = tmp_path / f"{task}.pt"
    result = e7.run_training(
        run_config,
        project_root=PROJECT_ROOT,
        checkpoint_path=checkpoint,
        device="auto",
    )
    metadata, validation, model = e7.evaluate_saved_validation_checkpoint(
        run_config,
        project_root=PROJECT_ROOT,
        checkpoint_path=checkpoint,
        device="auto",
    )
    assert result.parameter_update_observed
    assert result.epochs_completed == 1
    assert result.best_epoch == 1
    assert checkpoint.is_file()
    assert np.isfinite(result.history[0]["train_loss"])
    assert np.isfinite(result.history[0]["validation_loss"])
    assert validation.loss == pytest.approx(metadata["validation_loss"], rel=1e-5, abs=1e-6)
    assert validation.selection_value == pytest.approx(metadata["selection_value"], rel=1e-5, abs=1e-6)
    assert sum(parameter.numel() for parameter in model.parameters()) == e7.EXPECTED_PARAMETERS[task]
