from __future__ import annotations

import inspect
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from src.training import harness as e7
from src.training import optimization_pilot as e8


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "pilot-20260927t000000z-deadbeef"


def _history(candidate: e8.PilotCandidate, objective: float = 0.25):
    return [
        {
            "task": candidate.task,
            "seed": e8.PILOT_SEED,
            "learning_rate": candidate.learning_rate,
            "epoch": epoch,
            "train_loss": 1.0 / epoch,
            "validation_loss": 1.5 / epoch,
            "validation_selection_name": candidate.selection_name,
            "validation_selection_value": objective + epoch / 1000,
            "validation_selection_support_complete": True,
            "train_batches": e8.PILOT_MAX_TRAIN_BATCHES,
            "validation_batches": e8.PILOT_MAX_VALIDATION_BATCHES,
            "train_examples": e8.PILOT_BATCH_SIZE
            * e8.PILOT_MAX_TRAIN_BATCHES,
            "validation_examples": e8.PILOT_BATCH_SIZE
            * e8.PILOT_MAX_VALIDATION_BATCHES,
            "train_order_sha256": f"{epoch:064x}",
            "validation_order_sha256": f"{99:064x}",
            "best_checkpoint_updated": True,
        }
        for epoch in range(1, e8.PILOT_MAX_EPOCHS + 1)
    ]


def _complete_record(
    project_root: Path,
    candidate: e8.PilotCandidate,
    objective: float = 0.25,
    *,
    run_id: str = RUN_ID,
):
    paths = e8.candidate_paths(project_root, run_id, candidate)
    history = _history(candidate, objective)
    return {
        "candidate_id": candidate.candidate_id,
        "candidate_identity_sha256": candidate.identity_sha256,
        **candidate.contract_dict(),
        "run_id": run_id,
        "checkpoint_path": paths.checkpoint.relative_to(project_root).as_posix(),
        "log_path": paths.log.relative_to(project_root).as_posix(),
        "candidate_result_path": paths.result.relative_to(project_root).as_posix(),
        "test_accessed": False,
        "test_metrics_recorded": False,
        "status": "COMPLETE",
        "executor": "test",
        "device": "cpu",
        "epochs_executed": len(history),
        "stopped_early": False,
        "best_epoch": len(history),
        "best_validation_objective": objective + len(history) / 1000,
        "history": history,
        "parameter_update_observed": True,
        "parameter_count": e7.EXPECTED_PARAMETERS[candidate.task],
        "model_state_schema_sha256": "a" * 64,
        "checkpoint_sha256": "b" * 64,
        "log_sha256": "c" * 64,
        "elapsed_seconds": 1.0,
    }


def test_enumerates_only_approved_candidates_with_equal_budgets():
    candidates = e8.enumerate_candidates()
    assert len(candidates) == 9
    assert {(item.task, item.learning_rate) for item in candidates} == {
        (task, learning_rate)
        for task in e8.PILOT_TASKS
        for learning_rate in e8.PILOT_LEARNING_RATES
    }
    invariant = {
        (
            item.loss_mode,
            item.training_config().seed,
            item.training_config().batch_size,
            item.training_config().max_epochs,
            item.training_config().max_train_batches,
            item.training_config().max_validation_batches,
        )
        for item in candidates
    }
    assert invariant == {("unweighted", 2025, 32, 3, 16, 128)}


def test_rejects_unauthorized_learning_rate_and_task():
    with pytest.raises(ValueError, match="learning_rate"):
        e8.PilotCandidate("E1", 5e-4)
    with pytest.raises(ValueError, match="task"):
        e8.PilotCandidate("E4", 1e-4)


def test_candidate_identity_is_deterministic_and_configuration_bound():
    first = e8.PilotCandidate("E2", 3e-4)
    second = e8.PilotCandidate("E2", 3e-4)
    different = e8.PilotCandidate("E2", 1e-3)
    assert first.candidate_id == second.candidate_id
    assert first.identity_sha256 == second.identity_sha256
    assert first.candidate_id != different.candidate_id
    assert first.identity_sha256 != different.identity_sha256


def test_generated_run_id_satisfies_the_path_safety_contract():
    run_id = e8.new_run_id()
    assert e8._validate_run_id(run_id) == run_id


def test_paths_are_unique_fresh_and_never_alias(tmp_path):
    all_paths = []
    for candidate in e8.enumerate_candidates():
        paths = e8.candidate_paths(tmp_path, RUN_ID, candidate)
        assert paths.checkpoint != paths.log != paths.result
        all_paths.extend((paths.checkpoint, paths.log, paths.result))
    assert len(all_paths) == len(set(all_paths))
    assert not any(path.exists() for path in all_paths)


def test_preflight_rejects_path_reuse_before_any_candidate(tmp_path):
    candidate = e8.enumerate_candidates()[0]
    paths = e8.candidate_paths(tmp_path, RUN_ID, candidate)
    paths.log.parent.mkdir(parents=True)
    paths.log.write_text("existing", encoding="utf-8")
    with pytest.raises(e8.OptimizationPilotError, match="fresh output location"):
        e8._preflight_run(tmp_path, RUN_ID, tmp_path / "observations.json")


def test_correct_frozen_validation_objective_per_task():
    assert {
        candidate.task: candidate.selection_name
        for candidate in e8.enumerate_candidates()
    } == e8.EXPECTED_SELECTION_NAMES


def test_task_ranking_is_deterministic_and_uses_approved_lr_tie_order(tmp_path):
    candidates = [
        candidate for candidate in e8.enumerate_candidates() if candidate.task == "E1"
    ]
    records = [_complete_record(tmp_path, candidate, 0.5) for candidate in candidates]
    ranking = e8.rank_task_records(records)
    assert ranking == tuple(candidate.candidate_id for candidate in candidates)


def test_incomplete_result_is_rejected(tmp_path):
    candidate = e8.enumerate_candidates()[0]
    record = _complete_record(tmp_path, candidate)
    record.pop("history")
    with pytest.raises(e8.OptimizationPilotError, match="missing required fields"):
        e8.validate_candidate_record(record, candidate, allow_failed=False)


def test_failed_candidate_is_accounted_and_never_ranked(tmp_path):
    candidates = e8.enumerate_candidates()
    records = [_complete_record(tmp_path, candidate) for candidate in candidates]
    failed = dict(records[0])
    failed["status"] = "FAILED"
    failed["failure_type"] = "TrainingContractError"
    failed["failure_message"] = "nonfinite loss"
    records[0] = failed
    report = e8.build_observations(
        run_id=RUN_ID, executor="test", records=records
    )
    assert report["overall_status"] == "FAIL"
    assert report["task_observations"]["E1"]["status"] == (
        "NOT_RANKED_DUE_TO_FAILURE"
    )


def test_fairness_rejects_unequal_membership_order(tmp_path):
    candidates = e8.enumerate_candidates()
    records = [_complete_record(tmp_path, candidate) for candidate in candidates]
    mutated = dict(records[1])
    mutated_history = [dict(item) for item in mutated["history"]]
    mutated_history[0]["train_order_sha256"] = "f" * 64
    mutated["history"] = mutated_history
    records[1] = mutated
    with pytest.raises(e8.OptimizationPilotError, match="unequal train_order"):
        e8.build_observations(run_id=RUN_ID, executor="test", records=records)


def test_run_candidate_delegates_once_to_e7_without_split_or_test_access(
    tmp_path, monkeypatch
):
    candidate = e8.enumerate_candidates()[0]
    captured = []

    def fake_run_training(config, **kwargs):
        captured.append((config, kwargs))
        paths = e8.candidate_paths(tmp_path, RUN_ID, candidate)
        history = _history(candidate)
        paths.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        paths.log.parent.mkdir(parents=True, exist_ok=True)
        paths.checkpoint.write_bytes(b"checkpoint")
        paths.log.write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in history),
            encoding="utf-8",
        )
        return SimpleNamespace(
            task=candidate.task,
            selection_name=candidate.selection_name,
            history=tuple(history),
            epochs_completed=len(history),
            parameter_update_observed=True,
            best_epoch=len(history),
            best_selection_value=history[-1]["validation_selection_value"],
            stopped_early=False,
            device="cpu",
        )

    monkeypatch.setattr(e8.e7, "run_training", fake_run_training)
    monkeypatch.setattr(
        e8,
        "_verify_checkpoint",
        lambda *args, **kwargs: {
            "parameter_count": e7.EXPECTED_PARAMETERS[candidate.task],
            "model_state_schema_sha256": "d" * 64,
        },
    )
    record = e8.run_candidate(
        candidate,
        project_root=tmp_path,
        run_id=RUN_ID,
        device="cpu",
        executor="test",
    )
    assert record["status"] == "COMPLETE"
    assert record["test_accessed"] is False
    assert record["test_metrics_recorded"] is False
    assert len(captured) == 1
    assert set(captured[0][1]) == {
        "project_root",
        "checkpoint_path",
        "log_path",
        "device",
    }


def test_run_candidate_persists_explicit_failure(tmp_path, monkeypatch):
    candidate = e8.enumerate_candidates()[0]

    def fail(*args, **kwargs):
        raise e7.TrainingContractError("nonfinite gradient")

    monkeypatch.setattr(e8.e7, "run_training", fail)
    record = e8.run_candidate(
        candidate,
        project_root=tmp_path,
        run_id=RUN_ID,
        device="cpu",
        executor="test",
    )
    assert record["status"] == "FAILED"
    assert record["failure_type"] == "TrainingContractError"
    assert "nonfinite gradient" in record["failure_message"]
    path = e8.candidate_paths(tmp_path, RUN_ID, candidate).result
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "FAILED"


def test_aggregate_rejects_missing_candidate_results(tmp_path, monkeypatch):
    monkeypatch.setattr(e8, "require_frozen_contracts", lambda root: {})
    with pytest.raises(e8.OptimizationPilotError, match="incomplete"):
        e8.aggregate_existing_run(
            project_root=tmp_path,
            run_id=RUN_ID,
            output_path=tmp_path / "observations.json",
            executor="test",
        )


def test_frozen_e7_and_model_sources_remain_unchanged():
    assert e8.require_frozen_contracts(PROJECT_ROOT) == e8.FROZEN_E7_SOURCE_SHA256


def test_orchestrator_contains_no_duplicate_training_or_downstream_implementation():
    source = inspect.getsource(e8).lower()
    assert ".backward(" not in source
    assert "adamw(" not in source
    assert "lazygraphdataset(" not in source
    assert "stage_e9" not in source
    assert "evaluate_test" not in source
    assert "prun" not in source


def test_no_model_weights_are_tracked_by_git():
    completed = subprocess.run(
        ["git", "ls-files", "*.pt", "*.pth", "*.ckpt", "*.safetensors"],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    assert completed.stdout.strip() == ""


def test_observations_never_finalize_a_production_configuration(tmp_path):
    records = [
        _complete_record(tmp_path, candidate, objective=index / 100)
        for index, candidate in enumerate(e8.enumerate_candidates())
    ]
    report = e8.build_observations(
        run_id=RUN_ID, executor="test", records=records
    )
    assert report["overall_status"] == "PASS"
    assert report["production_configuration_finalized"] is False
    assert report["test_accessed"] is False
    assert report["test_metrics_recorded"] is False
    assert all(
        item["production_configuration_finalized"] is False
        for item in report["task_observations"].values()
    )


def test_observations_reject_corrupted_ranking_and_reused_path(tmp_path):
    records = [
        _complete_record(tmp_path, candidate, objective=index / 100)
        for index, candidate in enumerate(e8.enumerate_candidates())
    ]
    report = e8.build_observations(
        run_id=RUN_ID, executor="test", records=records
    )
    corrupted_ranking = dict(report)
    corrupted_ranking["task_observations"] = {
        key: dict(value) for key, value in report["task_observations"].items()
    }
    corrupted_ranking["task_observations"]["E1"][
        "strongest_bounded_observation"
    ] = "wrong"
    with pytest.raises(e8.OptimizationPilotError, match="ranking"):
        e8.validate_observations(corrupted_ranking)

    corrupted_records = [dict(record) for record in records]
    corrupted_records[1]["checkpoint_path"] = corrupted_records[0][
        "checkpoint_path"
    ]
    corrupted_paths = dict(report)
    corrupted_paths["candidate_results"] = corrupted_records
    with pytest.raises(e8.OptimizationPilotError, match="stored checkpoint_path"):
        e8.validate_observations(corrupted_paths)
