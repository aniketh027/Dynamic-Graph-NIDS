from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from src.graph import schema as d2
from src.training import full_training as e10


STUDY = "e10-official-20261001-v1"


def _candidate(
    *,
    family: str = "htgnn",
    task: str = "E2",
    lr: float = 1e-4,
    seed: int = 2025,
    phase: str = "lr_selection",
) -> e10.FullTrainingCandidate:
    return e10.FullTrainingCandidate(family, task, lr, seed, phase)


def _record(candidate: e10.FullTrainingCandidate, metric: float) -> dict:
    record = {
        "stage": e10.STAGE,
        "version": e10.FULL_TRAINING_VERSION,
        "official_study_id": STUDY,
        "run_class": e10.RUN_CLASS_OFFICIAL,
        "execution_authority": e10.EXECUTION_AUTHORITY,
        "candidate_id": candidate.candidate_id,
        "phase": candidate.phase,
        "configuration": candidate.config_dict(),
        "config_sha256": candidate.config_sha256,
        "status": "PASS",
        "best_validation_metric": metric,
        "best_validation_loss": 0.5,
        "best_epoch_training_loss": 0.6,
        "best_epoch": 1,
        "epochs_completed": 1,
        "history": [{}],
        "test_accessed": False,
        "test_metrics_recorded": False,
        "threshold_selected": False,
        "pruning_performed": False,
    }
    record["result_sha256"] = d2.canonical_sha256(record)
    return record


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(d2.canonical_json_bytes(payload) + b"\n")


def _provenance_fixture() -> dict:
    """Synthetic metadata with the reviewed execution-source identities."""

    return {
        "stage": e10.STAGE,
        "version": e10.FULL_TRAINING_VERSION,
        "official_study_id": STUDY,
        "source_git_identity": {
            "head": e10.FROZEN_START_COMMIT,
            "origin_main": e10.FROZEN_START_COMMIT,
            "tags_at_head": [e10.FROZEN_START_TAG],
            "dirty": True,
            "status_short": ["?? src/training/full_training.py"],
        },
        "environment_identity": {
            "python": "3.13.0",
            "python_implementation": "CPython",
            "torch": "2.14.0+cu130",
            "torch_geometric": "2.8.0.post1",
            "cuda_runtime": "13.0",
            "cuda_available": True,
            "gpu": "NVIDIA GeForce RTX 4050 Laptop GPU",
            "gpu_total_memory_bytes": 6438780928,
            "platform": "Windows-11-10.0.26300-SP0",
        },
        "stage_c_provenance": {
            "integrity": {"expected": 33, "matched": 33, "status": "PASS"},
            "split_artifact_sha256": {
                f"exp{index}/split_metadata.json": str(index) * 64
                for index in (1, 2, 3)
            },
        },
        "e10_source_sha256": {
            "src/training/full_training.py":
                e10.provenance_policy.HISTORICAL_RUNTIME_SHA256[1],
            "scripts/run_stage_e10_full_training.py":
                e10.provenance_policy.APPROVED_RUNNER_SHA256,
            "tests/test_stage_e10_full_training.py": "a" * 64,
            "docs/stage_e10_full_training.md": "a" * 64,
        },
        "upstream_evidence_sha256": {
            relative: "b" * 64 for relative in e10.UPSTREAM_EVIDENCE_PATHS
        },
        "preflight_sha256": "c" * 64,
    }


def _write_complete_result(
    root: Path, candidate: e10.FullTrainingCandidate, metric: float,
    *, provenance: dict | None = None,
) -> dict:
    paths = e10.candidate_paths(root, STUDY, candidate)
    paths.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    paths.checkpoint.write_bytes(b"checkpoint")
    paths.log.parent.mkdir(parents=True, exist_ok=True)
    paths.log.write_text("{}\n", encoding="utf-8")
    _write_json(paths.training_history, {"history": [{}]})
    _write_json(paths.validation_history, {"history": [{}]})
    manifest = {
        **_provenance_fixture(),
        "stage": e10.STAGE,
        "version": e10.FULL_TRAINING_VERSION,
        "official_study_id": STUDY,
        "run_class": e10.RUN_CLASS_OFFICIAL,
        "execution_authority": e10.EXECUTION_AUTHORITY,
        "phase": candidate.phase,
        "candidate_id": candidate.candidate_id,
        "configuration": candidate.config_dict(),
        "config_sha256": candidate.config_sha256,
    }
    manifest.update(provenance or {})
    manifest["manifest_sha256"] = d2.canonical_sha256(manifest)
    _write_json(paths.manifest, manifest)
    record = _record(candidate, metric)
    record["artifacts"] = {
        "checkpoint_sha256": e10._sha256(paths.checkpoint),
        "log_sha256": e10._sha256(paths.log),
        "manifest_sha256": e10._sha256(paths.manifest),
        "training_history_sha256": e10._sha256(paths.training_history),
        "validation_history_sha256": e10._sha256(paths.validation_history),
    }
    record.pop("result_sha256")
    record["result_sha256"] = d2.canonical_sha256(record)
    _write_json(paths.result, record)
    status = e10._status_payload(
        STUDY,
        candidate,
        "COMPLETE",
        epochs_completed=1,
        paths=paths,
    )
    _write_json(paths.status, status)
    return record


def _write_selection_plan(root: Path, records: list[dict]) -> Path:
    selected = e10.select_learning_rates(records)
    plan = e10._selection_plan(STUDY, selected)
    path = e10.report_paths(root, STUDY).selected_learning_rates
    _write_json(path, plan)
    return path


def test_official_matrix_is_exact_deterministic_and_nonduplicative() -> None:
    first = e10.enumerate_lr_selection_candidates()
    second = e10.enumerate_lr_selection_candidates()
    assert first == second
    assert len(first) == 27
    assert len({candidate.candidate_id for candidate in first}) == 27
    assert {candidate.task for candidate in first} == {"E1", "E2", "E3"}
    assert set(candidate.model_family for candidate in first) == set(e10.MODEL_FAMILIES)
    assert {candidate.learning_rate for candidate in first} == {1e-4, 3e-4, 1e-3}
    assert {candidate.seed for candidate in first} == {2025}
    assert all(candidate.training_config.batch_size == 256 for candidate in first)
    assert all(candidate.training_config.max_epochs == 30 for candidate in first)


def test_configuration_is_unbounded_train_validation_only() -> None:
    config = _candidate().config_dict()
    assert config["allowed_partitions"] == ["train", "validation"]
    assert config["max_train_batches"] is None
    assert config["max_validation_batches"] is None
    assert config["class_weighting"] is False
    assert config["test_predictive_array_access"] is False
    assert config["pruning"] is False
    assert "threshold" not in config


def test_official_identity_rejects_accidental_and_generic_run_ids() -> None:
    assert e10.validate_official_study_id(STUDY) == STUDY
    with pytest.raises(e10.FullTrainingError, match="pre-authorization"):
        e10.validate_official_study_id("e10-20261001-fulltrain")
    with pytest.raises(ValueError, match="official_study_id"):
        e10.validate_official_study_id("e10-another-generic-run")


def test_paths_isolate_phase_task_family_candidate_and_status(tmp_path: Path) -> None:
    first = e10.candidate_paths(tmp_path, STUDY, _candidate())
    second = e10.candidate_paths(
        tmp_path, STUDY, _candidate(family="final_flow_mlp")
    )
    assert first.checkpoint != second.checkpoint
    assert "lr_selection" in first.checkpoint.parts
    assert "e2" in first.checkpoint.parts
    assert "htgnn" in first.checkpoint.parts
    assert first.status.name == "status.json"


@pytest.mark.parametrize("name", ("X_test.npy", "y_test.npy"))
def test_predictive_test_lockout_fails_closed_for_features_and_labels(
    tmp_path: Path, name: str
) -> None:
    forbidden = (
        tmp_path / "data" / "processed" / "preprocessed" / "exp2" / name
    )
    forbidden.parent.mkdir(parents=True)
    forbidden.write_bytes(b"blocked")
    with pytest.raises(e10.FullTrainingError, match="test-array access blocked"):
        with e10.predictive_test_lockout(tmp_path) as accesses:
            forbidden.read_bytes()
    assert accesses == [str(forbidden.resolve())]


def test_lockout_allows_train_and_validation_arrays(tmp_path: Path) -> None:
    train = tmp_path / "data/processed/preprocessed/exp1/X_train.npy"
    validation = train.with_name("y_val.npy")
    train.parent.mkdir(parents=True)
    train.write_bytes(b"train")
    validation.write_bytes(b"validation")
    with e10.predictive_test_lockout(tmp_path) as accesses:
        assert train.read_bytes() == b"train"
        assert validation.read_bytes() == b"validation"
    assert accesses == []


def test_dry_run_does_not_preflight_create_model_or_train(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run crossed into an execution surface")

    monkeypatch.setattr(e10, "preflight", forbidden)
    monkeypatch.setattr(e10.e7, "create_model", forbidden)
    monkeypatch.setattr(e10.e7, "run_training", forbidden)
    jobs = e10.dry_run_jobs(tmp_path, STUDY, phase="lr_selection")
    assert len(jobs) == 27
    assert all(job["batch_size"] == 256 for job in jobs)
    assert all(job["official_study_id"] == STUDY for job in jobs)


def test_single_job_resolution_enforces_phase_seed_and_selected_lr(
    tmp_path: Path,
) -> None:
    records = [
        _record(candidate, {1e-4: 0.8, 3e-4: 0.7, 1e-3: 0.6}[candidate.learning_rate])
        for candidate in e10.enumerate_lr_selection_candidates()
    ]
    plan = _write_selection_plan(tmp_path, records)
    candidate = e10.resolve_single_job(
        tmp_path,
        STUDY,
        phase="final_seed",
        experiment="E1",
        family="htgnn",
        learning_rate=1e-4,
        seed=2026,
        selection_plan_path=plan,
    )
    assert candidate.phase == "final_seed"
    with pytest.raises(e10.FullTrainingError, match="does not match"):
        e10.resolve_single_job(
            tmp_path,
            STUDY,
            phase="final_seed",
            experiment="E1",
            family="htgnn",
            learning_rate=3e-4,
            seed=2026,
            selection_plan_path=plan,
        )
    with pytest.raises(ValueError, match="reused"):
        _candidate(seed=2025, phase="final_seed")

    final_candidates = e10.enumerate_final_seed_candidates_from_plan(
        tmp_path, STUDY, plan
    )
    assert len(final_candidates) == 18
    assert {item.seed for item in final_candidates} == {2026, 2027}
    assert len({item.candidate_id for item in final_candidates}) == 18


def test_owner_confirmation_and_fresh_paths_prevent_automatic_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = _candidate()
    with pytest.raises(e10.FullTrainingError, match="owner-only"):
        e10.execute_candidate(
            tmp_path, STUDY, candidate, preflight_result={}, device="cpu"
        )
    paths = e10.candidate_paths(tmp_path, STUDY, candidate)
    paths.status.parent.mkdir(parents=True)
    paths.status.write_text("preserve", encoding="utf-8")
    monkeypatch.setattr(
        e10, "_manifest", lambda *args, **kwargs: pytest.fail("must not create manifest")
    )
    with pytest.raises(e10.FullTrainingError, match="not fresh"):
        e10.execute_candidate(
            tmp_path,
            STUDY,
            candidate,
            preflight_result={},
            device="cpu",
            owner_confirmed_manual_execution=True,
        )
    assert paths.status.read_text(encoding="utf-8") == "preserve"


class _NoopMonitor:
    peak_rss_bytes = 1

    def __init__(self, path: Path, candidate: e10.FullTrainingCandidate):
        self.path = path
        self.candidate = candidate

    def start(self) -> None:
        return None

    def finish(self) -> tuple[float, ...]:
        return ()


def test_complete_epoch_log_record_prints_live_progress_only_after_newline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    log = tmp_path / "epochs.jsonl"
    candidate = _candidate(task="E1", family="htgnn", lr=1e-3)
    record = {
        "epoch": 3,
        "train_loss": 0.00821,
        "validation_loss": 2.01858,
        "validation_selection_name": "support_restricted_macro_f1",
        "validation_selection_value": 0.663095,
        "best_checkpoint_updated": True,
    }
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"))
    times = iter((100.0, 102.25))
    monkeypatch.setattr(e10.time, "perf_counter", lambda: next(times))
    monitor = e10._RuntimeMonitor(log, candidate)

    log.write_text(encoded, encoding="utf-8")
    monitor._sample()
    assert capsys.readouterr().out == ""

    log.write_text(encoded + "\n", encoding="utf-8")
    monitor._sample()
    assert capsys.readouterr().out == (
        "[E1][htgnn][lr=0.001][seed=2025] Epoch 3/30 | "
        "train_loss=0.008210 | val_loss=2.018580 | "
        "support_restricted_macro_f1=0.663095 | best=0.663095 | "
        "best_updated=yes | patience=0/5 | epoch_elapsed=2.250s\n"
    )
    assert monitor.epoch_completed_seconds == [2.25]


def test_informational_output_flushes_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(
        "builtins.print",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    e10._print_informational("progress")
    assert calls == [(("progress",), {"flush": True})]


def test_stdout_only_source_change_keeps_completed_run_semantics_compatible(
    tmp_path: Path,
) -> None:
    first = _candidate(task="E1", lr=1e-4)
    second = _candidate(task="E1", lr=3e-4)
    common = _provenance_fixture()
    earlier_sources = dict(common["e10_source_sha256"])
    earlier_sources["src/training/full_training.py"] = (
        e10.provenance_policy.HISTORICAL_RUNTIME_SHA256[0]
    )
    _write_json(
        e10.candidate_paths(tmp_path, STUDY, first).manifest,
        {
            **common,
            "e10_source_sha256": earlier_sources,
            "preflight_sha256": "2" * 64,
        },
    )
    _write_json(
        e10.candidate_paths(tmp_path, STUDY, second).manifest,
        {
            **common,
            "e10_source_sha256": common["e10_source_sha256"],
            "preflight_sha256": "4" * 64,
        },
    )
    assert e10._candidate_provenance_identity(
        tmp_path, STUDY, first
    ) == e10._candidate_provenance_identity(tmp_path, STUDY, second)
    _write_json(
        e10.candidate_paths(tmp_path, STUDY, second).manifest,
        {
            **common,
            "environment_identity": {**common["environment_identity"], "torch": "changed"},
            "e10_source_sha256": common["e10_source_sha256"],
            "preflight_sha256": "4" * 64,
        },
    )
    assert e10._candidate_provenance_identity(
        tmp_path, STUDY, first
    ) != e10._candidate_provenance_identity(tmp_path, STUDY, second)


def test_keyboard_interrupt_marks_partial_run_ineligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = _candidate()
    monkeypatch.setattr(
        e10,
        "_manifest",
        lambda *args, **kwargs: {"manifest_sha256": "a" * 64},
    )
    monkeypatch.setattr(e10, "_RuntimeMonitor", _NoopMonitor)
    monkeypatch.setattr(
        e10.e7,
        "run_training",
        lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    with pytest.raises(KeyboardInterrupt):
        e10.execute_candidate(
            tmp_path,
            STUDY,
            candidate,
            preflight_result={},
            device="cpu",
            owner_confirmed_manual_execution=True,
        )
    paths = e10.candidate_paths(tmp_path, STUDY, candidate)
    assert e10._read_json(paths.status)["state"] == "INTERRUPTED"
    assert not paths.result.exists()
    with pytest.raises(e10.FullTrainingError, match="missing"):
        e10.load_completed_results(tmp_path, STUDY, [candidate])


def test_mocked_test_access_marks_run_failed_without_real_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = _candidate()
    forbidden = (
        tmp_path / "data/processed/preprocessed/exp2/X_test.npy"
    )
    forbidden.parent.mkdir(parents=True)
    forbidden.write_bytes(b"blocked")
    monkeypatch.setattr(
        e10,
        "_manifest",
        lambda *args, **kwargs: {"manifest_sha256": "a" * 64},
    )
    monkeypatch.setattr(e10, "_RuntimeMonitor", _NoopMonitor)

    def fake_run(*args, **kwargs):
        forbidden.read_bytes()

    monkeypatch.setattr(e10.e7, "run_training", fake_run)
    with pytest.raises(e10.FullTrainingError, match="test-array access blocked"):
        e10.execute_candidate(
            tmp_path,
            STUDY,
            candidate,
            preflight_result={},
            device="cpu",
            owner_confirmed_manual_execution=True,
        )
    status = e10._read_json(e10.candidate_paths(tmp_path, STUDY, candidate).status)
    assert status["state"] == "FAILED"


def test_validation_only_lr_selection_uses_metric_then_frozen_lr_order() -> None:
    records = []
    for candidate in e10.enumerate_lr_selection_candidates():
        metric = {1e-4: 0.5, 3e-4: 0.5, 1e-3: 0.4}[candidate.learning_rate]
        records.append(_record(candidate, metric))
    selected = e10.select_learning_rates(records)
    assert len(selected) == 9
    assert all(
        record["configuration"]["learning_rate"] == 1e-4
        for record in selected.values()
    )
    assert all(record["test_accessed"] is False for record in selected.values())


def test_result_only_aggregation_reads_exact_complete_set_and_never_trains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("aggregation attempted execution")

    monkeypatch.setattr(e10, "preflight", forbidden)
    monkeypatch.setattr(e10.e7, "create_model", forbidden)
    monkeypatch.setattr(e10.e7, "run_training", forbidden)
    for candidate in e10.enumerate_lr_selection_candidates():
        metric = {1e-4: 0.9, 3e-4: 0.8, 1e-3: 0.7}[candidate.learning_rate]
        _write_complete_result(tmp_path, candidate, metric)
    summary = e10.aggregate_lr_selection(tmp_path, STUDY)
    assert summary["candidate_results_read"] == 27
    assert summary["training_executed"] is False
    plan = e10._read_json(e10.report_paths(tmp_path, STUDY).selected_learning_rates)
    assert len(plan["selected"]) == 9
    assert {entry["learning_rate"] for entry in plan["selected"]} == {1e-4}
    for candidate in e10.enumerate_final_seed_candidates_from_plan(tmp_path, STUDY):
        _write_complete_result(tmp_path, candidate, 0.85)
    final = e10.aggregate_final_report(tmp_path, STUDY)
    assert final["unique_full_runs"] == 45
    assert final["training_executed"] is False
    assert final["test_predictive_array_access"] is False


def test_partial_status_is_rejected_even_when_result_exists(tmp_path: Path) -> None:
    candidate = _candidate()
    _write_complete_result(tmp_path, candidate, 0.5)
    paths = e10.candidate_paths(tmp_path, STUDY, candidate)
    status = e10._status_payload(
        STUDY, candidate, "INTERRUPTED", epochs_completed=1, paths=paths
    )
    _write_json(paths.status, status)
    with pytest.raises(e10.FullTrainingError, match="not COMPLETE"):
        e10.load_completed_results(tmp_path, STUDY, [candidate])


def test_bulk_training_entrypoint_is_disabled(tmp_path: Path) -> None:
    assert not hasattr(e10, "run_full_study")


def _compatibility_hash(manifest: dict, study: str = STUDY) -> str:
    return d2.canonical_sha256(e10._provenance_compatibility_payload(manifest, study))


def test_six_audited_groups_are_compatible_without_mutating_metadata(tmp_path: Path):
    base_status = [
        "docs/stage_e10_full_training.md", "scripts/run_stage_e10_full_training.py",
        "src/training/full_training.py", "tests/test_stage_e10_full_training.py",
    ]
    additions = [[], [], [
        "e10-overnight-20261002-042253/", "e10-overnight-20261002-042528/",
        "e10-phase-a-overnight-20261002-043636/",
        "e10-verification-20261002-042253.zip", "e10-verification-20261002-042253/",
        "e10-verification-20261002-042528.zip", "e10-verification-20261002-042528/",
        "run_e10_phase_a_overnight.ps1",
    ], [
        "e10-interrupted-attempt-20261003-140737/",
        "e10-phase-a-resume-20261003-140507/", "e10-phase-a-resume-20261003-140810/",
        "e10-phase-a-resume-verification-20261003-140507.zip",
        "run_e10_phase_a_resume.ps1",
    ], [
        "e10-phase-a-final-resume-20261003-183405/",
        "e10-phase-a-resume-verification-20261003-140810.zip",
        "run_e10_phase_a_resume_after_job1.ps1",
    ], [
        "e10-phase-a-final-resume-verification-20261003-183405.zip",
        "e10-phase-a-final-two-20261003-190112/", "run_e10_phase_a_final_two.ps1",
    ]]
    groups = []
    for index, extra in enumerate(additions):
        base_status.extend(extra)
        manifest = _provenance_fixture()
        manifest["source_git_identity"]["status_short"] = [
            "?? " + path for path in sorted(base_status)
        ]
        if index == 0:
            manifest["environment_identity"]["platform"] = (
                "Windows-11-10.0.26200-SP0"
            )
        groups.append(manifest)
    assert [len(m["source_git_identity"]["status_short"]) for m in groups] == [4, 4, 12, 17, 20, 23]
    before = deepcopy(groups)
    assert len({_compatibility_hash(m) for m in groups}) == 1
    assert groups == before
    counts = [0] * 6
    candidates = e10.enumerate_lr_selection_candidates()
    for candidate in candidates:
        if candidate.model_family == "htgnn":
            index = 0 if candidate.task == "E1" else 1
        elif candidate.task != "E3" or (
            candidate.model_family == "final_flow_mlp" and candidate.learning_rate != 1e-3
        ):
            index = 2
        elif candidate.model_family == "final_flow_mlp":
            index = 3
        else:
            index = 4 if candidate.learning_rate == 1e-4 else 5
        counts[index] += 1
        provenance = deepcopy(groups[index])
        if index == 0 and candidate.learning_rate != 1e-3:
            provenance["e10_source_sha256"]["src/training/full_training.py"] = (
                e10.provenance_policy.HISTORICAL_RUNTIME_SHA256[0]
            )
        _write_complete_result(tmp_path, candidate, 0.8, provenance=provenance)
    assert counts == [3, 6, 14, 1, 1, 2]
    before_files = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert len(e10.load_completed_results(tmp_path, STUDY, candidates)) == 27
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before_files


def test_git_status_is_informational_but_stable_git_fields_are_not():
    first = _provenance_fixture()
    second = deepcopy(first)
    second["source_git_identity"]["status_short"] = ["?? another-wrapper.ps1"]
    assert _compatibility_hash(first) == _compatibility_hash(second)
    second["source_git_identity"]["dirty"] = False
    assert _compatibility_hash(first) != _compatibility_hash(second)
    second["source_git_identity"]["dirty"] = True
    second["source_git_identity"]["tags_at_head"].append("another-tag")
    assert _compatibility_hash(first) != _compatibility_hash(second)


@pytest.mark.parametrize("study,compatible", [(STUDY, True), ("e10-official-20261002-v1", False)])
def test_windows_build_exception_is_scoped_to_exact_study(study, compatible):
    first = _provenance_fixture()
    first["official_study_id"] = study
    second = deepcopy(first)
    second["environment_identity"]["platform"] = "Windows-11-10.0.26200-SP0"
    assert (_compatibility_hash(first, study) == _compatibility_hash(second, study)) is compatible


@pytest.mark.parametrize("study", [STUDY, "e10-official-20261002-v1"])
@pytest.mark.parametrize("platform", ["Windows-11-10.0.26400-SP0", "Linux", None])
def test_unknown_platform_is_not_implicitly_approved(study, platform):
    manifest = _provenance_fixture()
    manifest["official_study_id"] = study
    manifest["environment_identity"]["platform"] = platform
    with pytest.raises(e10.FullTrainingError, match="environment/platform"):
        _compatibility_hash(manifest, study)


@pytest.mark.parametrize("field,value", [
    ("python", "3.14.0"), ("python_implementation", "PyPy"),
    ("torch", "2.15.0+cu130"), ("torch_geometric", "2.9.0"),
    ("cuda_runtime", "13.1"), ("cuda_available", False),
    ("gpu", "another GPU"), ("gpu_total_memory_bytes", 1),
    ("extra_environment_field", "retained-and-compared"),
])
def test_environment_changes_fail_cross_candidate_loading(tmp_path: Path, field, value):
    candidates = [_candidate(lr=1e-4), _candidate(lr=3e-4)]
    _write_complete_result(tmp_path, candidates[0], 0.8)
    changed = _provenance_fixture()
    changed["environment_identity"][field] = value
    _write_complete_result(tmp_path, candidates[1], 0.8, provenance=changed)
    with pytest.raises(e10.FullTrainingError, match="one provenance identity"):
        e10.load_completed_results(tmp_path, STUDY, candidates)


@pytest.mark.parametrize("field", ["stage_c", "split", "upstream", "dirty", "tags"])
def test_methodology_provenance_differences_still_fail(tmp_path: Path, field):
    candidates = [_candidate(lr=1e-4), _candidate(lr=3e-4)]
    _write_complete_result(tmp_path, candidates[0], 0.8)
    changed = _provenance_fixture()
    if field == "stage_c":
        changed["stage_c_provenance"]["integrity"]["matched"] = 32
    elif field == "split":
        changed["stage_c_provenance"]["split_artifact_sha256"]["exp2/split_metadata.json"] = "d" * 64
    elif field == "upstream":
        changed["upstream_evidence_sha256"][e10.UPSTREAM_EVIDENCE_PATHS[0]] = "d" * 64
    elif field == "dirty":
        changed["source_git_identity"]["dirty"] = False
    else:
        changed["source_git_identity"]["tags_at_head"].append("different-tag")
    _write_complete_result(tmp_path, candidates[1], 0.8, provenance=changed)
    with pytest.raises(e10.FullTrainingError, match="one provenance identity"):
        e10.load_completed_results(tmp_path, STUDY, candidates)


@pytest.mark.parametrize("field", ["head", "origin_main"])
def test_changed_git_revision_is_rejected_even_for_single_result(tmp_path: Path, field):
    candidate = _candidate()
    changed = _provenance_fixture()
    changed["source_git_identity"][field] = "d" * 40
    _write_complete_result(tmp_path, candidate, 0.8, provenance=changed)
    with pytest.raises(e10.FullTrainingError, match="provenance"):
        e10.load_completed_results(tmp_path, STUDY, [candidate])


@pytest.mark.parametrize("relative", [
    "src/training/full_training.py", "scripts/run_stage_e10_full_training.py",
])
def test_unreviewed_execution_source_is_rejected(tmp_path: Path, relative):
    candidate = _candidate()
    changed = _provenance_fixture()
    changed["e10_source_sha256"][relative] = "f" * 64
    _write_complete_result(tmp_path, candidate, 0.8, provenance=changed)
    with pytest.raises(e10.FullTrainingError, match="unapproved E.10 training source"):
        e10.load_completed_results(tmp_path, STUDY, [candidate])


@pytest.mark.parametrize("field", ["stage", "version", "official_study_id"])
def test_stage_version_study_cannot_be_normalized(field):
    changed = _provenance_fixture()
    changed[field] = "changed"
    with pytest.raises(e10.FullTrainingError, match="stage/version/study"):
        _compatibility_hash(changed)


def test_incomplete_environment_and_git_metadata_fail_closed():
    changed = _provenance_fixture()
    del changed["environment_identity"]["torch"]
    with pytest.raises(e10.FullTrainingError, match="environment/platform"):
        _compatibility_hash(changed)
    changed = _provenance_fixture()
    del changed["source_git_identity"]["dirty"]
    with pytest.raises(e10.FullTrainingError, match="stable Git"):
        _compatibility_hash(changed)


def test_archived_interrupted_attempt_never_supplies_official_result(tmp_path: Path):
    candidate = _candidate(family="final_flow_mlp", task="E3", lr=1e-3)
    archive_root = tmp_path / "e10-interrupted-attempt-20261003-140737"
    paths = e10.candidate_paths(archive_root, STUDY, candidate)
    _write_json(paths.manifest, _provenance_fixture())
    _write_json(paths.status, {"state": "INTERRUPTED", "candidate_id": candidate.candidate_id})
    with pytest.raises(e10.FullTrainingError, match="missing"):
        e10.load_completed_results(tmp_path, STUDY, [candidate])
    _write_complete_result(tmp_path, candidate, 0.8)
    assert len(e10.load_completed_results(tmp_path, STUDY, [candidate])) == 1
    assert not paths.result.exists()
    assert e10._read_json(paths.status)["state"] == "INTERRUPTED"


@pytest.mark.parametrize("approved", [True, False])
def test_phase_c_source_change_requires_reviewed_semantics(tmp_path: Path, monkeypatch, approved):
    def forbidden(*args, **kwargs):
        raise AssertionError("synthetic aggregation must not execute models or training")

    monkeypatch.setattr(e10, "preflight", forbidden)
    monkeypatch.setattr(e10.e7, "create_model", forbidden)
    monkeypatch.setattr(e10.e7, "run_training", forbidden)
    records = []
    for candidate in e10.enumerate_lr_selection_candidates():
        provenance = _provenance_fixture()
        provenance["environment_identity"]["platform"] = "Windows-11-10.0.26200-SP0"
        provenance["e10_source_sha256"]["src/training/full_training.py"] = (
            e10.provenance_policy.HISTORICAL_RUNTIME_SHA256[0]
        )
        records.append(_write_complete_result(tmp_path, candidate, 0.8, provenance=provenance))
    # Synthetic selection plan only; no official aggregation or data access.
    _write_selection_plan(tmp_path, records)
    for candidate in e10.enumerate_final_seed_candidates_from_plan(tmp_path, STUDY):
        provenance = _provenance_fixture()
        provenance["source_git_identity"]["status_short"].append("?? phase-c-wrapper.ps1")
        provenance["e10_source_sha256"]["src/training/full_training.py"] = (
            e10.provenance_policy.CORRECTED_RUNTIME_SHA256 if approved else "f" * 64
        )
        _write_complete_result(tmp_path, candidate, 0.85, provenance=provenance)
    if approved:
        summary = e10.aggregate_final_report(tmp_path, STUDY)
        assert summary["unique_full_runs"] == 45
        assert summary["training_executed"] is False
    else:
        with pytest.raises(e10.FullTrainingError, match="unapproved E.10 training source"):
            e10.aggregate_final_report(tmp_path, STUDY)
        assert not e10.report_paths(tmp_path, STUDY).final_report.exists()


def test_corrected_runtime_digest_is_pinned_and_not_automatically_trusted():
    root = Path(e10.__file__).resolve().parents[2]
    hashes = e10._e10_source_hashes(root)
    assert hashes["src/training/full_training.py"] == e10.provenance_policy.CORRECTED_RUNTIME_SHA256
    manifest = _provenance_fixture()
    manifest["e10_source_sha256"] = hashes
    assert _compatibility_hash(manifest) == _compatibility_hash(_provenance_fixture())
