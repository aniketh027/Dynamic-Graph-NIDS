"""Read-only recovery audit and synthetic continuation tests; no real arrays."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.evaluation import official_test as e11, e11_failure_recovery as recovery
from src.training import full_training as e10
from test_stage_e11_test_evaluation import _metadata


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(e11.d2.canonical_json_bytes(value) + b"\n")


def _fixture(root, monkeypatch):
    for relative in recovery.RECOVERY_SOURCES:
        p = root / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"synthetic source")
    records = []
    for family in e10.MODEL_FAMILIES:
        for task in e10.TASKS:
            lr = 1e-3 if family != "flattened_five_flow_mlp" else (3e-4 if task == "E2" else 1e-4)
            for seed in e10.FINAL_SEEDS:
                c = e10.FullTrainingCandidate(family, task, lr, seed, "lr_selection" if seed == 2025 else "final_seed")
                records.append({"candidate_id": c.candidate_id, "configuration": c.config_dict(),
                                "artifacts": {"checkpoint_sha256": "a" * 64},
                                "checkpoint_provenance": {"synthetic": True}, "result_sha256": "b" * 64})
    splits = {}
    for task, directory in e11.d4.EXPERIMENT_DIRS.items():
        path = root / "data/processed/preprocessed" / directory / "X_test.npy"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"synthetic predictive stand-in")
        splits[task] = {"policy": e11.reporting_policy(_metadata(task), task),
                        "predictive_artifact_sha256": {f"{directory}/X_test.npy": e10._sha256(path)}}
    preparation = {"evaluation_records": records, "splits": splits, "e10_artifact_sha256": {},
                   "frozen_metadata_sha256": {}}
    source_hashes = {relative: e10._sha256(root / relative) for relative in e11.SOURCE_PATHS}
    marker_path = root / f"logs/stage_e11/{e11.STUDY}/test_consumption.json"
    log_dir = root / f"logs/stage_e11/{e11.STUDY}/{recovery.FAILED_RUN}"
    output_dir = root / f"results/stage_e11/{e11.STUDY}/{recovery.FAILED_RUN}"
    _write(marker_path, {"stage": "E.11", "e10_study": e11.STUDY, "run_id": recovery.FAILED_RUN,
                         "state": "CONSUMED_OR_RESERVED", "source_sha256": source_hashes,
                         "e10_report_sha256": e11.E10_REPORT_SHA256})
    manifest = {"source_sha256": source_hashes, "preparation": preparation,
                "environment": {}, "metric_library_versions": {"numpy": "fixture", "scikit_learn": "fixture"}}
    manifest["manifest_sha256"] = e11.d2.canonical_sha256(manifest)
    _write(log_dir / "manifest.json", manifest)
    _write(log_dir / "status.json", {"completed_checkpoints": 6, "error": "invalid frozen output shape or nonfinite logits",
                                    "error_type": "EvaluationError", "state": "FAILED"})
    paths = [marker_path, log_dir / "manifest.json", log_dir / "status.json"]
    for record in records[:6]:
        cfg = record["configuration"]
        policy = splits[cfg["task"]]["policy"]
        n = sum(policy["test_support"])
        result = {"candidate_id": record["candidate_id"], "checkpoint_sha256": record["artifacts"]["checkpoint_sha256"],
                  "checkpoint_provenance": record["checkpoint_provenance"], "e10_result_canonical_sha256": record["result_sha256"],
                  "model_state_unchanged": True, "examples": n,
                  "order_sha256": hashlib.sha256(np.arange(n, dtype="<i8").tobytes()).hexdigest(),
                  **{k: cfg[k] for k in ("task", "model_family", "seed", "learning_rate")},
                  "metrics": e11.multiclass_metrics(np.diag(policy["test_support"]).astype(np.int64), policy)}
        result["result_sha256"] = e11.d2.canonical_sha256(result)
        path = output_dir / (record["candidate_id"] + ".json")
        _write(path, result)
        paths.append(path)
    originals = {p.relative_to(root).as_posix(): e10._sha256(p) for p in paths}
    monkeypatch.setattr(recovery, "FAILED_FILES", originals)
    monkeypatch.setattr(recovery, "CORRECTED_EVALUATOR_SHA256", e10._sha256(root / "src/evaluation/official_test.py"))
    monkeypatch.setattr(e11, "audit_e10", lambda *args: deepcopy(preparation))
    monkeypatch.setattr(e10, "environment_identity", lambda: {})
    monkeypatch.setattr(e11, "package_version", lambda name: "fixture")
    monkeypatch.setattr(e11.e7, "resolve_device", lambda *args: torch.device("cpu"))
    monkeypatch.setattr(e11, "run_owner_evaluation", lambda *args, **kwargs: pytest.fail("original evaluate must remain blocked"))
    calls = []
    def evaluate(root, record, policy, scratch, device):
        assert (scratch / "recovery_authorization.json").is_file()
        assert e10._sha256(marker_path) == originals[marker_path.relative_to(root).as_posix()]
        task = record["configuration"]["task"]
        (root / "data/processed/preprocessed" / e11.d4.EXPERIMENT_DIRS[task] / "X_test.npy").read_bytes()
        calls.append(record["candidate_id"])
        return {"candidate_id": record["candidate_id"],
                **{k: record["configuration"][k] for k in ("task", "model_family", "seed", "learning_rate")},
                "metrics": {"pr_auc_average_precision": .5} if task == "E3" else {"primary": {"macro_f1": .5}}}
    monkeypatch.setattr(e11, "evaluate_checkpoint", evaluate)
    return preparation, originals, calls, paths


def test_failure_audit_is_read_only_and_validates_complete_prefix(tmp_path, monkeypatch):
    preparation, originals, calls, paths = _fixture(tmp_path, monkeypatch)
    before = {p.relative_to(tmp_path).as_posix(): e10._sha256(p) for p in tmp_path.rglob("*") if p.is_file()}
    result = recovery.audit_failure(tmp_path)
    after = {p.relative_to(tmp_path).as_posix(): e10._sha256(p) for p in tmp_path.rglob("*") if p.is_file()}
    assert result["audit_status"] == "PASS" and result["start_checkpoint"] == 7
    assert len(result["completed_results"]) == 6
    assert result["test_predictive_arrays_accessed"] is False
    assert calls == [] and before == after


@pytest.mark.parametrize("which", (0, 1, 2, 3, 8))
def test_changed_original_marker_manifest_status_or_result_blocks_recovery_audit(tmp_path, monkeypatch, which):
    preparation, originals, calls, paths = _fixture(tmp_path, monkeypatch)
    with paths[which].open("ab") as f:
        f.write(b" ")
    with pytest.raises(e11.EvaluationError, match="input changed"):
        recovery.audit_failure(tmp_path)
    assert calls == []


@pytest.mark.parametrize("fault", ("extra_result", "scratch", "evaluator", "upstream", "authority"))
def test_recovery_rejects_extra_partial_state_and_changed_source_or_split_authority(tmp_path, monkeypatch, fault):
    preparation, originals, calls, paths = _fixture(tmp_path, monkeypatch)
    if fault == "extra_result":
        _write(paths[-1].parent / "unexpected.json", {})
    elif fault == "scratch":
        (paths[1].parent / "scores-partial").mkdir()
    elif fault == "evaluator":
        (tmp_path / "src/evaluation/official_test.py").write_bytes(b"unknown implementation")
    elif fault == "upstream":
        (tmp_path / "src/graph/pyg_adapter.py").write_bytes(b"changed frozen adapter")
    else:
        preparation["splits"]["E3"]["policy"]["test_support"] = [99, 99]
    with pytest.raises(e11.EvaluationError):
        recovery.audit_failure(tmp_path)
    assert calls == []


@pytest.mark.parametrize("fault", ("status", "binding", "coverage", "order", "metric", "canonical"))
def test_semantic_forgeries_fail_even_when_fixture_physical_hashes_are_rebound(tmp_path, monkeypatch, fault):
    preparation, originals, calls, paths = _fixture(tmp_path, monkeypatch)
    path = paths[2] if fault == "status" else paths[3]
    value = e11._read(path)
    if fault == "status":
        value["completed_checkpoints"] = 7
    else:
        value.pop("result_sha256")
        if fault == "binding":
            value["checkpoint_sha256"] = "0" * 64
        elif fault == "coverage":
            value["examples"] -= 1
        elif fault == "order":
            value["order_sha256"] = "0" * 64
        elif fault == "metric":
            value["metrics"]["primary"]["macro_f1"] = .01
        value["result_sha256"] = "0" * 64 if fault == "canonical" else e11.d2.canonical_sha256(value)
    _write(path, value)
    originals[path.relative_to(tmp_path).as_posix()] = e10._sha256(path)
    with pytest.raises(e11.EvaluationError):
        recovery.audit_failure(tmp_path)
    assert calls == []


def test_synthetic_recovery_reuses_six_and_evaluates_only_suffix_without_touching_originals(tmp_path, monkeypatch):
    preparation, originals, calls, paths = _fixture(tmp_path, monkeypatch)
    result = recovery.run_owner_recovery(tmp_path, owner_confirmed_reviewed_failure_recovery=True)
    assert calls == [r["candidate_id"] for r in preparation["evaluation_records"][6:]]
    assert result["reused_checkpoints"] == 6 and result["newly_evaluated_checkpoints"] == 21
    assert len(result["results"]) == 27 and result["original_marker_preserved"] is True
    assert originals == {p.relative_to(tmp_path).as_posix(): e10._sha256(p) for p in paths}
    original_status = e11._read(paths[2])
    assert original_status["state"] == "FAILED" and original_status["completed_checkpoints"] == 6
    outputs = tmp_path / f"results/stage_e11/{e11.STUDY}/recovery/{recovery.RECOVERY_ID}"
    assert len(list(outputs.glob('e10-*.json'))) == 21
    assert not any((outputs / (r['candidate_id'] + '.json')).exists() for r in preparation['evaluation_records'][:6])
    e11._canonical_check(e11._read(outputs / 'stage_e11_report.json'), 'report_sha256')
    assert (outputs / 'stage_e11_report.sha256').read_text().split()[0] == e10._sha256(outputs / 'stage_e11_report.json')
    with pytest.raises(e11.EvaluationError, match="already reserved"):
        recovery.run_owner_recovery(tmp_path, owner_confirmed_reviewed_failure_recovery=True)


def test_recovery_requires_separate_owner_acknowledgement_before_any_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "audit_failure", lambda *args: pytest.fail("must not audit or read arrays"))
    with pytest.raises(e11.EvaluationError, match="acknowledgement"):
        recovery.run_owner_recovery(tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("fault", ("test_hash", "interrupt", "inference", "environment", "libraries"))
def test_recovery_failure_stops_and_retains_originals_and_new_reservation(tmp_path, monkeypatch, fault):
    preparation, originals, calls, paths = _fixture(tmp_path, monkeypatch)
    if fault == "test_hash":
        (tmp_path / 'data/processed/preprocessed/exp3/X_test.npy').write_bytes(b"changed")
    elif fault in ("interrupt", "inference"):
        def stop(*args):
            raise KeyboardInterrupt() if fault == "interrupt" else e11.EvaluationError("nonfinite synthetic logit")
        monkeypatch.setattr(e11, "evaluate_checkpoint", stop)
    elif fault == "environment":
        monkeypatch.setattr(e10, "environment_identity", lambda: {"torch": "changed"})
    else:
        monkeypatch.setattr(e11, "package_version", lambda name: "changed")
    with pytest.raises((e11.EvaluationError, KeyboardInterrupt)):
        recovery.run_owner_recovery(tmp_path, owner_confirmed_reviewed_failure_recovery=True)
    assert originals == {p.relative_to(tmp_path).as_posix(): e10._sha256(p) for p in paths}
    recovery_logs = tmp_path / f'logs/stage_e11/{e11.STUDY}/recovery/{recovery.RECOVERY_ID}'
    if fault in ("environment", "libraries"):
        assert not recovery_logs.exists()
    else:
        assert (recovery_logs / 'recovery_authorization.json').exists()
        assert e11._read(recovery_logs / 'status.json')['state'] == ('INTERRUPTED' if fault == 'interrupt' else 'FAILED')
        with pytest.raises(e11.EvaluationError, match="already reserved"):
            recovery.run_owner_recovery(tmp_path, owner_confirmed_reviewed_failure_recovery=True)


def test_recovery_cli_has_no_marker_bypass_start_subset_or_overwrite_switch():
    from scripts.recover_stage_e11_test_evaluation import build_parser
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(['recover'])
    for forbidden in ('--force', '--start-checkpoint', '--reset-marker', '--run-id'):
        with pytest.raises(SystemExit):
            parser.parse_args(['recover', '--owner-confirmed-reviewed-failure-recovery', forbidden, '7'])
