"""E.11 unit/synthetic tests. No real arrays or ML training are executed."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
import torch

from src.evaluation import official_test as e11
from src.graph import schema as d2
from src.training import full_training as e10, harness as e7
from test_htgnn_classifier import _synthetic_data
from test_stage_e10_full_training import _record, _provenance_fixture


def _metadata(task):
    binary = task == "E3"
    names = ["Benign", "Attack"] if binary else e11.LABEL_VOCABULARY
    train = [100, 100] if binary else [100, 100, 1] + [0] * 12
    val = [30, 30] if binary else [30, 30, 1] + [0] * 12
    test = [2, 2] if binary else [40, 40, 5] + [0] * 12
    if task == "E1":
        train[1] = 0
    m = {"experiment": e11.d4.EXPERIMENT_METADATA_IDENTITIES[task],
         "preprocessing": {"fit_split": "train", "validation_or_test_used_for_fit": False}}
    for split, counts in (("train", train), ("val", val), ("test", test)):
        m[f"n_{split}"] = sum(counts)
        m[f"{split}_{'binary' if binary else 'class'}_distribution"] = dict(zip(names, counts))
    if task == "E1":
        m["class_availability_matrix"] = {"classes_with_possible_supervised_evaluation": [names[0], names[2]]}
    if task == "E2":
        m["class_eligibility"] = {"thresholds": {"min_train": 100, "min_val": 30, "min_test": 30},
                "eligible_class_codes": [0, 1], "per_class": {
                    name: {"n_train": train[i], "n_val": val[i], "n_test": test[i], "eligible": i in (0, 1)}
                    for i, name in enumerate(names)}}
    return m


@pytest.mark.parametrize("task", e10.TASKS)
def test_reporting_policy_uses_frozen_support_and_output_contract(task):
    p = e11.reporting_policy(_metadata(task), task)
    assert p["reporting_class_codes"] == ([0, 2] if task == "E1" else [0, 1])
    assert p["output_logits"] == (1 if task == "E3" else 15)
    assert p["test_order"] == "sequential"


@pytest.mark.parametrize("fault", ("task", "fit", "count", "unknown_label", "eligibility", "threshold"))
def test_reporting_policy_rejects_inconsistent_frozen_metadata(fault):
    m = _metadata("E2")
    if fault == "task":
        m["experiment"] = "1_global_chronological"
    elif fault == "fit":
        m["preprocessing"]["validation_or_test_used_for_fit"] = True
    elif fault == "count":
        m["test_class_distribution"]["Benign"] = -1
    elif fault == "unknown_label":
        m["test_class_distribution"]["unknown"] = 2
    elif fault == "eligibility":
        m["class_eligibility"]["eligible_class_codes"] = [0, 1, 2]
    else:
        m["class_eligibility"]["thresholds"]["min_test"] = 29
    with pytest.raises(e11.EvaluationError):
        e11.reporting_policy(m, "E2")


def test_e2_primary_filters_true_rows_but_retains_ineligible_prediction_errors():
    p = e11.reporting_policy(_metadata("E2"), "E2")
    confusion = np.zeros((15, 15), dtype=np.int64)
    confusion[0, 0], confusion[0, 2] = 30, 10
    confusion[1, 1], confusion[1, 0] = 30, 10
    confusion[2, 0] = 5
    result = e11.multiclass_metrics(confusion, p)
    y = [0] * 40 + [1] * 40
    pred = [0] * 30 + [2] * 10 + [1] * 30 + [0] * 10
    assert result["primary"]["macro_f1"] == pytest.approx(f1_score(y, pred, labels=[0, 1], average="macro"))
    assert result["primary"]["accuracy"] == 60 / 80
    assert result["all_test_descriptive"]["accuracy"] == 60 / 85
    assert result["confusion_matrix"][2][0] == 5


def test_e1_primary_keeps_unseen_rows_when_counting_false_positives():
    p = e11.reporting_policy(_metadata("E1"), "E1")
    confusion = np.zeros((15, 15), dtype=np.int64)
    confusion[0, 0], confusion[1, 0], confusion[2, 2] = 40, 40, 5
    result = e11.multiclass_metrics(confusion, p)
    assert result["primary"]["macro_f1"] == pytest.approx((2 / 3 + 1) / 2)
    assert result["all_test_descriptive"]["per_class"][1]["status"].startswith("unseen")
    assert result["primary_true_row_filter"] == "all_test_rows"


@pytest.mark.parametrize("fault", ("shape", "negative", "support"))
def test_confusion_rejects_malformed_or_mismatched_support(fault):
    p = e11.reporting_policy(_metadata("E2"), "E2")
    c = np.diag(p["test_support"]).astype(np.int64)
    if fault == "shape":
        c = c[:14, :14]
    elif fault == "negative":
        c[0, 1] = -1
    else:
        c[0, 0] -= 1
    with pytest.raises(e11.EvaluationError):
        e11.multiclass_metrics(c, p)


def test_binary_ap_matches_sklearn_raw_logit_ap_and_natural_threshold():
    y = np.array([0, 1, 0, 1], dtype=np.int8)
    scores = np.array([-1., 0., 2., 3.])
    p = e11.reporting_policy(_metadata("E3"), "E3")
    result = e11.binary_metrics(y, scores, p)
    predictions = scores >= 0
    assert result["pr_auc_average_precision"] == average_precision_score(y, scores)
    assert result["precision"] == precision_score(y, predictions)
    assert result["recall"] == recall_score(y, predictions)
    assert result["f1"] == f1_score(y, predictions)
    assert result["confusion_matrix"] == [[1, 1], [0, 2]]
    assert result["threshold_tuned"] is False


def test_binary_single_class_ranking_is_null_and_json_finite():
    result = e11.binary_metrics(np.array([0, 0], dtype=np.int8), np.array([-2., -1.]), {"test_support": [2, 0]})
    assert result["pr_auc_average_precision"] is None
    assert result["roc_auc"] is None
    assert result["recall"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("scores,y", (([float("nan"), 0.], [0, 1]), ([0., 1.], [0, 2]), ([0.], [0, 1])))
def test_binary_rejects_bad_scores_domain_or_shape(scores, y):
    with pytest.raises(e11.EvaluationError):
        e11.binary_metrics(np.array(y), np.array(scores), {"test_support": [1, 1]})


@pytest.mark.parametrize("allow_test", (False, True))
def test_file_guard_denies_dataset_and_e10_writes_and_renames(tmp_path, allow_test):
    path = tmp_path / "logs/stage_e10/preserve.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"immutable")
    other = tmp_path / "logs/stage_e11/unit.json"
    other.parent.mkdir(parents=True)
    other.write_bytes(b"new")
    with e11.file_guard(tmp_path, allow_test=allow_test):
        with pytest.raises(e11.EvaluationError):
            path.write_bytes(b"corrupt")
        with pytest.raises(e11.EvaluationError):
            other.replace(path)
    assert path.read_bytes() == b"immutable"


def test_preparation_guard_blocks_numpy_loading_before_file_open(tmp_path):
    path = tmp_path / "data/processed/preprocessed/exp1/X_test.npy"
    path.parent.mkdir(parents=True)
    np.save(path, np.zeros((2, 5, 78), dtype=np.float32))
    with e11.file_guard(tmp_path):
        with pytest.raises(e11.EvaluationError, match="locked"):
            np.load(path, mmap_mode="r")
    with e11.file_guard(tmp_path, allow_test=True):
        with pytest.raises(e11.EvaluationError):
            np.load(path, mmap_mode="r+")


@pytest.mark.parametrize("name", ("X_train.npy", "y_val.npy", "other.npy"))
def test_evaluation_guard_only_allows_expected_readonly_test_arrays(tmp_path, name):
    path = tmp_path / "data/processed/preprocessed/exp1" / name
    path.parent.mkdir(parents=True)
    path.write_bytes(b"synthetic")
    with e11.file_guard(tmp_path, allow_test=True):
        with pytest.raises(e11.EvaluationError):
            path.read_bytes()


def _checkpoint(root, family="htgnn", task="E2", seed=2025):
    candidate = e10.FullTrainingCandidate(family, task, 1e-3, seed, "lr_selection" if seed == 2025 else "final_seed")
    model = e7.create_model(task, family)
    optimizer = e7.create_optimizer(model, 1e-3)
    # Manufacture valid serialized AdamW state directly: never take a step or
    # train even a synthetic model while preparing the first test evaluation.
    for parameter in model.parameters():
        optimizer.state[parameter] = {"step": torch.tensor(1.),
                                      "exp_avg": torch.zeros_like(parameter),
                                      "exp_avg_sq": torch.full_like(parameter, .001)}
    stats = e7.EpochStats(loss=.5, batches=1, examples=2, order_sha256="0" * 64, target_codes=(0, 1),
                        selection_value=.5, selection_name=e7.SELECTION_NAMES[task],
                        selection_support_complete=True, selection_class_codes=() if task == "E3" else (0, 1))
    path = e10.candidate_paths(root, e11.STUDY, candidate).checkpoint
    e7.save_checkpoint(path, config=candidate.training_config, model=model, optimizer=optimizer,
                       epoch=1, validation=stats, frozen_source_hashes=e7.e6.FROZEN_SOURCE_SHA256)
    metadata = torch.load(path, weights_only=True, map_location="cpu")["metadata"]
    return {"configuration": candidate.config_dict(), "checkpoint_provenance": metadata,
            "checkpoint_provenance_verified": True, "best_epoch": 1, "best_validation_metric": .5,
            "artifacts": {"checkpoint_sha256": e10._sha256(path)}, "result_sha256": "a" * 64}


@pytest.mark.parametrize("family", e10.MODEL_FAMILIES)
@pytest.mark.parametrize("task", e10.TASKS)
def test_synthetic_checkpoint_inference_all_nine_pairs(tmp_path, monkeypatch, family, task):
    record = _checkpoint(tmp_path, family, task)
    data = _synthetic_data(2)
    data.e2_audit["experiment"] = task
    data["sample"].y = torch.tensor([0, 1], dtype=torch.float32 if task == "E3" else torch.long)
    monkeypatch.setattr(e11, "iter_test_batches", lambda *args: iter([data]))
    def forbidden(*args, **kwargs):
        pytest.fail("training/optimization/selection must not execute")
    monkeypatch.setattr(e7, "run_training", forbidden)
    monkeypatch.setattr(e10, "select_learning_rates", forbidden)
    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden)
    policy = {"task": task, "test_support": [1, 1] if task == "E3" else [1, 1] + [0] * 13,
              "train_support": [100, 100] + [0] * 13, "validation_support": [30, 30] + [0] * 13,
              "reporting_class_codes": [0, 1]}
    result = e11.evaluate_checkpoint(tmp_path, record, policy, tmp_path, torch.device("cpu"))
    assert result["examples"] == 2
    assert result["model_state_unchanged"] is True
    assert result["checkpoint_sha256"] == record["artifacts"]["checkpoint_sha256"]
    assert not list(tmp_path.glob("scores-*"))


@pytest.mark.parametrize("family", e10.MODEL_FAMILIES[1:])
def test_dense_test_accessor_matches_frozen_training_accessor_without_relabelling(family):
    data = _synthetic_data(2)
    data.e2_audit["experiment"] = "E2"
    model = e7.create_model("E2", family).eval()
    with torch.inference_mode():
        actual = e11._inference_logits(model, data, "E2", family)
        with pytest.raises(Exception, match="rejects test"):
            model(data)
        training_data = deepcopy(data)
        training_data.e2_audit["split"] = "validation"
        expected = model(training_data)
    assert torch.equal(actual, expected)
    assert data.e2_audit["split"] == "test"


@pytest.mark.parametrize("fault", ("hash", "epoch", "metadata", "task"))
def test_checkpoint_binding_rejects_changed_provenance(tmp_path, fault):
    record = _checkpoint(tmp_path)
    if fault == "hash":
        record["artifacts"]["checkpoint_sha256"] = "0" * 64
    elif fault == "epoch":
        record["best_epoch"] = 2
    elif fault == "metadata":
        record["checkpoint_provenance"]["parameter_count"] += 1
    else:
        record["configuration"]["task"] = "E1"
    with pytest.raises((e11.EvaluationError, e10.FullTrainingError, FileNotFoundError, e7.TrainingContractError)):
        e11._load_bound_checkpoint(tmp_path, record)


@pytest.mark.parametrize("fault", ("order", "split", "task", "coverage", "nan", "shape"))
def test_checkpoint_evaluation_rejects_bad_batches(tmp_path, monkeypatch, fault):
    record = _checkpoint(tmp_path)
    data = _synthetic_data(2)
    data.e2_audit["experiment"] = "E2"
    if fault == "order":
        data.e2_audit["ordered_source_sample_indices"] = [1, 0]
    elif fault == "split":
        data.e2_audit["split"] = "validation"
    elif fault == "task":
        data.e2_audit["experiment"] = "E1"
    monkeypatch.setattr(e11, "iter_test_batches", lambda *args: iter([data]))
    if fault in ("nan", "shape"):
        monkeypatch.setattr(e11, "_inference_logits", lambda *args: torch.full((2, 15 if fault == "nan" else 14), float("nan") if fault == "nan" else 0.))
    policy = {"task": "E2", "test_support": [2, 1] + [0] * 13 if fault == "coverage" else [1, 1] + [0] * 13,
              "reporting_class_codes": [0, 1]}
    with pytest.raises(e11.EvaluationError):
        e11.evaluate_checkpoint(tmp_path, record, policy, tmp_path, torch.device("cpu"))


def _audit_fixture(root, monkeypatch):
    a = [_record(c, .5) for c in e10.enumerate_lr_selection_candidates()]
    selected = e10.select_learning_rates(a)
    c = [_record(c, .5) for c in e10.enumerate_additional_final_seed_candidates(selected)]
    for row in a + c:
        row["checkpoint_provenance"] = {"frozen_source_sha256": {}}
    plan = e10._selection_plan(e11.STUDY, selected)
    selection = e10._selection_report(e11.STUDY, selected, a)
    final = [r for entry in plan["selected"] for seed in e10.FINAL_SEEDS for r in a + c
             if r["configuration"]["model_family"] == entry["model_family"] and r["configuration"]["task"] == entry["task"]
             and r["configuration"]["seed"] == seed and r["configuration"]["learning_rate"] == entry["learning_rate"]]
    report = {"completed_runs": a + c, "overall_status": "PASS", "lr_selection": selection,
              "final_validation_selected_checkpoints": final, "test_access_audit": {"predictive_test_arrays_opened": [],
              "predictive_test_array_access": False, "test_metrics_recorded": False, "result": "PASS"}}
    report["report_sha256"] = d2.canonical_sha256(report)
    paths = e10.report_paths(root, e11.STUDY)
    paths.final_report_sha256.parent.mkdir(parents=True)
    paths.final_report_sha256.write_text(e11.E10_REPORT_SHA256 + "  stage_e10_report.json\n")
    manifest = _provenance_fixture()
    manifest["stage_c_provenance"]["split_artifact_sha256"] = {f"exp{i}/split_metadata.json": "f" * 64 for i in (1, 2, 3)}
    manifest["upstream_evidence_sha256"] = {}
    stage_c = {"status": "PASS", "artefact_sha256": {f"exp{i}/{name}": "f" * 64 for i in (1, 2, 3)
                                                   for name in ("split_metadata.json", "X_test.npy", "y_test.npy")}}
    values = {paths.final_report: report, paths.selected_learning_rates: plan, paths.lr_selection_report: selection,
              root / "results/stage_c_full_validation.json": stage_c}
    for task, directory in e11.d4.EXPERIMENT_DIRS.items():
        values[root / "data/processed/preprocessed" / directory / "split_metadata.json"] = _metadata(task)
    def read(path):
        return values[path] if path in values else manifest
    monkeypatch.setattr(e11, "_read", read)
    monkeypatch.setattr(e10, "load_selection_plan", lambda *args: {(entry["model_family"], entry["task"]): entry["learning_rate"] for entry in plan["selected"]})
    monkeypatch.setattr(e10, "load_completed_results", lambda root, study, candidates: a if len(candidates) == 27 else c)
    monkeypatch.setattr(e10, "_candidate_provenance_identity", lambda *args: "compatible")
    monkeypatch.setattr(e10, "_evidence_hashes", lambda *args: {})
    monkeypatch.setattr(e10, "_validate_frozen_e9_evidence", lambda *args: {"e9_source_sha256": {}, "frozen_upstream_source_sha256": {}})
    digests = {"e10_provenance_policy.py": e11.POLICY_SHA256, "full_training.py": e10.provenance_policy.CORRECTED_RUNTIME_SHA256,
               "run_stage_e10_full_training.py": e10.provenance_policy.APPROVED_RUNNER_SHA256,
               "stage_e10_report.json": e11.E10_REPORT_SHA256, "stage_c_full_validation.json": e11.STAGE_C_REPORT_SHA256}
    monkeypatch.setattr(e10, "_sha256", lambda path: digests.get(path.name, "f" * 64))
    loaded = []
    monkeypatch.setattr(e11, "_load_bound_checkpoint", lambda root, r: loaded.append(r["candidate_id"]))
    monkeypatch.setattr(e10, "select_learning_rates", lambda *args: pytest.fail("E.11 may not select LRs"))
    return report, plan, loaded, digests


def test_synthetic_e10_reconciliation_validates_exact_matrix_without_selecting_lr(tmp_path, monkeypatch):
    report, plan, loaded, digests = _audit_fixture(tmp_path, monkeypatch)
    result = e11.audit_e10(tmp_path)
    assert len(loaded) == 45
    assert len(result["evaluation_records"]) == 27
    assert result["preparation_test_predictive_access"] is False


@pytest.mark.parametrize("fault", ("duplicate", "missing_final", "failed", "test_access", "lr_source", "policy", "runtime", "report"))
def test_e10_audit_rejects_matrix_provenance_or_freeze_changes(tmp_path, monkeypatch, fault):
    report, plan, loaded, digests = _audit_fixture(tmp_path, monkeypatch)
    if fault == "duplicate":
        report["completed_runs"][1] = report["completed_runs"][0]
    elif fault == "missing_final":
        report["final_validation_selected_checkpoints"].pop()
    elif fault == "failed":
        report["overall_status"] = "FAIL"
    elif fault == "test_access":
        report["test_access_audit"]["predictive_test_array_access"] = True
    elif fault == "lr_source":
        plan["selected"][0]["source_result_sha256"] = "0" * 64
    else:
        name = {"policy": "e10_provenance_policy.py", "runtime": "full_training.py", "report": "stage_e10_report.json"}[fault]
        digests[name] = "0" * 64
    report.pop("report_sha256")
    report["report_sha256"] = d2.canonical_sha256(report)
    with pytest.raises((e11.EvaluationError, KeyError)):
        e11.audit_e10(tmp_path)


def _owner_fixture(root, monkeypatch):
    for relative in e11.SOURCE_PATHS:
        p = root / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"synthetic source")
    test_path = root / "data/processed/preprocessed/exp1/X_test.npy"
    test_path.parent.mkdir(parents=True)
    test_path.write_bytes(b"synthetic predictive stand-in")
    records = []
    for family in e10.MODEL_FAMILIES:
        for task in e10.TASKS:
            for seed in e10.FINAL_SEEDS:
                records.append({"configuration": {"model_family": family, "task": task, "seed": seed}})
    preparation = {"evaluation_records": records, "splits": {task: {"policy": {"task": task},
                    "predictive_artifact_sha256": {"exp1/X_test.npy": e10._sha256(test_path)}} for task in e10.TASKS},
                   "e10_artifact_sha256": {}, "frozen_metadata_sha256": {}, "training_environment": {}}
    monkeypatch.setattr(e11, "audit_e10", lambda *args: deepcopy(preparation))
    monkeypatch.setattr(e10, "environment_identity", lambda: {})
    called = []
    def evaluate(root, record, policy, scratch, device):
        assert (root / f"logs/stage_e11/{e11.STUDY}/test_consumption.json").is_file()
        test_path.read_bytes()
        cfg = record["configuration"]
        called.append(cfg)
        return {"candidate_id": f"{cfg['model_family']}-{cfg['task']}-{cfg['seed']}", **cfg,
                "metrics": {"pr_auc_average_precision": .5} if cfg["task"] == "E3" else {"primary": {"macro_f1": .5}}}
    monkeypatch.setattr(e11, "evaluate_checkpoint", evaluate)
    return called, test_path


def test_synthetic_owner_flow_reserves_before_test_reads_and_blocks_different_run_id(tmp_path, monkeypatch):
    called, test_path = _owner_fixture(tmp_path, monkeypatch)
    real_hash = e10._sha256
    def hash_check(path):
        if path == test_path:
            assert (tmp_path / f"logs/stage_e11/{e11.STUDY}/test_consumption.json").exists()
        return real_hash(path)
    monkeypatch.setattr(e10, "_sha256", hash_check)
    report = e11.run_owner_evaluation(tmp_path, "e11-official-20261006-v1", device="cpu", owner_confirmed_manual_test_evaluation=True)
    assert len(called) == 27
    assert report["overall_status"] == "PASS"
    assert len(report["seed_summary"]) == 9
    result_dir = tmp_path / f"results/stage_e11/{e11.STUDY}/e11-official-20261006-v1"
    report_path = result_dir / "stage_e11_report.json"
    assert (result_dir / "stage_e11_report.sha256").read_text().split()[0] == real_hash(report_path)
    e11._canonical_check(json.loads(report_path.read_text()), "report_sha256")
    with pytest.raises(e11.EvaluationError, match="consumed"):
        e11.run_owner_evaluation(tmp_path, "e11-official-20261007-v2", owner_confirmed_manual_test_evaluation=True)


def test_owner_acknowledgement_required_before_any_audit_or_test_access(tmp_path, monkeypatch):
    monkeypatch.setattr(e11, "audit_e10", lambda *args: pytest.fail("no audit without explicit owner launch"))
    with pytest.raises(e11.EvaluationError, match="acknowledgement"):
        e11.run_owner_evaluation(tmp_path, "e11-official-20261006-v1")
    assert not (tmp_path / "logs").exists()


@pytest.mark.parametrize("failure", ("interrupt", "bad_test_hash", "source_change"))
def test_owner_failure_retains_consumption_marker_and_never_automatically_retries(tmp_path, monkeypatch, failure):
    called, test_path = _owner_fixture(tmp_path, monkeypatch)
    original = e11.evaluate_checkpoint
    if failure == "bad_test_hash":
        test_path.write_bytes(b"changed")
    else:
        def evaluate(*args):
            if failure == "interrupt":
                raise KeyboardInterrupt()
            result = original(*args)
            # Simulate an external source mutation; restore before leaving test.
            with (tmp_path / e11.SOURCE_PATHS[0]).open("ab") as handle:
                handle.write(b"changed")
            return result
        if failure == "source_change":
            # In-process source mutation is already forbidden by the file guard.
            # The resulting failure is equally fail-closed and preserves marker.
            monkeypatch.setattr(e11, "evaluate_checkpoint", evaluate)
        else:
            monkeypatch.setattr(e11, "evaluate_checkpoint", evaluate)
    with pytest.raises((KeyboardInterrupt, e11.EvaluationError)):
        e11.run_owner_evaluation(tmp_path, "e11-official-20261006-v1", device="cpu", owner_confirmed_manual_test_evaluation=True)
    assert (tmp_path / f"logs/stage_e11/{e11.STUDY}/test_consumption.json").exists()
    status = json.loads((tmp_path / f"logs/stage_e11/{e11.STUDY}/e11-official-20261006-v1/status.json").read_text())
    assert status["state"] == ("INTERRUPTED" if failure == "interrupt" else "FAILED")


def test_cli_has_no_training_pruning_threshold_or_subset_options():
    from scripts.run_stage_e11_test_evaluation import build_parser
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["evaluate", "--official-study-id", e11.STUDY, "--run-id", "e11-official-20261006-v1"])
    with pytest.raises(SystemExit):
        parser.parse_args(["evaluate", "--official-study-id", e11.STUDY, "--run-id", "e11-official-20261006-v1",
                           "--owner-confirmed-manual-test-evaluation", "--threshold", ".7"])


def test_seed_summary_requires_all_three_and_reports_sample_std():
    rows = [{"model_family": family, "task": task, "seed": seed,
             "metrics": {"pr_auc_average_precision": value} if task == "E3" else {"primary": {"macro_f1": value}}}
            for family in e10.MODEL_FAMILIES for task in e10.TASKS
            for seed, value in zip(e10.FINAL_SEEDS, (.4, .5, .6))]
    result = e11.summarize_seeds(rows)
    assert result[0]["primary_mean"] == .5
    assert result[0]["primary_sample_std_ddof1"] == pytest.approx(.1)
    with pytest.raises(e11.EvaluationError):
        e11.summarize_seeds(rows[:-1])


@pytest.mark.parametrize("task", e10.TASKS)
def test_actual_lazy_d4_d6_e2_test_traversal_uses_only_synthetic_arrays(tmp_path, monkeypatch, task):
    root = Path(__file__).resolve().parents[1]
    for relative in ("results/stage_d1/feature_taxonomy.json", "results/stage_d2/graph_schema.json",
                     "results/stage_d3/structural_validation.json"):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((root / relative).read_bytes())
    directory = tmp_path / "data/processed/preprocessed" / e11.d4.EXPERIMENT_DIRS[task]
    directory.mkdir(parents=True)
    x = np.arange(257 * 5 * 78, dtype=np.float32).reshape(257, 5, 78) / 100000
    y = (np.arange(257) % 2).astype(np.int16)
    np.save(directory / "X_test.npy", x)
    np.save(directory / "y_test.npy", y)
    (directory / "split_metadata.json").write_text(json.dumps({"experiment": e11.d4.EXPERIMENT_METADATA_IDENTITIES[task], "n_test": 257}))
    ledger = tmp_path / f"logs/stage_e11/{e11.STUDY}/test_consumption.json"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(json.dumps({"e10_study": e11.STUDY, "state": "CONSUMED_OR_RESERVED"}))
    # Frozen D6 exposes this explicit fixture seam. Production E.11 never sets
    # it; only this synthetic test permits a temporary root and short length.
    real_build = e11.d6.build_logical_batch
    monkeypatch.setattr(e11.d6, "build_logical_batch", lambda dataset, indices:
                        real_build(dataset, indices, _allow_noncanonical_fixture=True))
    with e11.file_guard(tmp_path, allow_test=True):
        batches = list(e11.iter_test_batches(tmp_path, task))
    assert [len(data.e2_audit["ordered_source_sample_indices"]) for data in batches] == [256, 1]
    assert batches[0].e2_audit["ordered_source_sample_indices"] == list(range(256))
    assert batches[1].e2_audit["ordered_source_sample_indices"] == [256]
    expected = torch.tensor(x[:256].transpose(0, 2, 1).reshape(256 * 78, 5))
    assert torch.equal(batches[0]["feature"].x, expected)


def test_direct_test_traversal_fails_before_opening_any_arrays(tmp_path):
    with pytest.raises(e11.EvaluationError, match="reservation context"):
        next(e11.iter_test_batches(tmp_path, "E1"))


@pytest.mark.parametrize("action", ("overwrite", "delete", "rename"))
def test_consumption_marker_is_immutable_inside_evaluation_process(tmp_path, action):
    ledger = tmp_path / f"logs/stage_e11/{e11.STUDY}/test_consumption.json"
    ledger.parent.mkdir(parents=True)
    ledger.write_bytes(b"immutable")
    with e11.file_guard(tmp_path, allow_test=True):
        with pytest.raises(e11.EvaluationError, match="immutable"):
            if action == "overwrite":
                ledger.write_bytes(b"corrupt")
            elif action == "delete":
                ledger.unlink()
            else:
                ledger.rename(ledger.with_name("removed.json"))
    assert ledger.read_bytes() == b"immutable"


@pytest.mark.parametrize("task", e10.TASKS)
@pytest.mark.parametrize("size", (1, 2, 243, 256))
def test_frozen_task_specific_target_dtype_and_output_shape_are_accepted_unchanged(task, size):
    shape = (size,) if task == "E3" else (size, 15)
    logits = torch.arange(np.prod(shape), dtype=torch.float32).reshape(shape)
    targets = (torch.arange(size) % 2).to(torch.float32 if task == "E3" else torch.long)
    before_logits, before_targets = logits.clone(), targets.clone()
    e11._validate_inference_outputs(task, logits, targets, size)
    assert torch.equal(logits, before_logits) and torch.equal(targets, before_targets)


@pytest.mark.parametrize("shape,size", (((), 1), ((2, 1), 2), ((2, 2), 2), ((2, 15), 2),
                                      ((1, 2), 2), ((2, 1, 1), 2), ((1,), 2)))
def test_e3_rejects_noncanonical_head_axis_scalar_multiclass_and_broadcast_shapes(shape, size):
    # Frozen E.5 and E.9 both return [B]; [B,1] is an internal head shape,
    # not an alternate legitimate production output needing canonicalization.
    with pytest.raises(e11.EvaluationError, match="logits must have shape"):
        e11._validate_inference_outputs("E3", torch.zeros(shape), torch.zeros(size), size)


@pytest.mark.parametrize("task", e10.TASKS)
@pytest.mark.parametrize("bad", (float("nan"), float("inf"), -float("inf")))
def test_output_nonfinite_logits_still_fail_for_every_task(task, bad):
    logits = torch.zeros((2,) if task == "E3" else (2, 15))
    logits.reshape(-1)[0] = bad
    targets = torch.zeros(2, dtype=torch.float32 if task == "E3" else torch.long)
    with pytest.raises(e11.EvaluationError, match="finite float32"):
        e11._validate_inference_outputs(task, logits, targets, 2)


@pytest.mark.parametrize("task", e10.TASKS)
def test_wrong_target_dtype_is_rejected_without_casting_or_broadcasting(task):
    logits = torch.zeros((2,) if task == "E3" else (2, 15))
    wrong_dtype = torch.long if task == "E3" else torch.float32
    with pytest.raises(e11.EvaluationError, match="targets must have"):
        e11._validate_inference_outputs(task, logits, torch.zeros(2, dtype=wrong_dtype), 2)
    with pytest.raises(e11.EvaluationError, match="targets must have"):
        e11._validate_inference_outputs(task, logits,
            torch.zeros((2, 1), dtype=torch.float32 if task == "E3" else torch.long), 2)


@pytest.mark.parametrize("family", e10.MODEL_FAMILIES)
def test_e3_actual_adapter_float32_labels_reach_end_to_end_evaluator(tmp_path, monkeypatch, family):
    from test_pyg_adapter import _batch
    data = e11.e2.logical_batch_to_heterodata(_batch(tmp_path / "fixture", [0, 1], "E3"))
    # The fixture is a synthetic train split; expose its values as synthetic
    # test audit metadata solely inside this unit test.
    data.e2_audit["split"] = "test"
    assert data["sample"].y.dtype == torch.float32
    record = _checkpoint(tmp_path, family, "E3")
    monkeypatch.setattr(e11, "iter_test_batches", lambda *args: iter([data]))
    policy = {"task": "E3", "test_support": [1, 1], "reporting_class_codes": [0, 1]}
    result = e11.evaluate_checkpoint(tmp_path, record, policy, tmp_path, torch.device("cpu"))
    assert result["examples"] == 2
    assert result["metrics"]["support"] == [1, 1]


@pytest.mark.parametrize("invalid", (.5, 2., float("nan"), float("inf")))
def test_e3_float32_labels_are_not_silently_truncated_to_binary_codes(tmp_path, monkeypatch, invalid):
    record = _checkpoint(tmp_path, "htgnn", "E3")
    data = _synthetic_data(2)
    data.e2_audit["experiment"] = "E3"
    data["sample"].y = torch.tensor([0., invalid], dtype=torch.float32)
    monkeypatch.setattr(e11, "iter_test_batches", lambda *args: iter([data]))
    with pytest.raises(e11.EvaluationError, match="target domain violation"):
        e11.evaluate_checkpoint(tmp_path, record, {"task": "E3", "test_support": [1, 1]}, tmp_path, torch.device("cpu"))
