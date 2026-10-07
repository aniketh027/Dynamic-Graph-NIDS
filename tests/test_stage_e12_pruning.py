"""E.12 unit/synthetic tests; no real data, optimizer, backward or training."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import threading

import numpy as np
import pytest
import torch

from src.pruning import stage_e12 as e12
from test_htgnn_classifier import _synthetic_data


def data(b=2, split="validation", task="E2", start=0, device="cpu"):
    result = _synthetic_data(b, device=device)
    result.e2_audit.update(experiment=task, split=split, ordered_source_sample_indices=list(range(start, start + b)))
    result["sample"].y = torch.arange(b, device=device) % 2
    if task == "E3":
        result["sample"].y = result["sample"].y.float()
    return result


def model(task="E2", device="cpu"):
    return e12.HTGNNClassifier(task).requires_grad_(False).eval().to(device)


@pytest.fixture(autouse=True)
def forbid_training(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("optimizer/backprop/training executed")
    monkeypatch.setattr(torch.optim.AdamW, "__init__", forbidden)
    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(e12.e7, "create_optimizer", forbidden)
    monkeypatch.setattr(e12.e7, "run_training", forbidden)
    monkeypatch.setattr(e12.e10, "select_learning_rates", forbidden)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    with e12.deterministic_inference("cpu"):
        yield


@pytest.mark.parametrize("task", ("E1", "E2", "E3"))
@pytest.mark.parametrize("b", (1, 3))
@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_zero_pruning_is_exact_original_logits_and_weights(task, b, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    m, d = model(task, device), data(b, task=task, device=device)
    before = e12.state_sha256(m)
    with torch.inference_mode():
        expected = m(d)
    actual = e12.pruned_logits(m, d, e12.make_policy([1.] * 78, 0))
    assert torch.equal(expected, actual)
    assert e12.state_sha256(m) == before
    assert not actual.requires_grad


@pytest.mark.parametrize("ratio,count", ((0., 78), (.10, 73), (.25, 61), (.50, 40), (.75, 22)))
def test_policy_counts_subset_and_all_category_connectivity(ratio, count):
    policy = e12.make_policy(np.arange(78), ratio)
    assert len(policy.retained_feature_ids) == count
    record = policy.record()
    assert record["pruned_graph"]["nodes"] == count + 9
    assert record["pruned_graph"]["edges"] == count + 8
    assert set(c for _, c in record["retained_feature_category_edges"]) == set(range(8))
    assert record["retained_category_sample_edges"] == [[c, 0] for c in range(8)]


@pytest.mark.parametrize("bad", (-.1, 1, 1.2, float("nan"), float("inf"), True, "0.5", None))
def test_invalid_ratios_fail(bad):
    with pytest.raises(e12.PruningError):
        e12.make_policy([1.] * 78, bad)


@pytest.mark.parametrize("bad", ([1.] * 77, [float("nan")] * 78, [-1.] * 78))
def test_invalid_scores_fail(bad):
    with pytest.raises(e12.PruningError):
        e12.make_policy(bad, .5)


@pytest.mark.parametrize("ids", ((0,), tuple(range(77)), tuple(range(78)) + (77,), tuple(range(77)) + (78,)))
def test_manual_policy_cannot_remove_categories_add_nodes_or_violate_quotas(ids):
    with pytest.raises(e12.PruningError):
        e12.FeaturePolicy(.5, ids)


def test_deterministic_ranking_id_ties_and_head_permutation():
    scores = np.ones(78)
    first = e12.make_policy(scores, .5)
    assert first == e12.make_policy(scores.copy(), .5)
    for category in range(8):
        members = [i for i, c in enumerate(e12.d2.FEATURE_TO_CATEGORY_DESTINATION_INDICES) if c == category]
        kept = [i for i in first.retained_feature_ids if i in members]
        assert kept == members[:len(kept)]
    raw = np.arange(3 * 78 * 4).reshape(3, 78, 4)
    assert np.array_equal(raw.mean(axis=(0, 2)), raw[:, :, [3, 1, 0, 2]].mean(axis=(0, 2)))


@pytest.mark.parametrize("b", (1, 4))
def test_compaction_maps_original_edges_without_new_edges_or_cross_graphs(b):
    d = data(b)
    old = deepcopy(d)
    p = e12.make_policy(np.arange(78), .75)
    values, ids, edges = e12.compact_inputs(d, p, b)
    k = len(p.retained_feature_ids)
    assert values.shape == (b * k, 5)
    for g in range(b):
        assert ids[g * k:(g + 1) * k].tolist() == list(p.retained_feature_ids)
        for j, original in enumerate(p.retained_feature_ids):
            assert edges[:, g * k + j].tolist() == [g * k + j, g * 8 + e12.d2.FEATURE_TO_CATEGORY_DESTINATION_INDICES[original]]
            assert torch.equal(values[g * k + j], old["feature"].x[g * 78 + original])
    assert torch.equal(d[e12.e2.FEATURE_RELATION].edge_index, old[e12.e2.FEATURE_RELATION].edge_index)


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_pruned_execution_deterministic_batched_isolation_and_skips_removed_encoder_rows(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    m, d = model(device=device), data(3, device=device)
    p = e12.make_policy(np.arange(78), .5)
    before = e12.state_sha256(m)
    seen = []
    hook = m.feature_encoder.register_forward_pre_hook(lambda _, args: seen.append(args[0].shape[0]))
    first = e12.pruned_logits(m, d, p)
    assert torch.equal(first, e12.pruned_logits(m, d, p))
    assert seen == [3 * 40, 3 * 40]
    removed = p.record()["pruned_feature_ids"]
    altered = deepcopy(d)
    for g in range(3):
        altered["feature"].x[g * 78 + torch.tensor(removed, device=device)] += 1000
    assert torch.equal(first, e12.pruned_logits(m, altered, p))
    changed_graph = deepcopy(d)
    changed_graph["feature"].x[78:156] += 100
    changed = e12.pruned_logits(m, changed_graph, p)
    assert torch.equal(first[[0, 2]], changed[[0, 2]])
    assert not torch.equal(first[1], changed[1])
    assert e12.state_sha256(m) == before
    hook.remove()


@pytest.mark.parametrize("fault", ("train_mode", "grad", "test", "ownership", "feature_id", "redirect", "reorder", "cross_task"))
def test_malformed_inputs_fail_before_gat(fault):
    m, d = model(), data()
    p = e12.make_policy([1.] * 78, .5)
    if fault == "train_mode":
        m.train()
    elif fault == "grad":
        next(m.parameters()).requires_grad_(True)
    elif fault == "test":
        d.e2_audit["split"] = "test"
    elif fault == "ownership":
        d["feature"].batch[0] = 1
    elif fault == "feature_id":
        d["feature"].feature_id[0] = 1
    elif fault == "redirect":
        d[e12.e2.FEATURE_RELATION].edge_index[1, 0] = 7
    elif fault == "reorder":
        d[e12.e2.FEATURE_RELATION].edge_index = d[e12.e2.FEATURE_RELATION].edge_index.flip(1)
    else:
        d.e2_audit["experiment"] = "E1"
    handle = m.graph_encoder.feature_to_category.register_forward_pre_hook(lambda *args: pytest.fail("GAT ran before rejection"))
    with pytest.raises((e12.PruningError, e12.e7.TrainingContractError, ValueError)):
        e12.pruned_logits(m, d, p)
    handle.remove()


def test_attention_shape_mapping_and_eval_normalization_installed_pyg():
    m, d = model(), data(3, "train")
    with torch.inference_mode():
        _, raw = m(d, return_attention=True)
    feature, category = e12._attention_arrays(d, raw, 3)
    assert feature.shape == (3, 78, 4)
    assert category.shape == (3, 8, 4)
    np.testing.assert_allclose(category.sum(axis=1), 1., rtol=1e-5, atol=1e-5)
    corrupt = dict(raw)
    corrupt[e12.e2.FEATURE_RELATION] = (raw[e12.e2.FEATURE_RELATION][0].flip(1), raw[e12.e2.FEATURE_RELATION][1])
    with pytest.raises(e12.PruningError):
        e12._attention_arrays(d, corrupt, 3)


def test_train_ranking_is_deterministic_and_does_not_consume_validation_or_test():
    m = model()
    batches = [data(2, "train"), data(1, "train", start=2)]
    first = e12.collect_scores(m, iter(deepcopy(batches)), torch.device("cpu"))
    second = e12.collect_scores(m, iter(deepcopy(batches)), torch.device("cpu"))
    assert first == second
    assert first["examples"] == 3
    assert first["batches"] == 2
    assert np.asarray(first["feature_per_head_mean"]).shape == (78, 4)
    for split in ("validation", "test"):
        with pytest.raises(e12.PruningError, match="train-only"):
            e12.collect_scores(m, iter([data(split=split)]), torch.device("cpu"))


@pytest.mark.parametrize("task", ("E1", "E3"))
def test_other_tasks_cannot_supply_primary_ranking_or_selection(task):
    m = model(task)
    with pytest.raises(e12.PruningError, match="primary E2"):
        e12.collect_scores(m, iter([data(task=task, split="train")]), torch.device("cpu"))
    with pytest.raises(e12.PruningError, match="primary E2"):
        e12.evaluate_validation(m, iter([data(task=task)]), e12.make_policy([1.] * 78, 0), [0, 1], torch.device("cpu"))


def test_validation_metric_reuses_frozen_e7_full_row_false_positive_contract():
    m = model()
    d = data(3)
    result = e12.evaluate_validation(m, iter([d]), e12.make_policy([1.] * 78, 0), [0, 1], torch.device("cpu"))
    with torch.inference_mode():
        logits = m(d)
    expected = e12.e7._MetricAccumulator("E2")
    expected.update(logits, d["sample"].y)
    score, complete, codes = expected.selection(train_target_codes=(), validation_selection_class_codes=[0, 1])
    assert result["macro_f1"] == score
    assert result["examples"] == 3 and complete
    with pytest.raises(e12.PruningError):
        e12.evaluate_validation(m, iter([data(split="test")]), e12.make_policy([1.] * 78, 0), codes, torch.device("cpu"))


def rows():
    values = ((.8, .9, .7), (.79, .88, .69), (.77, .87, .68), (.78, .87, .60), (.50, .50, .50))
    return [{"ratio": ratio, "policy": e12.make_policy([1.] * 78, ratio).record(),
             "seed_results": [{"seed": seed, "macro_f1": value} for seed, value in zip(e12.SEEDS, metrics, strict=True)]}
            for ratio, metrics in zip(e12.RATIOS, values, strict=True)]


def test_selection_maximizes_reduction_under_every_seed_relative_tolerance():
    assert e12.select_candidate(rows(), .05)["ratio"] == .25
    assert e12.select_candidate(rows(), 0)["ratio"] == 0
    assert e12.select_candidate(rows(), .99)["ratio"] == .75


@pytest.mark.parametrize("fault", ("missing", "seed", "nan", "negative", "policy", "baseline"))
def test_selection_rejects_incomplete_or_poisoned_sweep(fault):
    sweep = rows()
    if fault == "missing":
        sweep.pop()
    elif fault == "seed":
        sweep[1]["seed_results"][0]["seed"] = 999
    elif fault == "nan":
        sweep[1]["seed_results"][0]["macro_f1"] = float("nan")
    elif fault == "negative":
        sweep[1]["seed_results"][0]["macro_f1"] = -.1
    elif fault == "policy":
        sweep[1]["policy"]["pruned_graph"]["nodes"] = 1
    else:
        sweep[0]["seed_results"][0]["macro_f1"] = 0
    with pytest.raises(e12.PruningError):
        e12.select_candidate(sweep, .05)


@pytest.mark.parametrize("task", ("exp1", "exp2", "exp3"))
@pytest.mark.parametrize("name", ("X_test.npy", "y_test.npy"))
def test_phase_a_test_arrays_locked_before_numpy_open(tmp_path, task, name):
    path = tmp_path / "data/processed/preprocessed" / task / name
    path.parent.mkdir(parents=True)
    np.save(path, np.zeros(2))
    with e12.phase_a_guard(tmp_path) as attempts:
        with pytest.raises(e12.PruningError, match="locked"):
            np.load(path, mmap_mode="r")
    assert len(attempts) == 1


def test_guard_blocks_e11_evidence_train_write_external_namespace_and_threads(tmp_path):
    train = tmp_path / "data/processed/preprocessed/exp2/X_train.npy"
    train.parent.mkdir(parents=True)
    np.save(train, np.zeros(2))
    e11 = tmp_path / "results/stage_e11/test_report.json"
    e11.parent.mkdir(parents=True)
    e11.write_text("{}")
    errors = []
    def worker():
        try:
            e11.read_bytes()
        except e12.PruningError as exc:
            errors.append(str(exc))
    with e12.phase_a_guard(tmp_path):
        assert np.load(train, mmap_mode="r").shape == (2,)
        with pytest.raises(e12.PruningError):
            os.open(train, os.O_RDWR)
        with pytest.raises(e12.PruningError):
            e11.read_bytes()
        with pytest.raises(e12.PruningError):
            (tmp_path / "PROJECT_HANDOFF.md").write_text("change")
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
    assert len(errors) == 1
    assert e11.read_text() == "{}"


def test_canonical_exclusive_stage_e12_publication(tmp_path):
    path = tmp_path / "results/stage_e12/proposed_policy.json"
    with e12.phase_a_guard(tmp_path):
        path.parent.mkdir(parents=True)
        payload = {"phase_b_enabled": False, "policy": e12.make_policy([1.] * 78, .5).record()}
        e12._write(path, payload)
        assert path.read_bytes() == e12.d2.canonical_json_bytes(payload)
        with pytest.raises(FileExistsError):
            e12._write(path, payload)


def test_owner_ack_and_tolerance_required_before_any_data_or_model_work(tmp_path, monkeypatch):
    monkeypatch.setattr(e12, "prepare", lambda *args: pytest.fail("real preflight executed"))
    with pytest.raises(e12.PruningError, match="owner approval"):
        e12.run_phase_a(tmp_path, run_id="e12a-20261006-v1", device="cpu", max_relative_f1_drop=.05, tolerance_rationale="proposal")
    with pytest.raises(e12.PruningError, match="tolerance"):
        e12.run_phase_a(tmp_path, run_id="e12a-20261006-v1", device="cpu", max_relative_f1_drop=float("nan"),
                        tolerance_rationale="proposal", owner_confirmed_validation_contract=True)


def synthetic_checkpoint(root):
    candidate = e12.e10.FullTrainingCandidate("htgnn", "E2", 1e-3, 2025, "lr_selection")
    m = model()
    identity = e12.e7.model_contract(m)
    metadata = {"schema_version": e12.e7.CHECKPOINT_SCHEMA_VERSION, "harness_version": e12.e7.HARNESS_VERSION,
                "task": "E2", "model_family": "htgnn", "model_contract_version": identity.architecture_version,
                "model_input_contract": identity.input_contract, "model_version": identity.architecture_version,
                "parameter_count": identity.parameter_count, "model_state_schema_sha256": e12.e7.model_state_schema_sha256(m),
                "training_config": candidate.training_config.checkpoint_dict(), "selection_name": e12.e7.SELECTION_NAMES["E2"],
                "frozen_source_sha256": dict(e12.e7.e6.FROZEN_SOURCE_SHA256), "epoch": 1, "selection_value": .5}
    for relative in metadata["frozen_source_sha256"]:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((Path(e12.__file__).resolve().parents[2] / relative).read_bytes())
    path = e12.e10.candidate_paths(root, e12.STUDY, candidate).checkpoint
    path.parent.mkdir(parents=True)
    torch.save({"metadata": metadata, "model_state_dict": m.state_dict()}, path)
    record = {"configuration": candidate.config_dict(), "artifacts": {"checkpoint_sha256": e12.e10._sha256(path)},
              "checkpoint_provenance": metadata, "checkpoint_provenance_verified": True, "best_epoch": 1, "best_validation_metric": .5}
    return m, path, record


def test_inference_checkpoint_loader_no_optimizer_state_bytes_unchanged(tmp_path):
    original, path, record = synthetic_checkpoint(tmp_path)
    raw = path.read_bytes()
    loaded = e12.load_bound_model(tmp_path, record)
    assert e12.state_sha256(loaded) == e12.state_sha256(original)
    assert path.read_bytes() == raw
    e12.pruned_logits(loaded, data(), e12.make_policy([1.] * 78, .75))
    assert e12.state_sha256(loaded) == e12.state_sha256(original)
    assert path.read_bytes() == raw


@pytest.mark.parametrize("fault", ("hash", "metadata", "task", "state", "source"))
def test_inference_loader_rejects_corrupted_binding(tmp_path, fault):
    _, path, record = synthetic_checkpoint(tmp_path)
    if fault == "hash":
        record["artifacts"]["checkpoint_sha256"] = "0" * 64
    elif fault == "metadata":
        record["checkpoint_provenance"]["epoch"] = 2
    elif fault == "task":
        record["configuration"]["task"] = "E1"
    elif fault == "source":
        (tmp_path / "src/graph/feature_encoder.py").write_text("changed")
    else:
        payload = torch.load(path, weights_only=True)
        payload["model_state_dict"].pop(next(iter(payload["model_state_dict"])))
        torch.save(payload, path)
        record["artifacts"]["checkpoint_sha256"] = e12.e10._sha256(path)
    with pytest.raises((e12.PruningError, e12.e7.TrainingContractError)):
        e12.load_bound_model(tmp_path, record)


def test_synthetic_phase_a_orchestration_outputs_and_retry_lockout(tmp_path, monkeypatch):
    records = [{"configuration": {"seed": seed}, "checkpoint_provenance": {"selection_class_codes": [0, 1]}} for seed in e12.SEEDS]
    preparation = {"records": records, "synthetic": True}
    models = {}
    for seed in e12.SEEDS:
        m = model()
        with torch.no_grad():
            m.head.weight.zero_()
            m.head.bias.fill_(-10)
            m.head.bias[0] = 10
        models[seed] = m
    originals = {seed: e12.state_sha256(m) for seed, m in models.items()}
    for relative in ("src/pruning/stage_e12.py", "scripts/run_stage_e12_pruning.py", "docs/stage_e12_attention_pruning.md", "tests/test_stage_e12_pruning.py"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic source identity")
    class Dataset:
        def __init__(self, split):
            self.split = split
        def __len__(self):
            return 3
    inputs = {split: data(3, split) for split in ("train", "validation")}
    monkeypatch.setattr(e12, "prepare", lambda *args: deepcopy(preparation))
    def reload(root, record):
        result = model()
        result.load_state_dict(models[record["configuration"]["seed"]].state_dict())
        return result
    monkeypatch.setattr(e12, "load_bound_model", reload)
    monkeypatch.setattr(e12, "_dataset", lambda root, split: Dataset(split))
    monkeypatch.setattr(e12, "_batches", lambda dataset, **kwargs: iter([deepcopy(inputs[dataset.split])]))
    monkeypatch.setattr(e12.e7, "_derive_e2_validation_selection_class_codes", lambda *args: (0, 1))
    kwargs = dict(run_id="e12a-20261006-v1", device="cpu", max_relative_f1_drop=.05,
                  tolerance_rationale="synthetic approved contract", owner_confirmed_validation_contract=True)
    report = e12.run_phase_a(tmp_path, **kwargs)
    output = tmp_path / "results/stage_e12" / e12.STUDY / kwargs["run_id"]
    assert report["status"] == "COMPLETE_PENDING_REVIEW"
    assert report["phase_b_enabled"] is False and report["test_arrays_opened"] == []
    assert report["selected_ratio"] == .75
    sweep = json.loads((output / "validation_sweep.json").read_text())
    assert len(sweep["rows"]) == 5 and all(len(row["seed_results"]) == 3 for row in sweep["rows"])
    assert all(result["benchmark"]["median_batch_seconds"] > 0 for row in sweep["rows"] for result in row["seed_results"])
    policy = json.loads((output / "proposed_policy.json").read_text())
    assert policy["status"] == "PROPOSED_REQUIRES_OWNER_REVIEW_AND_FREEZE"
    saved_hash = policy.pop("policy_sha256")
    assert saved_hash == e12.d2.canonical_sha256(policy)
    assert (output / "stage_e12a_report.sha256").read_text().strip() == hashlib.sha256((output / "stage_e12a_report.json").read_bytes()).hexdigest()
    assert {seed: e12.state_sha256(m) for seed, m in models.items()} == originals
    with pytest.raises(e12.PruningError, match="fresh"):
        e12.run_phase_a(tmp_path, **kwargs)


def test_deterministic_context_restores_runtime_settings():
    prior = torch.are_deterministic_algorithms_enabled()
    with e12.deterministic_inference("cpu"):
        assert torch.are_deterministic_algorithms_enabled()
    assert torch.are_deterministic_algorithms_enabled() == prior


def preparation_fixture(root, monkeypatch):
    a = [{"configuration": c.config_dict()} for c in e12.e10.enumerate_lr_selection_candidates()]
    c = [{"configuration": e12.e10.FullTrainingCandidate(family, task, 1e-3, seed, "final_seed").config_dict()}
         for family in e12.e10.MODEL_FAMILIES for task in e12.e10.TASKS for seed in (2026, 2027)]
    final = [r for r in a + c if r["configuration"]["learning_rate"] == 1e-3]
    final.sort(key=lambda r: (r["configuration"]["model_family"], r["configuration"]["task"], r["configuration"]["seed"]))
    report = {"completed_runs": a + c, "final_validation_selected_checkpoints": final, "overall_status": "PASS"}
    report["report_sha256"] = e12.d2.canonical_sha256(report)
    paths = e12.e10.report_paths(root, e12.STUDY)
    paths.final_report.parent.mkdir(parents=True)
    paths.final_report.write_bytes(e12.d2.canonical_json_bytes(report))
    real_hash = e12.e10._sha256
    monkeypatch.setattr(e12, "E10_REPORT_SHA256", real_hash(paths.final_report))
    paths.final_report_sha256.write_text(real_hash(paths.final_report))
    hashes = {}
    for name in ("X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy", "split_metadata.json"):
        path = root / "data/processed/preprocessed/exp2" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic integrity fixture; never loaded")
        hashes[f"exp2/{name}"] = real_hash(path)
    stage_c = root / "results/stage_c_full_validation.json"
    stage_c.parent.mkdir(exist_ok=True)
    stage_c.write_bytes(e12.d2.canonical_json_bytes({"status": "PASS", "artefact_sha256": hashes}))
    monkeypatch.setattr(e12, "STAGE_C_REPORT_SHA256", real_hash(stage_c))
    environment = {k: "synthetic" for k in ("python", "python_implementation", "torch", "torch_geometric", "cuda_runtime")}
    candidate = next(r["configuration"] for r in final if r["configuration"]["model_family"] == "htgnn" and r["configuration"]["task"] == "E2")
    candidate = e12.e10.FullTrainingCandidate(**{k: candidate[k] for k in ("model_family", "task", "learning_rate", "seed", "phase")})
    manifest_path = e12.e10.candidate_paths(root, e12.STUDY, candidate).manifest
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_bytes(e12.d2.canonical_json_bytes({"environment_identity": environment,
        "upstream_evidence_sha256": {}, "stage_c_provenance": {"split_artifact_sha256": {"exp2/split_metadata.json": hashes["exp2/split_metadata.json"]}}}))
    def digest(path):
        if path.name == "full_training.py":
            return e12.e10.provenance_policy.CORRECTED_RUNTIME_SHA256
        if path.name == "run_stage_e10_full_training.py":
            return e12.e10.provenance_policy.APPROVED_RUNNER_SHA256
        return real_hash(path)
    monkeypatch.setattr(e12.e10, "_sha256", digest)
    monkeypatch.setattr(e12.e10, "environment_identity", lambda: dict(environment))
    monkeypatch.setattr(e12.e10, "_validate_frozen_e9_evidence", lambda *args: {"frozen_upstream_source_sha256": {}, "e9_source_sha256": {}})
    monkeypatch.setattr(e12.e10, "_evidence_hashes", lambda *args: {})
    monkeypatch.setattr(e12.e10, "enumerate_final_seed_candidates_from_plan", lambda *args: tuple(range(18)))
    monkeypatch.setattr(e12.e10, "load_completed_results", lambda root, study, candidates: a if len(candidates) == 27 else c)
    return environment, paths


def test_synthetic_preparation_reconciles_pinned_records_and_allowed_inputs(tmp_path, monkeypatch):
    preparation_fixture(tmp_path, monkeypatch)
    with e12.phase_a_guard(tmp_path) as attempts:
        result = e12.prepare(tmp_path)
    assert len(result["records"]) == 3
    assert tuple(r["configuration"]["seed"] for r in result["records"]) == e12.SEEDS
    assert len(result["input_sha256"]) == 5 and attempts == []
    assert not any("test" in path for path in result["input_sha256"])


@pytest.mark.parametrize("fault", ("report", "environment", "train_array"))
def test_preparation_rejects_integrity_or_environment_drift(tmp_path, monkeypatch, fault):
    environment, paths = preparation_fixture(tmp_path, monkeypatch)
    if fault == "report":
        paths.final_report.write_text("changed")
    elif fault == "environment":
        environment["torch"] = "changed"
    else:
        (tmp_path / "data/processed/preprocessed/exp2/X_train.npy").write_bytes(b"changed")
    with e12.phase_a_guard(tmp_path), pytest.raises(e12.PruningError):
        e12.prepare(tmp_path)
