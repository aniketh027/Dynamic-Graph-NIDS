"""E.12B synthetic contracts only. Never open real predictive test arrays."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import threading

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pytest
import torch

from src.pruning import stage_e12b as b
from test_stage_e12_pruning import data, model
from test_stage_e11_test_evaluation import _metadata


@pytest.fixture(autouse=True)
def no_training(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("optimizer, backward or training executed")
    monkeypatch.setattr(torch.optim.Optimizer, "__init__", forbidden)
    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(torch.autograd, "backward", forbidden)
    monkeypatch.setattr(b.a.e7, "create_optimizer", forbidden)
    monkeypatch.setattr(b.a.e7, "run_training", forbidden)
    with b.a.deterministic_inference("cpu"):
        yield


def test_exact_mask_category_and_graph_contract():
    p = b.frozen_policy().record()
    assert p["pruned_feature_ids"] == [6, 10, 29, 32, 34, 35, 40, 41, 42, 50, 52, 57, 64, 70, 71, 72, 73]
    assert len(p["retained_feature_ids"]) == 61
    assert set(p["retained_feature_ids"]) | set(p["pruned_feature_ids"]) == set(range(78))
    assert not set(p["retained_feature_ids"]) & set(p["pruned_feature_ids"])
    assert p["pruned_graph"] == {"feature_nodes": 61, "category_nodes": 8, "sample_nodes": 1, "nodes": 70,
                                  "feature_category_edges": 61, "category_sample_edges": 8, "edges": 69}
    assert np.bincount([c for _, c in p["retained_feature_category_edges"]], minlength=8).tolist() == [2, 18, 18, 4, 9, 5, 3, 2]


def test_benchmark_protocol_cannot_be_mutated_through_returned_report():
    plan = b._benchmark_plan()
    plan["sequential_batch_starts"][0] = 999
    plan["other_scopes"].append("unapproved scope")
    assert b._benchmark_plan()["sequential_batch_starts"] == [0, 256, 512, 768]
    assert len(b._benchmark_plan()["other_scopes"]) == 2
    with pytest.raises(TypeError): b.BENCHMARK_PLAN["batch_size"] = 1


@pytest.mark.parametrize("size", (1, 3, 256))
def test_compaction_preserves_values_original_identity_edges_and_ownership(size):
    d = data(size)
    original = deepcopy(d)
    values, ids, edges = b.a.compact_inputs(d, b.frozen_policy(), size)
    assert values.shape == (61 * size, 5)
    for graph in range(size):
        selected = torch.tensor(b.RETAINED_IDS) + graph * 78
        assert torch.equal(values[graph * 61:(graph + 1) * 61], original["feature"].x[selected])
        assert ids[graph * 61:(graph + 1) * 61].tolist() == list(b.RETAINED_IDS)
        assert edges[0, graph * 61:(graph + 1) * 61].tolist() == list(range(graph * 61, (graph + 1) * 61))
        assert torch.equal(edges[1, graph * 61:(graph + 1) * 61], original[b.a.e2.FEATURE_RELATION].edge_index[1, selected])
    assert torch.equal(d["feature"].x, original["feature"].x)
    assert torch.equal(d[b.a.e2.CATEGORY_RELATION].edge_index, original[b.a.e2.CATEGORY_RELATION].edge_index)


@pytest.mark.parametrize("task", b.TASKS)
@pytest.mark.parametrize("size", (1, 3))
@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_zero_equivalence_fixed_mask_equivalence_and_unchanged_model(task, size, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    m, d = model(task, device), data(size, task=task, device=device)
    before = b.a.state_sha256(m)
    with torch.inference_mode():
        expected = m(d)
    assert torch.equal(b.forward(m, d, diagnostic_zero=True), expected)
    pruned = b.forward(m, d)
    assert torch.equal(pruned, b.a.pruned_logits(m, d, b.frozen_policy()))
    assert pruned.shape == ((size,) if task == "E3" else (size, 15))
    assert pruned.dtype == torch.float32 and not pruned.requires_grad
    assert b.a.state_sha256(m) == before
    assert all(p.grad is None and not p.requires_grad for p in m.parameters())


@pytest.mark.parametrize("task", b.TASKS)
def test_partial_batch_partition_and_graph_isolation(task):
    m, d = model(task), data(3, task=task)
    complete = b.forward(m, d)
    single = data(1, task=task)
    single["feature"].x = d["feature"].x[:78].clone()
    assert torch.allclose(b.forward(m, single)[0], complete[0], atol=1e-6, rtol=1e-5)
    changed = deepcopy(d)
    changed["feature"].x[78:156] += 9
    after = b.forward(m, changed)
    assert torch.allclose(after[[0, 2]], complete[[0, 2]], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("fault", ("task", "split", "feature_order", "ptr", "edge", "float_indices", "negative_indices", "bound", "target_dtype", "target_domain", "training", "gradients"))
def test_forward_rejects_malformed_contract(fault):
    m, d = model(), data()
    if fault == "task": d.e2_audit["experiment"] = "E1"
    elif fault == "split": d.e2_audit["split"] = "unknown"
    elif fault == "feature_order": d["feature"].feature_id[0] = 1
    elif fault == "ptr": d["category"].ptr[1] = 7
    elif fault == "edge": d[b.a.e2.FEATURE_RELATION].edge_index[1, 0] = 1
    elif fault == "float_indices": d.e2_audit["ordered_source_sample_indices"] = [0., 1.]
    elif fault == "negative_indices": d.e2_audit["ordered_source_sample_indices"] = [-1, 1]
    elif fault == "bound": d = data(257)
    elif fault == "target_dtype": d["sample"].y = d["sample"].y.float()
    elif fault == "target_domain": d["sample"].y[0] = 15
    elif fault == "training": m.train()
    else: m.requires_grad_(True)
    with pytest.raises((b.ComparisonError, b.a.PruningError, b.a.e7.TrainingContractError, ValueError)):
        b.forward(m, d)


def test_e3_float_targets_required():
    m, d = model("E3"), data(task="E3")
    assert b.forward(m, d).shape == (2,)
    d["sample"].y = d["sample"].y.long()
    with pytest.raises(b.ComparisonError, match="dtype"):
        b.forward(m, d)


def _reservation(state):
    authorization = {"policy_sha256": b.POLICY_SHA256, "freeze_sha256": b.FREEZE_SHA256,
                     "run_id": state.run_id, "state": "CONSUMED_OR_RESERVED",
                     "owner_confirmed_one_time_pruned_test_execution": True}
    authorization["authorization_sha256"] = b.a.d2.canonical_sha256(authorization)
    b._write(state.ledger, authorization)
    state.ledger_sha256 = b.a.e10._sha256(state.ledger)
    state.test_enabled = True


def _parents(root):
    for area in ("logs/stage_e12b", "results/stage_e12b"):
        (root / area / b.STUDY).mkdir(parents=True)


@pytest.mark.parametrize("task", b.TASKS)
def test_synthetic_test_adapter_requires_reservation_and_never_allows_zero(tmp_path, task):
    m, d = model(task), data(task=task, split="test")
    with pytest.raises(b.ComparisonError, match="reservation"):
        b.forward(m, d)
    _parents(tmp_path)
    with b.file_guard(tmp_path, "e12b-20261006-v1") as state:
        _reservation(state)
        assert b.forward(m, d).shape == ((2,) if task == "E3" else (2, 15))
        with pytest.raises(b.ComparisonError, match="unpruned test"):
            b.forward(m, d, diagnostic_zero=True)
    with pytest.raises(b.ComparisonError):
        b.forward(m, d)


@pytest.mark.parametrize("directory", ("exp1", "exp2", "exp3"))
@pytest.mark.parametrize("name", ("X_test.npy", "y_test.npy"))
def test_test_files_locked_until_explicit_synthetic_reservation(tmp_path, directory, name):
    p = tmp_path / "data/processed/preprocessed" / directory / name
    p.parent.mkdir(parents=True)
    np.save(p, np.zeros(2))
    with b.file_guard(tmp_path) as state:
        with pytest.raises(b.ComparisonError, match="locked"):
            np.load(p, mmap_mode="r")
        assert not state.opened_test and len(state.attempts) == 1
    _parents(tmp_path)
    with b.file_guard(tmp_path, "e12b-20261006-v1") as state:
        _reservation(state)
        assert np.load(p, mmap_mode="r").shape == (2,)
        assert p.relative_to(tmp_path).as_posix() in state.opened_test
        with pytest.raises(b.ComparisonError): np.load(p, mmap_mode="r+")
        with pytest.raises(b.ComparisonError): state.ledger.write_text("replace")
        with pytest.raises(b.ComparisonError): state.ledger.unlink()


@pytest.mark.parametrize("area", ("results/stage_e11", "logs/stage_e11", "results/stage_e12", "logs/stage_e12", "logs/stage_e10"))
def test_frozen_namespaces_and_baselines_immutable(tmp_path, area):
    p = tmp_path / area / "original.json"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"immutable")
    _parents(tmp_path)
    with b.file_guard(tmp_path, "e12b-20261006-v1"):
        assert p.read_bytes() == b"immutable"
        with pytest.raises(b.ComparisonError): p.write_bytes(b"change")
        with pytest.raises(b.ComparisonError): p.unlink()
        with pytest.raises(b.ComparisonError): p.rename(p.with_name("changed.json"))
    assert p.read_bytes() == b"immutable"


def test_lock_applies_to_worker_threads(tmp_path):
    p = tmp_path / "data/processed/preprocessed/exp2/X_test.npy"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"synthetic")
    errors = []
    def worker():
        try: p.read_bytes()
        except b.ComparisonError as exc: errors.append(str(exc))
    with b.file_guard(tmp_path):
        t = threading.Thread(target=worker); t.start(); t.join()
    assert len(errors) == 1


def _checkpoint(root, task):
    candidate = b.a.e10.FullTrainingCandidate("htgnn", task, 1e-3, 2025, "lr_selection")
    m = model(task)
    identity = b.a.e7.model_contract(m)
    metadata = {"schema_version": b.a.e7.CHECKPOINT_SCHEMA_VERSION, "harness_version": b.a.e7.HARNESS_VERSION,
                "task": task, "model_family": "htgnn", "model_contract_version": identity.architecture_version,
                "model_input_contract": identity.input_contract, "model_version": identity.architecture_version,
                "parameter_count": identity.parameter_count, "model_state_schema_sha256": b.a.e7.model_state_schema_sha256(m),
                "training_config": candidate.training_config.checkpoint_dict(), "selection_name": b.a.e7.SELECTION_NAMES[task],
                "frozen_source_sha256": dict(b.a.e7.e6.FROZEN_SOURCE_SHA256), "epoch": 1, "selection_value": .5}
    for relative in metadata["frozen_source_sha256"]:
        p = root / relative; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((Path(b.__file__).resolve().parents[2] / relative).read_bytes())
    path = b.a.e10.candidate_paths(root, b.STUDY, candidate).checkpoint
    path.parent.mkdir(parents=True)
    torch.save({"metadata": metadata, "model_state_dict": m.state_dict()}, path)
    record = {"candidate_id": candidate.candidate_id, "configuration": candidate.config_dict(),
              "artifacts": {"checkpoint_sha256": b.a.e10._sha256(path)}, "checkpoint_provenance": metadata,
              "checkpoint_provenance_verified": True, "best_epoch": 1, "best_validation_metric": .5}
    return m, path, record


@pytest.mark.parametrize("task", b.TASKS)
def test_all_task_checkpoint_loading_never_constructs_optimizer_and_preserves_bytes(tmp_path, task):
    original, path, record = _checkpoint(tmp_path, task)
    before = path.read_bytes()
    loaded = b.load_bound_model(tmp_path, record)
    assert b.a.state_sha256(loaded) == b.a.state_sha256(original)
    b.forward(loaded, data(task=task))
    assert b.a.state_sha256(loaded) == b.a.state_sha256(original) and path.read_bytes() == before


@pytest.mark.parametrize("fault", ("hash", "metadata", "state", "source", "family", "seed"))
def test_checkpoint_corruption_rejected(tmp_path, fault):
    _, path, record = _checkpoint(tmp_path, "E3")
    if fault == "hash": record["artifacts"]["checkpoint_sha256"] = "0" * 64
    elif fault == "metadata": record["checkpoint_provenance"]["epoch"] = 9
    elif fault == "source": (tmp_path / "src/graph/feature_encoder.py").write_bytes(b"changed")
    elif fault == "family": record["configuration"]["model_family"] = "final_flow_mlp"
    elif fault == "seed": record["configuration"]["seed"] = 2030
    else:
        payload = torch.load(path, weights_only=True)
        payload["model_state_dict"].pop("head.bias")
        torch.save(payload, path); record["artifacts"]["checkpoint_sha256"] = b.a.e10._sha256(path)
    with pytest.raises((b.ComparisonError, b.a.e7.TrainingContractError)):
        b.load_bound_model(tmp_path, record)


def _pairs():
    pruned, bindings = [], []
    for task in b.TASKS:
        policy = b.e11.reporting_policy(_metadata(task), task)
        if task == "E3":
            metrics = b.e11.binary_metrics(np.array([0, 1, 0, 1]), np.array([-1., 1., -2., 2.]), policy)
        else:
            metrics = b.e11.multiclass_metrics(np.diag(policy["test_support"]).astype(np.int64), policy)
        for seed in b.SEEDS:
            identity = f"{task}-{seed}"
            baseline = {"task": task, "seed": seed, "candidate_id": identity, "model_family": "htgnn", "model_state_unchanged": True,
                        "checkpoint_sha256": identity, "e10_result_canonical_sha256": identity, "examples": sum(policy["test_support"]),
                        "order_sha256": b._order_sha256(sum(policy["test_support"])), "metrics": deepcopy(metrics)}
            baseline["result_sha256"] = b.a.d2.canonical_sha256(baseline)
            bindings.append({"result": baseline, "canonical_sha256": baseline["result_sha256"], "physical_sha256": identity, "path": identity,
                             "reporting_policy": deepcopy(policy)})
            r = {k: deepcopy(baseline[k]) for k in ("task", "seed", "candidate_id", "model_family", "model_state_unchanged", "checkpoint_sha256", "examples", "order_sha256", "metrics")}
            r.update(policy_sha256=b.POLICY_SHA256, e10_result_sha256=identity, graph=b.frozen_policy().record(), model_state_sha256=identity)
            r["result_sha256"] = b.a.d2.canonical_sha256(r)
            pruned.append(r)
    return pruned, bindings


def test_e11_metric_parity_paired_deltas_and_sample_standard_deviation():
    pruned, bindings = _pairs()
    result = b.compare_results(pruned, bindings)
    assert len(result["paired"]) == 9 and len(result["seed_summary"]) == 3
    assert result["test_based_selection"] is False
    for row in result["paired"]:
        assert all(x["absolute_change"] == 0 for x in row["changes"].values())
    for summary in result["seed_summary"]:
        assert all(v["absolute_change"] == {"mean": 0., "sample_std_ddof1": 0., "defined_seeds": 3} for v in summary["metrics"].values())
    # Preserve E2 true-row filtering and E3 AP's fixed decision convention.
    assert pruned[0]["metrics"]["primary_true_row_filter"] == "eligible_classes"
    assert pruned[6]["metrics"]["primary_name"] == "pr_auc_average_precision"
    assert pruned[6]["metrics"]["logit_threshold"] == 0.


@pytest.mark.parametrize("fault", ("missing", "duplicate", "policy", "checkpoint", "coverage", "graph", "baseline", "family", "state"))
def test_comparison_rejects_changed_mask_or_baseline_binding(fault):
    pruned, bound = _pairs()
    if fault == "missing": pruned.pop()
    elif fault == "duplicate": pruned[1] = deepcopy(pruned[0])
    elif fault == "policy": pruned[0]["policy_sha256"] = "changed"
    elif fault == "checkpoint": pruned[0]["checkpoint_sha256"] = "changed"
    elif fault == "coverage": pruned[0]["examples"] -= 1
    elif fault == "graph": pruned[0]["graph"]["retained_feature_ids"][0] = 6
    elif fault == "baseline": bound[0]["result"]["metrics"]["primary"]["macro_f1"] = .1
    elif fault == "family": pruned[0]["model_family"] = "final_flow_mlp"
    else: pruned[0]["model_state_unchanged"] = False
    if len(pruned) == 9:
        for r in pruned:
            r.pop("result_sha256", None); r["result_sha256"] = b.a.d2.canonical_sha256(r)
    with pytest.raises(b.ComparisonError): b.compare_results(pruned, bound)


def test_benchmark_pairing_sync_timing_memory_and_scope(monkeypatch):
    m, d = model(), data(256)
    ticks, events = [0], []
    def clock():
        ticks[0] += 10_000; events.append("clock"); return ticks[0]
    def forward(m, d, diagnostic_zero=False):
        events.append("unpruned" if diagnostic_zero else "pruned")
        ticks[0] += 2_000_000 if diagnostic_zero else 1_000_000
        return torch.zeros(256, 15)
    monkeypatch.setattr(b.time, "perf_counter_ns", clock)
    monkeypatch.setattr(b, "forward", forward)
    monkeypatch.setattr(b.HeteroData, "to", lambda self, *args, **kwargs: self)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: events.append("sync"))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *args: events.append("reset"))
    for name, value in (("memory_allocated", 111), ("memory_reserved", 222), ("max_memory_allocated", 333), ("max_memory_reserved", 444)):
        monkeypatch.setattr(torch.cuda, name, lambda *args, value=value: value)
    result = b.benchmark(m, [d], torch.device("cuda"), warmups=1, pairs=4)
    assert [p["order"] for p in result["paired_measurements_ms"]] == [["unpruned", "pruned"], ["pruned", "unpruned"]] * 2
    assert result["forward"]["unpruned"]["median_ms"] == pytest.approx(2.01)
    assert result["forward"]["pruned"]["median_ms"] == pytest.approx(1.01)
    assert result["forward"]["pruned"]["examples_per_second"] == pytest.approx(256000 / 1.01)
    assert result["gpu_memory"]["pruned"]["reproducible"] is True
    assert result["gpu_memory"]["pruned"]["bytes"]["peak_reserved"] == 444
    assert events[:events.index("reset")].count("clock") == 2  # transfer scope only; warmups untimed
    for index, event in enumerate(events):
        if event == "reset":
            assert events[index - 1] == "sync"
            assert events[index + 1:index + 5] in (["clock", "unpruned", "sync", "clock"], ["clock", "pruned", "sync", "clock"])
    assert result["parameter_reduction"] is False and result["input_adapter_reduction_claimed"] is False
    allocation = [111]
    def varying_allocation(*args):
        allocation[0] += 1
        return allocation[0]
    monkeypatch.setattr(torch.cuda, "memory_allocated", varying_allocation)
    varied = b.benchmark(m, [d], torch.device("cuda"), warmups=1, pairs=2)
    assert varied["gpu_memory"]["pruned"]["reproducible"] is False
    assert varied["gpu_memory"]["pruned"]["bytes"] is None


@pytest.mark.parametrize("fault", ("test", "partial"))
def test_benchmark_never_accepts_test_or_partial_batch(fault):
    m, d = model(), data(256 if fault == "test" else 3, split="test" if fault == "test" else "validation")
    with pytest.raises(b.ComparisonError): b.benchmark(m, [d], torch.device("cpu"), warmups=1, pairs=2)


def test_owner_execution_ack_and_hashes_required_before_preflight(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "prepare", lambda *args: pytest.fail("preflight called"))
    kw = dict(run_id="e12b-20261006-v1", device="cpu", policy_sha256=b.POLICY_SHA256, freeze_sha256=b.FREEZE_SHA256, implementation_sha256="x")
    with pytest.raises(b.ComparisonError, match="acknowledgement"): b.run_owner(tmp_path, **kw)
    kw["policy_sha256"] = "wrong"
    with pytest.raises(b.ComparisonError, match="hashes"): b.run_owner(tmp_path, **kw, owner_confirmed_one_time_pruned_test_execution=True)


@pytest.mark.parametrize("fail", (False, True))
def test_synthetic_owner_orchestration_exclusive_ledger_and_failure_stop(tmp_path, monkeypatch, fail):
    # Fake records/results and synthetic callbacks; never invoke a real loader.
    _parents(tmp_path)
    pruned, bindings = _pairs()
    models = {r["candidate_id"]: model(r["task"]) for r in pruned}
    for r in pruned:
        r["model_state_sha256"] = b.a.state_sha256(models[r["candidate_id"]])
        r.pop("result_sha256"); r["result_sha256"] = b.a.d2.canonical_sha256(r)
    records = [{"candidate_id": r["candidate_id"], "configuration": {"task": r["task"], "seed": r["seed"]},
                "artifacts": {"checkpoint_sha256": r["checkpoint_sha256"]}} for r in pruned]
    preparation = {"implementation_sha256": "approved", "records": records, "baselines": bindings,
                   "splits": {t: {"policy": {}, "test_sha256": {}} for t in b.TASKS},
                   "model_state_sha256": {r["candidate_id"]: r["model_state_sha256"] for r in pruned}}
    monkeypatch.setattr(b, "prepare", lambda *args: deepcopy(preparation))
    monkeypatch.setattr(b, "load_bound_model", lambda root, rec: models[rec["candidate_id"]])
    monkeypatch.setattr(b, "validation_benchmark", lambda *args: {"synthetic": True})
    calls = []
    def evaluate(root, record, *args):
        b._test_access(root); calls.append(record["candidate_id"])
        if fail: raise b.ComparisonError("synthetic inference failure")
        return deepcopy(next(r for r in pruned if r["candidate_id"] == record["candidate_id"]))
    monkeypatch.setattr(b, "evaluate_pruned", evaluate)
    kwargs = dict(run_id="e12b-20261006-v1", device="cpu", policy_sha256=b.POLICY_SHA256,
                  freeze_sha256=b.FREEZE_SHA256, implementation_sha256="approved",
                  owner_confirmed_one_time_pruned_test_execution=True)
    if fail:
        with pytest.raises(b.ComparisonError, match="inference failure"): b.run_owner(tmp_path, **kwargs)
        assert len(calls) == 1
        assert (tmp_path / "logs/stage_e12b" / b.STUDY / kwargs["run_id"] / "failure.json").exists()
    else:
        report = b.run_owner(tmp_path, **kwargs)
        assert len(calls) == 9 and len(report["comparison"]["paired"]) == 9
        assert report["test_arrays_opened"] == [] and not report["unpruned_test_evaluation_repeated"]
    ledger = tmp_path / "logs/stage_e12b" / b.STUDY / "test_consumption.json"
    before = ledger.read_bytes()
    with pytest.raises(b.ComparisonError, match="retry"): b.run_owner(tmp_path, **kwargs)
    assert ledger.read_bytes() == before and len(calls) == (1 if fail else 9)


def test_frozen_attestation_verification_and_corruption(tmp_path):
    root = Path(b.__file__).resolve().parents[2]
    f = b.a._read(root / b.FREEZE_PATH)
    inputs = [b.FREEZE_PATH, str(Path(b.FREEZE_PATH).with_suffix(".sha256")),
              *f["e12a_artifact_physical_sha256"], *f["e12a_source_physical_sha256"]]
    for relative in inputs:
        target = tmp_path / relative; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / relative).read_bytes())
    assert b.verify_freeze(tmp_path)["policy_record"] == b.frozen_policy().record()
    policy = tmp_path / f"results/stage_e12/{b.STUDY}/e12a-20261006-v1/proposed_policy.json"
    policy.write_bytes(policy.read_bytes() + b" ")
    with pytest.raises(b.ComparisonError, match="changed"): b.verify_freeze(tmp_path)


def _authority_fixture(root, monkeypatch):
    """Copy compact provenance only; no predictive arrays/checkpoints or Git writes."""
    source = Path(b.__file__).resolve().parents[2]
    attestation = b.a._read(source / b.AUTHORITY_PATH)
    historical = b.a._read(source / b.E9_CONTRACT_PATH)
    paths = {b.AUTHORITY_PATH, str(Path(b.AUTHORITY_PATH).with_suffix(".sha256")),
             "PROJECT_HANDOFF.md", attestation["owner_authorization"]["evidence_path"],
             *attestation["authority_policy_physical_sha256"],
             *attestation["review_evidence_physical_sha256"],
             *historical["frozen_upstream_source_sha256"], *historical["e9_source_sha256"]}
    for relative in paths:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((source / relative).read_bytes())
    old_bytes = b.subprocess.check_output(
        ["git", "cat-file", "blob", f"{b.E9_FROZEN_COMMIT}:PROJECT_HANDOFF.md"], cwd=source)
    def historical_blob(command, *, cwd, timeout, stderr):
        assert command == ["git", "cat-file", "blob", f"{b.E9_FROZEN_COMMIT}:PROJECT_HANDOFF.md"]
        assert cwd == root.resolve() and timeout == 10
        return old_bytes
    monkeypatch.setattr(b.subprocess, "check_output", historical_blob)
    return attestation, historical


def _rewrite_authority(root, monkeypatch, attestation):
    # Rebind fixture physical/canonical hashes to exercise semantic checks too.
    attestation.pop("attestation_sha256", None)
    attestation["attestation_sha256"] = b.a.d2.canonical_sha256(attestation)
    path = root / b.AUTHORITY_PATH
    path.write_bytes(b.a.d2.canonical_json_bytes(attestation))
    physical = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".sha256").write_text(physical)
    monkeypatch.setattr(b, "AUTHORITY_SHA256", physical)


def test_authority_exact_old_to_new_binding_is_read_only_and_test_locked(tmp_path, monkeypatch):
    attestation, historical = _authority_fixture(tmp_path, monkeypatch)
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    monkeypatch.setattr(b.a, "run_phase_a", lambda *a, **k: pytest.fail("pruning selection executed"))
    monkeypatch.setattr(b, "run_owner", lambda *a, **k: pytest.fail("E.12B executed"))
    with b.file_guard(tmp_path) as access:
        result = b.verify_authority_reconciliation(tmp_path)
        assert result == attestation
        assert historical["e9_source_sha256"]["PROJECT_HANDOFF.md"] == b.HISTORICAL_HANDOFF_SHA256
        assert hashlib.sha256((tmp_path / "PROJECT_HANDOFF.md").read_bytes()).hexdigest() == b.CURRENT_HANDOFF_SHA256
        assert not access.test_enabled and access.opened_test == set() and access.attempts == []
        with pytest.raises(b.ComparisonError, match="reservation"): b._test_access(tmp_path)
    assert {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
    assert not (tmp_path / "logs/stage_e12b").exists()
    assert not (tmp_path / "results/stage_e12b").exists()


@pytest.mark.parametrize("fault", ("old_hash", "current_hash", "commit", "scope", "owner", "execution", "release"))
def test_authority_rejects_semantic_forgery_even_with_rebound_attestation_hash(tmp_path, monkeypatch, fault):
    f, _ = _authority_fixture(tmp_path, monkeypatch)
    if fault == "old_hash": f["historical_handoff_sha256"] = "f" * 64
    elif fault == "current_hash": f["current_handoff_sha256"] = "f" * 64
    elif fault == "commit": f["e9_frozen_commit"] = "f" * 40
    elif fault == "scope": f["scope"].append("src/training/baselines.py")
    elif fault == "owner": f["owner_authorization"]["authorized_scope"] = "any later source change"
    elif fault == "execution": f["e12b_execution_authorized"] = True
    else: f["git_release_authorized"] = True
    _rewrite_authority(tmp_path, monkeypatch, f)
    with b.file_guard(tmp_path), pytest.raises(b.ComparisonError): b.verify_authority_reconciliation(tmp_path)


@pytest.mark.parametrize("fault", ("missing", "corrupt", "malformed", "sidecar", "live_handoff", "owner_evidence", "historical_blob", "e9_evidence", "unrelated_source", "agents", "release_skill", "review_evidence"))
def test_authority_rejects_missing_corrupt_or_unrelated_changes(tmp_path, monkeypatch, fault):
    f, _ = _authority_fixture(tmp_path, monkeypatch)
    attestation = tmp_path / b.AUTHORITY_PATH
    if fault == "missing": attestation.unlink()
    elif fault == "corrupt": attestation.write_bytes(attestation.read_bytes() + b" ")
    elif fault == "malformed":
        attestation.write_text("{")
        monkeypatch.setattr(b, "AUTHORITY_SHA256", hashlib.sha256(attestation.read_bytes()).hexdigest())
    elif fault == "sidecar": attestation.with_suffix(".sha256").write_text("f" * 64)
    elif fault == "live_handoff": (tmp_path / "PROJECT_HANDOFF.md").write_bytes(b"arbitrary future edit")
    elif fault == "owner_evidence": (tmp_path / f["owner_authorization"]["evidence_path"]).write_text("forged")
    elif fault == "historical_blob": monkeypatch.setattr(b.subprocess, "check_output", lambda *a, **k: b"wrong historical bytes")
    elif fault == "e9_evidence": (tmp_path / b.E9_CONTRACT_PATH).write_bytes(b"changed")
    elif fault == "unrelated_source": (tmp_path / "src/training/baselines.py").write_bytes(b"changed")
    elif fault == "agents": (tmp_path / "AGENTS.md").write_bytes(b"changed")
    elif fault == "release_skill": (tmp_path / ".agents/skills/nids-release-audit/SKILL.md").write_bytes(b"changed")
    else: (tmp_path / b.E11_REPORT_PATH).write_bytes(b"changed")
    with b.file_guard(tmp_path), pytest.raises(b.ComparisonError): b.verify_authority_reconciliation(tmp_path)


def test_original_e9_handoff_pin_cannot_be_replaced_even_with_rebound_evidence(tmp_path, monkeypatch):
    f, e9 = _authority_fixture(tmp_path, monkeypatch)
    e9["e9_source_sha256"]["PROJECT_HANDOFF.md"] = b.CURRENT_HANDOFF_SHA256
    e9.pop("report_sha256")
    e9["report_sha256"] = b.a.d2.canonical_sha256(e9)
    path = tmp_path / b.E9_CONTRACT_PATH
    path.write_bytes(b.a.d2.canonical_json_bytes(e9))
    physical = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".sha256").write_text(physical)
    monkeypatch.setattr(b, "E9_CONTRACT_SHA256", physical)
    f["e9_contract_physical_sha256"] = physical
    f["e9_contract_canonical_sha256"] = e9["report_sha256"]
    for relative in (b.E9_CONTRACT_PATH, Path(b.E9_CONTRACT_PATH).with_suffix(".sha256").as_posix()):
        f["review_evidence_physical_sha256"][relative] = hashlib.sha256((tmp_path / relative).read_bytes()).hexdigest()
    _rewrite_authority(tmp_path, monkeypatch, f)
    with b.file_guard(tmp_path), pytest.raises(b.ComparisonError, match="original E.9 historical handoff pin"):
        b.verify_authority_reconciliation(tmp_path)


def test_reconciled_authority_never_unlocks_even_synthetic_test_arrays(tmp_path, monkeypatch):
    _authority_fixture(tmp_path, monkeypatch)
    for directory in b.a.d4.EXPERIMENT_DIRS.values():
        for name in ("X_test.npy", "y_test.npy"):
            path = tmp_path / "data/processed/preprocessed" / directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic sentinel; must never open")
    with b.file_guard(tmp_path) as access:
        b.verify_authority_reconciliation(tmp_path)
        for directory in b.a.d4.EXPERIMENT_DIRS.values():
            for name in ("X_test.npy", "y_test.npy"):
                with pytest.raises(b.ComparisonError, match="locked"):
                    np.load(tmp_path / "data/processed/preprocessed" / directory / name, mmap_mode="r")
        assert access.opened_test == set() and len(access.attempts) == 6
        assert not access.test_enabled and access.ledger_sha256 is None


def test_e12b_upstream_reuses_all_frozen_checks_with_only_authority_substitution(tmp_path, monkeypatch):
    from test_stage_e12_pruning import preparation_fixture
    preparation_fixture(tmp_path, monkeypatch)
    frozen = b.a.prepare(tmp_path)
    monkeypatch.setattr(b, "verify_authority_reconciliation", lambda root: {
        "attestation_sha256": "synthetic-authority", "e9_contract_canonical_sha256": "synthetic-e9"})
    monkeypatch.setattr(b.a.e10, "_validate_frozen_e9_evidence", lambda root: {"report_sha256": "synthetic-e9"})
    with b.file_guard(tmp_path) as access:
        result = b._prepare_upstream(tmp_path)
    binding = result.pop("authority_reconciliation")
    assert result == frozen and len(result["input_sha256"]) == 5
    assert binding["historical_handoff_sha256"] == b.HISTORICAL_HANDOFF_SHA256
    assert binding["current_handoff_sha256"] == b.CURRENT_HANDOFF_SHA256
    assert not access.opened_test and not access.attempts


@pytest.mark.parametrize("fault", ("report", "environment", "train_array", "split", "upstream_evidence", "e9_binding", "runtime_source"))
def test_e12b_upstream_retains_nonhandoff_rejections(tmp_path, monkeypatch, fault):
    from test_stage_e12_pruning import preparation_fixture
    environment, paths = preparation_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(b, "verify_authority_reconciliation", lambda root: {
        "attestation_sha256": "synthetic-authority", "e9_contract_canonical_sha256": "synthetic-e9"})
    monkeypatch.setattr(b.a.e10, "_validate_frozen_e9_evidence", lambda root: {"report_sha256": "synthetic-e9"})
    if fault == "report": paths.final_report.write_bytes(b"changed")
    elif fault == "environment": environment["torch"] = "changed"
    elif fault == "train_array": (tmp_path / "data/processed/preprocessed/exp2/X_train.npy").write_bytes(b"changed")
    elif fault == "split": (tmp_path / "data/processed/preprocessed/exp2/split_metadata.json").write_bytes(b"changed")
    elif fault == "upstream_evidence": monkeypatch.setattr(b.a.e10, "_evidence_hashes", lambda root: {"unknown": "f" * 64})
    elif fault == "e9_binding": monkeypatch.setattr(b.a.e10, "_validate_frozen_e9_evidence", lambda root: {"report_sha256": "wrong"})
    else:
        original = b.a.e10._sha256
        monkeypatch.setattr(b.a.e10, "_sha256", lambda p: "f" * 64 if p.name == "full_training.py" else original(p))
    with b.file_guard(tmp_path), pytest.raises((b.ComparisonError, b.a.PruningError)):
        b._prepare_upstream(tmp_path)


def test_missing_authority_blocks_e12b_prepare_before_any_loader_or_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "verify_freeze", lambda root: {})
    monkeypatch.setattr(b.a.e10, "load_completed_results", lambda *a, **k: pytest.fail("records loaded before authority check"))
    monkeypatch.setattr(b, "load_bound_model", lambda *a, **k: pytest.fail("model loaded before authority check"))
    with b.file_guard(tmp_path) as access, pytest.raises(b.ComparisonError, match="authority reconciliation"):
        b.prepare(tmp_path)
    assert not access.opened_test and not access.test_enabled
    assert not (tmp_path / "logs/stage_e12b").exists()


def test_authority_is_rechecked_after_upstream_preflight(tmp_path, monkeypatch):
    from test_stage_e12_pruning import preparation_fixture
    preparation_fixture(tmp_path, monkeypatch)
    calls = []
    def verify(root):
        calls.append(root)
        if len(calls) == 2:
            raise b.ComparisonError("live authority handoff changed")
        return {"attestation_sha256": "synthetic-authority", "e9_contract_canonical_sha256": "synthetic-e9"}
    monkeypatch.setattr(b, "verify_authority_reconciliation", verify)
    monkeypatch.setattr(b.a.e10, "_validate_frozen_e9_evidence", lambda root: {"report_sha256": "synthetic-e9"})
    with b.file_guard(tmp_path), pytest.raises(b.ComparisonError, match="live authority handoff changed"):
        b._prepare_upstream(tmp_path)
    assert len(calls) == 2


def test_old_owner_implementation_bundle_is_rejected_before_reservation(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "prepare", lambda root: {"implementation_sha256": "new-reviewed-bundle"})
    with pytest.raises(b.ComparisonError, match="implementation source bundle changed"):
        b.run_owner(tmp_path, run_id="e12b-20261006-v1", device="cpu",
                    policy_sha256=b.POLICY_SHA256, freeze_sha256=b.FREEZE_SHA256,
                    implementation_sha256="0fd1177ef4128a7384f5ef4c5b8cecdf0e1469e0448499d6f5df9d279e930002",
                    owner_confirmed_one_time_pruned_test_execution=True)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("task", b.TASKS)
@pytest.mark.parametrize("bad_order", (False, True))
def test_synthetic_pruned_streaming_partial_batches_and_e11_metric_parity(tmp_path, monkeypatch, task, bad_order):
    _parents(tmp_path)
    candidate = b.a.e10.FullTrainingCandidate("htgnn", task, 1e-3, 2025, "lr_selection")
    record = {"candidate_id": candidate.candidate_id, "configuration": candidate.config_dict(),
              "artifacts": {"checkpoint_sha256": "synthetic"}, "result_sha256": "synthetic"}
    m = model(task)
    with torch.no_grad():
        m.head.weight.zero_(); m.head.bias.zero_()
        if task != "E3": m.head.bias[0] = 1.
    p = b.e11.reporting_policy(_metadata(task), task)
    y = np.repeat(np.arange(len(p["test_support"])), p["test_support"])
    first, second = data(2, task=task, split="test"), data(len(y) - 2, task=task, split="test", start=2)
    first["sample"].y = torch.tensor(y[:2], dtype=torch.float32 if task == "E3" else torch.long)
    second["sample"].y = torch.tensor(y[2:], dtype=first["sample"].y.dtype)
    if bad_order: second.e2_audit["ordered_source_sample_indices"][0] = 1
    monkeypatch.setattr(b, "load_bound_model", lambda *args: m)
    monkeypatch.setattr(b, "iter_test_batches", lambda *args: iter((first, second)))
    with b.file_guard(tmp_path, "e12b-20261006-v1") as state:
        scratch = tmp_path / "logs/stage_e12b" / b.STUDY / state.run_id
        scratch.mkdir(); _reservation(state)
        if bad_order:
            with pytest.raises(b.ComparisonError, match="traversal"):
                b.evaluate_pruned(tmp_path, record, p, scratch, torch.device("cpu"))
        else:
            before = b.a.state_sha256(m)
            result = b.evaluate_pruned(tmp_path, record, p, scratch, torch.device("cpu"))
            if task == "E3":
                expected = b.e11.binary_metrics(y, np.zeros(len(y)), p)
            else:
                confusion = np.zeros((15, 15), dtype=np.int64); confusion[:, 0] = p["test_support"]
                expected = b.e11.multiclass_metrics(confusion, p)
            assert result["metrics"] == expected and result["examples"] == len(y)
            assert result["model_state_sha256"] == before == b.a.state_sha256(m)
            b._canonical(result, "result_sha256")
        assert list(scratch.iterdir()) == []  # score-only scratch maps closed and removed


def test_paired_statistics_use_three_seed_sample_std_and_reject_forged_metrics():
    pruned, bindings = _pairs()
    values = []
    for index, errors in enumerate((0, 10, 20)):
        r = pruned[index]
        c = np.asarray(r["metrics"]["confusion_matrix"], dtype=np.int64)
        c[0, 0] -= errors; c[0, 1] += errors
        r["metrics"] = b.e11.multiclass_metrics(c, bindings[index]["reporting_policy"])
        values.append(r["metrics"]["primary"]["macro_f1"])
        r.pop("result_sha256"); r["result_sha256"] = b.a.d2.canonical_sha256(r)
    measured = b.compare_results(pruned, bindings)["seed_summary"][0]["metrics"]["primary.macro_f1"]
    assert measured["pruned"]["mean"] == pytest.approx(np.mean(values))
    assert measured["pruned"]["sample_std_ddof1"] == pytest.approx(np.std(values, ddof=1))
    assert measured["absolute_change"]["sample_std_ddof1"] == pytest.approx(np.std(values, ddof=1))
    pruned[0]["metrics"]["primary"]["macro_f1"] = .01
    pruned[0].pop("result_sha256"); pruned[0]["result_sha256"] = b.a.d2.canonical_sha256(pruned[0])
    with pytest.raises(b.ComparisonError, match="metric/confusion"):
        b.compare_results(pruned, bindings)


def test_validation_benchmark_uses_predeclared_frozen_iterator_contract(tmp_path, monkeypatch):
    m = model()
    fake = object()
    monkeypatch.setattr(b, "_dataset", lambda root, task, split: fake)
    observed = []
    def batches(dataset, *, task, split, batch_size, epoch_seed, max_batches):
        assert dataset is fake and task == "E2" and split == "validation"
        assert batch_size == 256 and epoch_seed is None and max_batches == 4
        for start in (0, 256, 512, 768):
            observed.append(start); yield data(256, start=start)
    monkeypatch.setattr(b.a.e7, "iter_production_batches", batches)
    monkeypatch.setattr(b, "benchmark", lambda m, batches, device: {"batches": len(batches)})
    result = b.validation_benchmark(tmp_path, m, torch.device("cpu"))
    assert observed == [0, 256, 512, 768] and result["batches"] == 4
    assert result["loading_and_conversion"]["observations"] == 4
