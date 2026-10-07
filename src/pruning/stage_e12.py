"""E.12A owner-run validation-only pruning of frozen HTGNN weights.

No E.12B executor, optimizer, backward pass, training, or test reader exists.
The separate inference path compacts features while reusing the original modules.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import time
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch_geometric.data import HeteroData

from src.graph import adapter as d4, pyg_adapter as e2, schema as d2
from src.graph.htgnn_classifier import HTGNNClassifier, _validated_head
from src.training import full_training as e10, harness as e7

VERSION = "stage-e12a-attention-feature-pruning-v1"
STUDY = "e10-official-20261001-v1"
E10_REPORT_SHA256 = "8ff3f89c4aac3b2948936ea794e710059a7fe441c161b8c20cae0b8e51f6639c"
STAGE_C_REPORT_SHA256 = "d1d33012e0135533585952aa8234154e03870ead0a78aa48edae393a602b3718"
RATIOS = (0.0, 0.10, 0.25, 0.50, 0.75)
SEEDS = (2025, 2026, 2027)
BATCH_SIZE = 256
RANK_BATCHES = 32
RANK_SEED = 12012
_GUARDS: list[tuple[Path, list[str]]] = []
_HOOK_INSTALLED = False


class PruningError(ValueError):
    """Invalid pruning contract, provenance, or phase access."""


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _audit_hook(event: str, args: tuple[Any, ...]) -> None:
    if not _GUARDS:
        return
    writing = False
    if event == "open":
        paths = [args[0]]
        mode, flags = args[1:3]
        writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
            isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)))
    elif event in {"os.remove", "os.rmdir", "os.mkdir", "os.chmod", "os.utime"}:
        paths, writing = [args[0]], True
    elif event in {"os.rename", "os.link", "os.symlink"}:
        paths, writing = list(args[:2]), True
    else:
        return
    for value in paths:
        if not isinstance(value, (str, bytes, os.PathLike)):
            continue
        lexical = Path(os.fsdecode(value)).absolute()
        resolved = lexical.resolve()
        for root, attempts in tuple(_GUARDS):
            if not (_inside(lexical, root) or _inside(resolved, root)):
                continue
            approved_arrays = {root / "data/processed/preprocessed/exp2" / name
                               for name in ("X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy")}
            reason = None
            if event == "open" and any(_inside(p, root / area / "stage_e11")
                                        for p in (lexical, resolved) for area in ("logs", "results")):
                reason = "E.11 evidence is unavailable during policy selection"
            if event == "open" and any(_inside(p, root / "data") and p.suffix == ".npy"
                                        for p in (lexical, resolved)):
                if writing or lexical not in approved_arrays or resolved not in approved_arrays:
                    reason = "predictive array access locked; E.12A permits only E2 train/validation"
            creating_parent = event == "os.mkdir" and lexical == resolved and lexical in (root / "logs", root / "results")
            if writing and not creating_parent and not any(_inside(lexical, root / area / "stage_e12") and
                                   _inside(resolved, root / area / "stage_e12")
                                   for area in ("logs", "results")):
                reason = "writes outside stage_e12 are prohibited"
            if reason:
                attempts.append(f"{event}:{lexical}:{reason}")
                raise PruningError(reason)


@contextmanager
def phase_a_guard(root: Path) -> Iterator[list[str]]:
    """Process-wide accidental-access guard, including Python worker threads.

Not a security sandbox against arbitrary native/external code or hard-link attacks.
"""
    global _HOOK_INSTALLED
    if not _HOOK_INSTALLED:
        sys.addaudithook(_audit_hook)
        _HOOK_INSTALLED = True
    item = (Path(root).resolve(), [])
    _GUARDS.append(item)
    try:
        yield item[1]
    finally:
        _GUARDS.remove(item)


def _ratio(value: float) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value < 1:
        raise PruningError("pruning ratio must be finite and in [0,1)")
    return float(value)


@dataclass(frozen=True)
class FeaturePolicy:
    requested_ratio: float
    retained_feature_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        _ratio(self.requested_ratio)
        ids = self.retained_feature_ids
        if (not isinstance(ids, tuple) or not ids or any(type(i) is not int or not 0 <= i < 78 for i in ids)
                or tuple(sorted(set(ids))) != ids):
            raise PruningError("retained IDs must be unique canonical original feature IDs")
        categories = d2.FEATURE_TO_CATEGORY_DESTINATION_INDICES
        for category in range(8):
            members = [i for i in range(78) if categories[i] == category]
            expected = max(1, len(members) - math.floor(self.requested_ratio * len(members)))
            if sum(categories[i] == category for i in ids) != expected:
                raise PruningError("policy violates the per-category retention/connectivity quota")

    def record(self) -> dict[str, Any]:
        retained = set(self.retained_feature_ids)
        k = len(retained)
        return {
            "requested_per_category_pruning_ratio": self.requested_ratio,
            "actual_feature_pruning_ratio": (78 - k) / 78,
            "actual_total_node_pruning_ratio": (78 - k) / 87,
            "actual_total_edge_pruning_ratio": (78 - k) / 86,
            "original_graph": {"feature_nodes": 78, "category_nodes": 8, "sample_nodes": 1,
                               "nodes": 87, "feature_category_edges": 78, "category_sample_edges": 8, "edges": 86},
            "pruned_graph": {"feature_nodes": k, "category_nodes": 8, "sample_nodes": 1,
                             "nodes": k + 9, "feature_category_edges": k, "category_sample_edges": 8, "edges": k + 8},
            "retained_feature_ids": list(self.retained_feature_ids),
            "pruned_feature_ids": [i for i in range(78) if i not in retained],
            "retained_feature_category_edges": [[i, d2.FEATURE_TO_CATEGORY_DESTINATION_INDICES[i]] for i in self.retained_feature_ids],
            "retained_category_sample_edges": [[i, 0] for i in range(8)],
            "input_adapter_graph_stays_full": True,
            "inference_computation_compacts_features": True,
        }


def make_policy(scores: Sequence[float], ratio: float) -> FeaturePolicy:
    ratio = _ratio(ratio)
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (78,) or not np.isfinite(values).all() or (values < 0).any():
        raise PruningError("scores must be 78 finite nonnegative values")
    keep = []
    for category in range(8):
        members = [i for i, c in enumerate(d2.FEATURE_TO_CATEGORY_DESTINATION_INDICES) if c == category]
        count = max(1, len(members) - math.floor(ratio * len(members)))
        keep.extend(sorted(members, key=lambda i: (-values[i], i))[:count])
    return FeaturePolicy(ratio, tuple(sorted(keep)))


def state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(d2.canonical_json_bytes([name, str(value.dtype), list(value.shape)]))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def deterministic_inference(device: str) -> Iterator[None]:
    """Use the same deterministic kernels for baseline and reduced inference."""
    if device == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise PruningError("CUDA requires CUBLAS_WORKSPACE_CONFIG before process startup")
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    benchmark_before = torch.backends.cudnn.benchmark
    deterministic_before = torch.backends.cudnn.deterministic
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)
        torch.backends.cudnn.benchmark = benchmark_before
        torch.backends.cudnn.deterministic = deterministic_before


def _frozen(model: HTGNNClassifier) -> None:
    if not isinstance(model, HTGNNClassifier):
        raise PruningError("E.12 supports only the frozen HTGNN")
    if any(module.training for module in model.modules()) or any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise PruningError("all model modules must be eval with gradients disabled and absent")
    if next(model.parameters()).device.type == "cuda" and not torch.are_deterministic_algorithms_enabled():
        raise PruningError("CUDA pruning requires the deterministic inference context")
    e7.model_contract(model)


def _validate_batch(model: HTGNNClassifier, data: HeteroData) -> int:
    _frozen(model)
    audit = getattr(data, "e2_audit", {})
    split = audit.get("split")
    if split not in {"train", "validation"}:
        raise PruningError("E.12A rejects test and unknown splits")
    indices = e7._require_batch_audit(data, model.task, split)
    b = len(indices)
    x = data["feature"].x
    if x.shape != (b * 78, 5) or x.dtype != torch.float32 or not torch.isfinite(x).all():
        raise PruningError("invalid feature values")
    for kind, count, identity_name in (("feature", 78, "feature_id"), ("category", 8, "category_id"), ("sample", 1, None)):
        store = data[kind]
        expected_batch = torch.arange(b, device=x.device).repeat_interleave(count)
        expected_ptr = torch.arange(b + 1, device=x.device) * count
        if (store.batch.dtype != torch.long or store.ptr.dtype != torch.long or
                not torch.equal(store.batch, expected_batch) or not torch.equal(store.ptr, expected_ptr)):
            raise PruningError("invalid graph-major batch ownership")
        if identity_name:
            ids = store[identity_name]
            if ids.dtype != torch.long or not torch.equal(ids, torch.arange(count, device=x.device).repeat(b)):
                raise PruningError("original node identity/order mismatch")
    model.graph_encoder._validate_inputs(data, x.new_empty((b * 78, 32)))
    return b


def compact_inputs(data: HeteroData, policy: FeaturePolicy, b: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Use the frozen relation as authority; only remove feature rows/edges."""
    policy = FeaturePolicy(policy.requested_ratio, policy.retained_feature_ids)
    device = data["feature"].x.device
    ids = torch.tensor(policy.retained_feature_ids, device=device)
    rows = (torch.arange(b, device=device)[:, None] * 78 + ids).reshape(-1)
    edges = data[e2.FEATURE_RELATION].edge_index[:, rows].clone()
    edges[0] = torch.arange(rows.numel(), device=device)
    return data["feature"].x[rows], data["feature"].feature_id[rows], edges


@torch.inference_mode()
def pruned_logits(model: HTGNNClassifier, data: HeteroData, policy: FeaturePolicy) -> torch.Tensor:
    b = _validate_batch(model, data)
    policy = FeaturePolicy(policy.requested_ratio, policy.retained_feature_ids)
    if policy.requested_ratio == 0:
        return model(data)  # exact original call, including original kernel shapes
    values, ids, edges = compact_inputs(data, policy, b)
    feature_embeddings = model.feature_encoder(values, ids)
    graph = model.graph_encoder
    category_targets = graph.category_embedding(data["category"].category_id)
    message = graph.feature_to_category((feature_embeddings, category_targets), edges)
    categories = graph.feature_to_category_activation(
        graph.feature_to_category_norm(graph.feature_to_category_dropout(message) + category_targets))
    sample_targets = graph.sample_token.unsqueeze(0).expand(b, -1)
    message = graph.category_to_sample((categories, sample_targets), data[e2.CATEGORY_RELATION].edge_index)
    samples = graph.category_to_sample_activation(
        graph.category_to_sample_norm(graph.category_to_sample_dropout(message) + sample_targets))
    contract, head = _validated_head(model)
    logits = head(samples)
    return logits.squeeze(-1) if contract.output_channels == 1 else logits


def _attention_arrays(data: HeteroData, attention: Mapping[Any, Any], b: int) -> tuple[np.ndarray, np.ndarray]:
    if tuple(attention) != (e2.FEATURE_RELATION, e2.CATEGORY_RELATION):
        raise PruningError("attention relation contract mismatch")
    arrays = []
    for relation, count in ((e2.FEATURE_RELATION, 78), (e2.CATEGORY_RELATION, 8)):
        edges, alpha = attention[relation]
        if (not torch.equal(edges, data[relation].edge_index) or alpha.shape != (b * count, 4)
                or alpha.dtype != torch.float32 or not torch.isfinite(alpha).all() or (alpha < 0).any()):
            raise PruningError("attention shape, edge mapping, or values invalid")
        sums = alpha.new_zeros((b * (8 if count == 78 else 1), 4))
        sums.index_add_(0, edges[1], alpha)
        if not torch.allclose(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5):
            raise PruningError("attention must be evaluation-mode destination-normalized softmax")
        arrays.append(alpha.cpu().numpy().reshape(b, count, 4))
    return arrays[0], arrays[1]


@torch.inference_mode()
def collect_scores(model: HTGNNClassifier, batches: Iterator[HeteroData], device: torch.device) -> dict[str, Any]:
    if model.task != "E2":
        raise PruningError("E.12A ranking requires the primary E2 model")
    feature_sum, category_sum = np.zeros((78, 4)), np.zeros((8, 4))
    attention_digest, order_digest = hashlib.sha256(), hashlib.sha256()
    count, batch_count = 0, 0
    support = np.zeros(15, dtype=np.int64)
    for data in batches:
        if getattr(data, "e2_audit", {}).get("split") != "train":
            raise PruningError("ranking is train-only")
        data = data.to(device)
        b = _validate_batch(model, data)
        _, attention = model(data, return_attention=True)
        feature, category = _attention_arrays(data, attention, b)
        feature_sum += feature.astype(np.float64).sum(axis=0)
        category_sum += category.astype(np.float64).sum(axis=0)
        for relation in (e2.FEATURE_RELATION, e2.CATEGORY_RELATION):
            for tensor in attention[relation]:
                attention_digest.update(tensor.cpu().contiguous().numpy().tobytes())
        for i in data.e2_audit["ordered_source_sample_indices"]:
            order_digest.update(int(i).to_bytes(8, "little"))
        support += np.bincount(data["sample"].y.cpu().numpy().astype(np.int64), minlength=15)
        count += b
        batch_count += 1
    if not count:
        raise PruningError("ranking requires training observations")
    return {"examples": count, "batches": batch_count, "sample_order_sha256": order_digest.hexdigest(),
            "raw_attention_stream_sha256": attention_digest.hexdigest(),
            "feature_per_head_mean": (feature_sum / count).tolist(),
            "category_per_head_mean_diagnostic_only": (category_sum / count).tolist(),
            "train_support_observed": support.tolist()}


def _check_tolerance(value: float, rationale: str) -> None:
    if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value < 1 or not rationale.strip():
        raise PruningError("owner tolerance must be relative, finite in [0,1), with a rationale")


def select_candidate(rows: Sequence[Mapping[str, Any]], tolerance: float) -> Mapping[str, Any]:
    """Largest reduction with EACH seed F1 >= (1-tolerance)*its baseline."""
    _check_tolerance(tolerance, "validated contract")
    if len(rows) != len(RATIOS) or tuple(row["ratio"] for row in rows) != RATIOS:
        raise PruningError("selection requires the exact predeclared sweep")
    for row in rows:
        policy = FeaturePolicy(row["ratio"], tuple(row["policy"]["retained_feature_ids"]))
        if policy.record() != row["policy"]:
            raise PruningError("candidate policy content mismatch")
        if len(row["seed_results"]) != 3 or tuple(r["seed"] for r in row["seed_results"]) != SEEDS:
            raise PruningError("selection requires all three frozen seeds in canonical order")
        for result in row["seed_results"]:
            value = result["macro_f1"]
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise PruningError("invalid validation metric")
    baseline = [r["macro_f1"] for r in rows[0]["seed_results"]]
    if any(value <= 0 for value in baseline):
        raise PruningError("relative tolerance requires positive unpruned baselines")
    eligible = [row for row in rows if all(result["macro_f1"] >= (1 - tolerance) * base
                for result, base in zip(row["seed_results"], baseline, strict=True))]
    return max(eligible, key=lambda row: (78 - len(row["policy"]["retained_feature_ids"]), -row["ratio"]))


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PruningError("expected a JSON object")
    return value


def load_bound_model(root: Path, record: Mapping[str, Any]) -> HTGNNClassifier:
    """Inference-only loading of a record authenticated by the pinned E.10 report.

Reuses E.7 model contract and tensor preflight. The optimizer payload is never
instantiated/restored; historical optimizer validation remains E.10 provenance.
"""
    config = record["configuration"]
    if config["model_family"] != "htgnn" or config["task"] != "E2" or config["seed"] not in SEEDS:
        raise PruningError("E.12A accepts only frozen E2 HTGNN checkpoints")
    candidate = e10.FullTrainingCandidate(config["model_family"], config["task"], config["learning_rate"], config["seed"], config["phase"])
    if config != candidate.config_dict():
        raise PruningError("checkpoint configuration mismatch")
    path = e10.candidate_paths(root, STUDY, candidate).checkpoint
    expected_hash = record["artifacts"]["checkpoint_sha256"]
    if e10._sha256(path) != expected_hash:
        raise PruningError("checkpoint bytes changed")
    payload = torch.load(path, weights_only=True, map_location="cpu")
    metadata = payload["metadata"]
    if metadata != record["checkpoint_provenance"] or record.get("checkpoint_provenance_verified") is not True:
        raise PruningError("checkpoint metadata binding mismatch")
    model = HTGNNClassifier("E2")
    identity = e7.model_contract(model)
    expected = {"schema_version": e7.CHECKPOINT_SCHEMA_VERSION, "harness_version": e7.HARNESS_VERSION,
                "task": "E2", "model_family": "htgnn", "model_contract_version": identity.architecture_version,
                "model_input_contract": identity.input_contract, "model_version": identity.architecture_version,
                "parameter_count": identity.parameter_count, "model_state_schema_sha256": e7.model_state_schema_sha256(model),
                "training_config": candidate.training_config.checkpoint_dict(), "selection_name": e7.SELECTION_NAMES["E2"],
                "frozen_source_sha256": dict(e7.e6.FROZEN_SOURCE_SHA256),
                "epoch": record["best_epoch"], "selection_value": record["best_validation_metric"]}
    if any(metadata.get(k) != v for k, v in expected.items()):
        raise PruningError("frozen model/selection metadata mismatch")
    for relative, expected_digest in metadata["frozen_source_sha256"].items():
        if e10._sha256(root / relative) != expected_digest:
            raise PruningError("frozen model execution source changed")
    state = payload["model_state_dict"]
    e7._preflight_model_state_dict(model, state)
    if any(not torch.isfinite(t).all() for t in state.values()):
        raise PruningError("nonfinite checkpoint tensors")
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False).eval()
    if e10._sha256(path) != expected_hash:
        raise PruningError("checkpoint changed while loading")
    return model


def prepare(root: Path) -> dict[str, Any]:
    """Read authenticated E.10 records and metadata; never consult E.11."""
    paths = e10.report_paths(root, STUDY)
    if e10._sha256(paths.final_report) != E10_REPORT_SHA256 or paths.final_report_sha256.read_text().split()[0] != E10_REPORT_SHA256:
        raise PruningError("pinned E.10 report changed")
    report = _read(paths.final_report)
    unhashed = dict(report)
    if unhashed.pop("report_sha256", None) != d2.canonical_sha256(unhashed) or report["overall_status"] != "PASS":
        raise PruningError("E.10 report canonical/content validation failed")
    for path, expected in (("src/training/full_training.py", e10.provenance_policy.CORRECTED_RUNTIME_SHA256),
                           ("scripts/run_stage_e10_full_training.py", e10.provenance_policy.APPROVED_RUNNER_SHA256)):
        if e10._sha256(root / path) != expected:
            raise PruningError("reviewed E.10 execution source changed")
    completed = e10.load_completed_results(root, STUDY, e10.enumerate_lr_selection_candidates()) + e10.load_completed_results(
        root, STUDY, e10.enumerate_final_seed_candidates_from_plan(root, STUDY))
    if len(completed) != 45 or report["completed_runs"] != completed:
        raise PruningError("E.10 completed-record reconciliation failed")
    records = [r for r in report["final_validation_selected_checkpoints"]
               if r["configuration"]["model_family"] == "htgnn" and r["configuration"]["task"] == "E2"]
    if tuple(r["configuration"]["seed"] for r in records) != SEEDS or any(r not in completed for r in records):
        raise PruningError("final E2 checkpoint membership mismatch")
    frozen_e9 = e10._validate_frozen_e9_evidence(root)
    for relative, expected in {**frozen_e9["frozen_upstream_source_sha256"], **frozen_e9["e9_source_sha256"]}.items():
        if e10._sha256(root / relative) != expected:
            raise PruningError("frozen upstream source changed")
    manifest = _read(e10.candidate_paths(root, STUDY, e10.FullTrainingCandidate(**{
        k: records[0]["configuration"][k] for k in ("model_family", "task", "learning_rate", "seed", "phase")})).manifest)
    if e10._evidence_hashes(root) != manifest["upstream_evidence_sha256"]:
        raise PruningError("frozen upstream evidence changed")
    current_env = e10.environment_identity()
    for key in ("python", "python_implementation", "torch", "torch_geometric", "cuda_runtime"):
        if key not in manifest["environment_identity"] or current_env.get(key) != manifest["environment_identity"][key]:
            raise PruningError(f"execution environment changed: {key}")
    stage_c_path = root / "results/stage_c_full_validation.json"
    if e10._sha256(stage_c_path) != STAGE_C_REPORT_SHA256:
        raise PruningError("frozen Stage C report changed")
    stage_c = _read(stage_c_path)
    if stage_c.get("status") != "PASS":
        raise PruningError("frozen Stage C integrity evidence is not PASS")
    hashes = {}
    for name in ("X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy", "split_metadata.json"):
        relative = f"exp2/{name}"
        path = root / "data/processed/preprocessed" / relative
        expected = stage_c["artefact_sha256"][relative]
        if name == "split_metadata.json" and expected != manifest["stage_c_provenance"]["split_artifact_sha256"][relative]:
            raise PruningError("split metadata differs from E.10 provenance")
        if e10._sha256(path) != expected:
            raise PruningError("E2 train/validation artifact integrity mismatch")
        hashes[path.relative_to(root).as_posix()] = expected
    return {"records": records, "input_sha256": hashes, "environment": current_env,
            "e10_report_physical_sha256": E10_REPORT_SHA256}


def _dataset(root: Path, split: str) -> d4.LazyGraphDataset:
    if split not in {"train", "validation"}:
        raise PruningError("E.12A dataset split must be train or validation")
    return d4.LazyGraphDataset("E2", split, processed_root=root / "data/processed/preprocessed",
                              d1_manifest_path=root / "results/stage_d1/feature_taxonomy.json",
                              d2_manifest_path=root / "results/stage_d2/graph_schema.json",
                              d3_report_path=root / "results/stage_d3/structural_validation.json")


def _batches(dataset: d4.LazyGraphDataset, *, ranking: bool = False) -> Iterator[HeteroData]:
    yield from e7.iter_production_batches(dataset, task="E2", split=dataset.split, batch_size=BATCH_SIZE,
                                          epoch_seed=RANK_SEED if ranking else None,
                                          max_batches=RANK_BATCHES if ranking else None)


@torch.inference_mode()
def evaluate_validation(model: HTGNNClassifier, batches: Iterator[HeteroData], policy: FeaturePolicy,
                        codes: Sequence[int], device: torch.device) -> dict[str, Any]:
    if model.task != "E2":
        raise PruningError("E.12A selection requires the primary E2 model")
    metric = e7._MetricAccumulator("E2")
    count, order_digest = 0, hashlib.sha256()
    for data in batches:
        if getattr(data, "e2_audit", {}).get("split") != "validation":
            raise PruningError("candidate metrics require validation only")
        data = data.to(device)
        logits = pruned_logits(model, data, policy)
        targets = data["sample"].y
        if (targets.dtype != torch.long or targets.shape != (logits.shape[0],) or
                logits.shape != (targets.numel(), 15) or not torch.isfinite(logits).all() or
                torch.any(targets < 0) or torch.any(targets >= 15)):
            raise PruningError("invalid frozen E2 logits/targets")
        metric.update(logits, targets)
        count += targets.numel()
        for i in data.e2_audit["ordered_source_sample_indices"]:
            order_digest.update(int(i).to_bytes(8, "little"))
    score, complete, selected = metric.selection(train_target_codes=(), validation_selection_class_codes=codes)
    if not count or not complete:
        raise PruningError("complete validation selection support is required")
    return {"macro_f1": score, "examples": count, "class_codes": list(selected),
            "confusion_matrix": metric.confusion.tolist(), "sample_order_sha256": order_digest.hexdigest()}


@torch.inference_mode()
def benchmark(model: HTGNNClassifier, data: HeteroData, policy: FeaturePolicy,
              device: torch.device) -> dict[str, Any]:
    """Warmup 5, time 20 forwards; resident input, validation/compaction included."""
    b = _validate_batch(model, data)
    synchronize = lambda: torch.cuda.synchronize(device) if device.type == "cuda" else None
    for _ in range(5):
        pruned_logits(model, data, policy)
    synchronize()
    allocated_before = torch.cuda.memory_allocated(device) if device.type == "cuda" else None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    elapsed = []
    for _ in range(20):
        synchronize()
        start = time.perf_counter()
        logits = pruned_logits(model, data, policy)
        synchronize()
        elapsed.append(time.perf_counter() - start)
        del logits
    median = statistics.median(elapsed)
    return {"warmup_forwards": 5, "timed_forwards": 20, "batch_size": b, "median_batch_seconds": median,
            "p95_batch_seconds": float(np.percentile(elapsed, 95)), "throughput_samples_per_second": b / median,
            "cuda_allocated_before_bytes": allocated_before,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "cpu_memory_bytes": None, "cpu_memory_reason": "no reliable per-forward CPU allocator measurement",
            "scope": "resident full E2 input; contract checks, feature compaction, encoders and head; excludes I/O, H2D and ranking"}


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    content = d2.canonical_json_bytes(payload)
    # Validate canonical content before exclusive publication.
    if d2.canonical_json_bytes(json.loads(content)) != content:
        raise PruningError("noncanonical report")
    with path.open("xb") as stream:
        stream.write(content)


def run_phase_a(root: Path, *, run_id: str, device: str, max_relative_f1_drop: float,
                tolerance_rationale: str, owner_confirmed_validation_contract: bool = False) -> dict[str, Any]:
    if owner_confirmed_validation_contract is not True:
        raise PruningError("owner approval of the E.12A contract/tolerance is required")
    _check_tolerance(max_relative_f1_drop, tolerance_rationale)
    if not re.fullmatch(r"e12a-[0-9]{8}-v[1-9][0-9]*", run_id):
        raise PruningError("invalid E.12A run ID")
    if device not in {"cpu", "cuda"} or (device == "cuda" and not torch.cuda.is_available()):
        raise PruningError("requested device unavailable")
    root = Path(root).resolve()
    result_dir, log_dir = (root / area / "stage_e12" / STUDY / run_id for area in ("results", "logs"))
    if result_dir.exists() or log_dir.exists():
        raise PruningError("fresh E.12A namespaces required; overwrite/retry is forbidden")
    with deterministic_inference(device), phase_a_guard(root) as attempts:
        preparation = prepare(root)
        source_paths = ("src/pruning/stage_e12.py", "scripts/run_stage_e12_pruning.py", "docs/stage_e12_attention_pruning.md", "tests/test_stage_e12_pruning.py")
        source_hashes = {path: e10._sha256(root / path) for path in source_paths}
        records = preparation["records"]
        train, validation = _dataset(root, "train"), _dataset(root, "validation")
        codes = e7._derive_e2_validation_selection_class_codes(train, validation)
        if any(tuple(r["checkpoint_provenance"]["selection_class_codes"]) != codes for r in records):
            raise PruningError("E2 train/validation eligibility differs from frozen checkpoints")
        result_dir.mkdir(parents=True, exist_ok=False)
        log_dir.mkdir(parents=True, exist_ok=False)
        contract = {"stage": "E.12A", "version": VERSION, "study": STUDY, "run_id": run_id,
                    "ratios": list(RATIOS), "seeds": list(SEEDS), "max_relative_f1_drop": max_relative_f1_drop,
                    "tolerance_rationale": tolerance_rationale, "owner_confirmed_validation_contract": True,
                    "ranking": {"split": "train", "batch_size": BATCH_SIZE, "max_batches": RANK_BATCHES, "affine_seed": RANK_SEED,
                                "aggregation": "float64 mean over train examples, four heads, then three E2 seeds; rank within category; ID tie break"},
                    "validation": "full sequential E2; E7 train/validation-only class eligibility and metric",
                    "selection": "maximum removed feature nodes with every seed F1 >= (1-relative_tolerance)*its baseline; ties smaller ratio",
                    "transfer": "one E2-derived mask shared by E2 seeds; future unchanged mask transfer to E1/E3, no retuning",
                    "phase_b_enabled": False, "optimizer_created": False, "backpropagation": False, "training": False,
                    "execution_arithmetic": {"deterministic_algorithms": True, "cudnn_deterministic": True,
                                             "cudnn_benchmark": False, "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")},
                    "test_access": False, "source_sha256": source_hashes, "preparation": preparation}
        contract["manifest_sha256"] = d2.canonical_sha256(contract)
        _write(log_dir / "manifest.json", contract)
        try:
            ranks, states = [], []
            for record in records:
                model = load_bound_model(root, record).to(device)
                before = state_sha256(model)
                ranks.append(collect_scores(model, _batches(train, ranking=True), torch.device(device)))
                if state_sha256(model) != before:
                    raise PruningError("model state changed while collecting attention")
                states.append(before)
                del model
            if len({r["sample_order_sha256"] for r in ranks}) != 1 or len({r["examples"] for r in ranks}) != 1:
                raise PruningError("seed rankings must use identical training membership")
            pooled = np.mean(np.asarray([r["feature_per_head_mean"] for r in ranks], dtype=np.float64), axis=(0, 2))
            policies = [make_policy(pooled, ratio) for ratio in RATIOS]
            rank_report = {"per_seed": [{"seed": seed, **rank} for seed, rank in zip(SEEDS, ranks, strict=True)],
                           "pooled_feature_scores": pooled.tolist(), "feature_names": list(d2.MODEL_FEATURE_COLUMNS),
                           "policies": [p.record() for p in policies]}
            _write(result_dir / "attention_ranking.json", rank_report)
            rows = [{"ratio": ratio, "policy": policy.record(), "seed_results": []}
                    for ratio, policy in zip(RATIOS, policies, strict=True)]
            for seed_index, record in enumerate(records):
                model = load_bound_model(root, record).to(device)
                before = state_sha256(model)
                timed_data = next(_batches(validation)).to(device)
                for index, policy in enumerate(policies):
                    result = evaluate_validation(model, _batches(validation), policy, codes, torch.device(device))
                    if result["examples"] != len(validation):
                        raise PruningError("validation traversal is incomplete")
                    result.update({"seed": SEEDS[seed_index], "benchmark": benchmark(model, timed_data, policy, torch.device(device))})
                    rows[index]["seed_results"].append(result)
                if state_sha256(model) != before or before != states[seed_index]:
                    raise PruningError("model/checkpoint state changed")
                del model, timed_data
            if len({r["sample_order_sha256"] for row in rows for r in row["seed_results"]}) != 1:
                raise PruningError("candidate validation order differs")
            selected = select_candidate(rows, max_relative_f1_drop)
            # Reconcile every authenticated input/checkpoint again before publication.
            if prepare(root) != preparation or {p: e10._sha256(root / p) for p in source_paths} != source_hashes or attempts:
                raise PruningError("provenance changed or forbidden access attempted")
            _write(result_dir / "validation_sweep.json", {"rows": rows})
            proposed = {"status": "PROPOSED_REQUIRES_OWNER_REVIEW_AND_FREEZE", "phase": "E.12A", "version": VERSION,
                        "manifest_sha256": contract["manifest_sha256"], "selected_ratio": selected["ratio"],
                        "policy": selected["policy"], "max_relative_f1_drop": max_relative_f1_drop,
                        "ranking_canonical_sha256": d2.canonical_sha256(rank_report), "phase_b_enabled": False}
            proposed["policy_sha256"] = d2.canonical_sha256(proposed)
            _write(result_dir / "proposed_policy.json", proposed)
            report = {"stage": "E.12A", "version": VERSION, "status": "COMPLETE_PENDING_REVIEW", "manifest_sha256": contract["manifest_sha256"],
                      "selected_ratio": selected["ratio"], "policy_sha256": proposed["policy_sha256"], "test_support_consulted": False,
                      "test_arrays_opened": [], "forbidden_access_attempts": attempts, "model_states_unchanged": True,
                      "checkpoint_state_sha256": dict(zip(map(str, SEEDS), states, strict=True)),
                      "output_physical_sha256": {p.name: e10._sha256(p) for p in result_dir.iterdir()},
                      "phase_b_enabled": False}
            report["report_sha256"] = d2.canonical_sha256(report)
            _write(result_dir / "stage_e12a_report.json", report)
            with (result_dir / "stage_e12a_report.sha256").open("x", encoding="utf-8") as stream:
                stream.write(e10._sha256(result_dir / "stage_e12a_report.json") + "\n")
            _write(log_dir / "status.json", {"state": "COMPLETE_PENDING_REVIEW", "phase_b_enabled": False})
            return report
        except BaseException as exc:
            _write(log_dir / "failure.json", {"state": "FAILED", "error_type": type(exc).__name__, "error": str(exc)})
            raise
