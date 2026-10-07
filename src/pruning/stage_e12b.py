"""E.12B: owner-reserved fixed-mask inference; never select, train or retry.

Preparation and efficiency measurements are test-locked. Only run_owner's
explicit, hash-bound reservation unlocks the nine pruned predictive traversals.
Existing E.11 predictions and all E.12A evidence remain immutable inputs.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterator, Mapping, Sequence
from types import MappingProxyType

import numpy as np
import torch
from torch_geometric.data import HeteroData

from src.evaluation import official_test as e11
from src.graph.htgnn_classifier import HTGNNClassifier, _validated_head
from src.pruning import stage_e12 as a

VERSION = "stage-e12b-frozen-mask-comparison-v1"
STUDY = a.STUDY
TASKS = ("E2", "E1", "E3")
SEEDS = a.SEEDS
BATCH_SIZE = 256
POLICY_SHA256 = "f7a8f6aa41cdfb6ad2ec8644ef4570677b7ef172374592f84c8239d7ac0e00aa"
FREEZE_PATH = f"results/stage_e12/{STUDY}/e12a-20261006-v1-freeze/freeze_attestation.json"
FREEZE_SHA256 = "7f92187689993ea049cf4a364746301f1175d18ae25402e5a0518f2094032e51"
E11_REPORT_PATH = f"results/stage_e11/{STUDY}/recovery/e11-recovery-20261006-v1/stage_e11_report.json"
E11_REPORT_SHA256 = "ace0d585fbb89b0f3df4376b2f69e35bd3172ab663cc66b369b769e84a964aac"
BASELINE_PHYSICAL_SHA256 = {
    "e10-htgnn-e2-lr1e-3-s2025-b256-4c0dbe73d0e4": "9be44707b105403e784a9fb55cfa4397dfeb22db53d9d6c39f57e7ac36520b50",
    "e10-htgnn-e2-lr1e-3-s2026-b256-0a1122c0ee42": "6890e8c2a2c34fd9c4996810cc90ad2a7ecef1616cf410f6e1950fdca656baf1",
    "e10-htgnn-e2-lr1e-3-s2027-b256-3df7694884f0": "2b6091a02d9fc5baaa7b0d27e8d66ff0a5c165a298ce7ee378be4ffa28e5bacd",
    "e10-htgnn-e1-lr1e-3-s2025-b256-c2fd73b62871": "44f3bed7240d909a630d2425aab5726d20c764cd159ed0f5339fd0dc5e26ba4d",
    "e10-htgnn-e1-lr1e-3-s2026-b256-07fde048fbfd": "95686ffca2582c4022368c2421c0ff8231793f0426d6bde70318f06bd7ee2aef",
    "e10-htgnn-e1-lr1e-3-s2027-b256-79771d0a8e2c": "5e0f0e7b014e9fe4371d608d59d34b4ce840aaff7d75d16880ad64f5d8c03ca8",
    "e10-htgnn-e3-lr1e-3-s2025-b256-b0ed07a45fa6": "b0a69a4205328e31c003786479dd4972b63a2c79bd1bb728802e9d31e23f5f2e",
    "e10-htgnn-e3-lr1e-3-s2026-b256-3d991d8ab94c": "c8e1a3b6e21ac32a3ad641dc6de50faf3dd4b573997f671578a98e92b7d11d83",
    "e10-htgnn-e3-lr1e-3-s2027-b256-63b7a22cd5e9": "6c105d38a0e8b3da364d2183a640d63fd537091f8f87c63e8f27a867016664bf",
}
RETAINED_IDS = (0, 1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17,
                18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31, 33,
                36, 37, 38, 39, 43, 44, 45, 46, 47, 48, 49, 51, 53, 54,
                55, 56, 58, 59, 60, 61, 62, 63, 65, 66, 67, 68, 69, 74, 75, 76, 77)
SOURCE_PATHS = ("src/pruning/stage_e12b.py", "scripts/run_stage_e12b.py",
                "tests/test_stage_e12b.py", "docs/stage_e12b_final_comparison.md")
AUTHORITY_PATH = "results/provenance/authority/handoff-reconciliation-20261006-v1/attestation.json"
AUTHORITY_SHA256 = "af4b21ea289932dfad648c6affd13a4cc9f91944bcc8d4eca7a94397d3422895"
E9_FROZEN_COMMIT = "2fa823943ba318fe32b2867bc8e2b1f799d78ac1"
E9_CONTRACT_PATH = "results/stage_e9/stage_e9_contract.json"
E9_CONTRACT_SHA256 = "e569712855d068b97e5d5b1717faf7602e1a083b1016654b8b47ed003f0d2254"
HISTORICAL_HANDOFF_SHA256 = "0eb4f09d989a8c641736be9637f7677cdccf3e675d73425bbd0e7cf22a5df9f6"
CURRENT_HANDOFF_SHA256 = "58a22c8d6feba77a6f50cdef9b56ee653a5f21fa87c803253a16b4f89c5733c5"
BENCHMARK_PLAN = MappingProxyType({"split": "validation", "batch_size": 256,
                  "sequential_batch_starts": (0, 256, 512, 768),
                  "warmups_per_path_per_batch": 5, "pairs_per_batch": 20,
                  "pair_order": "alternate unpruned/pruned and pruned/unpruned",
                  "precision": "float32", "deterministic_algorithms": True,
                  "forward_scope": "resident full input; checks, compaction, encoders and head; excludes loading/conversion/H2D",
                  "other_scopes": ("combined lazy loading/D6/E2 conversion", "device transfer")})


def _benchmark_plan() -> dict[str, Any]:
    """Fresh serializable view; report consumers cannot mutate the protocol."""
    return {k: list(v) if isinstance(v, tuple) else v for k, v in BENCHMARK_PLAN.items()}


class ComparisonError(ValueError):
    """A policy, provenance, access, inference or measurement failure."""


def frozen_policy() -> a.FeaturePolicy:
    return a.FeaturePolicy(.25, RETAINED_IDS)


def _hashes(root: Path, paths: Sequence[str]) -> dict[str, str]:
    return {p: a.e10._sha256(root / p) for p in paths}


def _canonical(value: Mapping[str, Any], key: str) -> None:
    content = dict(value)
    expected = content.pop(key, None)
    if expected != a.d2.canonical_sha256(content):
        raise ComparisonError(f"canonical {key} mismatch")


def _bound_json(root: Path, relative: str, physical: str, key: str) -> dict[str, Any]:
    path = root / relative
    if a.e10._sha256(path) != physical:
        raise ComparisonError(f"physical artifact changed: {relative}")
    value = a._read(path)
    _canonical(value, key)
    return value


def verify_authority_reconciliation(root: Path) -> dict[str, Any]:
    """Bind historical E.9 bytes and the sole authorized live handoff revision.

    The separately pinned attestation grants no experimental/source exception.
    Git cat-file reads the historical object without changing refs or the index.
    No caller-supplied hash, future-handoff fallback or global bypass exists.
    """
    root = Path(root).resolve()
    try:
        attestation = _bound_json(root, AUTHORITY_PATH, AUTHORITY_SHA256, "attestation_sha256")
        if (root / AUTHORITY_PATH).read_bytes() != a.d2.canonical_json_bytes(attestation):
            raise ComparisonError("authority attestation serialization changed")
        if (root / AUTHORITY_PATH).with_suffix(".sha256").read_text(encoding="ascii").strip() != AUTHORITY_SHA256:
            raise ComparisonError("authority attestation sidecar mismatch")
        required = {
            "version": "project-handoff-authority-reconciliation-v1",
            "run_id": "handoff-reconciliation-20261006-v1",
            "scope": ["PROJECT_HANDOFF.md"], "e9_frozen_commit": E9_FROZEN_COMMIT,
            "e9_contract_path": E9_CONTRACT_PATH, "e9_contract_physical_sha256": E9_CONTRACT_SHA256,
            "historical_handoff_sha256": HISTORICAL_HANDOFF_SHA256,
            "current_handoff_sha256": CURRENT_HANDOFF_SHA256,
            "historical_handoff_git_object": f"{E9_FROZEN_COMMIT}:PROJECT_HANDOFF.md",
            "reason": "handoff status/documentation reconciliation after completed E.10/E.11/E.12A evidence",
            "supersession": "current handoff supersedes the historical handoff ONLY as the live authority/status record",
        }
        if any(attestation.get(k) != v for k, v in required.items()):
            raise ComparisonError("authority reconciliation scope/hash binding mismatch")
        for key in ("no_frozen_e9_experimental_artifact_changed", "independent_review_required"):
            if attestation.get(key) is not True:
                raise ComparisonError("authority reconciliation safeguards missing")
        for key in ("e12b_execution_authorized", "git_release_authorized"):
            if attestation.get(key) is not False:
                raise ComparisonError("authority reconciliation cannot grant execution/release approval")
        if not isinstance(attestation.get("reconciled_at_utc"), str):
            raise ComparisonError("authority reconciliation timestamp missing")
        authorization = attestation["owner_authorization"]
        if (authorization.get("actor") != "current project owner"
                or authorization.get("basis") != "explicit attached request in the current chat"
                or authorization.get("authorized_scope") != "PROJECT_HANDOFF.md authority reconciliation and E.12B provenance correction only"
                or authorization.get("evidence_path") != Path(AUTHORITY_PATH).with_name("owner_authorization.txt").as_posix()):
            raise ComparisonError("authority reconciliation owner scope mismatch")
        if a.e10._sha256(root / authorization["evidence_path"]) != authorization["evidence_physical_sha256"]:
            raise ComparisonError("authority reconciliation owner evidence changed")
        for field in ("authority_policy_physical_sha256", "review_evidence_physical_sha256"):
            for relative, expected in attestation[field].items():
                if a.e10._sha256(root / relative) != expected:
                    raise ComparisonError(f"authority reconciliation evidence changed: {relative}")
        historical = _bound_json(root, E9_CONTRACT_PATH, E9_CONTRACT_SHA256, "report_sha256")
        if (historical["report_sha256"] != attestation["e9_contract_canonical_sha256"]
                or historical["e9_source_sha256"].get("PROJECT_HANDOFF.md") != HISTORICAL_HANDOFF_SHA256):
            raise ComparisonError("original E.9 historical handoff pin changed")
        old_bytes = subprocess.check_output(
            ["git", "cat-file", "blob", attestation["historical_handoff_git_object"]],
            cwd=root, timeout=10, stderr=subprocess.PIPE)
        if hashlib.sha256(old_bytes).hexdigest() != HISTORICAL_HANDOFF_SHA256:
            raise ComparisonError("historical E.9 handoff object mismatch")
        if a.e10._sha256(root / "PROJECT_HANDOFF.md") != CURRENT_HANDOFF_SHA256:
            raise ComparisonError("live authority handoff changed")
        for relative, expected in {**historical["frozen_upstream_source_sha256"], **historical["e9_source_sha256"]}.items():
            if relative == "PROJECT_HANDOFF.md":
                # The original pin is verified above against its historical bytes.
                # Only this one path has a separately authenticated live revision.
                if expected != HISTORICAL_HANDOFF_SHA256:
                    raise ComparisonError("original E.9 historical handoff pin changed")
            elif a.e10._sha256(root / relative) != expected:
                raise ComparisonError(f"frozen upstream source changed: {relative}")
        return attestation
    except (OSError, KeyError, TypeError, ValueError, subprocess.SubprocessError) as exc:
        raise ComparisonError(f"authority reconciliation validation failed: {exc}") from exc


def verify_freeze(root: Path) -> dict[str, Any]:
    f = _bound_json(root, FREEZE_PATH, FREEZE_SHA256, "attestation_sha256")
    if (f.get("status") != "FROZEN" or f.get("review_status") != "PASS"
            or f.get("e12a_run_id") != "e12a-20261006-v1" or f.get("study") != STUDY
            or f.get("policy_canonical_sha256") != POLICY_SHA256
            or f.get("selected_ratio") != .25 or f.get("max_relative_f1_drop") != .05
            or f.get("owner_approved_relative_macro_f1_tolerance") is not True
            or f.get("owner_approved_freeze") is not True
            or f.get("retained_feature_ids") != list(RETAINED_IDS)
            or f.get("removed_feature_ids") != frozen_policy().record()["pruned_feature_ids"]
            or f.get("policy_record") != frozen_policy().record()
            or f.get("declared_candidates") != list(a.RATIOS)
            or f.get("admissible_candidates") != [0., .1, .25]
            or f.get("failed_candidates") != [.5, .75]):
        raise ComparisonError("freeze contract/mask differs from the reviewed policy")
    for p, expected in {**f["e12a_artifact_physical_sha256"], **f["e12a_source_physical_sha256"]}.items():
        if a.e10._sha256(root / p) != expected:
            raise ComparisonError(f"frozen E.12A input changed: {p}")
    policy_path = f"results/stage_e12/{STUDY}/e12a-20261006-v1/proposed_policy.json"
    policy = _bound_json(root, policy_path, f["policy_physical_sha256"], "policy_sha256")
    if policy["policy_sha256"] != POLICY_SHA256 or policy["policy"] != frozen_policy().record():
        raise ComparisonError("frozen policy content mismatch")
    if (root / FREEZE_PATH).with_suffix(".sha256").read_text().strip() != FREEZE_SHA256:
        raise ComparisonError("freeze physical sidecar mismatch")
    return f


@dataclass
class _Access:
    root: Path
    run_id: str | None
    attempts: list[str] = field(default_factory=list)
    opened_test: set[str] = field(default_factory=set)
    ledger_sha256: str | None = None
    test_enabled: bool = False

    @property
    def ledger(self) -> Path:
        return self.root / "logs/stage_e12b" / STUDY / "test_consumption.json"


_ACCESS: list[_Access] = []
_HOOK_INSTALLED = False


def _audit_hook(event: str, args: tuple[Any, ...]) -> None:
    if not _ACCESS:
        return
    if event == "open":
        paths = [args[0]]
        mode, flags = args[1:3]
        writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
            isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)))
    elif event in {"os.mkdir", "os.remove", "os.rmdir", "os.chmod", "os.utime"}:
        paths, writing = [args[0]], True
    elif event in {"os.rename", "os.link", "os.symlink"}:
        paths, writing = list(args[:2]), True
    else:
        return
    for value in paths:
        if not isinstance(value, (str, bytes, os.PathLike)):
            continue
        lexical = Path(os.path.abspath(os.fsdecode(value)))
        resolved = lexical.resolve()
        for state in tuple(_ACCESS):
            if not (a._inside(lexical, state.root) or a._inside(resolved, state.root)):
                continue
            reason = None
            if writing:
                run_dirs = [state.root / area / STUDY / state.run_id
                            for area in ("logs/stage_e12b", "results/stage_e12b")] if state.run_id else []
                parents = [state.root / area / part for area in ("logs/stage_e12b", "results/stage_e12b")
                           for part in ("", STUDY)] if state.run_id else []
                exclusive = event == "open" and isinstance(args[2], int) and bool(args[2] & os.O_EXCL) and bool(args[2] & os.O_CREAT)
                permitted = lexical == resolved and (
                    any(a._inside(lexical, p) for p in run_dirs)
                    or (event == "os.mkdir" and lexical in parents)
                    or (lexical == state.ledger and exclusive))
                if not permitted:
                    reason = "writes outside the new E.12B reservation are prohibited"
            if event == "open" and any(a._inside(p, state.root / "data") and p.suffix == ".npy" for p in (lexical, resolved)):
                allowed = {state.root / "data/processed/preprocessed" / d / name
                           for d in a.d4.EXPERIMENT_DIRS.values() for name in ("X_val.npy", "y_val.npy", "X_train.npy", "y_train.npy")}
                tests = {state.root / "data/processed/preprocessed" / d / name
                         for d in a.d4.EXPERIMENT_DIRS.values() for name in ("X_test.npy", "y_test.npy")}
                if lexical in tests and resolved == lexical and not writing:
                    if not state.test_enabled or not state.ledger_sha256:
                        reason = "test arrays locked until explicit owner reservation"
                    else:
                        state.opened_test.add(lexical.relative_to(state.root).as_posix())
                elif writing or lexical not in allowed or resolved != lexical:
                    reason = "unapproved predictive array access"
            if reason:
                state.attempts.append(f"{event}:{lexical}:{reason}")
                raise ComparisonError(reason)


@contextmanager
def file_guard(root: Path, run_id: str | None = None) -> Iterator[_Access]:
    """Python accidental-access guard; not a sandbox against arbitrary native code."""
    global _HOOK_INSTALLED
    root = Path(root).resolve()
    if run_id is not None and not re.fullmatch(r"e12b-[0-9]{8}-v[1-9][0-9]*", run_id):
        raise ComparisonError("invalid E.12B run ID")
    if any(s.root == root and (run_id is not None or s.test_enabled) for s in _ACCESS):
        raise ComparisonError("overlapping E.12B contexts forbidden")
    if not _HOOK_INSTALLED:
        sys.addaudithook(_audit_hook)
        _HOOK_INSTALLED = True
    state = _Access(root, run_id)
    _ACCESS.append(state)
    try:
        yield state
    finally:
        state.test_enabled = False
        _ACCESS.remove(state)


def _test_access(root: Path | None = None) -> _Access:
    states = [s for s in _ACCESS if s.test_enabled and s.ledger_sha256 and (root is None or s.root == root)]
    if len(states) != 1:
        raise ComparisonError("test inference requires the explicit owner reservation context")
    state = states[0]
    if a.e10._sha256(state.ledger) != state.ledger_sha256:
        raise ComparisonError("E.12B consumption ledger changed")
    ledger = a._read(state.ledger)
    _canonical(ledger, "authorization_sha256")
    if (ledger.get("policy_sha256") != POLICY_SHA256 or ledger.get("freeze_sha256") != FREEZE_SHA256
            or ledger.get("run_id") != state.run_id or ledger.get("state") != "CONSUMED_OR_RESERVED"
            or ledger.get("owner_confirmed_one_time_pruned_test_execution") is not True):
        raise ComparisonError("invalid E.12B owner reservation")
    return state


def _candidate(record: Mapping[str, Any]) -> a.e10.FullTrainingCandidate:
    c = record["configuration"]
    try:
        candidate = a.e10.FullTrainingCandidate(**{k: c[k] for k in ("model_family", "task", "learning_rate", "seed", "phase")})
    except (KeyError, ValueError, TypeError) as exc:
        raise ComparisonError("invalid frozen checkpoint configuration") from exc
    if (c != candidate.config_dict() or candidate.model_family != "htgnn"
            or candidate.task not in TASKS or candidate.seed not in SEEDS):
        raise ComparisonError("only nine frozen HTGNN configurations are allowed")
    return candidate


def load_bound_model(root: Path, record: Mapping[str, Any]) -> HTGNNClassifier:
    """No optimizer construction or restoration, including for provenance checks."""
    candidate = _candidate(record)
    if candidate.task == "E2":
        return a.load_bound_model(root, record)
    path = a.e10.candidate_paths(root, STUDY, candidate).checkpoint
    expected_hash = record["artifacts"]["checkpoint_sha256"]
    if a.e10._sha256(path) != expected_hash:
        raise ComparisonError("checkpoint physical hash mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = payload["metadata"]
    if metadata != record["checkpoint_provenance"] or record.get("checkpoint_provenance_verified") is not True:
        raise ComparisonError("checkpoint metadata/result binding mismatch")
    model = HTGNNClassifier(candidate.task)
    identity = a.e7.model_contract(model)
    expected = {"schema_version": a.e7.CHECKPOINT_SCHEMA_VERSION, "harness_version": a.e7.HARNESS_VERSION,
                "task": candidate.task, "model_family": "htgnn", "model_contract_version": identity.architecture_version,
                "model_input_contract": identity.input_contract, "model_version": identity.architecture_version,
                "parameter_count": identity.parameter_count, "model_state_schema_sha256": a.e7.model_state_schema_sha256(model),
                "training_config": candidate.training_config.checkpoint_dict(), "selection_name": a.e7.SELECTION_NAMES[candidate.task],
                "frozen_source_sha256": dict(a.e7.e6.FROZEN_SOURCE_SHA256),
                "epoch": record["best_epoch"], "selection_value": record["best_validation_metric"]}
    if any(metadata.get(k) != v for k, v in expected.items()):
        raise ComparisonError("frozen task/model/checkpoint contract mismatch")
    for relative, digest in metadata["frozen_source_sha256"].items():
        if a.e10._sha256(root / relative) != digest:
            raise ComparisonError("frozen model source changed")
    state = payload["model_state_dict"]
    a.e7._preflight_model_state_dict(model, state)
    if any(not torch.isfinite(t).all() for t in state.values()):
        raise ComparisonError("nonfinite checkpoint state")
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False).eval()
    if a.e10._sha256(path) != expected_hash:
        raise ComparisonError("checkpoint changed while loading")
    return model


def _order_sha256(n: int) -> str:
    digest = hashlib.sha256()
    for start in range(0, n, 4096):
        digest.update(np.arange(start, min(start + 4096, n), dtype="<i8").tobytes())
    return digest.hexdigest()


def _prepare_upstream(root: Path) -> dict[str, Any]:
    """E.12B replay of frozen E.12A compact preflight with authority binding.

    E.12A source cannot change after freeze. Keep its checks and return contract,
    reuse its E.9/E.10 APIs, and substitute only the historical/live handoff
    comparison. This creates no loader, split, topology or selection authority.
    """
    authority = verify_authority_reconciliation(root)
    paths = a.e10.report_paths(root, STUDY)
    if a.e10._sha256(paths.final_report) != a.E10_REPORT_SHA256 or paths.final_report_sha256.read_text().split()[0] != a.E10_REPORT_SHA256:
        raise a.PruningError("pinned E.10 report changed")
    report = a._read(paths.final_report)
    unhashed = dict(report)
    if unhashed.pop("report_sha256", None) != a.d2.canonical_sha256(unhashed) or report["overall_status"] != "PASS":
        raise a.PruningError("E.10 report canonical/content validation failed")
    for path, expected in (("src/training/full_training.py", a.e10.provenance_policy.CORRECTED_RUNTIME_SHA256),
                           ("scripts/run_stage_e10_full_training.py", a.e10.provenance_policy.APPROVED_RUNNER_SHA256)):
        if a.e10._sha256(root / path) != expected:
            raise a.PruningError("reviewed E.10 execution source changed")
    completed = a.e10.load_completed_results(root, STUDY, a.e10.enumerate_lr_selection_candidates()) + a.e10.load_completed_results(
        root, STUDY, a.e10.enumerate_final_seed_candidates_from_plan(root, STUDY))
    if len(completed) != 45 or report["completed_runs"] != completed:
        raise a.PruningError("E.10 completed-record reconciliation failed")
    records = [r for r in report["final_validation_selected_checkpoints"]
               if r["configuration"]["model_family"] == "htgnn" and r["configuration"]["task"] == "E2"]
    if tuple(r["configuration"]["seed"] for r in records) != a.SEEDS or any(r not in completed for r in records):
        raise a.PruningError("final E2 checkpoint membership mismatch")
    frozen_e9 = a.e10._validate_frozen_e9_evidence(root)
    if frozen_e9["report_sha256"] != authority["e9_contract_canonical_sha256"]:
        raise ComparisonError("frozen E.9 evidence differs from authority attestation")
    manifest = a._read(a.e10.candidate_paths(root, STUDY, a.e10.FullTrainingCandidate(**{
        k: records[0]["configuration"][k] for k in ("model_family", "task", "learning_rate", "seed", "phase")})).manifest)
    if a.e10._evidence_hashes(root) != manifest["upstream_evidence_sha256"]:
        raise a.PruningError("frozen upstream evidence changed")
    current_env = a.e10.environment_identity()
    for key in ("python", "python_implementation", "torch", "torch_geometric", "cuda_runtime"):
        if key not in manifest["environment_identity"] or current_env.get(key) != manifest["environment_identity"][key]:
            raise a.PruningError(f"execution environment changed: {key}")
    stage_c_path = root / "results/stage_c_full_validation.json"
    if a.e10._sha256(stage_c_path) != a.STAGE_C_REPORT_SHA256:
        raise a.PruningError("frozen Stage C report changed")
    stage_c = a._read(stage_c_path)
    if stage_c.get("status") != "PASS":
        raise a.PruningError("frozen Stage C integrity evidence is not PASS")
    hashes = {}
    for name in ("X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy", "split_metadata.json"):
        relative = f"exp2/{name}"
        path = root / "data/processed/preprocessed" / relative
        expected = stage_c["artefact_sha256"][relative]
        if name == "split_metadata.json" and expected != manifest["stage_c_provenance"]["split_artifact_sha256"][relative]:
            raise a.PruningError("split metadata differs from E.10 provenance")
        if a.e10._sha256(path) != expected:
            raise a.PruningError("E2 train/validation artifact integrity mismatch")
        hashes[path.relative_to(root).as_posix()] = expected
    if verify_authority_reconciliation(root) != authority:
        raise ComparisonError("authority reconciliation changed during upstream preflight")
    return {"records": records, "input_sha256": hashes, "environment": current_env,
            "e10_report_physical_sha256": a.E10_REPORT_SHA256,
            "authority_reconciliation": {"path": AUTHORITY_PATH, "physical_sha256": AUTHORITY_SHA256,
                                       "canonical_sha256": authority["attestation_sha256"],
                                       "historical_handoff_sha256": HISTORICAL_HANDOFF_SHA256,
                                       "current_handoff_sha256": CURRENT_HANDOFF_SHA256}}
def prepare(root: Path) -> dict[str, Any]:
    """Authenticate compact evidence/models and validation inputs; no test arrays."""
    root = Path(root).resolve()
    with file_guard(root):
        freeze = verify_freeze(root)
        upstream = _prepare_upstream(root)
        report = _bound_json(root, E11_REPORT_PATH, E11_REPORT_SHA256, "report_sha256")
        if (report.get("overall_status") != "PASS" or report.get("evaluated_checkpoints") != 27
                or len(report["results"]) != 27 or report.get("original_marker_preserved") is not True):
            raise ComparisonError("completed E.11 report contract mismatch")
        recovery = a._read(root / f"logs/stage_e11/{STUDY}/recovery/e11-recovery-20261006-v1/recovery_authorization.json")
        _canonical(recovery, "manifest_sha256")
        if recovery["manifest_sha256"] != report["recovery_manifest_canonical_sha256"]:
            raise ComparisonError("E.11 recovery provenance mismatch")
        for p, digest in {**report["original_file_sha256"], **recovery["source_sha256"]}.items():
            if a.e10._sha256(root / p) != digest:
                raise ComparisonError(f"E.11 baseline/source changed: {p}")
        libraries = {"numpy": version("numpy"), "scikit_learn": version("scikit-learn")}
        if libraries != recovery["metric_library_versions"]:
            raise ComparisonError("E.11 metric-library versions changed")
        current_env = a.e10.environment_identity()
        if current_env != recovery["environment"]:
            raise ComparisonError("E.11 hardware/runtime identity changed")
        e10_report = a._read(a.e10.report_paths(root, STUDY).final_report)
        records = [r for r in e10_report["final_validation_selected_checkpoints"] if r["configuration"]["model_family"] == "htgnn"]
        by_key = {(r["configuration"]["task"], r["configuration"]["seed"]): r for r in records}
        if len(records) != 9 or set(by_key) != {(t, s) for t in TASKS for s in SEEDS}:
            raise ComparisonError("E.10 final nine HTGNN membership mismatch")
        records = [by_key[(t, s)] for t in TASKS for s in SEEDS]
        stage_c = a._read(root / "results/stage_c_full_validation.json")
        metadata_hashes = recovery["audit"]["preparation"]["frozen_metadata_sha256"]
        for relative, digest in metadata_hashes.items():
            if a.e10._sha256(root / relative) != digest:
                raise ComparisonError(f"frozen preprocessing/upstream metadata changed: {relative}")
        splits, validation_hashes = {}, {}
        for task in TASKS:
            directory = a.d4.EXPERIMENT_DIRS[task]
            relative = f"data/processed/preprocessed/{directory}/split_metadata.json"
            expected = stage_c["artefact_sha256"][f"{directory}/split_metadata.json"]
            if a.e10._sha256(root / relative) != expected:
                raise ComparisonError("frozen split metadata changed")
            splits[task] = {"metadata_sha256": expected, "policy": e11.reporting_policy(a._read(root / relative), task),
                            "test_sha256": {f"data/processed/preprocessed/{directory}/{name}": stage_c["artefact_sha256"][f"{directory}/{name}"]
                                            for name in ("X_test.npy", "y_test.npy")}}
            for name in ("X_val.npy", "y_val.npy"):
                rel = f"data/processed/preprocessed/{directory}/{name}"
                digest = stage_c["artefact_sha256"][f"{directory}/{name}"]
                if a.e10._sha256(root / rel) != digest:
                    raise ComparisonError("validation benchmark input changed")
                validation_hashes[rel] = digest
        baseline_by_key = {(r["task"], r["seed"]): r for r in report["results"] if r["model_family"] == "htgnn"}
        if len(baseline_by_key) != 9:
            raise ComparisonError("E.11 baseline membership mismatch")
        baseline_bindings, states = [], {}
        for record in records:
            candidate = _candidate(record)
            baseline = baseline_by_key[(candidate.task, candidate.seed)]
            _canonical(baseline, "result_sha256")
            namespace = "recovery/e11-recovery-20261006-v1" if candidate.task == "E3" else "e11-official-20261006-v1"
            relative = f"results/stage_e11/{STUDY}/{namespace}/{candidate.candidate_id}.json"
            if (a.e10._sha256(root / relative) != BASELINE_PHYSICAL_SHA256[candidate.candidate_id]
                    or a._read(root / relative) != baseline):
                raise ComparisonError("E.11 physical result differs from authenticated report")
            policy = splits[candidate.task]["policy"]
            if (baseline["checkpoint_sha256"] != record["artifacts"]["checkpoint_sha256"]
                    or baseline["checkpoint_provenance"] != record["checkpoint_provenance"]
                    or baseline["e10_result_canonical_sha256"] != record["result_sha256"]
                    or baseline["model_state_unchanged"] is not True
                    or baseline["examples"] != sum(policy["test_support"])
                    or baseline["order_sha256"] != _order_sha256(baseline["examples"])):
                raise ComparisonError("E.11 baseline checkpoint/coverage mismatch")
            if candidate.task != "E3" and e11.multiclass_metrics(np.asarray(baseline["metrics"]["confusion_matrix"], dtype=np.int64), policy) != baseline["metrics"]:
                raise ComparisonError("E.11 baseline confusion/metrics mismatch")
            loaded = load_bound_model(root, record)
            states[candidate.candidate_id] = a.state_sha256(loaded)
            del loaded
            baseline_bindings.append({"path": relative, "physical_sha256": a.e10._sha256(root / relative),
                                      "canonical_sha256": baseline["result_sha256"], "reporting_policy": policy, "result": baseline})
        e2_frozen = {x["candidate_id"]: x for x in freeze["e2_checkpoints"]}
        for record in records:
            if record["configuration"]["task"] == "E2":
                item = e2_frozen[record["candidate_id"]]
                if (item["checkpoint_physical_sha256"] != record["artifacts"]["checkpoint_sha256"]
                        or item["model_state_sha256"] != states[record["candidate_id"]]):
                    raise ComparisonError("E2 checkpoint differs from E.12A freeze")
        sources = _hashes(root, SOURCE_PATHS)
        return {"stage": "E.12B", "version": VERSION, "policy_sha256": POLICY_SHA256,
                "freeze_sha256": FREEZE_SHA256, "e11_report_sha256": E11_REPORT_SHA256,
                "source_sha256": sources, "implementation_sha256": a.d2.canonical_sha256(sources),
                "environment": current_env, "metric_library_versions": libraries, "upstream": upstream,
                "records": records, "model_state_sha256": states, "baselines": baseline_bindings,
                "splits": splits, "validation_sha256": validation_hashes,
                "preprocessing_sha256": dict(metadata_hashes),
                "benchmark_plan": _benchmark_plan(), "test_arrays_opened": [],
                "training": False, "optimizer": False, "backpropagation": False}


def _validate_batch(model: HTGNNClassifier, data: HeteroData) -> int:
    split = getattr(data, "e2_audit", {}).get("split")
    if split in {"train", "validation"}:
        b = a._validate_batch(model, data)
    elif split == "test":
        _test_access()
        a._frozen(model)
        indices = a.e7._require_batch_audit(data, model.task, "test")
        b = len(indices)
        x = data["feature"].x
        if x.shape != (b * 78, 5) or x.dtype != torch.float32 or not torch.isfinite(x).all():
            raise ComparisonError("invalid original feature values")
        for kind, count, identity in (("feature", 78, "feature_id"), ("category", 8, "category_id"), ("sample", 1, None)):
            store = data[kind]
            if (store.batch.dtype != torch.long or store.ptr.dtype != torch.long
                    or not torch.equal(store.batch, torch.arange(b, device=x.device).repeat_interleave(count))
                    or not torch.equal(store.ptr, torch.arange(b + 1, device=x.device) * count)
                    or (identity and (store[identity].dtype != torch.long or not torch.equal(store[identity], torch.arange(count, device=x.device).repeat(b))))):
                raise ComparisonError("original node identity/ownership mismatch")
        model.graph_encoder._validate_inputs(data, x.new_empty((b * 78, 32)))
    else:
        raise ComparisonError("unknown inference split")
    raw = data.e2_audit["ordered_source_sample_indices"]
    if b > BATCH_SIZE or any(type(i) is not int or i < 0 for i in raw):
        raise ComparisonError("invalid batch bound/source indices")
    targets = data["sample"].y
    dtype = torch.float32 if model.task == "E3" else torch.long
    if targets.shape != (b,) or targets.dtype != dtype or not torch.isfinite(targets).all():
        raise ComparisonError("frozen target shape/dtype mismatch")
    if not torch.isin(targets, targets.new_tensor(range(2 if model.task == "E3" else 15))).all():
        raise ComparisonError("target domain mismatch")
    return b


@torch.inference_mode()
def forward(model: HTGNNClassifier, data: HeteroData, *, diagnostic_zero: bool = False) -> torch.Tensor:
    """Only fixed 61-feature policy; zero is restricted to non-test diagnostics."""
    b = _validate_batch(model, data)
    if diagnostic_zero:
        if data.e2_audit["split"] == "test":
            raise ComparisonError("unpruned test evaluation is prohibited")
        logits = model(data)
    else:
        values, ids, edges = a.compact_inputs(data, frozen_policy(), b)
        embeddings = model.feature_encoder(values, ids)
        graph = model.graph_encoder
        categories = graph.category_embedding(data["category"].category_id)
        message = graph.feature_to_category((embeddings, categories), edges)
        categories = graph.feature_to_category_activation(graph.feature_to_category_norm(graph.feature_to_category_dropout(message) + categories))
        samples = graph.sample_token.unsqueeze(0).expand(b, -1)
        message = graph.category_to_sample((categories, samples), data[a.e2.CATEGORY_RELATION].edge_index)
        samples = graph.category_to_sample_activation(graph.category_to_sample_norm(graph.category_to_sample_dropout(message) + samples))
        contract, head = _validated_head(model)
        logits = head(samples)
        if contract.output_channels == 1:
            logits = logits.squeeze(-1)
    e11._validate_inference_outputs(model.task, logits, data["sample"].y, b)
    return logits


def _dataset(root: Path, task: str, split: str) -> a.d4.LazyGraphDataset:
    if task not in TASKS or split not in {"validation", "test"}:
        raise ComparisonError("invalid E.12B dataset")
    if split == "test":
        _test_access(root)
    dataset = a.d4.LazyGraphDataset(task, split, processed_root=root / "data/processed/preprocessed",
                                  d1_manifest_path=root / "results/stage_d1/feature_taxonomy.json",
                                  d2_manifest_path=root / "results/stage_d2/graph_schema.json",
                                  d3_report_path=root / "results/stage_d3/structural_validation.json")
    if not dataset.is_lazy_read_only:
        raise ComparisonError("dataset must remain lazy/read-only")
    return dataset


def iter_test_batches(root: Path, task: str) -> Iterator[HeteroData]:
    _test_access(root)
    dataset = _dataset(root, task, "test")
    for indices in a.e7.iter_index_batches(len(dataset), BATCH_SIZE, shuffle=False):
        yield a.e2.logical_batch_to_heterodata(e11.d6.build_logical_batch(dataset, indices))


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _summary(values: Sequence[float]) -> dict[str, Any]:
    if not values or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ComparisonError("latencies must be finite and positive")
    median = statistics.median(values)
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.
    return {"observations": len(values), "median_ms": median, "p95_ms": float(np.percentile(values, 95)),
            "mean_ms": mean, "sample_std_ms_ddof1": std, "coefficient_of_variation": std / mean,
            "min_ms": min(values), "max_ms": max(values), "examples_per_second": BATCH_SIZE * 1000 / median}


def _memory(device: torch.device) -> tuple[int, int, int, int]:
    return (torch.cuda.memory_allocated(device), torch.cuda.memory_reserved(device),
            torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device))


@torch.inference_mode()
def benchmark(model: HTGNNClassifier, batches: Sequence[HeteroData], device: torch.device,
              *, warmups: int = 5, pairs: int = 20) -> dict[str, Any]:
    """Paired validation-only observations; no predictive evaluation or selection."""
    if not batches or warmups < 1 or pairs < 2:
        raise ComparisonError("benchmark needs batches, warmup and repeated pairs")
    state_before = a.state_sha256(model)
    times = {"unpruned": [], "pruned": []}
    memory = {"unpruned": [], "pruned": []}
    transfer_times, indices, paired = [], [], []
    for batch_index, cpu_data in enumerate(batches):
        if cpu_data.e2_audit["split"] != "validation" or _validate_batch(model, cpu_data) != BATCH_SIZE:
            raise ComparisonError("benchmark requires full 256-example validation batches")
        indices.append(cpu_data.e2_audit["ordered_source_sample_indices"])
        _sync(device)
        started = time.perf_counter_ns()
        data = cpu_data.clone().to(device)
        _sync(device)
        transfer_times.append((time.perf_counter_ns() - started) / 1e6)
        for _ in range(warmups):
            for name in ("unpruned", "pruned"):
                output = forward(model, data, diagnostic_zero=name == "unpruned")
                _sync(device)
                del output
        for pair in range(pairs):
            order = ("unpruned", "pruned") if pair % 2 == 0 else ("pruned", "unpruned")
            measurements = {}
            for name in order:
                _sync(device)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                    allocated, reserved = torch.cuda.memory_allocated(device), torch.cuda.memory_reserved(device)
                start = time.perf_counter_ns()
                output = forward(model, data, diagnostic_zero=name == "unpruned")
                _sync(device)
                elapsed = (time.perf_counter_ns() - start) / 1e6
                times[name].append(elapsed)
                measurements[name] = elapsed
                if device.type == "cuda":
                    end_allocated, end_reserved, peak_allocated, peak_reserved = _memory(device)
                    memory[name].append((allocated, reserved, end_allocated, end_reserved, peak_allocated, peak_reserved))
                del output
            paired.append({"batch": batch_index, "pair": pair, "order": list(order), **measurements})
        del data
    if a.state_sha256(model) != state_before:
        raise ComparisonError("model changed during benchmarking")
    memory_report = {}
    for name, observations in memory.items():
        reproducible = bool(observations) and len(set(observations)) == 1
        memory_report[name] = {"reproducible": reproducible, "bytes": dict(zip(
            ("allocated_before", "reserved_before", "allocated_after", "reserved_after", "peak_allocated", "peak_reserved"), observations[0])) if reproducible else None,
            "reason": None if reproducible else "CPU measurement unavailable or GPU allocator observations varied"}
    ratios = [p["unpruned"] / p["pruned"] for p in paired]
    return {"plan": _benchmark_plan(), "observed_warmups": warmups, "observed_pairs_per_batch": pairs,
            "validation_source_indices": indices, "validation_order_sha256": hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest(),
            "forward": {name: _summary(values) for name, values in times.items()}, "paired_measurements_ms": paired,
            "paired_speed_ratio_median": statistics.median(ratios),
            "paired_speed_ratio_sample_std_ddof1": statistics.stdev(ratios),
            "device_transfer": _summary(transfer_times), "gpu_memory": memory_report,
            "memory_scope": "model and resident full adapter input included; caching allocator only",
            "cpu_memory": None, "parameter_reduction": False, "input_adapter_reduction_claimed": False,
            "model_state_unchanged": True}


def validation_benchmark(root: Path, model: HTGNNClassifier, device: torch.device) -> dict[str, Any]:
    dataset = _dataset(root, model.task, "validation")
    iterator = a.e7.iter_production_batches(dataset, task=model.task, split="validation", batch_size=BATCH_SIZE, epoch_seed=None, max_batches=4)
    batches, loading = [], []
    for start in BENCHMARK_PLAN["sequential_batch_starts"]:
        beginning = time.perf_counter_ns()
        data = next(iterator)
        loading.append((time.perf_counter_ns() - beginning) / 1e6)
        if data.e2_audit["ordered_source_sample_indices"] != list(range(start, start + BATCH_SIZE)):
            raise ComparisonError("predeclared validation batch membership changed")
        batches.append(data)
    result = benchmark(model, batches, device)
    result["loading_and_conversion"] = _summary(loading)
    result["loading_scope"] = "next bounded lazy batch plus D6/E2 conversion; dataset initialization excluded"
    return result


def evaluate_pruned(root: Path, record: Mapping[str, Any], reporting: Mapping[str, Any],
                    scratch: Path, device: torch.device) -> dict[str, Any]:
    _test_access(root)
    candidate = _candidate(record)
    model = load_bound_model(root, record).to(device)
    before = a.state_sha256(model)
    n, position = sum(reporting["test_support"]), 0
    confusion = np.zeros((15, 15), dtype=np.int64)
    digest = hashlib.sha256()
    with tempfile.TemporaryDirectory(prefix="scores-", dir=scratch) as temporary:
        scores = targets = None
        if candidate.task == "E3":
            scores = np.memmap(Path(temporary) / "scores.bin", mode="w+", dtype=np.float64, shape=(n,))
            targets = np.memmap(Path(temporary) / "targets.bin", mode="w+", dtype=np.int8, shape=(n,))
        try:
            for data in iter_test_batches(root, candidate.task):
                indices = data.e2_audit["ordered_source_sample_indices"]
                if (any(type(i) is not int for i in indices) or indices != list(range(position, position + len(indices)))
                        or not 0 < len(indices) <= BATCH_SIZE or position + len(indices) > n):
                    raise ComparisonError("test traversal must be complete, sequential and unique")
                digest.update(np.asarray(indices, dtype="<i8").tobytes())
                logits = forward(model, data.to(device))
                y = data["sample"].y.cpu().numpy()
                if candidate.task == "E3":
                    scores[position:position + len(indices)] = logits.cpu().numpy()
                    targets[position:position + len(indices)] = y
                else:
                    predictions = logits.argmax(dim=1).cpu().numpy()
                    confusion += np.bincount(y.astype(np.int64) * 15 + predictions, minlength=225).reshape(15, 15)
                position += len(indices)
            if position != n or digest.hexdigest() != _order_sha256(n) or a.state_sha256(model) != before:
                raise ComparisonError("incomplete traversal or changed model state")
            metrics = e11.binary_metrics(targets, scores, reporting) if candidate.task == "E3" else e11.multiclass_metrics(confusion, reporting)
            result = {"candidate_id": candidate.candidate_id, "task": candidate.task, "seed": candidate.seed,
                      "model_family": "htgnn", "checkpoint_sha256": record["artifacts"]["checkpoint_sha256"],
                      "e10_result_sha256": record["result_sha256"], "policy_sha256": POLICY_SHA256,
                      "examples": position, "order_sha256": digest.hexdigest(), "metrics": metrics,
                      "model_state_sha256": before, "model_state_unchanged": True, "graph": frozen_policy().record()}
            result["result_sha256"] = a.d2.canonical_sha256(result)
            return result
        finally:
            for array in (scores, targets):
                if array is not None:
                    array.flush()
                    array._mmap.close()


def _metric_values(task: str, metrics: Mapping[str, Any]) -> dict[str, float | None]:
    if task == "E3":
        return {k: metrics[k] for k in ("pr_auc_average_precision", "roc_auc", "precision", "recall", "f1", "specificity",
                                        "false_positive_rate", "balanced_accuracy", "mcc", "accuracy")}
    return {f"{scope}.{key}": metrics[scope][key] for scope in ("primary", "all_test_descriptive")
            for key in ("macro_precision", "macro_recall", "macro_f1", "accuracy", "weighted_f1")}


def compare_results(pruned: Sequence[Mapping[str, Any]], bindings: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected = {(t, s) for t in TASKS for s in SEEDS}
    keys = {(r["task"], r["seed"]) for r in pruned}
    base = {(b["result"]["task"], b["result"]["seed"]): b for b in bindings}
    if len(pruned) != 9 or len(bindings) != 9 or keys != expected or set(base) != expected:
        raise ComparisonError("comparison requires exactly nine paired HTGNN results")
    paired, summaries = [], []
    for r in pruned:
        _canonical(r, "result_sha256")
        bound = base[(r["task"], r["seed"])]
        b = bound["result"]
        _canonical(b, "result_sha256")
        if (r["model_family"] != "htgnn" or b["model_family"] != "htgnn"
                or r["model_state_unchanged"] is not True or b["model_state_unchanged"] is not True
                or r["e10_result_sha256"] != b["e10_result_canonical_sha256"]
                or bound["canonical_sha256"] != b["result_sha256"]
                or r["policy_sha256"] != POLICY_SHA256 or r["checkpoint_sha256"] != b["checkpoint_sha256"]
                or r["candidate_id"] != b["candidate_id"] or r["examples"] != b["examples"]
                or r["order_sha256"] != b["order_sha256"] or r["graph"] != frozen_policy().record()):
            raise ComparisonError("pruned/unpruned identity, coverage or graph mismatch")
        if r["task"] != "E3" and e11.multiclass_metrics(
                np.asarray(r["metrics"]["confusion_matrix"], dtype=np.int64), bound["reporting_policy"]) != r["metrics"]:
            raise ComparisonError("pruned metric/confusion mismatch")
        rv, bv = _metric_values(r["task"], r["metrics"]), _metric_values(r["task"], b["metrics"])
        changes = {k: {"unpruned": bv[k], "pruned": rv[k], "absolute_change": None if bv[k] is None or rv[k] is None else rv[k] - bv[k],
                       "relative_drop": None if bv[k] is None or rv[k] is None or bv[k] == 0 else (bv[k] - rv[k]) / bv[k]} for k in rv}
        paired.append({"task": r["task"], "seed": r["seed"], "candidate_id": r["candidate_id"],
                       "role": "PRIMARY" if r["task"] == "E2" else "diagnostic transfer" if r["task"] == "E1" else "binary transfer",
                       "baseline_path": bound["path"], "baseline_physical_sha256": bound["physical_sha256"],
                       "baseline_canonical_sha256": bound["canonical_sha256"], "changes": changes})
    for task in TASKS:
        rows = [r for r in paired if r["task"] == task]
        metrics = {}
        for key in rows[0]["changes"]:
            metrics[key] = {}
            for field_name in ("unpruned", "pruned", "absolute_change", "relative_drop"):
                values = [r["changes"][key][field_name] for r in rows]
                valid = all(v is not None and math.isfinite(v) for v in values)
                metrics[key][field_name] = {"mean": statistics.mean(values) if valid else None,
                                          "sample_std_ddof1": statistics.stdev(values) if valid else None,
                                          "defined_seeds": sum(v is not None for v in values)}
        summaries.append({"task": task, "seeds": list(SEEDS), "metrics": metrics})
    return {"paired": paired, "seed_summary": summaries, "test_based_selection": False}


def _write(path: Path, content: Mapping[str, Any]) -> None:
    payload = a.d2.canonical_json_bytes(content) + b"\n"
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def run_owner(root: Path, *, run_id: str, device: str, policy_sha256: str, freeze_sha256: str,
              implementation_sha256: str, owner_confirmed_one_time_pruned_test_execution: bool = False) -> dict[str, Any]:
    """Explicit OWNER execution only. No caller performs automatic retries."""
    if owner_confirmed_one_time_pruned_test_execution is not True:
        raise ComparisonError("explicit owner one-time pruned test execution acknowledgement required")
    if policy_sha256 != POLICY_SHA256 or freeze_sha256 != FREEZE_SHA256:
        raise ComparisonError("owner command policy/freeze hashes mismatch")
    if device not in {"cpu", "cuda"} or (device == "cuda" and not torch.cuda.is_available()):
        raise ComparisonError("requested device unavailable")
    root = Path(root).resolve()
    preparation = prepare(root)
    if preparation["implementation_sha256"] != implementation_sha256:
        raise ComparisonError("owner-reviewed implementation source bundle changed")
    logs, outputs = (root / area / STUDY / run_id for area in ("logs/stage_e12b", "results/stage_e12b"))
    ledger = root / "logs/stage_e12b" / STUDY / "test_consumption.json"
    if logs.exists() or outputs.exists() or ledger.exists():
        raise ComparisonError("E.12B already reserved or namespace exists; retry/overwrite prohibited")
    with file_guard(root, run_id) as access, a.deterministic_inference(device):
        for directory in (logs.parent, outputs.parent):
            directory.mkdir(parents=True, exist_ok=True)
        logs.mkdir(exist_ok=False)
        try:
            outputs.mkdir(exist_ok=False)
            manifest = {"stage": "E.12B", "version": VERSION, "run_id": run_id, "preparation": preparation,
                        "device": device, "arithmetic": {"deterministic_algorithms": True, "cudnn_deterministic": True,
                        "cudnn_benchmark": False, "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")},
                        "unpruned_predictive_evaluation_repeated": False, "training": False, "optimizer": False,
                        "backpropagation": False, "policy_changes_allowed": False,
                        "historical_e11_arithmetic_bitwise_equivalence_certified": False}
            manifest["manifest_sha256"] = a.d2.canonical_sha256(manifest)
            _write(logs / "manifest.json", manifest)
            benchmarks = []
            for record in preparation["records"]:
                model = load_bound_model(root, record).to(device)
                measured = validation_benchmark(root, model, torch.device(device))
                benchmarks.append({"candidate_id": record["candidate_id"], "task": record["configuration"]["task"],
                                   "seed": record["configuration"]["seed"], **measured})
                del model
            _write(outputs / "validation_benchmarks.json", {"measurements": benchmarks})
            if prepare(root) != preparation or access.attempts:
                raise ComparisonError("provenance changed before test reservation")
            authorization = {"stage": "E.12B", "state": "CONSUMED_OR_RESERVED", "run_id": run_id,
                             "policy_sha256": POLICY_SHA256, "freeze_sha256": FREEZE_SHA256,
                             "manifest_sha256": manifest["manifest_sha256"], "implementation_sha256": implementation_sha256,
                             "owner_confirmed_one_time_pruned_test_execution": True,
                             "checkpoints": [r["artifacts"]["checkpoint_sha256"] for r in preparation["records"]],
                             "unpruned_test_evaluation_authorized": False, "policy_changes_authorized": False}
            authorization["authorization_sha256"] = a.d2.canonical_sha256(authorization)
            _write(ledger, authorization)
            access.ledger_sha256 = a.e10._sha256(ledger)
            access.test_enabled = True
            test_hashes = {p: h for item in preparation["splits"].values() for p, h in item["test_sha256"].items()}
            for p, expected in test_hashes.items():
                if a.e10._sha256(root / p) != expected:
                    raise ComparisonError("authorized test artifact integrity mismatch")
            pruned = []
            for record in preparation["records"]:
                result = evaluate_pruned(root, record, preparation["splits"][record["configuration"]["task"]]["policy"], logs, torch.device(device))
                if result["model_state_sha256"] != preparation["model_state_sha256"][record["candidate_id"]]:
                    raise ComparisonError("checkpoint/model state differs from preflight")
                _write(outputs / (record["candidate_id"] + ".json"), result)
                pruned.append(result)
            for p, expected in test_hashes.items():
                if a.e10._sha256(root / p) != expected:
                    raise ComparisonError("test artifact changed during evaluation")
            access.test_enabled = False
            if prepare(root) != preparation or access.attempts:
                raise ComparisonError("provenance changed during evaluation")
            report = {"stage": "E.12B", "version": VERSION, "run_id": run_id, "status": "COMPLETE_PENDING_REVIEW",
                      "policy_sha256": POLICY_SHA256, "freeze_sha256": FREEZE_SHA256, "manifest_sha256": manifest["manifest_sha256"],
                      "authorization_physical_sha256": access.ledger_sha256, "pruned_results": pruned,
                      "comparison": compare_results(pruned, preparation["baselines"]), "graph": frozen_policy().record(),
                      "test_arrays_opened": sorted(access.opened_test), "forbidden_access_attempts": access.attempts,
                      "model_states_unchanged": True, "unpruned_test_evaluation_repeated": False,
                      "training": False, "optimizer": False, "backpropagation": False,
                      "output_physical_sha256": {p.name: a.e10._sha256(p) for p in outputs.iterdir()}}
            report["report_sha256"] = a.d2.canonical_sha256(report)
            _write(outputs / "stage_e12b_report.json", report)
            with (outputs / "stage_e12b_report.sha256").open("x", encoding="ascii") as stream:
                stream.write(a.e10._sha256(outputs / "stage_e12b_report.json") + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            _write(logs / "status.json", {"state": "COMPLETE_PENDING_REVIEW", "completed_checkpoints": 9})
            return report
        except BaseException as exc:
            access.test_enabled = False
            _write(logs / "failure.json", {"state": "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED",
                    "error_type": type(exc).__name__, "error": str(exc), "automatic_retry_allowed": False,
                    "test_arrays_opened": sorted(access.opened_test), "forbidden_access_attempts": access.attempts})
            raise
