"""Focused contract tests for the E.2 provenance-validation cache."""

from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import threading
import time
from typing import Any

import numpy as np
import pytest
import torch

from src.graph import adapter as d4
from src.graph import batching as d6
from src.graph import pyg_adapter as e2
from tests.test_pyg_adapter import _batch


ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "67889f3315bcb227b89c3c3e929afcde534ca800"
VALIDATION_REPORT = ROOT / "results" / "stage_e2" / "provenance_cache_validation.json"
VALIDATION_SHA = VALIDATION_REPORT.with_suffix(".sha256")
OBSERVATIONS = ROOT / "results" / "stage_e2" / "provenance_cache_observations.json"
BASELINE_FINGERPRINTS = {
    ("E1", "train"): "78ddb7ee851030b8402782b3f8e4aaf3999e42ab02d3fa3585ceeab310fc9135",
    ("E1", "validation"): "cc35a7fc7fbe19b83f77ec9ffd95c59b4653e0a0841f91412bb8a263a98fb3de",
    ("E1", "test"): "6c7d7498adff9fd03924f23882c583ef0f572f52a62521695260f5cc6c9a7232",
    ("E2", "train"): "bb50cddec01eb3e234cdab21f263d56557887695448b924babd83735717ec0d4",
    ("E2", "validation"): "74078a5ac4f6b6da456ff97c0577ac68ceb5cc0aa83a2bd427ea4d32daecd142",
    ("E2", "test"): "a1938b65ea97ce919bc6c9f98b3f602edd53f8615c28edea7af0368b5dc2dee9",
    ("E3", "train"): "dc58ad9926a861441deaf5e88270c2b874e609b71a64a6f7ed7fe510ebabbce2",
    ("E3", "validation"): "98b046bc430900fbd38b855ef5145526b0e733657dc70a48f990cd84d187034b",
    ("E3", "test"): "01690c32631e2b822a7ba11e02c21242b33e7c0d65cc86ff51e3bd9a5116a469",
}


@pytest.fixture(autouse=True)
def _isolated_cache() -> None:
    e2._reset_frozen_identity_cache()
    yield
    e2._reset_frozen_identity_cache()


def _copy_provenance_root(destination: Path) -> Path:
    for _name, relative in e2._FROZEN_IDENTITY_ARTIFACTS:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    return destination


def _output_fingerprint(data: Any) -> str:
    digest = hashlib.sha256()
    for node_type in data.node_types:
        digest.update(node_type.encode("utf-8"))
        for key, value in sorted(data[node_type].items()):
            digest.update(key.encode("utf-8"))
            if isinstance(value, torch.Tensor):
                array = value.detach().cpu().contiguous().numpy()
                digest.update(str(array.dtype).encode("ascii"))
                digest.update(
                    json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
                )
                digest.update(array.tobytes())
            else:
                digest.update(repr(value).encode("utf-8"))
    for edge_type in data.edge_types:
        digest.update(repr(edge_type).encode("utf-8"))
        for key, value in sorted(data[edge_type].items()):
            digest.update(key.encode("utf-8"))
            if isinstance(value, torch.Tensor):
                array = value.detach().cpu().contiguous().numpy()
                digest.update(str(array.dtype).encode("ascii"))
                digest.update(
                    json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
                )
                digest.update(array.tobytes())
            else:
                digest.update(repr(value).encode("utf-8"))
    digest.update(
        json.dumps(data.e2_audit, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    return digest.hexdigest()


def _cache_has_forbidden_payload(value: object) -> bool:
    if isinstance(value, (np.ndarray, torch.Tensor, d6.LogicalGraphBatch)):
        return True
    if isinstance(value, (tuple, list)):
        return any(_cache_has_forbidden_payload(item) for item in value)
    if isinstance(value, dict):
        return any(
            _cache_has_forbidden_payload(key) or _cache_has_forbidden_payload(item)
            for key, item in value.items()
        )
    return False


def test_first_conversion_validates_and_repeated_conversion_hits_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = _batch(tmp_path, [1, 3])
    original = e2._load_frozen_identities_uncached
    calls = 0

    def counted(project_root: Path) -> dict[str, str]:
        nonlocal calls
        calls += 1
        return original(project_root)

    monkeypatch.setattr(e2, "_load_frozen_identities_uncached", counted)
    first = e2.logical_batch_to_heterodata(batch, project_root=ROOT)
    second = e2.logical_batch_to_heterodata(batch, project_root=ROOT)
    assert calls == 1
    assert _output_fingerprint(first) == _output_fingerprint(second)


def test_cache_key_is_deterministic_and_changes_with_artifact_identity(
    tmp_path: Path,
) -> None:
    root = _copy_provenance_root(tmp_path / "root")
    first = e2._frozen_identity_cache_key(root)
    assert first == e2._frozen_identity_cache_key(root)
    e1_path = root / "results" / "stage_e1" / "environment-report.json"
    e1_path.write_bytes(e1_path.read_bytes() + b"\n")
    assert e2._frozen_identity_cache_key(root) != first


def test_changed_d8_report_invalidates_cache_and_is_rejected(
    tmp_path: Path,
) -> None:
    root = _copy_provenance_root(tmp_path / "root")
    batch = _batch(tmp_path / "batch", [1])
    e2.logical_batch_to_heterodata(batch, project_root=root)
    d8_path = root / "results" / "stage_d8" / "full_split_integration.json"
    report = json.loads(d8_path.read_text(encoding="utf-8"))
    report["overall_status"] = "FAIL"
    d8_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(e2.HeteroDataAdapterError, match="frozen D8 report is invalid"):
        e2.logical_batch_to_heterodata(batch, project_root=root)


def test_changed_e1_report_invalidates_cache_and_is_rejected(
    tmp_path: Path,
) -> None:
    root = _copy_provenance_root(tmp_path / "root")
    batch = _batch(tmp_path / "batch", [1])
    e2.logical_batch_to_heterodata(batch, project_root=root)
    e1_path = root / "results" / "stage_e1" / "environment-report.json"
    e1_path.write_bytes(e1_path.read_bytes() + b"\n")
    with pytest.raises(e2.HeteroDataAdapterError, match="hash is not approved"):
        e2.logical_batch_to_heterodata(batch, project_root=root)


def test_artifact_change_during_cold_validation_is_not_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _copy_provenance_root(tmp_path / "root")
    original = e2._load_frozen_identities_uncached

    def changing(project_root: Path) -> dict[str, str]:
        identities = original(project_root)
        e1_path = root / "results" / "stage_e1" / "environment-report.json"
        e1_path.write_bytes(e1_path.read_bytes() + b"\n")
        return identities

    monkeypatch.setattr(e2, "_load_frozen_identities_uncached", changing)
    with pytest.raises(e2.HeteroDataAdapterError, match="changed during validation"):
        e2._load_frozen_identities(root)
    assert e2._FROZEN_IDENTITY_CACHE is None


def test_cached_authority_is_immutable_bounded_and_returns_fresh_dicts(
    tmp_path: Path,
) -> None:
    expected = e2._load_frozen_identities(ROOT)
    expected["d1_taxonomy_sha256"] = "0" * 64
    assert e2._load_frozen_identities(ROOT)["d1_taxonomy_sha256"] != "0" * 64
    for index in range(12):
        root = _copy_provenance_root(tmp_path / f"root-{index}")
        e2._load_frozen_identities(root)
        assert e2._FROZEN_IDENTITY_CACHE is not None
        assert not _cache_has_forbidden_payload(e2._FROZEN_IDENTITY_CACHE)
        assert e2._FROZEN_IDENTITY_CACHE[0][0] == str(root.resolve())


def test_concurrent_cache_miss_performs_one_full_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = e2._load_frozen_identities_uncached
    calls = 0
    counter_lock = threading.Lock()

    def counted(project_root: Path) -> dict[str, str]:
        nonlocal calls
        with counter_lock:
            calls += 1
        time.sleep(0.01)
        return original(project_root)

    monkeypatch.setattr(e2, "_load_frozen_identities_uncached", counted)
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _index: e2._load_frozen_identities(ROOT), range(24)))
    assert calls == 1
    assert all(result == results[0] for result in results)


def test_warm_cache_does_not_bypass_experiment_split_or_batch_validation(
    tmp_path: Path,
) -> None:
    batch = _batch(tmp_path, [1])
    e2.logical_batch_to_heterodata(batch, project_root=ROOT)
    with pytest.raises(e2.HeteroDataAdapterError, match="unsupported experiment"):
        e2.logical_batch_to_heterodata(
            replace(batch, experiment="E4"), project_root=ROOT
        )
    with pytest.raises(e2.HeteroDataAdapterError, match="unsupported split"):
        e2.logical_batch_to_heterodata(
            replace(batch, split="holdout"), project_root=ROOT
        )


def test_cold_and_warm_outputs_do_not_alias_or_mutate_source(tmp_path: Path) -> None:
    batch = _batch(tmp_path, [1, 6])
    source_snapshot = np.array(batch.feature_x, copy=True)
    cold = e2.logical_batch_to_heterodata(batch, project_root=ROOT)
    warm = e2.logical_batch_to_heterodata(batch, project_root=ROOT)
    assert _output_fingerprint(cold) == _output_fingerprint(warm)
    cold["feature"].x[0, 0] += 1
    assert not torch.equal(cold["feature"].x, warm["feature"].x)
    assert np.array_equal(batch.feature_x, source_snapshot, equal_nan=True)


@pytest.mark.parametrize("experiment", tuple(d4.EXPERIMENT_DIRS))
@pytest.mark.parametrize("split", tuple(d4.SPLIT_STEMS))
def test_all_nine_pairs_match_frozen_pre_correction_fingerprint(
    experiment: str, split: str
) -> None:
    dataset = d4.LazyGraphDataset(experiment, split)
    indices = [0, len(dataset) // 2, len(dataset) - 1]
    batch = d6.build_logical_batch(dataset, indices)
    data = e2.logical_batch_to_heterodata(batch, project_root=ROOT)
    assert _output_fingerprint(data) == BASELINE_FINGERPRINTS[(experiment, split)]


def test_adapter_still_contains_no_model_or_training_components() -> None:
    tree = ast.parse((ROOT / "src" / "graph" / "pyg_adapter.py").read_text("utf-8"))
    forbidden = {
        "DataLoader",
        "Module",
        "Parameter",
        "Embedding",
        "Linear",
        "Conv1d",
        "GATv2Conv",
        "Optimizer",
        "AdamW",
        "backward",
        "prune",
    }
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert not (forbidden & names)


def test_saved_cache_evidence_is_canonical_complete_and_separates_timings() -> None:
    payload = VALIDATION_REPORT.read_bytes()
    report = json.loads(payload)
    supplied = report.pop("report_sha256")
    assert d6.d2.canonical_sha256(report) == supplied
    report["report_sha256"] = supplied
    assert d6.d2.canonical_json_bytes(report) == payload
    assert report["overall_status"] == "PASS"
    assert report["frozen_semantic_oracle"]["commit"] == BASELINE_COMMIT
    assert report["semantic_equivalence"]["pair_count"] == 9
    assert report["semantic_equivalence"]["all_pairs_equivalent"] is True
    assert report["cache_contract"]["maximum_entries"] == 1
    assert report["cache_contract"]["logical_batches_or_tensors_cached"] is False
    assert report["e6_regression"]["all_contracts_pass"] is True
    assert report["e6_regression"]["runtime_measurements_are_noncanonical"] is True
    physical = hashlib.sha256(payload).hexdigest()
    assert VALIDATION_SHA.read_text(encoding="ascii") == (
        f"{physical}  {VALIDATION_REPORT.name}\n"
    )
    observations = json.loads(OBSERVATIONS.read_text(encoding="utf-8"))
    assert observations["canonical_or_byte_reproducible"] is False
    assert observations["hardware_specific"] is True
    assert observations["h2d_or_forward_optimization_performed"] is False
