"""Application equivalence/adversarial checks; all windows are synthetic.

The only real predictive artifact loaded is the owner-selected frozen model.
The existing E.2 helper writes tiny NONCANONICAL fixtures outside the project.
No official partition, raw data, ranking, metric or benchmark is used.
"""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from backend.app.ml.application_graph import validate_application_graph
from backend.app.ml.application_inference import (
    ApplicationInferenceAdapter, CHECKPOINT_ID, CHECKPOINT_SHA256,
)
from backend.app.ml.external_windows import (
    ADAPTER_VERSION, ApplicationContractError, ExternalWindows, TransformIdentity,
)
from backend.app.services import artifacts
from src.graph import pyg_adapter as e2, schema
from src.pruning import stage_e12 as e12

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("application_e2_fixture_support", ROOT / "tests/test_pyg_adapter.py")
FIXTURES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIXTURES)


@pytest.fixture(scope="module")
def adapter():
    # Clearly synthetic caller-declared identity; no scaler state is opened.
    return ApplicationInferenceAdapter(ROOT, transform_state=TransformIdentity("synthetic-post-transform-state", "a" * 64))


def numeric(b=2):
    return (np.arange(b * 5 * 78, dtype=np.float32).reshape(b, 5, 78) / np.float32(1000))


def windows(adapter, values=None, max_windows=3):
    return adapter.windows(numeric() if values is None else values, source_id="synthetic-source",
                           analysis_id="synthetic-analysis", replay_order_semantics="synthetic deterministic array order",
                           max_windows=max_windows)


def graph(adapter):
    return adapter.graph(windows(adapter), max_windows=3)


def reject_before_encoder(adapter, data, *, policy=None):
    def forbidden(*args):
        pytest.fail("feature encoder ran before application rejection")
    handle = adapter.model.feature_encoder.register_forward_pre_hook(forbidden)
    try:
        with torch.inference_mode(), pytest.raises((ApplicationContractError, e12.PruningError, ValueError, RuntimeError)):
            adapter.logits(data, max_windows=3, policy=policy)
    finally:
        handle.remove()


def test_selected_checkpoint_policy_and_authoritative_identity(adapter):
    identity = adapter.identity
    assert identity.application_adapter_version == ADAPTER_VERSION
    assert identity.checkpoint_id == CHECKPOINT_ID
    assert identity.checkpoint_sha256 == CHECKPOINT_SHA256
    assert identity.deployment_seed == 2026 and identity.model_task == "E2"
    assert identity.pruning_policy_sha256 == "f7a8f6aa41cdfb6ad2ec8644ef4570677b7ef172374592f84c8239d7ac0e00aa"
    assert len(adapter.policy.retained_feature_ids) == 61
    record = adapter.policy.record()
    assert record["pruned_feature_ids"] == [6, 10, 29, 32, 34, 35, 40, 41, 42, 50, 52, 57, 64, 70, 71, 72, 73]
    assert record["pruned_graph"]["nodes"] == 70 and record["pruned_graph"]["edges"] == 69
    assert sum(p.numel() for p in adapter.model.parameters()) == 7183
    assert not any(m.training for m in adapter.model.modules())
    assert not any(p.requires_grad or p.grad is not None for p in adapter.model.parameters())


@pytest.mark.parametrize("indices", ([3], [6, 1], [6, 1, 4]))
def test_exact_graph_unpruned_pruned_repeat_and_compaction_equivalence(adapter, tmp_path, indices, monkeypatch):
    # Repository helper -> authorized noncanonical D4/D6 fixture -> genuine E.2.
    logical = FIXTURES._batch(tmp_path, indices, "E2")
    research = e2.logical_batch_to_heterodata(logical)
    values = np.array(logical.feature_x.transpose(0, 2, 1), order="C", copy=True)
    owned = windows(adapter, values)
    application = adapter.graph(owned, max_windows=3)
    assert tuple(application.node_types) == tuple(research.node_types)
    assert tuple(application.edge_types) == tuple(research.edge_types)
    assert "e2_audit" not in application._global_store and "y" not in application["sample"]
    for kind, fields in {"feature": ("x", "feature_id", "batch", "ptr"),
                         "category": ("category_id", "batch", "ptr"), "sample": ("batch", "ptr")}.items():
        assert application[kind].num_nodes == research[kind].num_nodes
        for field in fields:
            left, right = application[kind][field], research[kind][field]
            assert left.dtype == right.dtype and left.device == right.device
            assert torch.equal(left, right), (kind, field)
    for relation in schema.RELATION_TYPES:
        assert torch.equal(application[relation].edge_index, research[relation].edge_index)
    assert application.num_nodes == len(indices) * 87
    assert application.num_edges == len(indices) * 86
    for left, right in zip(e12.compact_inputs(application, adapter.policy, len(indices)),
                           e12.compact_inputs(research, adapter.policy, len(indices)), strict=True):
        assert torch.equal(left, right)
    before = e12.state_sha256(adapter.model)
    with e12.deterministic_inference("cpu"), torch.inference_mode():
        assert torch.equal(adapter.model(application), adapter.model(research))
        expected = e12.pruned_logits(adapter.model, research, adapter.policy)
        # Ensure application never calls the research validation/inference entry.
        def forbidden(*args, **kwargs):
            pytest.fail("application called a research-only entry point")
        monkeypatch.setattr(e12, "pruned_logits", forbidden)
        monkeypatch.setattr(e12, "_validate_batch", forbidden)
        original_compact = e12.compact_inputs
        calls = []
        def observed(*args):
            calls.append(args[2])
            return original_compact(*args)
        monkeypatch.setattr(e12, "compact_inputs", observed)
        actual = adapter.logits(application, max_windows=3)
        assert torch.equal(actual, expected)
        assert torch.equal(actual, adapter.logits(application, max_windows=3))
        assert torch.equal(actual, adapter.infer(owned, max_windows=3))
        assert calls == [len(indices)] * 3
        assert actual.shape == (len(indices), 15) and not actual.requires_grad
    assert e12.state_sha256(adapter.model) == before
    assert all(p.grad is None for p in adapter.model.parameters())


@pytest.mark.parametrize("shape", ((5, 78), (0, 5, 78), (1, 4, 78), (1, 6, 78), (1, 5, 77), (1, 5, 79), (1, 5, 78, 1)))
def test_invalid_window_dimensions(adapter, shape):
    with pytest.raises(ApplicationContractError):
        windows(adapter, np.zeros(shape, dtype=np.float32))


@pytest.mark.parametrize("dtype", (np.float64, np.float16, np.int64, np.int32, object, ">f4"))
def test_wrong_dtype(adapter, dtype):
    with pytest.raises(ApplicationContractError):
        windows(adapter, np.zeros((1, 5, 78), dtype=dtype))


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), float("-inf")))
def test_nonfinite_windows(adapter, bad):
    values = numeric()
    values[-1, -1, -1] = bad
    with pytest.raises(ApplicationContractError):
        windows(adapter, values)


@pytest.mark.parametrize("bound", (0, -1, True, 1.5, "3", None, 1))
def test_explicit_positive_caller_bound(adapter, bound):
    with pytest.raises(ApplicationContractError):
        windows(adapter, max_windows=bound)


def test_non_numpy_input_rejected(adapter):
    with pytest.raises(ApplicationContractError):
        windows(adapter, numeric().tolist())


@pytest.mark.parametrize("layout", ("contiguous", "strided", "fortran", "readonly"))
def test_owned_snapshot_graph_and_noncontiguous_normalization(adapter, layout):
    values = numeric()
    if layout == "strided":
        values = values[:, :, ::-1]
    elif layout == "fortran":
        values = np.asfortranarray(values)
    elif layout == "readonly":
        values.flags.writeable = False
    expected = values.copy()
    owned = windows(adapter, values)
    application = adapter.graph(owned, max_windows=3)
    assert np.array_equal(values, expected)
    assert owned.values.flags.c_contiguous and not owned.values.flags.writeable
    with pytest.raises(ValueError):
        owned.values.flags.writeable = True
    values.flags.writeable = True
    values[:] = 999
    assert np.array_equal(owned.values, expected)
    assert torch.equal(application["feature"].x, torch.tensor(expected.transpose(0, 2, 1).copy()).reshape(-1, 5))
    application["feature"].x[0, 0] += 1
    assert np.array_equal(owned.values, expected)
    assert not torch.equal(application["feature"].x, adapter.graph(owned, max_windows=3)["feature"].x)


def test_typed_provenance_is_deeply_immutable(adapter):
    owned = windows(adapter)
    for target, field, value in ((owned, "payload", b""), (owned.provenance, "source_kind", "validation"),
                                  (owned.provenance.identity, "model_task", "E1"),
                                  (owned.provenance.identity.transform_state, "state_id", "changed")):
        with pytest.raises(FrozenInstanceError):
            setattr(target, field, value)


@pytest.mark.parametrize("field,value", (
    ("application_adapter_version", "other"), ("feature_order_sha256", "0" * 64),
    ("feature_schema_sha256", "0" * 64), ("graph_taxonomy_sha256", "0" * 64),
    ("graph_schema_sha256", "0" * 64), ("graph_taxonomy_version", "other"),
    ("graph_schema_version", "other"), ("pruning_policy_sha256", "0" * 64),
    ("pruning_policy_record_sha256", "0" * 64), ("checkpoint_id", "wrong"),
    ("checkpoint_sha256", "0" * 64), ("deployment_seed", 2025), ("model_task", "E1"),
    ("model_task", "E3"), ("temporal_window_size", 4), ("window_semantics", "packets"),
    ("target_semantics", "first flow"), ("class_mapping_sha256", "0" * 64),
    ("transform_state", TransformIdentity("other", "b" * 64)),
    ("temporal_window_size", 5.0), ("deployment_seed", 2026.0),
))
def test_wrong_application_identity_rejected_before_model(adapter, field, value):
    application = graph(adapter)
    application.application_audit = replace(application.application_audit, identity=replace(adapter.identity, **{field: value}))
    reject_before_encoder(adapter, application)


@pytest.mark.parametrize("fault", ("missing", "research_dict", "research_audit", "source_kind", "payload_hash", "batch_size"))
def test_missing_or_fake_application_provenance(adapter, fault):
    application = graph(adapter)
    if fault == "missing":
        del application.application_audit
    elif fault == "research_dict":
        application.application_audit = {"experiment": "E2", "split": "validation"}
    elif fault == "research_audit":
        application.e2_audit = {"experiment": "E2", "split": "validation"}
    elif fault == "source_kind":
        with pytest.raises(ApplicationContractError):
            replace(application.application_audit, source_kind="validation")
        return
    elif fault == "payload_hash":
        application.application_audit = replace(application.application_audit, transformed_payload_sha256="0" * 64)
    else:
        application.application_audit = replace(application.application_audit, batch_size=1)
    reject_before_encoder(adapter, application)


@pytest.mark.parametrize("kind", schema.NODE_KINDS)
@pytest.mark.parametrize("field", ("batch", "ptr", "num_nodes"))
def test_ownership_and_counts_reject_mutation(adapter, kind, field):
    application = graph(adapter)
    if field == "num_nodes":
        application[kind].num_nodes += 1
    else:
        application[kind][field][0] += 1
    reject_before_encoder(adapter, application)


@pytest.mark.parametrize("kind,field", (("feature", "feature_id"), ("category", "category_id"),
                                        ("feature", "batch"), ("category", "ptr"), ("sample", "batch")))
@pytest.mark.parametrize("fault", ("mutation", "dtype", "shape"))
def test_identifiers_tensor_dtype_and_shape(adapter, kind, field, fault):
    application = graph(adapter)
    if fault == "mutation":
        application[kind][field][0] += 1
    elif fault == "dtype":
        application[kind][field] = application[kind][field].int()
    else:
        application[kind][field] = application[kind][field][:-1]
    reject_before_encoder(adapter, application)


@pytest.mark.parametrize("relation", schema.RELATION_TYPES)
@pytest.mark.parametrize("fault", ("missing", "added", "destination", "cross_sample", "negative", "out_of_range", "reordered", "dtype"))
def test_exact_edges_reject_all_topology_corruption(adapter, relation, fault):
    application = graph(adapter)
    edges = application[relation].edge_index
    if fault == "missing":
        application[relation].edge_index = edges[:, :-1].contiguous()
    elif fault == "added":
        application[relation].edge_index = torch.cat((edges, edges[:, :1]), dim=1)
    elif fault == "destination":
        edges[1, 0] = (int(edges[1, 0]) + 1) % schema.NODE_COUNTS[relation[2]]
        if relation == schema.CATEGORY_TO_SAMPLE_RELATION:
            edges[0, 0] = 1
    elif fault == "cross_sample":
        edges[1, 0] += schema.NODE_COUNTS[relation[2]]
    elif fault == "negative":
        edges[0, 0] = -1
    elif fault == "out_of_range":
        edges[1, 0] = application[relation[2]].num_nodes
    elif fault == "reordered":
        application[relation].edge_index = edges.flip(1)
    else:
        application[relation].edge_index = edges.int()
    reject_before_encoder(adapter, application)


@pytest.mark.parametrize("fault", ("extra_node", "reverse_edge", "label", "extra_feature", "dtype", "nan", "inf", "value", "gradient", "wrong_device"))
def test_predictive_graph_contract_is_strict(adapter, fault):
    application = graph(adapter)
    if fault == "extra_node":
        application["timestamp"].num_nodes = 2
    elif fault == "reverse_edge":
        application[("category", "reverse", "feature")].edge_index = application[schema.FEATURE_TO_CATEGORY_RELATION].edge_index.flip(0)
    elif fault == "label":
        application["sample"].y = torch.zeros(2, dtype=torch.long)
    elif fault == "extra_feature":
        application["feature"].timestamp = torch.zeros(156)
    elif fault == "dtype":
        application["feature"].x = application["feature"].x.double()
    elif fault in {"nan", "inf", "value"}:
        application["feature"].x[0, 0] = {"nan": float("nan"), "inf": float("inf"), "value": 99.}[fault]
    elif fault == "gradient":
        application["feature"].x.requires_grad_(True)
    else:
        application["category"].category_id = application["category"].category_id.to("meta")
    reject_before_encoder(adapter, application)


def test_wrong_mask_and_valid_same_quota_swapped_mask_rejected(adapter):
    reject_before_encoder(adapter, graph(adapter), policy=e12.FeaturePolicy(0., tuple(range(78))))
    retained = list(adapter.policy.retained_feature_ids)
    removed = adapter.policy.record()["pruned_feature_ids"]
    categories = schema.FEATURE_TO_CATEGORY_DESTINATION_INDICES
    old, new = next((i, j) for i in retained for j in removed if categories[i] == categories[j])
    retained.remove(old)
    retained.append(new)
    other = e12.FeaturePolicy(adapter.policy.requested_ratio, tuple(sorted(retained)))
    reject_before_encoder(adapter, graph(adapter), policy=other)
    reject_before_encoder(adapter, graph(adapter), policy=adapter.policy.record())


@pytest.mark.parametrize("fault", ("training", "child_training", "requires_grad", "grad_present", "weights", "task"))
def test_selected_model_state_mode_and_task_guards(adapter, fault, monkeypatch):
    m = adapter.model
    original = deepcopy(m.state_dict())
    try:
        if fault == "training":
            m.train()
        elif fault == "child_training":
            m.graph_encoder.feature_to_category.train()
        elif fault == "requires_grad":
            next(m.parameters()).requires_grad_(True)
        elif fault == "grad_present":
            next(m.parameters()).grad = torch.zeros_like(next(m.parameters()))
        elif fault == "weights":
            with torch.no_grad():
                next(m.parameters()).reshape(-1)[0] += 1
        else:
            monkeypatch.setattr(adapter, "_model", e12.HTGNNClassifier("E1").requires_grad_(False).eval())
        reject_before_encoder(adapter, graph(adapter))
    finally:
        m.load_state_dict(original)
        m.requires_grad_(False).eval()
        for p in m.parameters():
            p.grad = None


def test_gradient_enabled_caller_rejected(adapter):
    with torch.enable_grad(), pytest.raises(ApplicationContractError, match="gradients disabled"):
        adapter.logits(graph(adapter), max_windows=3)
    with torch.enable_grad(), pytest.raises(ApplicationContractError, match="gradients disabled"):
        adapter.infer(windows(adapter), max_windows=3)


def test_no_grad_execution_and_determinism_settings_are_restored(adapter):
    before = torch.are_deterministic_algorithms_enabled()
    with torch.no_grad():
        output = adapter.logits(graph(adapter), max_windows=3)
    assert not output.requires_grad
    assert torch.are_deterministic_algorithms_enabled() == before


@pytest.mark.parametrize("split", ("test", "unknown", "external_application"))
def test_frozen_research_split_guard_still_rejects(adapter, tmp_path, split):
    data = e2.logical_batch_to_heterodata(FIXTURES._batch(tmp_path, [1], "E2"))
    data.e2_audit["split"] = split
    with torch.inference_mode(), pytest.raises(e12.PruningError, match="test and unknown"):
        e12.pruned_logits(adapter.model, data, adapter.policy)


def test_frozen_research_entry_rejects_application_audit(adapter):
    with torch.inference_mode(), pytest.raises(e12.PruningError):
        e12.pruned_logits(adapter.model, graph(adapter), adapter.policy)


@pytest.mark.parametrize("fault", ("truncated", "changed", "mutable"))
def test_owned_window_payload_tampering(adapter, fault):
    owned = windows(adapter)
    if fault == "truncated":
        payload = owned.payload[:-4]
    elif fault == "changed":
        payload = bytes([owned.payload[0] ^ 1]) + owned.payload[1:]
    else:
        payload = bytearray(owned.payload)
    with pytest.raises(ApplicationContractError):
        adapter.graph(ExternalWindows(payload, owned.provenance), max_windows=3)


@pytest.mark.parametrize("field", ("source_id", "analysis_id", "replay_order_semantics"))
@pytest.mark.parametrize("value", ("", "  ", 42, "x" * 257))
def test_invalid_provenance_text(adapter, field, value):
    kwargs = dict(source_id="synthetic", analysis_id="synthetic", replay_order_semantics="synthetic array order", max_windows=3)
    kwargs[field] = value
    with pytest.raises(ApplicationContractError):
        adapter.windows(numeric(), **kwargs)


@pytest.mark.parametrize("fault", ("checkpoint_id", "checkpoint_hash", "policy_file", "freeze_file"))
def test_authentication_fails_closed_on_corrupted_artifacts(tmp_path, adapter, fault, monkeypatch):
    if fault.startswith("checkpoint"):
        deployment = artifacts.discover(ROOT)
        if fault == "checkpoint_id":
            deployment.record["candidate_id"] = "other"
        else:
            deployment.record["artifacts"]["checkpoint_sha256"] = "0" * 64
        monkeypatch.setattr("backend.app.ml.application_inference.discover", lambda root: deployment)
        with pytest.raises(ApplicationContractError):
            ApplicationInferenceAdapter(ROOT, transform_state=adapter.identity.transform_state)
    else:
        from backend.app.services.artifacts import bound_json, FREEZE_PATH, FREEZE_PHYSICAL_SHA256, POLICY_PATH
        relative = FREEZE_PATH if fault == "freeze_file" else POLICY_PATH
        expected_hash = FREEZE_PHYSICAL_SHA256 if fault == "freeze_file" else artifacts.discover(ROOT).freeze["policy_physical_sha256"]
        target = tmp_path / relative
        target.parent.mkdir(parents=True)
        target.write_text("{}", encoding="utf-8")
        with pytest.raises(artifacts.ArtifactError):
            bound_json(tmp_path, relative, expected_hash)


def test_graph_bound_rechecked_at_every_entry(adapter):
    application = graph(adapter)
    with pytest.raises(ApplicationContractError):
        validate_application_graph(application, adapter.identity, max_windows=1)
    with torch.inference_mode(), pytest.raises(ApplicationContractError):
        adapter.logits(application, max_windows=1)
