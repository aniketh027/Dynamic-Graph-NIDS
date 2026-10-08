"""Authenticated CPU application inference, separate from research split guards."""
from pathlib import Path

import torch

from backend.app.ml.application_graph import application_graph, validate_application_graph
from backend.app.ml.external_windows import (
    ADAPTER_VERSION, TARGET_SEMANTICS, WINDOW_SEMANTICS,
    ApplicationContractError, ApplicationIdentity, ExternalWindows, TransformIdentity, snapshot_windows,
)
from backend.app.services.artifacts import discover
from backend.app.services.inference import load_frozen
from src.graph import adapter as d4, schema, taxonomy
from src.graph.htgnn_classifier import HTGNNClassifier, _validated_head
from src.preprocessing.labels import INT_TO_LABEL
from src.preprocessing.schema import MODEL_FEATURE_COLUMNS
from src.pruning import stage_e12 as e12
from src.training import harness
from torch_geometric.data import HeteroData
import numpy as np

# Owner-pinned application deployment choice; scientific definitions stay in src.
CHECKPOINT_ID = "e10-htgnn-e2-lr1e-3-s2026-b256-0a1122c0ee42"
CHECKPOINT_SHA256 = "ef7976f4beb58813a596453d65b448cb1a09f3998ac5865b667d51795dcf135f"
DEPLOYMENT_SEED = 2026


class ApplicationInferenceAdapter:
    """Load through C1, bind immutable identities, and return raw E2 logits.

    Call logits/infer inside torch.inference_mode() or torch.no_grad(). This
    rejects accidental gradient-enabled use instead of silently changing it.
    Only CPU is supported in this version, matching the C1 selected loader.
    """

    def __init__(self, root: Path, *, transform_state: TransformIdentity):
        if type(transform_state) is not TransformIdentity:
            raise ApplicationContractError("a trusted typed expected transform identity is required")
        transform_state.__post_init__()
        deployment = discover(Path(root))
        record = deployment.record
        if (record["candidate_id"] != CHECKPOINT_ID
                or record["artifacts"]["checkpoint_sha256"] != CHECKPOINT_SHA256
                or record["configuration"]["seed"] != DEPLOYMENT_SEED
                or record["configuration"]["task"] != "E2"):
            raise ApplicationContractError("C1 discovery did not authenticate the owner-selected deployment")
        self._model, self._policy = load_frozen(Path(root), deployment)
        d1 = taxonomy.build_manifest(MODEL_FEATURE_COLUMNS)
        d2 = schema.build_manifest(d1)
        if (d1["taxonomy_sha256"] != d4.STATIC_GRAPH_METADATA.d1_taxonomy_sha256
                or d2["schema_sha256"] != d4.STATIC_GRAPH_METADATA.d2_schema_sha256):
            raise ApplicationContractError("scientific taxonomy/schema identities changed")
        self._identity = ApplicationIdentity(
            application_adapter_version=ADAPTER_VERSION,
            feature_schema_sha256=schema.canonical_sha256({"feature_names": list(MODEL_FEATURE_COLUMNS), "dtype": "float32"}),
            feature_order_sha256=d1["stage_c_feature_order"]["sha256"],
            transform_state=transform_state,
            temporal_window_size=schema.TEMPORAL_SEMANTICS["sequence_length"],
            window_semantics=WINDOW_SEMANTICS, target_semantics=TARGET_SEMANTICS,
            graph_taxonomy_version=taxonomy.TAXONOMY_VERSION, graph_taxonomy_sha256=d1["taxonomy_sha256"],
            graph_schema_version=schema.SCHEMA_VERSION, graph_schema_sha256=d2["schema_sha256"],
            pruning_policy_sha256=deployment.freeze["policy_canonical_sha256"],
            pruning_policy_record_sha256=schema.canonical_sha256(self._policy.record()),
            checkpoint_id=record["candidate_id"], checkpoint_sha256=record["artifacts"]["checkpoint_sha256"],
            deployment_seed=record["configuration"]["seed"], model_task=record["configuration"]["task"],
            class_mapping_sha256=schema.canonical_sha256(INT_TO_LABEL),
        )
        self._state_sha256 = e12.state_sha256(self._model)
        self._model_contract = harness.model_contract(self._model)
        self._validate_policy(self._policy)
        self._validate_model(self._model)

    @property
    def identity(self) -> ApplicationIdentity:
        return self._identity

    @property
    def model(self) -> HTGNNClassifier:
        return self._model

    @property
    def policy(self) -> e12.FeaturePolicy:
        return self._policy

    def windows(self, values: np.ndarray, *, source_id: str, analysis_id: str,
                replay_order_semantics: str, max_windows: int) -> ExternalWindows:
        return snapshot_windows(values, self.identity, source_id=source_id, analysis_id=analysis_id,
                                replay_order_semantics=replay_order_semantics, max_windows=max_windows)

    def graph(self, windows: ExternalWindows, *, max_windows: int) -> HeteroData:
        return application_graph(windows, self.identity, max_windows=max_windows)

    def _validate_policy(self, policy) -> None:
        if type(policy) is not e12.FeaturePolicy:
            raise ApplicationContractError("the frozen FeaturePolicy is required")
        authenticated = e12.FeaturePolicy(policy.requested_ratio, policy.retained_feature_ids)
        if schema.canonical_sha256(authenticated.record()) != self.identity.pruning_policy_record_sha256:
            raise ApplicationContractError("policy differs from the authenticated frozen E.12A artifact")

    def _validate_model(self, model) -> None:
        if not isinstance(model, HTGNNClassifier) or harness.model_contract(model) != self._model_contract:
            raise ApplicationContractError("model differs from the authenticated E2 model contract")
        contract, _ = _validated_head(model)
        if contract.task != self.identity.model_task or contract.output_channels != len(INT_TO_LABEL):
            raise ApplicationContractError("the authoritative E2 class head is required")
        if any(module.training for module in model.modules()):
            raise ApplicationContractError("all model modules must be in eval mode")
        tensors = tuple(model.parameters()) + tuple(model.buffers())
        if any(t.device != torch.device("cpu") or t.dtype != torch.float32 for t in tensors):
            raise ApplicationContractError("the authenticated CPU float32 model is required")
        if any(p.requires_grad or p.grad is not None for p in model.parameters()):
            raise ApplicationContractError("model gradients must be disabled and absent")
        if e12.state_sha256(model) != self._state_sha256:
            raise ApplicationContractError("model state differs from the authenticated selected checkpoint")

    def logits(self, data: HeteroData, *, max_windows: int,
               policy: e12.FeaturePolicy | None = None) -> torch.Tensor:
        if torch.is_grad_enabled():
            raise ApplicationContractError("application inference requires gradients disabled by the caller")
        self._validate_model(self.model)
        b = validate_application_graph(data, self.identity, max_windows=max_windows)
        selected_policy = self.policy if policy is None else policy
        self._validate_policy(selected_policy)
        with e12.deterministic_inference("cpu"):
            # compact_inputs has no split assumptions. The application validator
            # above independently authenticates every input it depends on.
            values, ids, edges = e12.compact_inputs(data, selected_policy, b)
            feature_embeddings = self.model.feature_encoder(values, ids)
            graph = self.model.graph_encoder
            category_targets = graph.category_embedding(data["category"].category_id)
            message = graph.feature_to_category((feature_embeddings, category_targets), edges)
            categories = graph.feature_to_category_activation(
                graph.feature_to_category_norm(graph.feature_to_category_dropout(message) + category_targets))
            sample_targets = graph.sample_token.unsqueeze(0).expand(b, -1)
            message = graph.category_to_sample((categories, sample_targets), data[schema.CATEGORY_TO_SAMPLE_RELATION].edge_index)
            samples = graph.category_to_sample_activation(
                graph.category_to_sample_norm(graph.category_to_sample_dropout(message) + sample_targets))
            _, head = _validated_head(self.model)
            logits = head(samples)
        if logits.shape != (b, len(INT_TO_LABEL)) or not torch.isfinite(logits).all():
            raise ApplicationContractError("invalid raw E2 logits")
        return logits

    def infer(self, windows: ExternalWindows, *, max_windows: int) -> torch.Tensor:
        if torch.is_grad_enabled():
            raise ApplicationContractError("application inference requires gradients disabled by the caller")
        return self.logits(self.graph(windows, max_windows=max_windows), max_windows=max_windows)
