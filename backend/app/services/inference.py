"""Application facade for authenticated frozen model loading.

C1 establishes metadata/loading only. External windows cannot be disguised as
research-split batches. C2 must resolve that request/provenance boundary first.
"""
import logging
import math
from pathlib import Path
from typing import Callable, Sequence

from backend.app.core.errors import ApplicationError
from backend.app.schemas.contracts import APIError, ModelResponse, SchemaResponse, StatusResponse
from backend.app.services.artifacts import Deployment, SELECTION_RULE, discover
from src.graph import schema as graph_schema
from src.preprocessing.labels import LABEL_VOCABULARY
from src.preprocessing.schema import MODEL_FEATURE_COLUMNS
from backend.app.services.csv_schema import csv_descriptor

logger = logging.getLogger("graphnids.adapter")


def schema_metadata() -> SchemaResponse:
    window = graph_schema.TEMPORAL_SEMANTICS["sequence_length"]
    return SchemaResponse(
        model_input_features=len(MODEL_FEATURE_COLUMNS), feature_names=MODEL_FEATURE_COLUMNS,
        temporal_window=window, shape=(window, len(MODEL_FEATURE_COLUMNS)),
        window_semantics="Application replay preserves uploaded CSV row order; five non-overlapping flow records; not packets",
        prediction_target="fifth/final flow", research_basis="CICFlowMeter / CSE-CIC-IDS2018",
        csv_compatibility=csv_descriptor(),
    )


def load_frozen(root: Path, deployment: Deployment):
    # Deferred imports: module import and health checks never initialize Torch/model.
    from src.pruning import stage_e12
    model = stage_e12.load_bound_model(root, deployment.record)
    frozen = deployment.freeze
    policy = stage_e12.FeaturePolicy(frozen["selected_ratio"], tuple(frozen["retained_feature_ids"]))
    if policy.record() != frozen["policy_record"]:
        raise ApplicationError("BACKEND_NOT_READY", "Frozen policy compatibility failed.")
    return model, policy


class FrozenInferenceService:
    def __init__(self, root: Path, *, discoverer: Callable = discover, loader: Callable = load_frozen):
        self.root = root
        self._discover, self._loader = discoverer, loader
        self._deployment = None
        self._model = self._policy = None
        self.state = "uninitialized"
        self.error = None

    def initialize(self, *, load_model: bool) -> None:
        self.state, self.error = "loading", None
        self._model = self._policy = self._deployment = None
        try:
            self._deployment = self._discover(self.root)
            if load_model:
                self._model, self._policy = self._loader(self.root, self._deployment)
                self.state = "ready"
            else:
                self.state = "uninitialized"
        except Exception:
            self.state = "error"
            self._model = self._policy = None
            self.error = APIError(code="BACKEND_NOT_READY", message="Frozen model initialization failed. See backend logs.")
            logger.exception("Frozen adapter initialization failed")

    def close(self) -> None:
        self._model = self._policy = None
        self.state = "uninitialized"

    def status(self) -> StatusResponse:
        return StatusResponse(model_adapter_state=self.state, model_loaded=self._model is not None, error=self.error)

    def metadata(self) -> ModelResponse:
        if self._deployment is None:
            raise ApplicationError("BACKEND_NOT_READY", "Frozen model metadata is unavailable.")
        deployment = self._deployment
        config = deployment.record["configuration"]
        policy = deployment.freeze["policy_record"]
        original, pruned = policy["original_graph"], policy["pruned_graph"]
        return ModelResponse(
            model_family="HTGNN", task=config["task"], class_count=len(LABEL_VOCABULARY),
            model_input_features=len(MODEL_FEATURE_COLUMNS), retained_feature_nodes=pruned["feature_nodes"],
            removed_feature_nodes=len(policy["pruned_feature_ids"]), category_nodes=pruned["category_nodes"],
            total_graph_nodes_unpruned=original["nodes"], total_graph_nodes_pruned=pruned["nodes"],
            edges_unpruned=original["edges"], edges_pruned=pruned["edges"],
            temporal_window=graph_schema.TEMPORAL_SEMANTICS["sequence_length"],
            deployment_seed=config["seed"], checkpoint_id=deployment.record["candidate_id"],
            checkpoint_sha256=deployment.record["artifacts"]["checkpoint_sha256"], learning_rate=config["learning_rate"],
            learning_rate_provenance="E.10 validation-only learning-rate selection at seed 2025",
            policy_sha256=deployment.freeze["policy_canonical_sha256"], selection_rule=SELECTION_RULE,
            pruned_validation_macro_f1_by_seed=deployment.scores,
        )

    def validate_input_contract(self, feature_names: Sequence[str], window: Sequence[Sequence[float]]) -> None:
        """Validate bounded numeric dimensions only; this certifies no CSV/cleaning/provenance."""
        expected = schema_metadata()
        if (not isinstance(feature_names, (list, tuple)) or len(feature_names) != len(MODEL_FEATURE_COLUMNS)
                or tuple(feature_names) != MODEL_FEATURE_COLUMNS
                or not isinstance(window, (list, tuple)) or len(window) != expected.temporal_window):
            raise ApplicationError("SCHEMA_MISMATCH", "Expected the canonical 78-feature order and five flow records.", 422)
        for row in window:
            valid = isinstance(row, (list, tuple)) and len(row) == len(MODEL_FEATURE_COLUMNS)
            try:
                valid = valid and all(type(v) in {int, float} and math.isfinite(v) for v in row)
            except (OverflowError, TypeError):
                valid = False
            if not valid:
                raise ApplicationError("SCHEMA_MISMATCH", "Each flow requires 78 finite numeric feature values.", 422)

    def predict_window(self, feature_names: Sequence[str], window: Sequence[Sequence[float]]):
        self.validate_input_contract(feature_names, window)
        if self._model is None:
            raise ApplicationError("MODEL_NOT_LOADED", "The frozen model is not loaded.")
        raise ApplicationError("CAPABILITY_NOT_IMPLEMENTED", "Application window prediction is pending the replay input adapter.", 501)
