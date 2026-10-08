"""Bounded application replay responses; no tensors or uploaded contents."""
from typing import Literal

from pydantic import Field

from backend.app.schemas.contracts import Contract

API_VERSION = "v1"
SCORE_SEMANTICS = "uncalibrated normalized model score"


class ValidationResponse(Contract):
    validation_id: str
    filename: str
    size_bytes: int
    sha256: str
    flow_row_count: int
    canonical_feature_count: Literal[78] = 78
    columns_reordered: bool
    schema_valid: Literal[True] = True
    complete_window_count: int
    trailing_flow_count: int
    temporal_window: Literal[5] = 5
    replay_order: Literal["csv-row-order"] = "csv-row-order"
    transform_state_sha256: str
    model_inference_performed: Literal[False] = False


class Prediction(Contract):
    window_index: int
    target_flow_index: int
    prediction_class_id: int
    prediction_label: str
    model_score: float = Field(ge=0, le=1, allow_inf_nan=False)
    is_attack: bool


class ReplayModel(Contract):
    task: Literal["E2"] = "E2"
    family: Literal["HTGNN"] = "HTGNN"
    deployment_seed: Literal[2026] = 2026
    checkpoint_id: str
    checkpoint_sha256: str
    inference_mode: Literal["attention-pruned"] = "attention-pruned"
    model_score_semantics: Literal["uncalibrated normalized model score"] = SCORE_SEMANTICS


class Preprocessing(Contract):
    feature_count: Literal[78] = 78
    transform_state_id: str
    transform_state_sha256: str
    temporal_window: Literal[5] = 5


class ReplayGraph(Contract):
    input_feature_nodes: int
    retained_feature_nodes: int
    category_nodes: int
    total_nodes_per_graph: int
    edges_per_graph: int
    parameter_count: int


class ReplayProvenance(Contract):
    source_kind: Literal["external_application"] = "external_application"
    analysis_id: str
    uploaded_file_sha256: str
    application_adapter_version: str
    feature_schema_sha256: str
    feature_order_sha256: str
    transform_state_id: str
    transform_state_sha256: str
    temporal_window: Literal[5] = 5
    replay_order_semantics: Literal["csv-row-order"] = "csv-row-order"
    graph_schema_version: str
    graph_schema_sha256: str
    graph_taxonomy_version: str
    graph_taxonomy_sha256: str
    pruning_policy_sha256: str
    pruning_policy_record_sha256: str
    checkpoint_id: str
    checkpoint_sha256: str
    deployment_seed: Literal[2026] = 2026
    model_task: Literal["E2"] = "E2"
    class_mapping_sha256: str
    api_version: Literal["v1"] = API_VERSION


class AnalysisResponse(Contract):
    analysis_id: str
    source_filename: str
    source_sha256: str
    source_size_bytes: int
    input_flow_count: int
    complete_window_count: int
    trailing_flow_count: int
    columns_reordered: bool
    replay_order_semantics: Literal["csv-row-order"] = "csv-row-order"
    flow_index_base: Literal[0] = 0
    benign_prediction_count: int
    intrusion_prediction_count: int
    prediction_class_counts: dict[str, int]
    predictions: list[Prediction] = Field(max_length=5000)
    model: ReplayModel
    preprocessing: Preprocessing
    graph: ReplayGraph
    provenance: ReplayProvenance
    provenance_sha256: str
