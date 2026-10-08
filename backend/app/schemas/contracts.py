"""Typed public responses. No local artifact paths or exception text."""
from typing import Literal

from pydantic import BaseModel, ConfigDict

AdapterState = Literal["uninitialized", "loading", "ready", "error"]
ErrorCode = Literal[
    "BACKEND_NOT_READY", "MODEL_NOT_LOADED", "SCHEMA_MISMATCH",
    "INVALID_REQUEST", "CAPABILITY_NOT_IMPLEMENTED", "INTERNAL_ERROR",
    "INVALID_FILE_TYPE", "FILE_TOO_LARGE", "EMPTY_FILE", "CSV_PARSE_ERROR",
    "DUPLICATE_COLUMN", "UNEXPECTED_COLUMN", "MISSING_COLUMN", "NON_NUMERIC_FEATURE",
    "TOO_MANY_ROWS", "INSUFFICIENT_FLOWS", "TRANSFORM_STATE_INVALID", "TRANSFORM_FAILED",
    "MODEL_NOT_READY", "ANALYSIS_BUSY", "INFERENCE_FAILED",
]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class APIError(Contract):
    code: ErrorCode
    message: str


class ErrorResponse(Contract):
    error: APIError


class HealthResponse(Contract):
    status: Literal["ok"] = "ok"
    service: Literal["graphnids-backend"] = "graphnids-backend"


class StatusResponse(Contract):
    backend_connected: Literal[True] = True
    model_adapter_state: AdapterState
    model_loaded: bool
    inference_ready: bool = False
    live_capture_supported: Literal[False] = False
    file_replay_supported: Literal[True] = True
    mode: Literal["offline_application_preview"] = "offline_application_preview"
    error: APIError | None = None


class ModelResponse(Contract):
    model_family: str
    task: Literal["E2"]
    class_count: int
    inference_mode: Literal["attention-pruned"] = "attention-pruned"
    model_input_features: int
    retained_feature_nodes: int
    removed_feature_nodes: int
    category_nodes: int
    total_graph_nodes_unpruned: int
    total_graph_nodes_pruned: int
    edges_unpruned: int
    edges_pruned: int
    temporal_window: int
    prediction_target: Literal["fifth/final flow"] = "fifth/final flow"
    model_parameters_unchanged: Literal[True] = True
    deployment_seed: int
    checkpoint_id: str
    checkpoint_sha256: str
    learning_rate: float
    learning_rate_provenance: str
    policy_sha256: str
    selection_rule: str
    pruned_validation_macro_f1_by_seed: dict[str, float]
    selection_uses_test_evidence: Literal[False] = False


class SchemaResponse(Contract):
    model_input_features: int
    feature_names: tuple[str, ...]
    temporal_window: int
    shape: tuple[int, int]
    window_semantics: str
    prediction_target: str
    research_basis: str
    preprocessing_required: Literal[True] = True
    arbitrary_61_column_input_supported: Literal[False] = False
    application_input_available: Literal[True] = True


class CapabilitiesResponse(Contract):
    backend_api: Literal[True] = True
    live_packet_capture: Literal[False] = False
    csv_file_replay_inference: Literal[True] = True
    strict_78_feature_application_csv: Literal[True] = True
    raw_pcap: Literal[False] = False
    arbitrary_cicflowmeter_csv: Literal[False] = False
    frontend_replay_integration: Literal[True] = True
    demo_frontend: Literal["frontend-only"] = "frontend-only"
    model_inference_adapter: AdapterState
    application_prediction_endpoint: Literal[True] = True
