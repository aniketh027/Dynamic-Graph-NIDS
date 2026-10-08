"""Owned, post-transform windows and immutable application provenance.

Transform identity is a trusted caller configuration, not proof that a raw
file was cleaned or transformed. Raw ingestion must establish that in C2A.
"""
from dataclasses import dataclass, fields
import hashlib
import re

import numpy as np

from src.graph import schema
from src.preprocessing.schema import MODEL_FEATURE_COLUMNS

ADAPTER_VERSION = "external-application-graph-adapter-v1"
SOURCE_KIND = "external_application"
WINDOW_SEMANTICS = "five consecutive clean flow records in deterministic capture order; not packets"
TARGET_SEMANTICS = "fifth/final flow"


class ApplicationContractError(ValueError):
    """Invalid external values, provenance, topology or deployment binding."""


def require_text(value: str, name: str) -> None:
    if type(value) is not str or not value.strip() or len(value) > 256:
        raise ApplicationContractError(f"{name} must be a nonempty string of at most 256 characters")


def require_digest(value: str, name: str) -> None:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ApplicationContractError(f"{name} must be a lowercase SHA-256")


def require_bound(batch_size: int, max_windows: int) -> None:
    if type(max_windows) is not int or max_windows <= 0:
        raise ApplicationContractError("max_windows must be an explicit positive integer")
    if type(batch_size) is not int or not 0 < batch_size <= max_windows:
        raise ApplicationContractError("B must be positive and within max_windows")


@dataclass(frozen=True, slots=True)
class TransformIdentity:
    state_id: str
    sha256: str

    def __post_init__(self) -> None:
        require_text(self.state_id, "transform state ID")
        require_digest(self.sha256, "transform state hash")


@dataclass(frozen=True, slots=True)
class ApplicationIdentity:
    application_adapter_version: str
    feature_schema_sha256: str
    feature_order_sha256: str
    transform_state: TransformIdentity
    temporal_window_size: int
    window_semantics: str
    target_semantics: str
    graph_taxonomy_version: str
    graph_taxonomy_sha256: str
    graph_schema_version: str
    graph_schema_sha256: str
    pruning_policy_sha256: str
    pruning_policy_record_sha256: str
    checkpoint_id: str
    checkpoint_sha256: str
    deployment_seed: int
    model_task: str
    class_mapping_sha256: str


@dataclass(frozen=True, slots=True)
class ApplicationProvenance:
    identity: ApplicationIdentity
    source_id: str
    analysis_id: str
    replay_order_semantics: str
    transformed_payload_sha256: str
    batch_size: int
    source_kind: str = SOURCE_KIND

    def __post_init__(self) -> None:
        if (type(self.identity) is not ApplicationIdentity or type(self.source_kind) is not str
                or self.source_kind != SOURCE_KIND):
            raise ApplicationContractError("application provenance requires a separate application identity")
        for name in ("source_id", "analysis_id", "replay_order_semantics"):
            require_text(getattr(self, name), name)
        require_digest(self.transformed_payload_sha256, "transformed payload hash")
        if type(self.batch_size) is not int or self.batch_size <= 0:
            raise ApplicationContractError("provenance B must be positive")


def payload_sha256(payload: bytes, batch_size: int) -> str:
    digest = hashlib.sha256(schema.canonical_json_bytes({
        "shape": [batch_size, schema.TEMPORAL_SEMANTICS["sequence_length"], len(MODEL_FEATURE_COLUMNS)],
        "dtype": "float32", "order": "C", "axes": "graph,flow,feature",
    }))
    digest.update(payload)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ExternalWindows:
    """Immutable bytes own the bounded C-order snapshot; views are read-only."""
    payload: bytes
    provenance: ApplicationProvenance

    @property
    def values(self) -> np.ndarray:
        b = self.provenance.batch_size
        return np.frombuffer(self.payload, dtype=np.float32).reshape(
            b, schema.TEMPORAL_SEMANTICS["sequence_length"], len(MODEL_FEATURE_COLUMNS))


def validate_provenance(provenance: ApplicationProvenance, expected: ApplicationIdentity,
                        *, max_windows: int) -> None:
    if type(provenance) is not ApplicationProvenance:
        raise ApplicationContractError("missing typed application provenance; research audits are unsupported")
    provenance.__post_init__()
    if (provenance.identity != expected
            or any(type(getattr(provenance.identity, field.name)) is not type(getattr(expected, field.name))
                   for field in fields(ApplicationIdentity))):
        raise ApplicationContractError("application schema/transform/policy/checkpoint identity mismatch")
    provenance.identity.transform_state.__post_init__()
    require_bound(provenance.batch_size, max_windows)


def snapshot_windows(values: np.ndarray, identity: ApplicationIdentity, *, source_id: str,
                     analysis_id: str, replay_order_semantics: str, max_windows: int) -> ExternalWindows:
    if type(values) is not np.ndarray or values.ndim != 3:
        raise ApplicationContractError("windows must be a NumPy array of shape (B,5,78)")
    require_bound(values.shape[0], max_windows)
    if values.shape[1:] != (identity.temporal_window_size, len(MODEL_FEATURE_COLUMNS)):
        raise ApplicationContractError("windows must have the canonical temporal and feature dimensions")
    if values.dtype != np.dtype(np.float32) or not np.isfinite(values).all():
        raise ApplicationContractError("post-transform windows must be float32 and entirely finite")
    # tobytes owns a bounded copy even for contiguous/read-only caller arrays.
    payload = values.tobytes(order="C")
    provenance = ApplicationProvenance(identity, source_id, analysis_id, replay_order_semantics,
                                       payload_sha256(payload, values.shape[0]), values.shape[0])
    result = ExternalWindows(payload, provenance)
    validate_windows(result, identity, max_windows=max_windows)
    return result


def validate_windows(windows: ExternalWindows, identity: ApplicationIdentity, *, max_windows: int) -> None:
    if type(windows) is not ExternalWindows or type(windows.payload) is not bytes:
        raise ApplicationContractError("windows must own immutable payload bytes")
    validate_provenance(windows.provenance, identity, max_windows=max_windows)
    expected_bytes = windows.provenance.batch_size * identity.temporal_window_size * len(MODEL_FEATURE_COLUMNS) * 4
    if len(windows.payload) != expected_bytes or payload_sha256(windows.payload, windows.provenance.batch_size) != windows.provenance.transformed_payload_sha256:
        raise ApplicationContractError("window payload does not match its application provenance")
    if not np.isfinite(windows.values).all():
        raise ApplicationContractError("post-transform values must be finite")
