"""Direct bounded PyG conversion, using the frozen Stage D edge templates."""
import numpy as np
import torch
from torch_geometric.data import HeteroData

from src.graph import schema

from backend.app.ml.external_windows import (
    ApplicationContractError, ApplicationIdentity, ExternalWindows,
    payload_sha256, validate_provenance, validate_windows,
)


def expected_edges(relation: tuple[str, str, str], b: int) -> torch.Tensor:
    if relation == schema.FEATURE_TO_CATEGORY_RELATION:
        sources = schema.FEATURE_TO_CATEGORY_SOURCE_INDICES
        destinations = schema.FEATURE_TO_CATEGORY_DESTINATION_INDICES
    elif relation == schema.CATEGORY_TO_SAMPLE_RELATION:
        sources = schema.CATEGORY_TO_SAMPLE_SOURCE_INDICES
        destinations = schema.CATEGORY_TO_SAMPLE_DESTINATION_INDICES
    else:
        raise ApplicationContractError("unknown canonical relation")
    offsets = torch.arange(b, dtype=torch.long).repeat_interleave(len(sources))
    return torch.stack((
        torch.tensor(sources, dtype=torch.long).repeat(b) + offsets * schema.NODE_COUNTS[relation[0]],
        torch.tensor(destinations, dtype=torch.long).repeat(b) + offsets * schema.NODE_COUNTS[relation[2]],
    ))


def application_graph(windows: ExternalWindows, identity: ApplicationIdentity, *, max_windows: int) -> HeteroData:
    validate_windows(windows, identity, max_windows=max_windows)
    b = windows.provenance.batch_size
    data = HeteroData()
    for kind in schema.NODE_KINDS:
        count = schema.NODE_COUNTS[kind]
        store = data[kind]
        if kind == "feature":
            # Explicit tensor copy owns storage independent of caller and snapshot.
            ordered = np.array(windows.values.transpose(0, 2, 1), order="C", copy=True)
            store.x = torch.tensor(ordered, dtype=torch.float32).reshape(b * count, identity.temporal_window_size)
            store.feature_id = torch.tensor(schema.NODE_IDS[kind], dtype=torch.long).repeat(b)
        elif kind == "category":
            store.category_id = torch.tensor(schema.NODE_IDS[kind], dtype=torch.long).repeat(b)
        store.batch = torch.arange(b, dtype=torch.long).repeat_interleave(count)
        store.ptr = torch.arange(b + 1, dtype=torch.long) * count
        store.num_nodes = b * count
    for relation in schema.RELATION_TYPES:
        data[relation].edge_index = expected_edges(relation, b)
    data.application_audit = windows.provenance
    validate_application_graph(data, identity, max_windows=max_windows)
    return data


def _tensor(value, expected: torch.Tensor, name: str) -> None:
    if (type(value) is not torch.Tensor or value.device != expected.device or value.dtype != expected.dtype
            or value.layout != torch.strided or value.requires_grad or not value.is_contiguous()
            or value.shape != expected.shape or not torch.equal(value, expected)):
        raise ApplicationContractError(f"{name} violates exact canonical dtype/device/shape/order/ownership/topology")


def validate_application_graph(data: HeteroData, identity: ApplicationIdentity, *, max_windows: int) -> int:
    if type(data) is not HeteroData:
        raise ApplicationContractError("application graphs must be HeteroData")
    if tuple(data.node_types) != schema.NODE_KINDS or tuple(data.edge_types) != schema.RELATION_TYPES:
        raise ApplicationContractError("application graph node/relation types must be exactly canonical")
    if set(data._global_store.keys()) != {"application_audit"}:
        raise ApplicationContractError("application graphs require only application_audit; research namespaces are forbidden")
    audit = data.application_audit
    validate_provenance(audit, identity, max_windows=max_windows)
    b = audit.batch_size
    for kind in schema.NODE_KINDS:
        count = schema.NODE_COUNTS[kind]
        store = data[kind]
        keys = {"num_nodes", "batch", "ptr"}
        if kind == "feature":
            keys.update(("x", "feature_id"))
        elif kind == "category":
            keys.add("category_id")
        if set(store.keys()) != keys or type(store.num_nodes) is not int or store.num_nodes != b * count:
            raise ApplicationContractError(f"{kind} store keys/node count differ; labels and extra nodes are forbidden")
        _tensor(store.batch, torch.arange(b, dtype=torch.long).repeat_interleave(count), f"{kind}.batch")
        _tensor(store.ptr, torch.arange(b + 1, dtype=torch.long) * count, f"{kind}.ptr")
        if kind in {"feature", "category"}:
            _tensor(store[f"{kind}_id"], torch.tensor(schema.NODE_IDS[kind], dtype=torch.long).repeat(b), f"{kind}_id")
    x = data["feature"].x
    if (type(x) is not torch.Tensor or x.device != torch.device("cpu") or x.dtype != torch.float32
            or x.layout != torch.strided or x.requires_grad or not x.is_contiguous()
            or x.shape != (b * schema.NODE_COUNTS["feature"], identity.temporal_window_size)
            or not torch.isfinite(x).all()):
        raise ApplicationContractError("feature.x must be finite owned CPU float32 [B*78,5] without gradients")
    payload = x.detach().numpy().reshape(b, schema.NODE_COUNTS["feature"], identity.temporal_window_size).transpose(0, 2, 1).tobytes(order="C")
    if payload_sha256(payload, b) != audit.transformed_payload_sha256:
        raise ApplicationContractError("graph values/order no longer match the transformed payload identity")
    for relation in schema.RELATION_TYPES:
        if set(data[relation].keys()) != {"edge_index"}:
            raise ApplicationContractError("only canonical edge_index is supported")
        _tensor(data[relation].edge_index, expected_edges(relation, b), str(relation))
    if not data.validate(raise_on_error=False):
        raise ApplicationContractError("invalid type-local graph endpoints")
    return b
