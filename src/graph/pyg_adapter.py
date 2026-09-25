"""Stage E.2 bounded conversion from one D6 batch to one PyG HeteroData.

The adapter copies only the supplied logical batch into CPU-owned tensors. It
does not read Stage C, construct per-sample PyG graphs, define model parameters,
or alter the frozen Stage D topology.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading
from typing import Any, Mapping

import numpy as np
import torch
import torch_geometric
from torch_geometric.data import HeteroData

from src.graph import adapter as d4
from src.graph import batching as d6
from src.graph import full_split_integration as d8
from src.graph import schema as d2


ADAPTER_VERSION = "stage-e2-d6-to-heterodata-v1"
FROZEN_STAGE_D_COMMIT = "469fe31a4ae1361ab1d390f6f7b91ac651ef4aaf"
FROZEN_STAGE_D_TAG = "v0.4-stage-d-graph-input"
APPROVED_D8_REPORT_SHA256 = (
    "d1a757459062fe8ca08f0be5951a5a728898a20d28eafbb12f10ce4646047b1f"
)
APPROVED_E1_REPORT_SHA256 = (
    "10af77a05a28d760a9cbf52916e8b3205f5590646af5608882538966e2e36f6a"
)
FEATURE_RELATION = ("feature", "contributes_to", "category")
CATEGORY_RELATION = ("category", "contributes_to", "sample")
NODE_TYPES = ("feature", "category", "sample")
EDGE_TYPES = (FEATURE_RELATION, CATEGORY_RELATION)
_FROZEN_IDENTITY_ARTIFACTS = (
    ("d8_report", Path("results/stage_d8/full_split_integration.json")),
    ("e1_environment_report", Path("results/stage_e1/environment-report.json")),
)
_FROZEN_IDENTITY_CACHE_LOCK = threading.RLock()
_FROZEN_IDENTITY_CACHE: tuple[tuple[object, ...], tuple[tuple[str, str], ...]] | None = None


class HeteroDataAdapterError(ValueError):
    """Raised when a D6 batch or frozen provenance cannot be converted safely."""


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_identity(path: Path) -> tuple[object, ...]:
    """Return cheap identity fields that change on normal writes/replacements."""

    try:
        resolved = path.resolve(strict=True)
        stat = resolved.stat()
    except OSError as exc:
        raise HeteroDataAdapterError(
            f"cannot inspect frozen provenance artifact {path}: {exc}"
        ) from exc
    if not resolved.is_file():
        raise HeteroDataAdapterError(
            f"frozen provenance artifact is not a regular file: {resolved}"
        )
    return (
        str(resolved),
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_mode),
        int(stat.st_nlink),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


def _frozen_identity_cache_key(project_root: Path) -> tuple[object, ...]:
    """Bind cached validation to its root, authorities, and artifact identities."""

    try:
        resolved_root = Path(project_root).resolve(strict=True)
    except OSError as exc:
        raise HeteroDataAdapterError(f"cannot resolve project root: {exc}") from exc
    artifact_identities = tuple(
        (name, _artifact_identity(resolved_root / relative_path))
        for name, relative_path in _FROZEN_IDENTITY_ARTIFACTS
    )
    return (
        str(resolved_root),
        tuple(sorted(d8.APPROVED_PROVENANCE.items())),
        APPROVED_D8_REPORT_SHA256,
        APPROVED_E1_REPORT_SHA256,
        artifact_identities,
    )


def _load_frozen_identities_uncached(project_root: Path) -> dict[str, str]:
    """Fully read and validate the compact frozen provenance artifacts."""

    expected_batch_provenance = {
        key: d8.APPROVED_PROVENANCE[key]
        for key in (
            "d1_taxonomy_sha256",
            "d2_schema_sha256",
            "d3_report_sha256",
            "d4_report_sha256",
            "d5_report_sha256",
        )
    }
    d8.validate_stage_provenance(d8.APPROVED_PROVENANCE)

    d8_path = project_root / "results" / "stage_d8" / "full_split_integration.json"
    try:
        d8_report = json.loads(d8_path.read_text(encoding="utf-8"))
        d8.validate_report(d8_report)
    except Exception as exc:
        raise HeteroDataAdapterError(f"frozen D8 report is invalid: {exc}") from exc
    if d8_report.get("report_sha256") != APPROVED_D8_REPORT_SHA256:
        raise HeteroDataAdapterError("frozen D8 report hash is not approved")

    e1_path = project_root / "results" / "stage_e1" / "environment-report.json"
    try:
        e1_hash = _sha256(e1_path)
    except OSError as exc:
        raise HeteroDataAdapterError(f"cannot read E.1 environment report: {exc}") from exc
    if e1_hash != APPROVED_E1_REPORT_SHA256:
        raise HeteroDataAdapterError("E.1 environment report hash is not approved")

    return {
        **expected_batch_provenance,
        "d6_report_sha256": d8.APPROVED_PROVENANCE["d6_report_sha256"],
        "d7_report_sha256": d8.APPROVED_PROVENANCE["d7_report_sha256"],
        "d8_report_sha256": APPROVED_D8_REPORT_SHA256,
        "e1_environment_report_sha256": e1_hash,
    }


def _load_frozen_identities(project_root: Path) -> dict[str, str]:
    """Return validated identities, revalidating after any detectable change.

    The single-entry cache stores only immutable identity strings. Every caller
    receives a fresh dictionary, so normal caller mutation cannot alter cached
    authority. Holding the lock across a cache miss makes cold validation
    atomic for concurrent callers and prevents duplicate or partial entries.
    """

    global _FROZEN_IDENTITY_CACHE
    with _FROZEN_IDENTITY_CACHE_LOCK:
        try:
            resolved_root = Path(project_root).resolve(strict=True)
        except OSError as exc:
            raise HeteroDataAdapterError(f"cannot resolve project root: {exc}") from exc
        key_before = _frozen_identity_cache_key(resolved_root)
        cached = _FROZEN_IDENTITY_CACHE
        if cached is not None and cached[0] == key_before:
            return dict(cached[1])

        identities = _load_frozen_identities_uncached(resolved_root)
        key_after = _frozen_identity_cache_key(resolved_root)
        if key_after != key_before:
            raise HeteroDataAdapterError(
                "frozen provenance artifacts changed during validation"
            )
        immutable_identities = tuple(identities.items())
        _FROZEN_IDENTITY_CACHE = (key_after, immutable_identities)
        return dict(immutable_identities)


def _reset_frozen_identity_cache() -> None:
    """Clear process-local validation state for tests and cold measurements."""

    global _FROZEN_IDENTITY_CACHE
    with _FROZEN_IDENTITY_CACHE_LOCK:
        _FROZEN_IDENTITY_CACHE = None


def _require_array(
    value: object, name: str, shape: tuple[int, ...], dtype: np.dtype[Any]
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise HeteroDataAdapterError(f"{name} must be a NumPy array")
    if value.shape != shape:
        raise HeteroDataAdapterError(
            f"{name} has shape {value.shape}; expected {shape}"
        )
    if value.dtype != dtype:
        raise HeteroDataAdapterError(
            f"{name} has dtype {value.dtype}; expected {dtype}"
        )
    if value.flags.writeable:
        raise HeteroDataAdapterError(f"{name} must be a read-only D6 array")
    return value


def _owned_tensor(array: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    """Return bounded Torch-owned storage without aliasing the D6 NumPy array."""

    return torch.from_numpy(np.array(array, copy=True)).to(dtype=dtype).clone()


def _validate_logical_batch(
    batch: d6.LogicalGraphBatch, project_root: Path
) -> dict[str, str]:
    if not isinstance(batch, d6.LogicalGraphBatch):
        raise TypeError("batch must be a D6 LogicalGraphBatch")
    identities = _load_frozen_identities(project_root)
    expected_batch_provenance = {
        key: identities[key]
        for key in (
            "d1_taxonomy_sha256",
            "d2_schema_sha256",
            "d3_report_sha256",
            "d4_report_sha256",
            "d5_report_sha256",
        )
    }
    if dict(batch.provenance) != expected_batch_provenance:
        raise HeteroDataAdapterError("corrupted or incompatible D6 provenance")
    if batch.static is not d4.STATIC_GRAPH_METADATA:
        raise HeteroDataAdapterError("batch does not use the approved Stage D topology")
    if batch.experiment not in d4.EXPERIMENT_DIRS:
        raise HeteroDataAdapterError("unsupported experiment")
    if batch.split not in d4.SPLIT_STEMS:
        raise HeteroDataAdapterError("unsupported split")

    size = batch.batch_size
    if size <= 0:
        raise HeteroDataAdapterError("empty D6 batches are not supported")
    _require_array(batch.feature_x, "feature_x", (size, 78, 5), np.dtype(np.float32))
    _require_array(batch.targets, "targets", (size,), np.dtype(np.int16))
    _require_array(batch.sample_indices, "sample_indices", (size,), np.dtype(np.int64))
    _require_array(batch.feature_batch, "feature_batch", (size * 78,), np.dtype(np.int64))
    _require_array(batch.category_batch, "category_batch", (size * 8,), np.dtype(np.int64))
    _require_array(batch.sample_batch, "sample_batch", (size,), np.dtype(np.int64))
    _require_array(
        batch.feature_to_category_sources,
        "feature_to_category_sources",
        (size * 78,),
        np.dtype(np.int64),
    )
    _require_array(
        batch.feature_to_category_destinations,
        "feature_to_category_destinations",
        (size * 78,),
        np.dtype(np.int64),
    )
    _require_array(
        batch.category_to_sample_sources,
        "category_to_sample_sources",
        (size * 8,),
        np.dtype(np.int64),
    )
    _require_array(
        batch.category_to_sample_destinations,
        "category_to_sample_destinations",
        (size * 8,),
        np.dtype(np.int64),
    )

    graph_ids = np.arange(size, dtype=np.int64)
    if not np.array_equal(batch.feature_batch, np.repeat(graph_ids, 78)):
        raise HeteroDataAdapterError("malformed feature ownership")
    if not np.array_equal(batch.category_batch, np.repeat(graph_ids, 8)):
        raise HeteroDataAdapterError("malformed category ownership")
    if not np.array_equal(batch.sample_batch, graph_ids):
        raise HeteroDataAdapterError("malformed sample ownership")
    if np.any(batch.sample_indices < 0) or np.any(
        batch.sample_indices >= batch.source_dataset_length
    ):
        raise HeteroDataAdapterError("sample index is outside the source dataset")
    if len(np.unique(batch.sample_indices)) != size:
        raise HeteroDataAdapterError("duplicate source sample index")
    if not isinstance(batch.source_dataset_length, int) or batch.source_dataset_length <= 0:
        raise HeteroDataAdapterError("invalid source dataset length")

    ftc_sources = batch.feature_to_category_sources
    ftc_destinations = batch.feature_to_category_destinations
    cts_sources = batch.category_to_sample_sources
    cts_destinations = batch.category_to_sample_destinations
    if (
        np.any(ftc_sources < 0)
        or np.any(ftc_sources >= size * 78)
        or np.any(ftc_destinations < 0)
        or np.any(ftc_destinations >= size * 8)
        or np.any(cts_sources < 0)
        or np.any(cts_sources >= size * 8)
        or np.any(cts_destinations < 0)
        or np.any(cts_destinations >= size)
    ):
        raise HeteroDataAdapterError("edge endpoint is outside its type-local range")
    if not np.array_equal(ftc_sources // 78, ftc_destinations // 8):
        raise HeteroDataAdapterError("cross-sample feature-to-category edge")
    if not np.array_equal(cts_sources // 8, cts_destinations):
        raise HeteroDataAdapterError("cross-sample category-to-sample edge")
    if not np.array_equal(batch.feature_batch, ftc_sources // 78):
        raise HeteroDataAdapterError("feature edges disagree with ownership")
    if not np.array_equal(batch.category_batch, cts_sources // 8):
        raise HeteroDataAdapterError("category edges disagree with ownership")
    if batch.total_edge_count != size * 86:
        raise HeteroDataAdapterError("total edge count is not B*86")

    feature_offsets = np.repeat(graph_ids * 78, 78)
    category_offsets_for_features = np.repeat(graph_ids * 8, 78)
    category_offsets = np.repeat(graph_ids * 8, 8)
    expected_ftc_sources = np.tile(
        np.asarray(batch.static.feature_to_category_sources, dtype=np.int64), size
    ) + feature_offsets
    expected_ftc_destinations = np.tile(
        np.asarray(batch.static.feature_to_category_destinations, dtype=np.int64), size
    ) + category_offsets_for_features
    expected_cts_sources = np.tile(
        np.asarray(batch.static.category_to_sample_sources, dtype=np.int64), size
    ) + category_offsets
    expected_cts_destinations = np.repeat(graph_ids, 8)
    if not all(
        (
            np.array_equal(ftc_sources, expected_ftc_sources),
            np.array_equal(ftc_destinations, expected_ftc_destinations),
            np.array_equal(cts_sources, expected_cts_sources),
            np.array_equal(cts_destinations, expected_cts_destinations),
        )
    ):
        raise HeteroDataAdapterError("batched topology differs from approved Stage D edges")

    if batch.experiment == "E3":
        values = set(int(value) for value in batch.targets.tolist())
        if not values.issubset({0, 1}):
            raise HeteroDataAdapterError("E3 targets must be binary 0/1 values")
    elif np.any(batch.targets < 0) or np.any(batch.targets >= 15):
        raise HeteroDataAdapterError("E1/E2 targets must use the frozen 15-class vocabulary")
    return identities


def logical_batch_to_heterodata(
    batch: d6.LogicalGraphBatch, *, project_root: Path | None = None
) -> HeteroData:
    """Copy one validated D6 logical batch into one CPU PyG HeteroData."""

    root = _project_root() if project_root is None else Path(project_root)
    identities = _validate_logical_batch(batch, root)
    size = batch.batch_size
    data = HeteroData()

    data["feature"].x = _owned_tensor(batch.feature_x, torch.float32).reshape(
        size * 78, 5
    )
    data["feature"].feature_id = torch.arange(78, dtype=torch.long).repeat(size)
    data["feature"].batch = _owned_tensor(batch.feature_batch, torch.long)
    data["feature"].ptr = torch.arange(0, (size + 1) * 78, 78, dtype=torch.long)
    data["feature"].num_nodes = size * 78

    data["category"].category_id = torch.arange(8, dtype=torch.long).repeat(size)
    data["category"].batch = _owned_tensor(batch.category_batch, torch.long)
    data["category"].ptr = torch.arange(0, (size + 1) * 8, 8, dtype=torch.long)
    data["category"].num_nodes = size * 8

    data["sample"].batch = _owned_tensor(batch.sample_batch, torch.long)
    data["sample"].ptr = torch.arange(size + 1, dtype=torch.long)
    data["sample"].num_nodes = size
    target_dtype = torch.float32 if batch.experiment == "E3" else torch.long
    data["sample"].y = _owned_tensor(batch.targets, target_dtype)

    data[FEATURE_RELATION].edge_index = torch.stack(
        (
            _owned_tensor(batch.feature_to_category_sources, torch.long),
            _owned_tensor(batch.feature_to_category_destinations, torch.long),
        )
    )
    data[CATEGORY_RELATION].edge_index = torch.stack(
        (
            _owned_tensor(batch.category_to_sample_sources, torch.long),
            _owned_tensor(batch.category_to_sample_destinations, torch.long),
        )
    )

    data.e2_audit = {
        "adapter_version": ADAPTER_VERSION,
        "experiment": batch.experiment,
        "split": batch.split,
        "ordered_source_sample_indices": [int(x) for x in batch.sample_indices],
        "source_dataset_length": int(batch.source_dataset_length),
        "stage_d_commit": FROZEN_STAGE_D_COMMIT,
        "stage_d_tag": FROZEN_STAGE_D_TAG,
        "graph_schema_identity": "78-feature__8-category__1-sample__86-directed-edges",
        "provenance": identities,
    }

    if tuple(data.node_types) != NODE_TYPES:
        raise HeteroDataAdapterError("unexpected HeteroData node stores")
    if tuple(data.edge_types) != EDGE_TYPES:
        raise HeteroDataAdapterError("unexpected HeteroData relation stores")
    if not data.validate(raise_on_error=False):
        raise HeteroDataAdapterError("PyG HeteroData validation failed")
    return data


def report_bytes(report: Mapping[str, Any]) -> bytes:
    return d2.canonical_json_bytes(report)
