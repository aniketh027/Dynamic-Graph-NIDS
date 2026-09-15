"""Deterministic Stage D2 hierarchical graph schema.

The module defines one reusable, type-local topology template shared by every
future Stage C sample.  It consumes the validated D1 taxonomy and does not load
sample arrays, labels, or any learned quantity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from src.graph.taxonomy import (
    CATEGORY_NAMES,
    EXPECTED_CATEGORY_COUNTS,
    FEATURE_INDEX_TO_CATEGORY_ID,
    MODEL_FEATURE_COLUMNS,
    TAXONOMY_VERSION,
    build_manifest as build_taxonomy_manifest,
    validate_manifest as validate_taxonomy_manifest,
)


SCHEMA_VERSION = "stage-d2-graph-schema-v1"

NODE_KINDS: tuple[str, ...] = ("feature", "category", "sample")
NODE_COUNTS: Mapping[str, int] = MappingProxyType(
    {"feature": 78, "category": 8, "sample": 1}
)
NODE_IDS: Mapping[str, tuple[int, ...]] = MappingProxyType(
    {
        kind: tuple(range(count))
        for kind, count in NODE_COUNTS.items()
    }
)
TOTAL_NODE_COUNT = sum(NODE_COUNTS.values())

FEATURE_TO_CATEGORY_RELATION: tuple[str, str, str] = (
    "feature",
    "contributes_to",
    "category",
)
CATEGORY_TO_SAMPLE_RELATION: tuple[str, str, str] = (
    "category",
    "contributes_to",
    "sample",
)
RELATION_TYPES: tuple[tuple[str, str, str], ...] = (
    FEATURE_TO_CATEGORY_RELATION,
    CATEGORY_TO_SAMPLE_RELATION,
)

FEATURE_TO_CATEGORY_SOURCE_INDICES: tuple[int, ...] = tuple(range(78))
# Preserve object identity with the D1 tuple: D2 does not redefine membership.
FEATURE_TO_CATEGORY_DESTINATION_INDICES = FEATURE_INDEX_TO_CATEGORY_ID
CATEGORY_TO_SAMPLE_SOURCE_INDICES: tuple[int, ...] = tuple(range(8))
CATEGORY_TO_SAMPLE_DESTINATION_INDICES: tuple[int, ...] = (0,) * 8

RELATION_EDGE_COUNTS: Mapping[tuple[str, str, str], int] = MappingProxyType(
    {
        FEATURE_TO_CATEGORY_RELATION: len(FEATURE_TO_CATEGORY_SOURCE_INDICES),
        CATEGORY_TO_SAMPLE_RELATION: len(CATEGORY_TO_SAMPLE_SOURCE_INDICES),
    }
)
TOTAL_EDGE_COUNT = sum(RELATION_EDGE_COUNTS.values())

TEMPORAL_SEMANTICS: Mapping[str, Any] = MappingProxyType(
    {
        "representation": "per-feature ordered value sequence",
        "future_feature_node_input": "[x1, x2, x3, x4, x5]",
        "sequence_length": 5,
        "position_semantics": "five consecutive capture-order FLOW RECORDS",
        "explicit_temporal_edges": False,
        "temporal_encoder_implemented": False,
    }
)
TARGET_SEMANTICS: Mapping[str, Any] = MappingProxyType(
    {
        "task_level": "graph",
        "target_source": "fifth/final Stage C flow label",
        "node_level_labels": False,
        "labels_loaded_by_schema": False,
    }
)


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON-compatible *value* deterministically for hashing."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Return SHA-256 over :func:`canonical_json_bytes`."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def load_validated_d1_manifest(path: Path) -> dict[str, Any]:
    """Load D1 provenance and fail if it differs from the frozen taxonomy."""

    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_taxonomy_manifest(manifest)
    expected = build_taxonomy_manifest(MODEL_FEATURE_COLUMNS)
    checks = {
        "taxonomy_version": TAXONOMY_VERSION,
        "taxonomy_sha256": expected["taxonomy_sha256"],
        "category_names": list(CATEGORY_NAMES),
        "expected_category_counts": dict(EXPECTED_CATEGORY_COUNTS),
        "feature_index_to_category_id": list(FEATURE_INDEX_TO_CATEGORY_ID),
    }
    for field, expected_value in checks.items():
        if manifest.get(field) != expected_value:
            raise ValueError(f"D1 provenance mismatch for {field}")
    if manifest["stage_c_feature_order"] != expected["stage_c_feature_order"]:
        raise ValueError("D1 Stage C feature-order provenance mismatch")
    return manifest


def _validate_static_topology() -> None:
    if NODE_KINDS != tuple(NODE_COUNTS):
        raise AssertionError("node counts are not in canonical node-kind order")
    if tuple(NODE_IDS) != NODE_KINDS:
        raise AssertionError("node IDs are not in canonical node-kind order")
    if any(NODE_IDS[kind] != tuple(range(NODE_COUNTS[kind])) for kind in NODE_KINDS):
        raise AssertionError("type-local node IDs must be contiguous from zero")
    if TOTAL_NODE_COUNT != 87:
        raise AssertionError(f"expected 87 logical nodes, got {TOTAL_NODE_COUNT}")
    if RELATION_TYPES != (
        FEATURE_TO_CATEGORY_RELATION,
        CATEGORY_TO_SAMPLE_RELATION,
    ):
        raise AssertionError("unexpected relation declaration")
    if FEATURE_TO_CATEGORY_SOURCE_INDICES != tuple(range(78)):
        raise AssertionError("feature sources must be exactly 0..77")
    if len(FEATURE_TO_CATEGORY_DESTINATION_INDICES) != 78:
        raise AssertionError("every feature must have exactly one category destination")
    if any(destination not in NODE_IDS["category"] for destination in FEATURE_TO_CATEGORY_DESTINATION_INDICES):
        raise AssertionError("feature edge has an invalid category destination")
    received = Counter(FEATURE_TO_CATEGORY_DESTINATION_INDICES)
    expected_received = {
        category_id: EXPECTED_CATEGORY_COUNTS[category_name]
        for category_id, category_name in enumerate(CATEGORY_NAMES)
    }
    if dict(received) != expected_received:
        raise AssertionError("category indegrees differ from the D1 taxonomy counts")
    if CATEGORY_TO_SAMPLE_SOURCE_INDICES != tuple(range(8)):
        raise AssertionError("category sources must be exactly 0..7")
    if CATEGORY_TO_SAMPLE_DESTINATION_INDICES != (0,) * 8:
        raise AssertionError("every category must point to type-local sample node 0")
    if TOTAL_EDGE_COUNT != 86:
        raise AssertionError(f"expected 86 structural edges, got {TOTAL_EDGE_COUNT}")


_validate_static_topology()


def build_manifest(d1_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Build the D2 schema manifest from validated D1 provenance."""

    validate_taxonomy_manifest(d1_manifest)
    expected_d1 = build_taxonomy_manifest(MODEL_FEATURE_COLUMNS)
    if dict(d1_manifest) != expected_d1:
        raise ValueError("D1 manifest does not match the frozen taxonomy")
    _validate_static_topology()

    content: dict[str, Any] = {
        "schema": "stage_d2_hierarchical_graph_schema",
        "schema_version": SCHEMA_VERSION,
        "hash_algorithm": "sha256",
        "canonical_serialization": "utf8-json-sort-keys-compact-v1",
        "d1_provenance": {
            "taxonomy_version": d1_manifest["taxonomy_version"],
            "taxonomy_sha256": d1_manifest["taxonomy_sha256"],
            "stage_c_feature_order_sha256": d1_manifest["stage_c_feature_order"]["sha256"],
        },
        "indexing": {
            "mode": "type-local",
            "feature_ids": list(NODE_IDS["feature"]),
            "category_ids": list(NODE_IDS["category"]),
            "sample_ids": list(NODE_IDS["sample"]),
        },
        "node_kinds": list(NODE_KINDS),
        "node_counts": dict(NODE_COUNTS),
        "total_node_count": TOTAL_NODE_COUNT,
        "node_semantics": {
            "feature": {
                "identity_source": "D1 taxonomy and frozen Stage C feature order",
                "future_input": "[x1, x2, x3, x4, x5]",
            },
            "category": {
                "identity_source": "D1 canonical category order and membership",
            },
            "sample": {
                "role": "virtual graph-level readout",
                "represents": "one complete Stage C T=5 sample/window",
                "network_entity": False,
            },
        },
        "category_order": list(CATEGORY_NAMES),
        "relation_types": [list(relation) for relation in RELATION_TYPES],
        "relations": [
            {
                "type": list(FEATURE_TO_CATEGORY_RELATION),
                "source_indices": list(FEATURE_TO_CATEGORY_SOURCE_INDICES),
                "destination_indices": list(FEATURE_TO_CATEGORY_DESTINATION_INDICES),
                "edge_count": RELATION_EDGE_COUNTS[FEATURE_TO_CATEGORY_RELATION],
            },
            {
                "type": list(CATEGORY_TO_SAMPLE_RELATION),
                "source_indices": list(CATEGORY_TO_SAMPLE_SOURCE_INDICES),
                "destination_indices": list(CATEGORY_TO_SAMPLE_DESTINATION_INDICES),
                "edge_count": RELATION_EDGE_COUNTS[CATEGORY_TO_SAMPLE_RELATION],
            },
        ],
        "relation_edge_counts": [
            {
                "type": list(relation),
                "edge_count": RELATION_EDGE_COUNTS[relation],
            }
            for relation in RELATION_TYPES
        ],
        "total_edge_count": TOTAL_EDGE_COUNT,
        "topology_scope": "one reusable static template shared by every sample and experiment",
        "temporal_semantics": dict(TEMPORAL_SEMANTICS),
        "target_semantics": dict(TARGET_SEMANTICS),
    }
    content["schema_sha256"] = canonical_sha256(content)
    return content


def manifest_bytes(d1_manifest: Mapping[str, Any]) -> bytes:
    """Return stable human-readable bytes for the D2 manifest."""

    return (json.dumps(build_manifest(d1_manifest), ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def write_manifest(d1_manifest_path: Path, output_path: Path) -> dict[str, Any]:
    """Validate D1 and write only the reusable D2 schema manifest."""

    d1_manifest = load_validated_d1_manifest(d1_manifest_path)
    data = manifest_bytes(d1_manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(data)
    return json.loads(data)


def validate_manifest(
    manifest: Mapping[str, Any], d1_manifest: Mapping[str, Any]
) -> None:
    """Validate the D2 hash, structure, and exact D1-derived regeneration."""

    supplied_hash = manifest.get("schema_sha256")
    if not isinstance(supplied_hash, str):
        raise ValueError("manifest schema_sha256 is missing")
    unhashed = dict(manifest)
    del unhashed["schema_sha256"]
    if canonical_sha256(unhashed) != supplied_hash:
        raise ValueError("manifest schema_sha256 does not match its canonical content")
    if dict(manifest) != build_manifest(d1_manifest):
        raise ValueError("manifest does not match the frozen Stage D2 schema")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the deterministic Stage D2 graph schema")
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=Path("results/stage_d1/feature_taxonomy.json"),
        help="approved D1 taxonomy manifest",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/stage_d2/graph_schema.json"),
        help="Stage D2 schema manifest destination",
    )
    args = parser.parse_args(argv)
    manifest = write_manifest(args.taxonomy, args.output)
    print(f"wrote {args.output}")
    print(f"schema_sha256={manifest['schema_sha256']}")
    print(f"d1_taxonomy_sha256={manifest['d1_provenance']['taxonomy_sha256']}")
    print(
        "stage_c_feature_order_sha256="
        f"{manifest['d1_provenance']['stage_c_feature_order_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
