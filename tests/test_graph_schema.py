"""Stage D2 tests for the static logical schema only."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from src.graph import taxonomy as d1
from src.graph.schema import (
    CATEGORY_TO_SAMPLE_DESTINATION_INDICES,
    CATEGORY_TO_SAMPLE_RELATION,
    CATEGORY_TO_SAMPLE_SOURCE_INDICES,
    FEATURE_TO_CATEGORY_DESTINATION_INDICES,
    FEATURE_TO_CATEGORY_RELATION,
    FEATURE_TO_CATEGORY_SOURCE_INDICES,
    NODE_COUNTS,
    NODE_IDS,
    NODE_KINDS,
    RELATION_EDGE_COUNTS,
    RELATION_TYPES,
    SCHEMA_VERSION,
    TARGET_SEMANTICS,
    TEMPORAL_SEMANTICS,
    TOTAL_EDGE_COUNT,
    TOTAL_NODE_COUNT,
    build_manifest,
    canonical_sha256,
    load_validated_d1_manifest,
    manifest_bytes,
    validate_manifest,
    write_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
D1_MANIFEST_PATH = ROOT / "results" / "stage_d1" / "feature_taxonomy.json"
D2_MANIFEST_PATH = ROOT / "results" / "stage_d2" / "graph_schema.json"
EXPECTED_D1_SHA256 = "fe1332dc22727cf952dc1d2117fec46439294821e2a4654537d910df901fab7e"
EXPECTED_FEATURE_ORDER_SHA256 = "7ae8c123332c05b1136e1093c53d710b69b5c397ad15309e207667b104f5fe4c"
EXPECTED_SCHEMA_SHA256 = "0950126394099acea7573e14357f8bd01b22eb91c2a9d0b4f9298ea208ecebea"


@pytest.fixture(scope="module")
def d1_manifest() -> dict[str, object]:
    return load_validated_d1_manifest(D1_MANIFEST_PATH)


@pytest.fixture(scope="module")
def manifest(d1_manifest: dict[str, object]) -> dict[str, object]:
    return build_manifest(d1_manifest)


def _fingerprint_tree(root: Path) -> dict[str, tuple[int, int, str]]:
    result = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        stat = path.stat()
        result[path.relative_to(root).as_posix()] = (
            stat.st_size,
            stat.st_mtime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    return result


def test_node_kinds_and_counts_are_exact() -> None:
    assert NODE_KINDS == ("feature", "category", "sample")
    assert dict(NODE_COUNTS) == {"feature": 78, "category": 8, "sample": 1}
    assert TOTAL_NODE_COUNT == 87


def test_type_local_ids_are_valid_and_contiguous() -> None:
    assert NODE_IDS["feature"] == tuple(range(78))
    assert NODE_IDS["category"] == tuple(range(8))
    assert NODE_IDS["sample"] == (0,)


def test_relation_types_are_exact_and_only_forward() -> None:
    assert RELATION_TYPES == (
        ("feature", "contributes_to", "category"),
        ("category", "contributes_to", "sample"),
    )
    reverse = {(destination, name, source) for source, name, destination in RELATION_TYPES}
    assert not reverse.intersection(RELATION_TYPES)


def test_excluded_relation_families_are_absent() -> None:
    assert all(source != destination for source, _, destination in RELATION_TYPES)
    assert not any(source == destination == "feature" for source, _, destination in RELATION_TYPES)
    assert not any(source == destination == "category" for source, _, destination in RELATION_TYPES)
    assert not any("temporal" in name.lower() for _, name, _ in RELATION_TYPES)
    assert not any("correlation" in name.lower() for _, name, _ in RELATION_TYPES)


def test_edge_counts_are_exact() -> None:
    assert RELATION_EDGE_COUNTS[FEATURE_TO_CATEGORY_RELATION] == 78
    assert RELATION_EDGE_COUNTS[CATEGORY_TO_SAMPLE_RELATION] == 8
    assert TOTAL_EDGE_COUNT == 86


def test_every_feature_occurs_once_and_has_one_d1_destination() -> None:
    assert FEATURE_TO_CATEGORY_SOURCE_INDICES == tuple(range(78))
    assert len(set(FEATURE_TO_CATEGORY_SOURCE_INDICES)) == 78
    assert len(FEATURE_TO_CATEGORY_DESTINATION_INDICES) == 78
    assert tuple(zip(FEATURE_TO_CATEGORY_SOURCE_INDICES, FEATURE_TO_CATEGORY_DESTINATION_INDICES)) == tuple(
        enumerate(d1.FEATURE_INDEX_TO_CATEGORY_ID)
    )


def test_d2_reuses_d1_membership_tuple_by_identity() -> None:
    assert FEATURE_TO_CATEGORY_DESTINATION_INDICES is d1.FEATURE_INDEX_TO_CATEGORY_ID


def test_category_indegrees_match_approved_d1_counts() -> None:
    received = Counter(FEATURE_TO_CATEGORY_DESTINATION_INDICES)
    expected = {
        category_id: d1.EXPECTED_CATEGORY_COUNTS[category_name]
        for category_id, category_name in enumerate(d1.CATEGORY_NAMES)
    }
    assert dict(received) == expected


def test_every_category_points_once_to_sample_zero() -> None:
    assert CATEGORY_TO_SAMPLE_SOURCE_INDICES == tuple(range(8))
    assert len(set(CATEGORY_TO_SAMPLE_SOURCE_INDICES)) == 8
    assert CATEGORY_TO_SAMPLE_DESTINATION_INDICES == (0,) * 8


def test_temporal_and_target_metadata_are_schema_only() -> None:
    assert TEMPORAL_SEMANTICS["sequence_length"] == 5
    assert TEMPORAL_SEMANTICS["explicit_temporal_edges"] is False
    assert TEMPORAL_SEMANTICS["temporal_encoder_implemented"] is False
    assert TARGET_SEMANTICS == {
        "task_level": "graph",
        "target_source": "fifth/final Stage C flow label",
        "node_level_labels": False,
        "labels_loaded_by_schema": False,
    }


def test_manifest_has_exact_schema_and_provenance(manifest: dict[str, object]) -> None:
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["node_kinds"] == list(NODE_KINDS)
    assert manifest["node_counts"] == dict(NODE_COUNTS)
    assert manifest["total_node_count"] == 87
    assert manifest["node_semantics"]["sample"] == {
        "role": "virtual graph-level readout",
        "represents": "one complete Stage C T=5 sample/window",
        "network_entity": False,
    }
    assert manifest["relation_types"] == [list(relation) for relation in RELATION_TYPES]
    assert manifest["total_edge_count"] == 86
    assert manifest["d1_provenance"] == {
        "taxonomy_version": d1.TAXONOMY_VERSION,
        "taxonomy_sha256": EXPECTED_D1_SHA256,
        "stage_c_feature_order_sha256": EXPECTED_FEATURE_ORDER_SHA256,
    }


def test_manifest_contains_complete_static_edge_template(manifest: dict[str, object]) -> None:
    feature_relation, category_relation = manifest["relations"]
    assert feature_relation["source_indices"] == list(range(78))
    assert feature_relation["destination_indices"] == list(d1.FEATURE_INDEX_TO_CATEGORY_ID)
    assert feature_relation["edge_count"] == 78
    assert category_relation["source_indices"] == list(range(8))
    assert category_relation["destination_indices"] == [0] * 8
    assert category_relation["edge_count"] == 8


def test_generation_is_byte_identical_and_checked_in(manifest: dict[str, object], d1_manifest: dict[str, object]) -> None:
    first = manifest_bytes(d1_manifest)
    second = manifest_bytes(d1_manifest)
    assert first == second
    assert json.loads(first)["schema_sha256"] == EXPECTED_SCHEMA_SHA256
    assert manifest["schema_sha256"] == EXPECTED_SCHEMA_SHA256
    assert D2_MANIFEST_PATH.read_bytes() == first


def test_schema_hash_changes_with_every_structural_contract_family(
    manifest: dict[str, object],
) -> None:
    original = dict(manifest)
    original_hash = original.pop("schema_sha256")
    assert canonical_sha256(original) == original_hash == EXPECTED_SCHEMA_SHA256
    mutations = []
    for field, replacement in (
        ("node_counts", {"feature": 77, "category": 8, "sample": 1}),
        ("category_order", list(reversed(d1.CATEGORY_NAMES))),
        ("relation_types", [["feature", "changed", "category"]]),
        ("temporal_semantics", {"sequence_length": 4}),
        ("target_semantics", {"task_level": "node"}),
    ):
        changed = json.loads(json.dumps(original))
        changed[field] = replacement
        mutations.append(changed)
    changed_edge = json.loads(json.dumps(original))
    changed_edge["relations"][0]["destination_indices"][0] = 1
    mutations.append(changed_edge)
    assert all(canonical_sha256(changed) != original_hash for changed in mutations)


def test_manifest_round_trip_validation_succeeds(
    manifest: dict[str, object], d1_manifest: dict[str, object]
) -> None:
    round_tripped = json.loads(json.dumps(manifest))
    validate_manifest(round_tripped, d1_manifest)


def test_tampered_schema_is_rejected(
    manifest: dict[str, object], d1_manifest: dict[str, object]
) -> None:
    tampered = json.loads(json.dumps(manifest))
    tampered["total_edge_count"] = 85
    with pytest.raises(ValueError, match="schema_sha256"):
        validate_manifest(tampered, d1_manifest)


def test_invalid_d1_provenance_is_rejected(tmp_path: Path) -> None:
    tampered = json.loads(D1_MANIFEST_PATH.read_text(encoding="utf-8"))
    tampered["taxonomy_sha256"] = "0" * 64
    path = tmp_path / "tampered_d1.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="taxonomy_sha256"):
        load_validated_d1_manifest(path)


def test_d2_generation_modifies_neither_stage_c_nor_raw(tmp_path: Path) -> None:
    stage_c = tmp_path / "stage_c"
    raw = tmp_path / "raw"
    stage_c.mkdir()
    raw.mkdir()
    (stage_c / "frozen.bin").write_bytes(b"stage-c")
    (raw / "frozen.csv").write_bytes(b"raw")
    before_stage_c = _fingerprint_tree(stage_c)
    before_raw = _fingerprint_tree(raw)

    write_manifest(D1_MANIFEST_PATH, tmp_path / "stage_d2" / "graph_schema.json")

    assert _fingerprint_tree(stage_c) == before_stage_c
    assert _fingerprint_tree(raw) == before_raw
