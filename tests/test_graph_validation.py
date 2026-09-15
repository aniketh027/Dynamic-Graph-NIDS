"""Adversarial endpoint-level tests for Stage D3 structural validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.graph import schema as d2
from src.graph import taxonomy as d1
from src.graph.validation import (
    APPROVED_D1_TAXONOMY_SHA256,
    APPROVED_D2_SCHEMA_SHA256,
    VALIDATION_VERSION,
    StructuralValidationError,
    analyze_manifest,
    report_bytes,
    validate_manifest,
    write_report,
)


ROOT = Path(__file__).resolve().parents[1]
D1_PATH = ROOT / "results" / "stage_d1" / "feature_taxonomy.json"
D2_PATH = ROOT / "results" / "stage_d2" / "graph_schema.json"
D3_PATH = ROOT / "results" / "stage_d3" / "structural_validation.json"
EXPECTED_REPORT_SHA256 = "40f4e23e9e0937f5182ebcdb6d6363301012e05c98a90d8b3985968bd1e605bb"


@pytest.fixture()
def manifests() -> tuple[dict[str, object], dict[str, object]]:
    return (
        json.loads(D1_PATH.read_text(encoding="utf-8")),
        json.loads(D2_PATH.read_text(encoding="utf-8")),
    )


def _relation(schema: dict[str, object], relation_type: tuple[str, str, str]) -> dict[str, object]:
    return next(
        relation
        for relation in schema["relations"]
        if tuple(relation["type"]) == relation_type
    )


def _rehash(schema: dict[str, object]) -> None:
    schema.pop("schema_sha256", None)
    schema["schema_sha256"] = d2.canonical_sha256(schema)


def _add_relation(
    schema: dict[str, object], relation_type: tuple[str, str, str], source: int, destination: int
) -> None:
    schema["relation_types"].append(list(relation_type))
    schema["relations"].append(
        {
            "type": list(relation_type),
            "source_indices": [source],
            "destination_indices": [destination],
            "edge_count": 1,
        }
    )
    schema["relation_edge_counts"].append(
        {"type": list(relation_type), "edge_count": 1}
    )
    schema["total_edge_count"] += 1


def _assert_rejected(
    d1_manifest: dict[str, object], schema: dict[str, object], diagnostic: str
) -> StructuralValidationError:
    _rehash(schema)
    with pytest.raises(StructuralValidationError) as caught:
        validate_manifest(schema, d1_manifest)
    assert diagnostic in str(caught.value)
    assert caught.value.report["overall_status"] == "FAIL"
    return caught.value


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


def test_valid_schema_proves_all_required_invariants(manifests) -> None:
    d1_manifest, schema = manifests
    report = validate_manifest(schema, d1_manifest)
    assert report["validation_version"] == VALIDATION_VERSION
    assert report["node_counts"] == {"feature": 78, "category": 8, "sample": 1}
    assert report["total_node_count"] == 87
    assert report["actual_edge_count"] == report["unique_edge_count"] == 86
    assert report["category_member_counts"] == dict(d1.EXPECTED_CATEGORY_COUNTS)
    assert report["degree_summary"]["features"]["all_indegree_0_outdegree_1"] is True
    assert report["degree_summary"]["categories"]["matches_d1_indegrees_and_outdegree_1"] is True
    assert report["degree_summary"]["sample"] == {"indegree": 8, "outdegree": 0}
    assert report["weak_connectivity"] == {
        "component_count": 1,
        "is_weakly_connected": True,
    }
    assert report["directed_acyclicity"] == {
        "is_dag": True,
        "self_loop_count": 0,
        "all_edges_increase_level_by_one": True,
    }
    assert all(report["path_validation"].values())
    assert report["duplicate_edge_validation"]["all_edges_unique"] is True
    assert all(report["endpoint_range_validation"].values())
    assert all(report["provenance_validation"].values())
    assert report["overall_status"] == "PASS"
    assert report["errors"] == []


def test_provenance_hashes_are_actual_approved_artifacts(manifests) -> None:
    d1_manifest, schema = manifests
    report = validate_manifest(schema, d1_manifest)
    assert d1_manifest["taxonomy_sha256"] == APPROVED_D1_TAXONOMY_SHA256
    assert schema["schema_sha256"] == APPROVED_D2_SCHEMA_SHA256
    assert report["references"] == {
        "d1_taxonomy_sha256": APPROVED_D1_TAXONOMY_SHA256,
        "d2_schema_sha256": APPROVED_D2_SCHEMA_SHA256,
    }


def test_report_generation_is_deterministic_and_checked_in(manifests, tmp_path: Path) -> None:
    d1_manifest, schema = manifests
    first = analyze_manifest(schema, d1_manifest)
    second = analyze_manifest(schema, d1_manifest)
    assert first == second
    assert first["report_sha256"] == EXPECTED_REPORT_SHA256
    assert report_bytes(first) == report_bytes(second) == D3_PATH.read_bytes()
    output = tmp_path / "nested" / "report.json"
    write_report(D1_PATH, D2_PATH, output)
    assert output.read_bytes() == D3_PATH.read_bytes()


def test_missing_feature_category_edge_is_rejected(manifests) -> None:
    d1_manifest, schema = manifests
    relation = _relation(schema, d2.FEATURE_TO_CATEGORY_RELATION)
    relation["source_indices"].pop()
    relation["destination_indices"].pop()
    relation["edge_count"] -= 1
    schema["total_edge_count"] -= 1
    _assert_rejected(d1_manifest, schema, "feature 77 does not point exactly once")


def test_duplicate_feature_category_edge_is_rejected(manifests) -> None:
    d1_manifest, schema = manifests
    relation = _relation(schema, d2.FEATURE_TO_CATEGORY_RELATION)
    relation["source_indices"].append(0)
    relation["destination_indices"].append(0)
    relation["edge_count"] += 1
    schema["total_edge_count"] += 1
    error = _assert_rejected(d1_manifest, schema, "duplicate structural edges found: 1")
    assert error.report["duplicate_edge_validation"]["all_edges_unique"] is False


def test_feature_assigned_to_two_categories_is_rejected(manifests) -> None:
    d1_manifest, schema = manifests
    relation = _relation(schema, d2.FEATURE_TO_CATEGORY_RELATION)
    relation["source_indices"].append(0)
    relation["destination_indices"].append(1)
    relation["edge_count"] += 1
    schema["total_edge_count"] += 1
    _assert_rejected(d1_manifest, schema, "feature 0 does not point exactly once")


def test_feature_assigned_to_wrong_category_is_rejected(manifests) -> None:
    d1_manifest, schema = manifests
    _relation(schema, d2.FEATURE_TO_CATEGORY_RELATION)["destination_indices"][0] = 1
    _assert_rejected(d1_manifest, schema, "feature 0 does not point exactly once to D1 category 0")


def test_missing_category_sample_edge_is_rejected(manifests) -> None:
    d1_manifest, schema = manifests
    relation = _relation(schema, d2.CATEGORY_TO_SAMPLE_RELATION)
    relation["source_indices"].pop()
    relation["destination_indices"].pop()
    relation["edge_count"] -= 1
    schema["total_edge_count"] -= 1
    _assert_rejected(d1_manifest, schema, "category indegrees/outdegrees differ")


def test_duplicate_category_sample_edge_is_rejected(manifests) -> None:
    d1_manifest, schema = manifests
    relation = _relation(schema, d2.CATEGORY_TO_SAMPLE_RELATION)
    relation["source_indices"].append(0)
    relation["destination_indices"].append(0)
    relation["edge_count"] += 1
    schema["total_edge_count"] += 1
    _assert_rejected(d1_manifest, schema, "duplicate structural edges found: 1")


def test_direct_feature_sample_edge_is_rejected(manifests) -> None:
    d1_manifest, schema = manifests
    _add_relation(schema, ("feature", "contributes_to", "sample"), 0, 0)
    _assert_rejected(d1_manifest, schema, "does not increase hierarchy level by exactly 1")


def test_category_feature_reverse_edge_is_rejected(manifests) -> None:
    d1_manifest, schema = manifests
    _add_relation(schema, ("category", "contributes_to", "feature"), 0, 0)
    _assert_rejected(d1_manifest, schema, "does not increase hierarchy level by exactly 1")


def test_sample_category_reverse_edge_is_rejected(manifests) -> None:
    d1_manifest, schema = manifests
    _add_relation(schema, ("sample", "contributes_to", "category"), 0, 0)
    error = _assert_rejected(d1_manifest, schema, "structural graph contains a directed cycle")
    assert error.report["degree_summary"]["sample"]["outdegree"] == 1


def test_self_loop_is_rejected(manifests) -> None:
    d1_manifest, schema = manifests
    _add_relation(schema, ("feature", "contributes_to", "feature"), 0, 0)
    _assert_rejected(d1_manifest, schema, "self-loop count is 1")


def test_out_of_range_feature_id_is_rejected(manifests) -> None:
    d1_manifest, schema = manifests
    _relation(schema, d2.FEATURE_TO_CATEGORY_RELATION)["source_indices"][0] = 78
    _assert_rejected(d1_manifest, schema, "out-of-range feature source ID 78")


def test_out_of_range_category_id_is_rejected(manifests) -> None:
    d1_manifest, schema = manifests
    _relation(schema, d2.FEATURE_TO_CATEGORY_RELATION)["destination_indices"][0] = 8
    _assert_rejected(d1_manifest, schema, "out-of-range category destination ID 8")


def test_invalid_sample_id_is_rejected(manifests) -> None:
    d1_manifest, schema = manifests
    _relation(schema, d2.CATEGORY_TO_SAMPLE_RELATION)["destination_indices"][0] = 1
    _assert_rejected(d1_manifest, schema, "out-of-range sample destination ID 1")


def test_disconnected_category_component_is_rejected(manifests) -> None:
    d1_manifest, schema = manifests
    relation = _relation(schema, d2.CATEGORY_TO_SAMPLE_RELATION)
    relation["source_indices"].pop()
    relation["destination_indices"].pop()
    relation["edge_count"] -= 1
    schema["total_edge_count"] -= 1
    error = _assert_rejected(d1_manifest, schema, "weak component count is 2")
    assert error.report["weak_connectivity"]["is_weakly_connected"] is False


def test_directed_cycle_is_rejected(manifests) -> None:
    d1_manifest, schema = manifests
    _add_relation(schema, ("sample", "contributes_to", "category"), 0, 1)
    error = _assert_rejected(d1_manifest, schema, "structural graph contains a directed cycle")
    assert error.report["directed_acyclicity"]["is_dag"] is False


def test_incorrect_category_member_count_is_rejected(manifests) -> None:
    d1_manifest, schema = manifests
    _relation(schema, d2.FEATURE_TO_CATEGORY_RELATION)["destination_indices"][0] = 1
    error = _assert_rejected(d1_manifest, schema, "category indegrees/outdegrees differ")
    assert error.report["category_member_counts"]["Context"] == 1
    assert error.report["category_member_counts"]["Time"] == 24


def test_corrupted_d1_provenance_is_rejected(manifests) -> None:
    d1_manifest, schema = manifests
    d1_manifest["taxonomy_sha256"] = "0" * 64
    with pytest.raises(StructuralValidationError, match="D1 provenance invalid"):
        validate_manifest(schema, d1_manifest)


def test_corrupted_d2_schema_hash_is_rejected(manifests) -> None:
    d1_manifest, schema = manifests
    schema["total_edge_count"] = 85
    with pytest.raises(StructuralValidationError, match="D2 schema SHA-256"):
        validate_manifest(schema, d1_manifest)


def test_corrupted_manifest_structure_is_rejected(manifests) -> None:
    d1_manifest, schema = manifests
    _relation(schema, d2.FEATURE_TO_CATEGORY_RELATION).pop("destination_indices")
    _rehash(schema)
    with pytest.raises(StructuralValidationError, match="endpoints must be lists"):
        validate_manifest(schema, d1_manifest)


def test_validation_generation_does_not_modify_stage_c_or_raw(tmp_path: Path) -> None:
    stage_c = tmp_path / "stage_c"
    raw = tmp_path / "raw"
    stage_c.mkdir()
    raw.mkdir()
    (stage_c / "artifact.bin").write_bytes(b"frozen-stage-c")
    (raw / "source.csv").write_bytes(b"frozen-raw")
    before_stage_c = _fingerprint_tree(stage_c)
    before_raw = _fingerprint_tree(raw)

    write_report(D1_PATH, D2_PATH, tmp_path / "stage_d3" / "report.json")

    assert _fingerprint_tree(stage_c) == before_stage_c
    assert _fingerprint_tree(raw) == before_raw
