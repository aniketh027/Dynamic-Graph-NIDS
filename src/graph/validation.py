"""Independent Stage D3 validation of the D2 logical graph schema.

The validator reconstructs the tiny typed graph from manifest endpoints and
checks degrees, paths, connectivity, acyclicity, uniqueness, ID safety, and
provenance.  It never constructs Stage C sample graphs or reads sample arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.graph import schema as d2
from src.graph import taxonomy as d1


VALIDATION_VERSION = "stage-d3-structural-validation-v1"
APPROVED_D1_TAXONOMY_SHA256 = (
    "fe1332dc22727cf952dc1d2117fec46439294821e2a4654537d910df901fab7e"
)
APPROVED_D2_SCHEMA_SHA256 = (
    "0950126394099acea7573e14357f8bd01b22eb91c2a9d0b4f9298ea208ecebea"
)

Node = tuple[str, int]
Edge = tuple[Node, str, Node]


class StructuralValidationError(ValueError):
    """Raised with the complete deterministic report when D3 finds defects."""

    def __init__(self, report: Mapping[str, Any]):
        self.report = dict(report)
        errors = self.report.get("errors", [])
        detail = "; ".join(str(error) for error in errors)
        super().__init__(f"Stage D3 structural validation failed: {detail}")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _record(errors: list[str], condition: bool, message: str) -> bool:
    if not condition:
        errors.append(message)
    return condition


def _expected_nodes() -> tuple[Node, ...]:
    return tuple(
        (kind, node_id)
        for kind in d2.NODE_KINDS
        for node_id in d2.NODE_IDS[kind]
    )


def _parse_edges(
    schema_manifest: Mapping[str, Any], errors: list[str]
) -> tuple[list[Edge], dict[str, int], bool]:
    edges: list[Edge] = []
    actual_relation_counts: Counter[str] = Counter()
    ids_valid = True
    relations = schema_manifest.get("relations")
    if not isinstance(relations, list):
        errors.append("relations must be a list")
        return edges, {}, False

    valid_ids = {kind: set(d2.NODE_IDS[kind]) for kind in d2.NODE_KINDS}
    for relation_index, relation in enumerate(relations):
        if not isinstance(relation, Mapping):
            errors.append(f"relation {relation_index} must be an object")
            ids_valid = False
            continue
        relation_type = relation.get("type")
        if (
            not isinstance(relation_type, list)
            or len(relation_type) != 3
            or not all(isinstance(item, str) for item in relation_type)
        ):
            errors.append(f"relation {relation_index} has an invalid type triple")
            ids_valid = False
            continue
        source_kind, relation_name, destination_kind = relation_type
        relation_key = "|".join(relation_type)
        sources = relation.get("source_indices")
        destinations = relation.get("destination_indices")
        if not isinstance(sources, list) or not isinstance(destinations, list):
            errors.append(f"relation {relation_key} endpoints must be lists")
            ids_valid = False
            continue
        if len(sources) != len(destinations):
            errors.append(f"relation {relation_key} source/destination lengths differ")
            ids_valid = False
        if relation.get("edge_count") != len(sources):
            errors.append(f"relation {relation_key} declared edge_count differs from endpoints")

        for endpoint_index, (source_id, destination_id) in enumerate(
            zip(sources, destinations)
        ):
            if not isinstance(source_id, int) or isinstance(source_id, bool):
                errors.append(f"relation {relation_key} source {endpoint_index} is not an integer")
                ids_valid = False
                continue
            if not isinstance(destination_id, int) or isinstance(destination_id, bool):
                errors.append(
                    f"relation {relation_key} destination {endpoint_index} is not an integer"
                )
                ids_valid = False
                continue
            if source_kind not in valid_ids or source_id not in valid_ids.get(source_kind, set()):
                errors.append(
                    f"out-of-range {source_kind} source ID {source_id} in {relation_key}"
                )
                ids_valid = False
            if (
                destination_kind not in valid_ids
                or destination_id not in valid_ids.get(destination_kind, set())
            ):
                errors.append(
                    f"out-of-range {destination_kind} destination ID {destination_id} in {relation_key}"
                )
                ids_valid = False
            if source_kind in valid_ids and destination_kind in valid_ids:
                edges.append(
                    (
                        (source_kind, source_id),
                        relation_name,
                        (destination_kind, destination_id),
                    )
                )
                actual_relation_counts[relation_key] += 1

    return edges, dict(actual_relation_counts), ids_valid


def _weak_component_count(nodes: Sequence[Node], edges: Sequence[Edge]) -> int:
    neighbors: dict[Node, set[Node]] = {node: set() for node in nodes}
    for source, _, destination in edges:
        if source in neighbors and destination in neighbors:
            neighbors[source].add(destination)
            neighbors[destination].add(source)
    remaining = set(nodes)
    components = 0
    while remaining:
        components += 1
        start = next(iter(remaining))
        queue = [start]
        remaining.remove(start)
        while queue:
            node = queue.pop()
            for neighbor in neighbors[node]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
    return components


def _is_dag(nodes: Sequence[Node], edges: Sequence[Edge]) -> bool:
    adjacency: dict[Node, list[Node]] = {node: [] for node in nodes}
    indegree: Counter[Node] = Counter({node: 0 for node in nodes})
    for source, _, destination in edges:
        if source in adjacency and destination in adjacency:
            adjacency[source].append(destination)
            indegree[destination] += 1
    queue = deque(node for node in nodes if indegree[node] == 0)
    visited = 0
    while queue:
        node = queue.popleft()
        visited += 1
        for destination in adjacency[node]:
            indegree[destination] -= 1
            if indegree[destination] == 0:
                queue.append(destination)
    return visited == len(nodes)


def _simple_path_lengths(
    adjacency: Mapping[Node, Sequence[Node]], start: Node, target: Node, max_nodes: int
) -> list[int]:
    lengths: list[int] = []

    def visit(node: Node, visited: set[Node], hops: int) -> None:
        if node == target:
            lengths.append(hops)
            return
        if len(visited) >= max_nodes:
            return
        for destination in adjacency.get(node, ()):
            if destination not in visited:
                visit(destination, visited | {destination}, hops + 1)

    visit(start, {start}, 0)
    return lengths


def analyze_manifest(
    schema_manifest: Mapping[str, Any], d1_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Analyze actual D2 endpoints and return a deterministic PASS/FAIL report."""

    errors: list[str] = []

    d1_valid = True
    try:
        d1.validate_manifest(d1_manifest)
    except Exception as exc:  # report provenance failure alongside topology defects
        d1_valid = False
        errors.append(f"D1 provenance invalid: {exc}")
    recorded_d1_hash = schema_manifest.get("d1_provenance", {}).get("taxonomy_sha256")
    recorded_feature_hash = schema_manifest.get("d1_provenance", {}).get(
        "stage_c_feature_order_sha256"
    )
    d1_hash_matches = (
        d1_valid
        and d1_manifest.get("taxonomy_sha256") == APPROVED_D1_TAXONOMY_SHA256
        and recorded_d1_hash == d1_manifest.get("taxonomy_sha256")
    )
    _record(errors, d1_hash_matches, "D1 taxonomy SHA-256 provenance mismatch")
    feature_hash_matches = (
        d1_valid
        and recorded_feature_hash
        == d1_manifest.get("stage_c_feature_order", {}).get("sha256")
    )
    _record(errors, feature_hash_matches, "Stage C feature-order SHA-256 provenance mismatch")

    supplied_schema_hash = schema_manifest.get("schema_sha256")
    unhashed_schema = dict(schema_manifest)
    unhashed_schema.pop("schema_sha256", None)
    schema_hash_valid = (
        isinstance(supplied_schema_hash, str)
        and d2.canonical_sha256(unhashed_schema) == supplied_schema_hash
    )
    _record(errors, schema_hash_valid, "D2 schema SHA-256 does not match manifest content")
    approved_schema_hash = supplied_schema_hash == APPROVED_D2_SCHEMA_SHA256
    _record(errors, approved_schema_hash, "D2 schema SHA-256 differs from approved D2")
    d2_module_consistent = False
    try:
        d2.validate_manifest(schema_manifest, d1_manifest)
        d2_module_consistent = True
    except Exception as exc:
        errors.append(f"D2 artifact/module consistency failed: {exc}")

    expected_nodes = _expected_nodes()
    node_kinds_valid = schema_manifest.get("node_kinds") == list(d2.NODE_KINDS)
    _record(errors, node_kinds_valid, "node kinds differ from feature/category/sample")
    node_counts_valid = schema_manifest.get("node_counts") == dict(d2.NODE_COUNTS)
    _record(errors, node_counts_valid, "node counts differ from 78/8/1")
    _record(errors, schema_manifest.get("total_node_count") == 87, "total node count is not 87")
    indexing = schema_manifest.get("indexing", {})
    indexing_valid = (
        indexing.get("mode") == "type-local"
        and indexing.get("feature_ids") == list(range(78))
        and indexing.get("category_ids") == list(range(8))
        and indexing.get("sample_ids") == [0]
    )
    _record(errors, indexing_valid, "type-local IDs are not canonical and contiguous")

    edges, relation_counts, endpoint_ids_valid = _parse_edges(schema_manifest, errors)
    actual_relation_types = tuple(
        tuple(relation.get("type", ()))
        for relation in schema_manifest.get("relations", [])
        if isinstance(relation, Mapping)
    )
    declared_relation_types = schema_manifest.get("relation_types")
    relations_exact = (
        declared_relation_types == [list(relation) for relation in d2.RELATION_TYPES]
        and actual_relation_types == d2.RELATION_TYPES
    )
    _record(errors, relations_exact, "relations are not exactly the two approved forward types")

    unique_edges = set(edges)
    duplicate_count = len(edges) - len(unique_edges)
    duplicates_valid = duplicate_count == 0
    _record(errors, duplicates_valid, f"duplicate structural edges found: {duplicate_count}")
    _record(errors, len(edges) == 86, f"actual structural edge count is {len(edges)}, not 86")
    _record(errors, len(unique_edges) == 86, f"unique structural edge count is {len(unique_edges)}, not 86")
    _record(errors, schema_manifest.get("total_edge_count") == 86, "declared total edge count is not 86")

    valid_node_set = set(expected_nodes)
    adjacency: dict[Node, list[Node]] = defaultdict(list)
    indegree: Counter[Node] = Counter({node: 0 for node in expected_nodes})
    outdegree: Counter[Node] = Counter({node: 0 for node in expected_nodes})
    level = {"feature": 0, "category": 1, "sample": 2}
    levels_valid = True
    self_loops = 0
    for source, _, destination in edges:
        if source == destination:
            self_loops += 1
        if (
            source[0] not in level
            or destination[0] not in level
            or level[destination[0]] - level[source[0]] != 1
        ):
            levels_valid = False
        if source in valid_node_set and destination in valid_node_set:
            adjacency[source].append(destination)
            outdegree[source] += 1
            indegree[destination] += 1
    _record(errors, self_loops == 0, f"self-loop count is {self_loops}, not 0")
    _record(errors, levels_valid, "an edge does not increase hierarchy level by exactly 1")

    feature_outdegrees = [outdegree[("feature", feature_id)] for feature_id in range(78)]
    feature_indegrees = [indegree[("feature", feature_id)] for feature_id in range(78)]
    feature_degrees_valid = all(value == 1 for value in feature_outdegrees) and all(
        value == 0 for value in feature_indegrees
    )
    _record(errors, feature_degrees_valid, "every feature must have indegree 0 and outdegree 1")

    membership_valid = True
    for feature_id, expected_category_id in enumerate(d1.FEATURE_INDEX_TO_CATEGORY_ID):
        destinations = adjacency.get(("feature", feature_id), [])
        if destinations != [("category", expected_category_id)]:
            membership_valid = False
            errors.append(
                f"feature {feature_id} does not point exactly once to D1 category {expected_category_id}"
            )

    category_indegrees = [indegree[("category", category_id)] for category_id in range(8)]
    category_outdegrees = [outdegree[("category", category_id)] for category_id in range(8)]
    expected_category_counts = [
        d1.EXPECTED_CATEGORY_COUNTS[name] for name in d1.CATEGORY_NAMES
    ]
    category_degrees_valid = (
        category_indegrees == expected_category_counts
        and category_outdegrees == [1] * 8
    )
    _record(
        errors,
        category_degrees_valid,
        "category indegrees/outdegrees differ from D1 counts and one-to-sample contract",
    )

    sample_indegree = indegree[("sample", 0)]
    sample_outdegree = outdegree[("sample", 0)]
    sample_degrees_valid = sample_indegree == 8 and sample_outdegree == 0
    _record(errors, sample_degrees_valid, "sample node must have indegree 8 and outdegree 0")

    weak_components = _weak_component_count(expected_nodes, edges)
    weakly_connected = weak_components == 1
    _record(errors, weakly_connected, f"weak component count is {weak_components}, not 1")

    dag = _is_dag(expected_nodes, edges)
    _record(errors, dag, "structural graph contains a directed cycle")

    target = ("sample", 0)
    feature_path_lengths = [
        _simple_path_lengths(adjacency, ("feature", feature_id), target, 87)
        for feature_id in range(78)
    ]
    category_path_lengths = [
        _simple_path_lengths(adjacency, ("category", category_id), target, 87)
        for category_id in range(8)
    ]
    paths_valid = all(lengths == [2] for lengths in feature_path_lengths) and all(
        lengths == [1] for lengths in category_path_lengths
    )
    _record(
        errors,
        paths_valid,
        "every feature must have exactly one two-hop path and every category one one-hop path to sample",
    )

    relation_count_summary = {
        "feature|contributes_to|category": relation_counts.get(
            "feature|contributes_to|category", 0
        ),
        "category|contributes_to|sample": relation_counts.get(
            "category|contributes_to|sample", 0
        ),
    }
    category_count_summary = {
        name: category_indegrees[category_id]
        for category_id, name in enumerate(d1.CATEGORY_NAMES)
    }
    report: dict[str, Any] = {
        "validation_version": VALIDATION_VERSION,
        "references": {
            "d1_taxonomy_sha256": d1_manifest.get("taxonomy_sha256"),
            "d2_schema_sha256": supplied_schema_hash,
        },
        "node_counts": (
            dict(schema_manifest.get("node_counts", {}))
            if isinstance(schema_manifest.get("node_counts"), Mapping)
            else schema_manifest.get("node_counts")
        ),
        "total_node_count": schema_manifest.get("total_node_count"),
        "relation_edge_counts": relation_count_summary,
        "actual_edge_count": len(edges),
        "unique_edge_count": len(unique_edges),
        "category_member_counts": category_count_summary,
        "degree_summary": {
            "features": {
                "count": 78,
                "indegree_min": min(feature_indegrees),
                "indegree_max": max(feature_indegrees),
                "outdegree_min": min(feature_outdegrees),
                "outdegree_max": max(feature_outdegrees),
                "all_indegree_0_outdegree_1": feature_degrees_valid,
            },
            "categories": {
                "indegrees": category_indegrees,
                "outdegrees": category_outdegrees,
                "matches_d1_indegrees_and_outdegree_1": category_degrees_valid,
            },
            "sample": {
                "indegree": sample_indegree,
                "outdegree": sample_outdegree,
            },
        },
        "weak_connectivity": {
            "component_count": weak_components,
            "is_weakly_connected": weakly_connected,
        },
        "directed_acyclicity": {
            "is_dag": dag,
            "self_loop_count": self_loops,
            "all_edges_increase_level_by_one": levels_valid,
        },
        "path_validation": {
            "all_features_exactly_one_path_of_two_hops": all(
                lengths == [2] for lengths in feature_path_lengths
            ),
            "all_categories_exactly_one_path_of_one_hop": all(
                lengths == [1] for lengths in category_path_lengths
            ),
        },
        "duplicate_edge_validation": {
            "duplicate_count": duplicate_count,
            "all_edges_unique": duplicates_valid,
        },
        "endpoint_range_validation": {
            "type_local_ids_declared_correctly": indexing_valid,
            "all_endpoints_in_range": endpoint_ids_valid,
        },
        "provenance_validation": {
            "d1_manifest_valid": d1_valid,
            "d1_hash_matches": d1_hash_matches,
            "stage_c_feature_order_hash_matches": feature_hash_matches,
            "d2_schema_hash_valid": schema_hash_valid,
            "d2_approved_hash_matches": approved_schema_hash,
            "d2_artifact_matches_module": d2_module_consistent,
        },
        "membership_validation": {
            "all_features_match_d1": membership_valid,
            "node_kinds_valid": node_kinds_valid,
            "node_counts_valid": node_counts_valid,
            "relations_exact": relations_exact,
        },
        "overall_status": "PASS" if not errors else "FAIL",
        "errors": errors,
    }
    report["report_sha256"] = _canonical_sha256(report)
    return report


def validate_manifest(
    schema_manifest: Mapping[str, Any], d1_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Return a PASS report or raise with all detected structural defects."""

    report = analyze_manifest(schema_manifest, d1_manifest)
    if report["overall_status"] != "PASS":
        raise StructuralValidationError(report)
    return report


def report_bytes(report: Mapping[str, Any]) -> bytes:
    """Return deterministic human-readable report bytes after hash validation."""

    supplied_hash = report.get("report_sha256")
    unhashed = dict(report)
    unhashed.pop("report_sha256", None)
    if not isinstance(supplied_hash, str) or _canonical_sha256(unhashed) != supplied_hash:
        raise ValueError("D3 report SHA-256 does not match its canonical content")
    return (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def write_report(
    d1_manifest_path: Path, d2_manifest_path: Path, output_path: Path
) -> dict[str, Any]:
    """Validate the approved artifacts and write only the compact D3 report."""

    d1_manifest = d2.load_validated_d1_manifest(d1_manifest_path)
    d2_manifest = json.loads(d2_manifest_path.read_text(encoding="utf-8"))
    report = validate_manifest(d2_manifest, d1_manifest)
    data = report_bytes(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(data)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the Stage D2 hierarchy")
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=Path("results/stage_d1/feature_taxonomy.json"),
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("results/stage_d2/graph_schema.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/stage_d3/structural_validation.json"),
    )
    args = parser.parse_args(argv)
    report = write_report(args.taxonomy, args.schema, args.output)
    print(f"wrote {args.output}")
    print(f"overall_status={report['overall_status']}")
    print(f"report_sha256={report['report_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
