"""Contract tests for Stage D1 only; no graph construction is exercised."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.graph.taxonomy import (
    CATEGORY_MEMBER_INDICES,
    CATEGORY_MEMBER_NAMES,
    CATEGORY_NAMES,
    EXPECTED_CATEGORY_COUNTS,
    FEATURE_INDEX_TO_CATEGORY_ID,
    FEATURE_TO_CATEGORY,
    MODEL_FEATURE_COLUMNS,
    TAXONOMY_VERSION,
    build_manifest,
    manifest_bytes,
    read_authoritative_feature_names,
    validate_manifest,
    write_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
STAGE_C_FEATURES = ROOT / "data" / "processed" / "preprocessed" / "feature_names.json"
D1_MANIFEST = ROOT / "results" / "stage_d1" / "feature_taxonomy.json"
EXPECTED_NAMES = (
    "Context",
    "Time",
    "Packets and size",
    "Rates and ratios",
    "Flags",
    "Bulk",
    "Subflow",
    "Window",
)
EXPECTED_COUNTS = (2, 23, 24, 5, 12, 6, 4, 2)
EXPECTED_MEMBER_INDICES = {
    "Context": (0, 1),
    "Time": (2, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 70, 71, 72, 73, 74, 75, 76, 77),
    "Packets and size": (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 35, 36, 39, 40, 41, 42, 43, 53, 54, 55, 68, 69),
    "Rates and ratios": (15, 16, 37, 38, 52),
    "Flags": (31, 32, 33, 34, 44, 45, 46, 47, 48, 49, 50, 51),
    "Bulk": (56, 57, 58, 59, 60, 61),
    "Subflow": (62, 63, 64, 65),
    "Window": (66, 67),
}
EXPECTED_TAXONOMY_SHA256 = "fe1332dc22727cf952dc1d2117fec46439294821e2a4654537d910df901fab7e"
EXPECTED_STAGE_C_FEATURE_ORDER_SHA256 = "7ae8c123332c05b1136e1093c53d710b69b5c397ad15309e207667b104f5fe4c"


@pytest.fixture(scope="module")
def authoritative_names() -> tuple[str, ...]:
    return read_authoritative_feature_names(STAGE_C_FEATURES)


@pytest.fixture(scope="module")
def manifest(authoritative_names: tuple[str, ...]) -> dict[str, object]:
    return build_manifest(authoritative_names)


def _tree_fingerprint(root: Path) -> dict[str, tuple[int, int, str]]:
    result = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        stat = path.stat()
        result[path.relative_to(root).as_posix()] = (
            stat.st_size,
            stat.st_mtime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    return result


def test_authoritative_stage_c_order_has_exactly_78_unique_features(
    authoritative_names: tuple[str, ...],
) -> None:
    assert len(authoritative_names) == 78
    assert len(set(authoritative_names)) == 78
    assert authoritative_names == MODEL_FEATURE_COLUMNS


def test_taxonomy_assigns_every_authoritative_feature_once_and_no_unknowns(
    authoritative_names: tuple[str, ...],
) -> None:
    assigned = [name for members in CATEGORY_MEMBER_NAMES.values() for name in members]
    assert len(FEATURE_TO_CATEGORY) == 78
    assert len(assigned) == 78
    assert len(set(assigned)) == 78
    assert set(assigned) == set(authoritative_names)


def test_category_order_counts_and_total_are_frozen() -> None:
    assert CATEGORY_NAMES == EXPECTED_NAMES
    assert tuple(EXPECTED_CATEGORY_COUNTS[name] for name in CATEGORY_NAMES) == EXPECTED_COUNTS
    assert sum(EXPECTED_COUNTS) == 78
    assert tuple(len(CATEGORY_MEMBER_NAMES[name]) for name in CATEGORY_NAMES) == EXPECTED_COUNTS


def test_index_mapping_is_derived_in_authoritative_feature_order(
    authoritative_names: tuple[str, ...],
) -> None:
    expected = tuple(CATEGORY_NAMES.index(FEATURE_TO_CATEGORY[name]) for name in authoritative_names)
    assert FEATURE_INDEX_TO_CATEGORY_ID == expected
    assert FEATURE_INDEX_TO_CATEGORY_ID == tuple(expected)


def test_member_indices_reproduce_approved_assignments(
    authoritative_names: tuple[str, ...],
) -> None:
    flattened = []
    for category_id, category_name in enumerate(CATEGORY_NAMES):
        indices = CATEGORY_MEMBER_INDICES[category_name]
        assert indices == EXPECTED_MEMBER_INDICES[category_name]
        assert tuple(authoritative_names[index] for index in indices) == CATEGORY_MEMBER_NAMES[category_name]
        assert all(FEATURE_INDEX_TO_CATEGORY_ID[index] == category_id for index in indices)
        flattened.extend(indices)
    assert sorted(flattened) == list(range(78))


def test_manifest_contains_complete_contract(manifest: dict[str, object]) -> None:
    assert manifest["taxonomy_version"] == TAXONOMY_VERSION
    assert manifest["category_names"] == list(EXPECTED_NAMES)
    assert manifest["feature_index_to_category_id"] == list(FEATURE_INDEX_TO_CATEGORY_ID)
    assert [item["member_count"] for item in manifest["categories"]] == list(EXPECTED_COUNTS)
    validate_manifest(manifest)


def test_manifest_regenerates_byte_identically_and_hash_is_stable(
    authoritative_names: tuple[str, ...], tmp_path: Path
) -> None:
    first = manifest_bytes(authoritative_names)
    second = manifest_bytes(authoritative_names)
    assert first == second
    assert json.loads(first)["taxonomy_sha256"] == EXPECTED_TAXONOMY_SHA256
    assert json.loads(second)["taxonomy_sha256"] == EXPECTED_TAXONOMY_SHA256
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "nested" / "second.json"
    write_manifest(STAGE_C_FEATURES, first_path)
    write_manifest(STAGE_C_FEATURES, second_path)
    assert first_path.read_bytes() == second_path.read_bytes() == first


def test_stage_c_feature_order_hash_matches_canonical_list(manifest: dict[str, object]) -> None:
    ordered_names = manifest["stage_c_feature_order"]["feature_names"]
    canonical = json.dumps(
        ordered_names, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    assert manifest["stage_c_feature_order"]["sha256"] == hashlib.sha256(canonical).hexdigest()
    assert manifest["stage_c_feature_order"]["sha256"] == EXPECTED_STAGE_C_FEATURE_ORDER_SHA256


def test_checked_in_manifest_is_exact_regeneration(
    authoritative_names: tuple[str, ...],
) -> None:
    assert D1_MANIFEST.read_bytes() == manifest_bytes(authoritative_names)
    validate_manifest(json.loads(D1_MANIFEST.read_text(encoding="utf-8")))


def test_generation_does_not_modify_stage_c_or_raw_tree(tmp_path: Path) -> None:
    stage_c = tmp_path / "stage_c"
    raw = tmp_path / "raw"
    stage_c.mkdir()
    raw.mkdir()
    authority = stage_c / "feature_names.json"
    authority.write_text(
        json.dumps({"n_features": 78, "feature_names": list(MODEL_FEATURE_COLUMNS)}),
        encoding="utf-8",
    )
    (stage_c / "sentinel.bin").write_bytes(b"frozen-stage-c")
    (raw / "sentinel.csv").write_bytes(b"raw,must,remain,untouched\n")
    before_stage_c = _tree_fingerprint(stage_c)
    before_raw = _tree_fingerprint(raw)

    write_manifest(authority, tmp_path / "stage_d1" / "feature_taxonomy.json")

    assert _tree_fingerprint(stage_c) == before_stage_c
    assert _tree_fingerprint(raw) == before_raw


@pytest.mark.parametrize(
    "names",
    [
        MODEL_FEATURE_COLUMNS[:-1],
        MODEL_FEATURE_COLUMNS[:-1] + (MODEL_FEATURE_COLUMNS[0],),
        MODEL_FEATURE_COLUMNS[1:] + MODEL_FEATURE_COLUMNS[:1],
        MODEL_FEATURE_COLUMNS[:-1] + ("Unknown Feature",),
    ],
)
def test_invalid_feature_authorities_are_rejected(names: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        build_manifest(names)


def test_tampered_manifest_hash_is_rejected(manifest: dict[str, object]) -> None:
    tampered = json.loads(json.dumps(manifest))
    tampered["category_names"][0] = "Changed"
    with pytest.raises(ValueError, match="taxonomy_sha256"):
        validate_manifest(tampered)
