"""Deterministic Stage D1 feature taxonomy.

This module maps the frozen 78-feature Stage C model-input order into the
eight owner-approved semantic categories.  It performs no data inspection,
fitting, feature selection, or graph construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.preprocessing.schema import MODEL_FEATURE_COLUMNS


TAXONOMY_VERSION = "stage-d1-taxonomy-v1"
CATEGORY_NAMES: tuple[str, ...] = (
    "Context",
    "Time",
    "Packets and size",
    "Rates and ratios",
    "Flags",
    "Bulk",
    "Subflow",
    "Window",
)

# This is the approved semantic assignment, not a second feature-order
# authority.  Every order-dependent representation below is derived from the
# frozen Stage C MODEL_FEATURE_COLUMNS sequence.
_APPROVED_CATEGORY_MEMBERS: Mapping[str, tuple[str, ...]] = {
    "Context": (
        "Dst Port",
        "Protocol",
    ),
    "Time": (
        "Flow Duration",
        "Flow IAT Mean",
        "Flow IAT Std",
        "Flow IAT Max",
        "Flow IAT Min",
        "Fwd IAT Tot",
        "Fwd IAT Mean",
        "Fwd IAT Std",
        "Fwd IAT Max",
        "Fwd IAT Min",
        "Bwd IAT Tot",
        "Bwd IAT Mean",
        "Bwd IAT Std",
        "Bwd IAT Max",
        "Bwd IAT Min",
        "Active Mean",
        "Active Std",
        "Active Max",
        "Active Min",
        "Idle Mean",
        "Idle Std",
        "Idle Max",
        "Idle Min",
    ),
    "Packets and size": (
        "Tot Fwd Pkts",
        "Tot Bwd Pkts",
        "TotLen Fwd Pkts",
        "TotLen Bwd Pkts",
        "Fwd Pkt Len Max",
        "Fwd Pkt Len Min",
        "Fwd Pkt Len Mean",
        "Fwd Pkt Len Std",
        "Bwd Pkt Len Max",
        "Bwd Pkt Len Min",
        "Bwd Pkt Len Mean",
        "Bwd Pkt Len Std",
        "Fwd Header Len",
        "Bwd Header Len",
        "Pkt Len Min",
        "Pkt Len Max",
        "Pkt Len Mean",
        "Pkt Len Std",
        "Pkt Len Var",
        "Pkt Size Avg",
        "Fwd Seg Size Avg",
        "Bwd Seg Size Avg",
        "Fwd Act Data Pkts",
        "Fwd Seg Size Min",
    ),
    "Rates and ratios": (
        "Flow Byts/s",
        "Flow Pkts/s",
        "Fwd Pkts/s",
        "Bwd Pkts/s",
        "Down/Up Ratio",
    ),
    "Flags": (
        "Fwd PSH Flags",
        "Bwd PSH Flags",
        "Fwd URG Flags",
        "Bwd URG Flags",
        "FIN Flag Cnt",
        "SYN Flag Cnt",
        "RST Flag Cnt",
        "PSH Flag Cnt",
        "ACK Flag Cnt",
        "URG Flag Cnt",
        "CWE Flag Count",
        "ECE Flag Cnt",
    ),
    "Bulk": (
        "Fwd Byts/b Avg",
        "Fwd Pkts/b Avg",
        "Fwd Blk Rate Avg",
        "Bwd Byts/b Avg",
        "Bwd Pkts/b Avg",
        "Bwd Blk Rate Avg",
    ),
    "Subflow": (
        "Subflow Fwd Pkts",
        "Subflow Fwd Byts",
        "Subflow Bwd Pkts",
        "Subflow Bwd Byts",
    ),
    "Window": (
        "Init Fwd Win Byts",
        "Init Bwd Win Byts",
    ),
}

EXPECTED_CATEGORY_COUNTS: Mapping[str, int] = {
    "Context": 2,
    "Time": 23,
    "Packets and size": 24,
    "Rates and ratios": 5,
    "Flags": 12,
    "Bulk": 6,
    "Subflow": 4,
    "Window": 2,
}


def _canonical_json_bytes(value: Any) -> bytes:
    """Serialize *value* deterministically for hashing."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_canonical(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _validate_assignment(feature_names: Sequence[str]) -> None:
    if len(feature_names) != 78:
        raise ValueError(f"expected 78 authoritative Stage C features, got {len(feature_names)}")
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("authoritative Stage C feature order contains duplicates")
    if tuple(_APPROVED_CATEGORY_MEMBERS) != CATEGORY_NAMES:
        raise AssertionError("approved category declaration is not in canonical order")
    if dict(EXPECTED_CATEGORY_COUNTS) != {
        name: len(_APPROVED_CATEGORY_MEMBERS[name]) for name in CATEGORY_NAMES
    }:
        raise AssertionError("approved category member counts do not match the contract")

    assigned = [
        feature
        for category_name in CATEGORY_NAMES
        for feature in _APPROVED_CATEGORY_MEMBERS[category_name]
    ]
    duplicates = sorted(name for name, count in Counter(assigned).items() if count != 1)
    missing = sorted(set(feature_names) - set(assigned))
    unknown = sorted(set(assigned) - set(feature_names))
    if duplicates or missing or unknown or len(assigned) != len(feature_names):
        raise ValueError(
            "taxonomy must assign each authoritative feature exactly once; "
            f"duplicates={duplicates}, missing={missing}, unknown={unknown}"
        )


_validate_assignment(MODEL_FEATURE_COLUMNS)

CATEGORY_IDS: Mapping[str, int] = {
    category_name: category_id
    for category_id, category_name in enumerate(CATEGORY_NAMES)
}
FEATURE_TO_CATEGORY: Mapping[str, str] = {
    feature_name: category_name
    for category_name in CATEGORY_NAMES
    for feature_name in _APPROVED_CATEGORY_MEMBERS[category_name]
}
FEATURE_INDEX_TO_CATEGORY_ID: tuple[int, ...] = tuple(
    CATEGORY_IDS[FEATURE_TO_CATEGORY[feature_name]]
    for feature_name in MODEL_FEATURE_COLUMNS
)
CATEGORY_MEMBER_INDICES: Mapping[str, tuple[int, ...]] = {
    category_name: tuple(
        index
        for index, feature_name in enumerate(MODEL_FEATURE_COLUMNS)
        if FEATURE_TO_CATEGORY[feature_name] == category_name
    )
    for category_name in CATEGORY_NAMES
}
CATEGORY_MEMBER_NAMES: Mapping[str, tuple[str, ...]] = {
    category_name: tuple(
        MODEL_FEATURE_COLUMNS[index]
        for index in CATEGORY_MEMBER_INDICES[category_name]
    )
    for category_name in CATEGORY_NAMES
}


def read_authoritative_feature_names(path: Path) -> tuple[str, ...]:
    """Read and validate the frozen Stage C ``feature_names.json`` artifact."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    names = payload.get("feature_names")
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError("Stage C feature_names.json must contain a string feature_names list")
    if payload.get("n_features") != len(names):
        raise ValueError("Stage C feature_names.json n_features does not match its list")
    if tuple(names) != MODEL_FEATURE_COLUMNS:
        raise ValueError("Stage C artifact feature order does not match the canonical schema")
    _validate_assignment(names)
    return tuple(names)


def build_manifest(authoritative_feature_names: Sequence[str]) -> dict[str, Any]:
    """Build the Stage D1 manifest without reading data or writing files."""

    feature_names = tuple(authoritative_feature_names)
    if feature_names != MODEL_FEATURE_COLUMNS:
        raise ValueError("authoritative feature order differs from frozen Stage C schema")
    _validate_assignment(feature_names)

    feature_to_category = {
        feature_name: FEATURE_TO_CATEGORY[feature_name] for feature_name in feature_names
    }
    feature_to_category_id = {
        feature_name: CATEGORY_IDS[feature_to_category[feature_name]]
        for feature_name in feature_names
    }
    index_to_category_id = [
        feature_to_category_id[feature_name] for feature_name in feature_names
    ]
    categories = []
    for category_id, category_name in enumerate(CATEGORY_NAMES):
        member_indices = [
            index
            for index, feature_name in enumerate(feature_names)
            if feature_to_category[feature_name] == category_name
        ]
        categories.append(
            {
                "id": category_id,
                "name": category_name,
                "member_count": len(member_indices),
                "member_indices": member_indices,
                "member_names": [feature_names[index] for index in member_indices],
            }
        )

    content: dict[str, Any] = {
        "schema": "stage_d1_feature_taxonomy",
        "taxonomy_version": TAXONOMY_VERSION,
        "hash_algorithm": "sha256",
        "canonical_serialization": "utf8-json-sort-keys-compact-v1",
        "stage_c_feature_order": {
            "source": "data/processed/preprocessed/feature_names.json",
            "n_features": len(feature_names),
            "sha256": _sha256_canonical(list(feature_names)),
            "feature_names": list(feature_names),
        },
        "category_names": list(CATEGORY_NAMES),
        "category_ids": dict(CATEGORY_IDS),
        "expected_category_counts": dict(EXPECTED_CATEGORY_COUNTS),
        "feature_to_category": feature_to_category,
        "feature_to_category_id": feature_to_category_id,
        "feature_index_to_category_id": index_to_category_id,
        "categories": categories,
    }
    content["taxonomy_sha256"] = _sha256_canonical(content)
    return content


def manifest_bytes(authoritative_feature_names: Sequence[str]) -> bytes:
    """Return the stable, human-readable bytes persisted for the manifest."""

    manifest = build_manifest(authoritative_feature_names)
    return (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def write_manifest(authoritative_path: Path, output_path: Path) -> dict[str, Any]:
    """Validate Stage C authority and write only the requested D1 manifest."""

    feature_names = read_authoritative_feature_names(authoritative_path)
    data = manifest_bytes(feature_names)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(data)
    return json.loads(data)


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Reject a manifest that is not a byte-independent D1 regeneration."""

    supplied_hash = manifest.get("taxonomy_sha256")
    if not isinstance(supplied_hash, str):
        raise ValueError("manifest taxonomy_sha256 is missing")
    unhashed = dict(manifest)
    del unhashed["taxonomy_sha256"]
    if _sha256_canonical(unhashed) != supplied_hash:
        raise ValueError("manifest taxonomy_sha256 does not match its canonical content")
    expected = build_manifest(MODEL_FEATURE_COLUMNS)
    if dict(manifest) != expected:
        raise ValueError("manifest does not match the frozen Stage D1 taxonomy")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the deterministic Stage D1 taxonomy")
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/processed/preprocessed/feature_names.json"),
        help="frozen Stage C feature_names.json authority",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/stage_d1/feature_taxonomy.json"),
        help="Stage D1 manifest destination",
    )
    args = parser.parse_args(argv)
    manifest = write_manifest(args.features, args.output)
    print(f"wrote {args.output}")
    print(f"taxonomy_sha256={manifest['taxonomy_sha256']}")
    print(f"stage_c_feature_order_sha256={manifest['stage_c_feature_order']['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
