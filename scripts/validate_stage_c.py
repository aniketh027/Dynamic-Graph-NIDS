"""Read-only validation for completed Stage C artefacts.

The validator memory-maps the large NumPy arrays and scans them in bounded
chunks.  It never modifies raw data or Stage C outputs.  When a second output
directory is supplied, every relative artefact path and SHA-256 digest must
match exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.labels import (  # noqa: E402
    BINARY_ATTACK,
    BINARY_BENIGN,
    INT_TO_LABEL,
    LABEL_TO_INT,
    LABEL_VOCABULARY,
)
from src.preprocessing.transforms import FitTransformState  # noqa: E402


class ValidationError(RuntimeError):
    """Raised when a persisted Stage C contract check fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    _require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def _sha256(path: Path, block_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def _file_digests(root: Path) -> Dict[str, str]:
    return {
        p.relative_to(root).as_posix(): _sha256(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _finite_in_chunks(array: np.ndarray, chunk_windows: int) -> bool:
    for start in range(0, len(array), chunk_windows):
        if not np.isfinite(array[start : start + chunk_windows]).all():
            return False
    return True


def _multiclass_distribution(y: np.ndarray) -> Dict[str, int]:
    codes, counts = np.unique(y, return_counts=True)
    return {INT_TO_LABEL[int(code)]: int(count) for code, count in zip(codes, counts)}


def _binary_distribution(y: np.ndarray) -> Dict[str, int]:
    codes, counts = np.unique(y, return_counts=True)
    by_code = {int(code): int(count) for code, count in zip(codes, counts)}
    return {
        "Benign": by_code.get(BINARY_BENIGN, 0),
        "Attack": by_code.get(BINARY_ATTACK, 0),
    }


def _validate_transform_state(path: Path, expected_fit: str) -> Dict[str, Any]:
    payload = _load_json(path)
    state = FitTransformState.from_dict(payload)
    for name in (
        "center",
        "scale",
        "train_median",
        "train_max_finite",
        "train_min_finite",
    ):
        values = getattr(state, name)
        _require(values.shape == (78,), f"{path}: {name} shape is {values.shape}")
        _require(bool(np.isfinite(values).all()), f"{path}: {name} is non-finite")
    provenance = payload.get("fit_provenance", {})
    _require(
        provenance.get("fit_experiment") == expected_fit,
        f"{path}: expected fit_experiment={expected_fit}",
    )
    _require(provenance.get("fit_split") == "train", f"{path}: fit split is not train")
    return {
        "loadable": True,
        "fit_experiment": provenance["fit_experiment"],
        "fit_split": provenance["fit_split"],
        "fit_window_count": int(provenance["fit_window_count"]),
        "all_statistics_finite": True,
    }


def _validate_experiment(
    output_dir: Path,
    exp_name: str,
    n_windows: int,
    chunk_windows: int,
) -> Dict[str, Any]:
    exp_dir = output_dir / exp_name
    metadata = _load_json(exp_dir / "split_metadata.json")
    expected_experiment = {
        "exp1": "1_global_chronological",
        "exp2": "2_per_class_chronological",
        "exp3": "3_binary_chronological",
    }[exp_name]
    _require(metadata.get("experiment") == expected_experiment, f"{exp_name}: wrong identity")
    _require(metadata.get("n_windows_total") == n_windows, f"{exp_name}: total mismatch")

    split_report: Dict[str, Any] = {}
    expected_fit = "E2" if exp_name == "exp2" else "E1"
    state_report = _validate_transform_state(exp_dir / "scaler_state.json", expected_fit)
    preprocessing = metadata.get("preprocessing", {})
    _require(preprocessing.get("fit_experiment") == expected_fit, f"{exp_name}: fit provenance")
    _require(preprocessing.get("fit_split") == "train", f"{exp_name}: fit split")
    _require(
        preprocessing.get("validation_or_test_used_for_fit") is False,
        f"{exp_name}: validation/test fitting flag is not false",
    )

    total = 0
    for split in ("train", "val", "test"):
        X_path = exp_dir / f"X_{split}.npy"
        y_path = exp_dir / f"y_{split}.npy"
        X = np.load(X_path, mmap_mode="r")
        y = np.load(y_path, mmap_mode="r")
        expected_n = int(metadata[f"n_{split}"])

        _require(X.shape == (expected_n, 5, 78), f"{X_path}: shape {X.shape}")
        _require(y.shape == (expected_n,), f"{y_path}: shape {y.shape}")
        _require(X.dtype == np.float32, f"{X_path}: dtype {X.dtype}")
        _require(y.dtype == np.int16, f"{y_path}: dtype {y.dtype}")
        _require(_finite_in_chunks(X, chunk_windows), f"{X_path}: NaN/inf found")

        codes = set(np.unique(y).tolist())
        if exp_name == "exp3":
            _require(codes <= {0, 1}, f"{y_path}: non-binary codes {codes}")
            measured_dist = _binary_distribution(y)
            expected_dist = metadata[f"{split}_binary_distribution"]
        else:
            _require(codes <= set(INT_TO_LABEL), f"{y_path}: unknown codes {codes}")
            measured_dist = _multiclass_distribution(y)
            expected_dist = metadata[f"{split}_class_distribution"]
        _require(measured_dist == expected_dist, f"{exp_name}/{split}: distribution mismatch")

        split_report[split] = {
            "X_shape": list(X.shape),
            "y_shape": list(y.shape),
            "X_dtype": str(X.dtype),
            "y_dtype": str(y.dtype),
            "all_X_finite": True,
            "label_codes": sorted(int(code) for code in codes),
            "label_distribution": measured_dist,
        }
        total += expected_n

    _require(total == n_windows, f"{exp_name}: split counts do not sum to total")
    _require(
        state_report["fit_window_count"] == int(metadata["n_train"]),
        f"{exp_name}: transform fit count does not equal training count",
    )
    return {
        "metadata_identity": expected_experiment,
        "split_counts_reconcile": True,
        "preprocessing": state_report,
        "splits": split_report,
    }


def _verify_e1_e3_equivalence(output_dir: Path, chunk_windows: int) -> Dict[str, Any]:
    for split in ("train", "val", "test"):
        e1_X = output_dir / "exp1" / f"X_{split}.npy"
        e3_X = output_dir / "exp3" / f"X_{split}.npy"
        _require(_sha256(e1_X) == _sha256(e3_X), f"E1/E3 X_{split} differs")

        y1 = np.load(output_dir / "exp1" / f"y_{split}.npy", mmap_mode="r")
        y3 = np.load(output_dir / "exp3" / f"y_{split}.npy", mmap_mode="r")
        _require(y1.shape == y3.shape, f"E1/E3 y_{split} shape differs")
        for start in range(0, len(y1), chunk_windows):
            mc = np.asarray(y1[start : start + chunk_windows])
            expected = np.where(mc == LABEL_TO_INT["Benign"], 0, 1).astype(np.int16)
            _require(
                np.array_equal(expected, y3[start : start + chunk_windows]),
                f"E3 y_{split} is not the binary mapping of E1",
            )
    return {
        "feature_arrays_sha256_identical": True,
        "binary_targets_match_multiclass_targets": True,
    }


def _raw_fingerprints(raw_dir: Path) -> Dict[str, Dict[str, Any]]:
    paths = sorted(raw_dir.glob("*.csv"))
    _require(len(paths) == 10, f"Expected 10 raw CSVs in {raw_dir}, found {len(paths)}")
    return {
        path.name: {
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "sha256": _sha256(path),
        }
        for path in paths
    }


def validate(
    output_dir: Path,
    raw_dir: Path,
    compare_dir: Optional[Path] = None,
    chunk_windows: int = 100_000,
) -> Dict[str, Any]:
    output_dir = output_dir.resolve()
    raw_dir = raw_dir.resolve()
    _require(output_dir.is_dir(), f"Stage C output directory not found: {output_dir}")

    cleaning = _load_json(output_dir / "cleaning_stats.json")
    duplicates = _load_json(output_dir / "duplicate_accounting.json")
    windowing = _load_json(output_dir / "windowing_stats.json")

    usable = int(cleaning["usable_rows"])
    retained = int(cleaning["retained_rows"])
    _require(
        usable
        == int(cleaning["raw_rows"])
        - int(cleaning["embedded_header_rows"])
        - int(cleaning["unknown_label_rows"]),
        "Raw/header/unknown-label accounting does not reconcile",
    )
    _require(
        usable
        == int(cleaning["conflicting_rows"])
        + int(cleaning["exact_duplicate_rows_removed"])
        + retained,
        "Three final duplicate states do not reconcile",
    )
    _require(
        retained
        == int(cleaning["temporal_duplicate_rows_retained"])
        + int(cleaning["unique_rows"]),
        "Retained temporal/unique breakdown does not reconcile",
    )
    _require(retained == int(cleaning["clean_rows"]), "Retained and clean rows differ")
    _require(duplicates.get("reconciles") is True, "duplicate_accounting.json does not reconcile")

    T = int(windowing["window_size"])
    n_windows = int(windowing["n_windows"])
    tail = int(windowing["tail_rows_discarded"])
    _require(T == 5, f"Expected T=5, found T={T}")
    _require(n_windows == retained // T, "Window total is not floor(clean/T)")
    _require(tail == retained % T, "Trailing-row count is inconsistent")

    mapping = _load_json(output_dir / "class_mapping.json")
    binary_mapping = _load_json(output_dir / "binary_class_mapping.json")
    _require(mapping.get("n_classes") == 15, "Multiclass mapping is not 15-way")
    _require(
        mapping.get("label_to_int") == LABEL_TO_INT,
        "Persisted multiclass mapping differs from the canonical mapping",
    )
    _require(
        set(binary_mapping.get("label_to_binary", {})) == set(LABEL_VOCABULARY),
        "Binary mapping omits or adds labels",
    )
    _require(binary_mapping["label_to_binary"]["Benign"] == 0, "Benign is not binary 0")
    _require(
        all(
            binary_mapping["label_to_binary"][label] == 1
            for label in LABEL_VOCABULARY if label != "Benign"
        ),
        "At least one attack label is not binary 1",
    )

    experiments = {
        exp: _validate_experiment(output_dir, exp, n_windows, chunk_windows)
        for exp in ("exp1", "exp2", "exp3")
    }

    artefact_sha256 = _file_digests(output_dir)
    comparison: Optional[Dict[str, Any]] = None
    if compare_dir is not None:
        compare_dir = compare_dir.resolve()
        _require(compare_dir.is_dir(), f"Comparison directory not found: {compare_dir}")
        compare_sha256 = _file_digests(compare_dir)
        missing = sorted(set(artefact_sha256) - set(compare_sha256))
        extra = sorted(set(compare_sha256) - set(artefact_sha256))
        differing = sorted(
            path for path in set(artefact_sha256) & set(compare_sha256)
            if artefact_sha256[path] != compare_sha256[path]
        )
        _require(not missing and not extra and not differing, (
            f"Full-run outputs differ: missing={missing}, extra={extra}, differing={differing}"
        ))
        comparison = {
            "comparison_dir": str(compare_dir),
            "relative_file_sets_identical": True,
            "all_sha256_identical": True,
            "n_files_compared": len(artefact_sha256),
            "missing": [],
            "extra": [],
            "differing": [],
        }

    return {
        "status": "PASS",
        "output_dir": str(output_dir),
        "duplicate_accounting": {
            "raw_rows": int(cleaning["raw_rows"]),
            "embedded_header_rows": int(cleaning["embedded_header_rows"]),
            "usable_rows": usable,
            "conflicting_rows": int(cleaning["conflicting_rows"]),
            "conflicting_groups": int(cleaning["conflicting_groups"]),
            "exact_duplicate_rows_removed": int(cleaning["exact_duplicate_rows_removed"]),
            "retained_rows": retained,
            "temporal_duplicate_rows_retained": int(
                cleaning["temporal_duplicate_rows_retained"]
            ),
            "unique_rows": int(cleaning["unique_rows"]),
            "final_clean_rows": int(cleaning["clean_rows"]),
            "reconciles": True,
        },
        "windowing": {
            "window_size": T,
            "n_windows": n_windows,
            "tail_rows_discarded": tail,
            "final_flow_labeling": windowing.get("label_source"),
        },
        "label_mappings_loadable": True,
        "experiments": experiments,
        "e1_e3_equivalence": _verify_e1_e3_equivalence(output_dir, chunk_windows),
        "artefact_sha256": artefact_sha256,
        "comparison": comparison,
        "raw_file_fingerprints": _raw_fingerprints(raw_dir),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--compare-dir", type=Path, default=None)
    parser.add_argument("--chunk-windows", type=int, default=100_000)
    parser.add_argument("--report", type=Path, default=None)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = validate(
            args.output_dir,
            args.raw_dir,
            compare_dir=args.compare_dir,
            chunk_windows=args.chunk_windows,
        )
    except (OSError, ValueError, KeyError, ValidationError) as exc:
        print(f"Stage C validation FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    rendered = json.dumps(report, indent=2)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
