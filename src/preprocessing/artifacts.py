"""Write and verify all Stage C output artefacts.

Artefact layout under ``<processed_data_dir>/<output_subdir>/``:

    feature_names.json          -- ordered list of 78 model feature column names
    class_mapping.json          -- int ↔ label mapping + absent-class annotations
    binary_class_mapping.json   -- canonical E3 target: Benign→0, 14 attacks→1
    cleaning_stats.json         -- canonical duplicate accounting (levels 1-3)
    duplicate_accounting.json   -- the same accounting as a flat report table
    class_eligibility.json      -- E2 per-class eligibility + exact reasons
    class_availability.json     -- E1 train/val/test class availability matrix
    scaler_state.json           -- E1/E3 train-fitted transform state + provenance
    exp1/                       -- strict chronological multiclass DIAGNOSTIC
        X_train.npy  y_train.npy
        X_val.npy    y_val.npy
        X_test.npy   y_test.npy
        split_metadata.json     -- counts, distributions, availability matrix
        scaler_state.json       -- copy of the E1-train-fitted state
    exp2/                       -- multiclass separability (PRIMARY A)
        X_train.npy  y_train.npy
        X_val.npy    y_val.npy
        X_test.npy   y_test.npy
        split_metadata.json     -- counts, distributions, eligibility table
        scaler_state.json       -- independently fit on E2 train only
    exp3/                       -- strict chronological binary (PRIMARY B)
        X_train.npy  y_train.npy   (y is binary: 0 = Benign, 1 = Attack)
        X_val.npy    y_val.npy
        X_test.npy   y_test.npy
        split_metadata.json     -- counts, binary + multiclass composition
        scaler_state.json       -- E1-train-fitted state (same feature split)

Arrays are stored as .npy files (NumPy binary) for fast, portable loading.
No Parquet is used in the temporal pipeline.

Integrity
---------
After writing, each .npy file is reloaded and its shape is checked against the
expected dimensions.  If the re-read shape does not match, a ``RuntimeError``
is raised so the caller knows the write failed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.preprocessing.labels import binary_mapping_dict, class_mapping_dict
from src.preprocessing.schema import MODEL_FEATURE_COLUMNS
from src.preprocessing.transforms import FitTransformState, apply_transforms


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def write_feature_names(out_dir: Path) -> Path:
    """Write ``feature_names.json`` to *out_dir*."""
    path = out_dir / "feature_names.json"
    path.write_text(
        json.dumps(
            {"n_features": len(MODEL_FEATURE_COLUMNS), "feature_names": MODEL_FEATURE_COLUMNS},
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def write_class_mapping(out_dir: Path) -> Path:
    """Write ``class_mapping.json`` to *out_dir*."""
    path = out_dir / "class_mapping.json"
    path.write_text(json.dumps(class_mapping_dict(), indent=2), encoding="utf-8")
    return path


def write_cleaning_stats(stats_dict: Dict[str, Any], out_dir: Path) -> Path:
    """Write ``cleaning_stats.json`` to *out_dir*."""
    path = out_dir / "cleaning_stats.json"
    path.write_text(json.dumps(stats_dict, indent=2), encoding="utf-8")
    return path


def write_binary_class_mapping(out_dir: Path) -> Path:
    """Write ``binary_class_mapping.json`` (the canonical E3 target) to *out_dir*."""
    path = out_dir / "binary_class_mapping.json"
    path.write_text(json.dumps(binary_mapping_dict(), indent=2), encoding="utf-8")
    return path


def write_duplicate_accounting(stats_dict: Dict[str, Any], out_dir: Path) -> Path:
    """Write ``duplicate_accounting.json``: the canonical accounting, flattened.

    ``cleaning_stats.json`` is the full record.  This file restates the same
    numbers as an ordered report table plus the reconciliation identities, so
    the accounting can be read without navigating the whole stats blob.  Both
    files come from the single duplicate-classification pass, so they cannot
    disagree.
    """
    table = stats_dict.get("accounting_table", [])
    path = out_dir / "duplicate_accounting.json"
    payload: Dict[str, Any] = {
        "source": "single canonical duplicate-classification pass (cleaning.build_keep_index)",
        "computed_before_windowing": True,
        "accounting_table": table,
        "identities": stats_dict.get("identities", {}),
        "policy": stats_dict.get("policy", {}),
        "reconciliation": {
            "usable_rows": stats_dict.get("usable_rows"),
            "sum_of_final_states": (
                (stats_dict.get("conflicting_rows") or 0)
                + (stats_dict.get("exact_duplicate_rows_removed") or 0)
                + (stats_dict.get("retained_rows") or 0)
            ),
            "retained_rows": stats_dict.get("retained_rows"),
            "retained_breakdown": (
                (stats_dict.get("temporal_duplicate_rows_retained") or 0)
                + (stats_dict.get("unique_rows") or 0)
            ),
            "clean_rows": stats_dict.get("clean_rows"),
            "sum_of_retained_categories": (
                (stats_dict.get("temporal_duplicate_rows_retained") or 0)
                + (stats_dict.get("unique_rows") or 0)
            ),
        },
    }
    rec = payload["reconciliation"]
    payload["reconciles"] = bool(
        rec["usable_rows"] == rec["sum_of_final_states"]
        and rec["retained_rows"] == rec["retained_breakdown"]
        and rec["clean_rows"] == rec["sum_of_retained_categories"]
    )
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_class_eligibility(eligibility: Dict[str, Any], out_dir: Path) -> Path:
    """Write ``class_eligibility.json`` (E2 primary-reporting eligibility)."""
    path = out_dir / "class_eligibility.json"
    path.write_text(json.dumps(eligibility, indent=2), encoding="utf-8")
    return path


def write_class_availability(availability: Dict[str, Any], out_dir: Path) -> Path:
    """Write ``class_availability.json`` (E1 train/val/test availability matrix)."""
    path = out_dir / "class_availability.json"
    path.write_text(json.dumps(availability, indent=2), encoding="utf-8")
    return path


def write_scaler_state(state_dict: Dict[str, Any], out_dir: Path) -> Path:
    """Write ``scaler_state.json`` to *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "scaler_state.json"
    path.write_text(json.dumps(state_dict, indent=2), encoding="utf-8")
    return path


def write_windowing_stats(stats_dict: Dict[str, Any], out_dir: Path) -> Path:
    """Write ``windowing_stats.json`` to *out_dir*."""
    path = out_dir / "windowing_stats.json"
    path.write_text(json.dumps(stats_dict, indent=2), encoding="utf-8")
    return path


def write_split(
    exp_dir: Path,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    metadata: Dict[str, Any],
    verify: bool = True,
) -> Dict[str, Path]:
    """Write the six .npy arrays and ``split_metadata.json`` for one experiment.

    Parameters
    ----------
    exp_dir:
        Destination directory (created if absent).
    verify:
        If ``True``, reload each .npy and confirm the shape matches what was
        written.

    Returns
    -------
    Dict mapping artefact name to its path.
    """
    exp_dir.mkdir(parents=True, exist_ok=True)

    arrays = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }

    written: Dict[str, Path] = {}
    for name, arr in arrays.items():
        path = exp_dir / f"{name}.npy"
        np.save(path, arr)
        written[name] = path

    meta_path = exp_dir / "split_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    written["split_metadata"] = meta_path

    if verify:
        for name, arr in arrays.items():
            loaded = np.load(written[name])
            if loaded.shape != arr.shape:
                raise RuntimeError(
                    f"Artefact integrity check failed: {written[name]} has shape "
                    f"{loaded.shape}, expected {arr.shape}"
                )

    return written


def write_split_streaming(
    exp_dir: Path,
    X_source: np.ndarray,
    selectors: Dict[str, object],
    y_splits: Dict[str, np.ndarray],
    state: FitTransformState,
    metadata: Dict[str, Any],
    chunk_windows: int = 25_000,
    reuse_feature_paths: Optional[Dict[str, Path]] = None,
    verify: bool = True,
) -> Dict[str, Path]:
    """Transform and write a split without materialising full X arrays in RAM.

    ``selectors`` maps ``train``/``val``/``test`` to either a contiguous slice
    (E1/E3) or an int64 window-index array (E2).  Each output X file is created
    as an ``.npy`` memmap and filled in bounded chunks.  E3 may reuse E1's
    already-written feature arrays as NTFS hard links because the feature
    membership and transform state are identical; only y differs.
    """
    if chunk_windows <= 0:
        raise ValueError("chunk_windows must be positive")
    exp_dir.mkdir(parents=True, exist_ok=True)
    reuse_feature_paths = reuse_feature_paths or {}
    written: Dict[str, Path] = {}

    for split in ("train", "val", "test"):
        selector = selectors[split]
        y = np.asarray(y_splits[split], dtype=np.int16)
        n = len(y)

        y_path = exp_dir / f"y_{split}.npy"
        np.save(y_path, y)
        written[f"y_{split}"] = y_path

        X_path = exp_dir / f"X_{split}.npy"
        reuse = reuse_feature_paths.get(split)
        if reuse is not None:
            if X_path.exists():
                X_path.unlink()
            os.link(reuse, X_path)
        else:
            out = np.lib.format.open_memmap(
                X_path,
                mode="w+",
                dtype=np.float32,
                shape=(n, X_source.shape[1], X_source.shape[2]),
            )
            for start in range(0, n, chunk_windows):
                end = min(start + chunk_windows, n)
                if isinstance(selector, slice):
                    if selector.step not in (None, 1):
                        raise ValueError("streaming slice selectors must have step 1")
                    base = 0 if selector.start is None else int(selector.start)
                    block = X_source[base + start : base + end]
                else:
                    idx = np.asarray(selector, dtype=np.int64)[start:end]
                    block = X_source[idx]
                out[start:end] = apply_transforms(block, state)
            out.flush()
            del out
        written[f"X_{split}"] = X_path

    meta_path = exp_dir / "split_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    written["split_metadata"] = meta_path

    if verify:
        for split in ("train", "val", "test"):
            expected_n = len(y_splits[split])
            X = np.load(written[f"X_{split}"], mmap_mode="r")
            y = np.load(written[f"y_{split}"], mmap_mode="r")
            if X.shape != (expected_n, X_source.shape[1], X_source.shape[2]):
                raise RuntimeError(
                    f"Artefact integrity check failed: {written[f'X_{split}']} "
                    f"has shape {X.shape}"
                )
            if y.shape != (expected_n,):
                raise RuntimeError(
                    f"Artefact integrity check failed: {written[f'y_{split}']} "
                    f"has shape {y.shape}"
                )

    return written


#: Global (non-experiment) artefact filenames, in report order.
GLOBAL_ARTEFACT_FILES = (
    "feature_names.json",
    "class_mapping.json",
    "binary_class_mapping.json",
    "cleaning_stats.json",
    "duplicate_accounting.json",
    "class_eligibility.json",
    "class_availability.json",
    "scaler_state.json",
    "windowing_stats.json",
)


def collect_artefact_summary(
    out_dir: Path,
    exp1_written: Dict[str, Path],
    exp2_written: Dict[str, Path],
    exp3_written: Optional[Dict[str, Path]] = None,
    extra_paths: Optional[List[Path]] = None,
) -> Dict[str, Any]:
    """Build a human-readable summary of all written artefacts.

    *exp3_written* is optional so a run that produced only E1 and E2 (e.g. a
    unit test) still summarises cleanly; when it is absent the ``exp3`` key is
    omitted rather than reported as empty.
    """
    def _entry(p: Path) -> Dict[str, Any]:
        stat = p.stat()
        return {"path": str(p.relative_to(out_dir.parent)), "size_bytes": stat.st_size}

    experiments: List[tuple] = [("exp1", exp1_written), ("exp2", exp2_written)]
    if exp3_written is not None:
        experiments.append(("exp3", exp3_written))

    summary: Dict[str, Any] = {
        "output_root": str(out_dir),
        "global": {},
    }
    for exp_key, _ in experiments:
        summary[exp_key] = {}

    for fname in GLOBAL_ARTEFACT_FILES:
        p = out_dir / fname
        if p.is_file():
            summary["global"][fname] = _entry(p)

    for exp_key, written in experiments:
        for name, path in written.items():
            summary[exp_key][name] = _entry(path)

    if extra_paths:
        summary["extra"] = {p.name: _entry(p) for p in extra_paths if p.is_file()}

    return summary
