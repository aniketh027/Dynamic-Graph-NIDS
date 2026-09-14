"""Top-level orchestrator for Stage C preprocessing.

Runs the full pipeline end-to-end:

1. Discover day CSV files in capture-date order.
2. Global cleaning: one canonical, mutually exclusive duplicate-classification
   pass (level 1 conflicting → level 2 exact → level 3 temporal).  Conflicting
   and exact-duplicate rows are excluded; temporal duplicates are retained.
   The accounting is reconciled before anything downstream runs.
3. Assemble the global capture-order sequence and build T=5 non-overlapping
   windows.
4. Experiment 1 — strict chronological multiclass DIAGNOSTIC (70/10/20).
5. Fit RobustScaler + finite-replacement statistics on Experiment 1 train only.
6. Apply that state to E1 (and later E3, whose feature split is identical).
7. Experiment 3 — strict chronological BINARY, PRIMARY B (same ordering and cut
   points as E1; target is Benign→0 / Attack→1).
8. Experiment 2 — per-class chronological multiclass, PRIMARY A, plus the
   class eligibility table.  Fit a separate transform state on E2 train only.
9. Write global artefacts.

E1 and E3 share an E1-train-fitted state because their feature arrays and cut
points are identical.  E2 has a different training membership and therefore
gets an independently fitted E2-train state.  This prevents E1-train windows
that belong to E2 validation/test from influencing E2 preprocessing.

Nothing under ``data/raw/`` is modified.  No graph edges, no model training,
no SMOTE or synthetic flows are produced here.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from src.config import Config
from src.preprocessing import artifacts as art
from src.preprocessing.cleaning import build_keep_index
from src.preprocessing.splitting import (
    ELIGIBILITY_MIN_TEST,
    ELIGIBILITY_MIN_TRAIN,
    ELIGIBILITY_MIN_VAL,
    experiment1_global_chronological,
    experiment2_per_class_chronological_indices,
    experiment3_binary_chronological,
)
from src.preprocessing.transforms import fit_transforms, fit_transforms_indexed
from src.preprocessing.windowing import build_windows


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

#: Capture-date order: 10 filenames in the exact measured sequence.
_CORPUS_ORDER: Tuple[str, ...] = (
    "02-14-2018.csv",
    "02-15-2018.csv",
    "02-16-2018.csv",
    "02-20-2018.csv",
    "02-21-2018.csv",
    "02-22-2018.csv",
    "02-23-2018.csv",
    "02-28-2018.csv",
    "03-01-2018.csv",
    "03-02-2018.csv",
)


def discover_csvs(cfg: Config) -> List[Path]:
    """Return day CSVs in deterministic capture-date order.

    Raises ``FileNotFoundError`` if a file in the canonical sequence is absent.
    """
    csv_dir = cfg.paths.raw_data_dir / cfg.profiling.csv_subdir
    paths: List[Path] = []
    for name in _CORPUS_ORDER:
        p = csv_dir / name
        if not p.is_file():
            raise FileNotFoundError(
                f"Expected corpus file not found: {p}\n"
                f"Raw data directory: {csv_dir}"
            )
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """All artefacts produced by the pipeline run."""

    out_dir: Path
    #: Paths written during this run.
    written: Dict[str, Path] = field(default_factory=dict)
    #: Human-readable summary dict.
    summary: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    cfg: Config,
    row_limit: Optional[int] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> PipelineResult:
    """Execute the full Stage C preprocessing pipeline.

    Parameters
    ----------
    cfg:
        Project configuration.  The ``preprocessing`` section drives chunk size,
        rare-class threshold and output directory.
    row_limit:
        Cap on rows read per file (debug runs only).
    progress:
        Optional callable for status messages (e.g. ``print``).
    """
    def _log(msg: str) -> None:
        if progress:
            progress(msg)

    def _close_memmap(arr: object) -> None:
        """Flush and close an ndarray's underlying NumPy memmap, if any.

        Reshape/slice views keep the Windows file mapping alive through their
        ``base`` chain.  Explicitly closing the mapping after all artefacts have
        been written makes deletion of the pipeline-owned scratch file
        deterministic instead of relying on garbage-collection timing.
        """
        seen: set[int] = set()
        current = arr
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if isinstance(current, np.memmap):
                current.flush()
                mmap_obj = getattr(current, "_mmap", None)
                if mmap_obj is not None and not mmap_obj.closed:
                    mmap_obj.close()
                    return
            current = getattr(current, "base", None)

    pc = cfg.preprocessing
    out_dir = cfg.paths.processed_data_dir / pc.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    result = PipelineResult(out_dir=out_dir)
    work_dir = out_dir / ".stage_c_work"

    # ------------------------------------------------------------------
    # 1. Discover CSVs
    # ------------------------------------------------------------------
    _log("=" * 70)
    _log("Stage C preprocessing pipeline")
    _log("=" * 70)
    csv_paths = discover_csvs(cfg)
    _log(f"Found {len(csv_paths)} day CSVs in capture-date order.")

    # ------------------------------------------------------------------
    # 2. Global cleaning — the single canonical duplicate pass
    # ------------------------------------------------------------------
    _log("\n--- Step 2: global cleaning (canonical duplicate cascade) ---")
    cleaning_result = build_keep_index(
        csv_paths,
        chunk_size=pc.chunk_size,
        row_limit=row_limit,
        progress=_log,
    )
    stats = cleaning_result.stats
    # build_keep_index already called stats.validate(); repeat it here so a
    # future refactor that loosens the internal check still cannot let an
    # inconsistent accounting reach the artefacts.
    stats.validate()

    _log("  canonical duplicate accounting (computed before windowing):")
    for row in stats.accounting_table():
        _log(f"    {row['quantity']:<34} {row['count']:>14,}  [{row['disposition']}]")
    _log(
        f"    conflicting feature-vector groups  {stats.conflicting_groups:>14,}"
    )

    stats_dict = stats.to_dict()
    if pc.save_cleaning_stats:
        p = art.write_cleaning_stats(stats_dict, out_dir)
        result.written["cleaning_stats.json"] = p
        _log(f"  wrote {p.name}")

    p = art.write_duplicate_accounting(stats_dict, out_dir)
    result.written["duplicate_accounting.json"] = p
    _log(f"  wrote {p.name}")

    # ------------------------------------------------------------------
    # 3. Windowing
    # ------------------------------------------------------------------
    T = cfg.window_size
    _log(f"\n--- Step 3: T={T} non-overlapping windowing ---")
    window_result = build_windows(
        csv_paths,
        kept_per_file=cleaning_result.kept_per_file,
        window_size=T,
        chunk_size=pc.chunk_size,
        progress=_log,
        work_dir=work_dir,
    )
    wstats = window_result.stats
    _log(
        f"  windows     : {wstats.n_windows:,}\n"
        f"  tail discard: {wstats.tail_rows_discarded} rows\n"
        f"  X shape     : {window_result.X.shape}\n"
        f"  y shape     : {window_result.y.shape}"
    )

    # The keep-index is consumed; release it before the split tensors appear.
    cleaning_result.kept_per_file = []

    p = art.write_windowing_stats(wstats.to_dict(), out_dir)
    result.written["windowing_stats.json"] = p

    X_all = window_result.X   # (N, T, 78) float32
    y_all = window_result.y   # (N,)       int16 multiclass codes

    # ------------------------------------------------------------------
    # 4. Experiment 1 — strict chronological multiclass diagnostic (unscaled)
    # ------------------------------------------------------------------
    _log("\n--- Step 4: Experiment 1 — strict chronological multiclass diagnostic ---")
    e1 = experiment1_global_chronological(
        X_all, y_all,
        train_ratio=cfg.split.train,
        val_ratio=cfg.split.validation,
        rare_class_min_test_windows=pc.rare_class_min_test_windows,
    )
    a1 = e1.arrays
    _log(
        f"  train={len(a1.y_train):,}  val={len(a1.y_val):,}  "
        f"test={len(a1.y_test):,}"
    )
    availability = e1.metadata["class_availability_matrix"]
    unseen = availability["classes_unseen_during_training"]
    _log(
        f"  classes unseen during training: {len(unseen)}"
        + (f" -> {unseen}" if unseen else "")
    )
    _log(
        "  E1 is a diagnostic, not a primary supervised multiclass benchmark; "
        "per-class supervised evaluation is impossible for the unseen classes."
    )
    if e1.metadata.get("rare_classes"):
        _log(
            f"  rare classes (<{pc.rare_class_min_test_windows} test windows): "
            f"{e1.metadata['rare_classes']}"
        )

    p = art.write_class_availability(availability, out_dir)
    result.written["class_availability.json"] = p
    _log(f"  wrote {p.name}")

    # ------------------------------------------------------------------
    # 5. Fit transforms on E1 train split
    # ------------------------------------------------------------------
    _log("\n--- Step 5: fitting transforms on E1 train split ---")
    transform_state = fit_transforms(a1.X_train)
    _log(f"  RobustScaler fitted on {len(a1.y_train):,} train windows")

    shared_transform_payload = transform_state.to_dict()
    shared_transform_payload["fit_provenance"] = {
        "fit_experiment": "E1",
        "fit_split": "train",
        "fit_window_count": int(len(a1.y_train)),
        "fit_flow_value_count": int(len(a1.y_train) * T),
        "applies_to_experiments": ["E1", "E3"],
        "reason": (
            "E1 and E3 use identical globally chronological feature splits; "
            "only their target encodings differ. No validation or test window "
            "is used to learn finite replacements or scaling statistics."
        ),
    }
    e1.metadata["preprocessing"] = {
        "state_artifact": "exp1/scaler_state.json",
        "fit_experiment": "E1",
        "fit_split": "train",
        "fit_window_count": int(len(a1.y_train)),
        "validation_or_test_used_for_fit": False,
    }

    # Kept at the output root for backward compatibility; provenance makes its
    # E1/E3 scope explicit.  Each experiment also receives its own state file.
    p = art.write_scaler_state(shared_transform_payload, out_dir)
    result.written["scaler_state.json"] = p
    _log(f"  wrote {p.name}")

    # ------------------------------------------------------------------
    # 6. Apply transforms to E1 splits and write E1
    # ------------------------------------------------------------------
    _log("\n--- Step 6: applying transforms to E1 splits ---")
    exp1_dir = out_dir / "exp1"
    p = art.write_scaler_state(shared_transform_payload, exp1_dir)
    train_end = int(e1.metadata["split_boundaries"]["train_end_idx"])
    val_end = int(e1.metadata["split_boundaries"]["val_end_idx"])
    chronological_selectors = {
        "train": slice(0, train_end),
        "val": slice(train_end, val_end),
        "test": slice(val_end, len(y_all)),
    }
    e1_written = art.write_split_streaming(
        exp1_dir,
        X_all,
        chronological_selectors,
        {"train": a1.y_train, "val": a1.y_val, "test": a1.y_test},
        transform_state,
        metadata=e1.metadata,
        verify=True,
    )
    e1_written["scaler_state"] = p
    result.written.update({f"exp1/{k}": v for k, v in e1_written.items()})
    _log(f"  wrote E1 artefacts to {exp1_dir}")

    # The E1 scaled copies are on disk now.  Release them before the next
    # experiment allocates its own — holding two experiments' scaled tensors at
    # once exceeded RAM.
    a1.X_train = a1.X_val = a1.X_test = None

    # ------------------------------------------------------------------
    # 7. Experiment 3 — strict chronological binary (PRIMARY B)
    #
    # Runs before E2 so E1 and E3 can share the already-written transformed
    # feature files.  E2 later uses index-selected bounded chunks and never
    # materialises a second full feature tensor.
    # ------------------------------------------------------------------
    _log("\n--- Step 7: Experiment 3 — strict chronological binary (primary B) ---")
    e3 = experiment3_binary_chronological(
        X_all, y_all,
        train_ratio=cfg.split.train,
        val_ratio=cfg.split.validation,
    )
    a3 = e3.arrays
    _log(
        f"  train={len(a3.y_train):,}  val={len(a3.y_val):,}  "
        f"test={len(a3.y_test):,}"
    )
    _log(f"  train binary: {e3.metadata['train_binary_distribution']}")
    _log(f"  val   binary: {e3.metadata['val_binary_distribution']}")
    _log(f"  test  binary: {e3.metadata['test_binary_distribution']}")

    e3.metadata["preprocessing"] = {
        "state_artifact": "exp3/scaler_state.json",
        "fit_experiment": "E1",
        "fit_split": "train",
        "fit_window_count": int(len(a1.y_train)),
        "shared_with": "E1",
        "shared_state_reason": (
            "E3 has exactly the same feature windows and chronological split "
            "boundaries as E1; only y is mapped from multiclass to binary."
        ),
        "validation_or_test_used_for_fit": False,
    }

    a3.X_train = None
    a3.X_val = None
    a3.X_test = None

    exp3_dir = out_dir / "exp3"
    p = art.write_scaler_state(shared_transform_payload, exp3_dir)
    e3_written = art.write_split_streaming(
        exp3_dir,
        X_all,
        chronological_selectors,
        {"train": a3.y_train, "val": a3.y_val, "test": a3.y_test},
        transform_state,
        metadata=e3.metadata,
        reuse_feature_paths={
            "train": e1_written["X_train"],
            "val": e1_written["X_val"],
            "test": e1_written["X_test"],
        },
        verify=True,
    )
    e3_written["scaler_state"] = p
    result.written.update({f"exp3/{k}": v for k, v in e3_written.items()})
    _log(f"  wrote E3 artefacts to {exp3_dir}")

    # ------------------------------------------------------------------
    # 8. Experiment 2 — per-class chronological multiclass (PRIMARY A)
    # ------------------------------------------------------------------
    _log("\n--- Step 8: Experiment 2 — multiclass separability (primary A) ---")
    e2 = experiment2_per_class_chronological_indices(
        y_all,
        train_ratio=cfg.split.train,
        val_ratio=cfg.split.validation,
        rare_class_min_test_windows=pc.rare_class_min_test_windows,
        min_train=ELIGIBILITY_MIN_TRAIN,
        min_val=ELIGIBILITY_MIN_VAL,
        min_test=ELIGIBILITY_MIN_TEST,
    )
    i2 = e2.indices
    y2_train = np.asarray(y_all[i2.train], dtype=np.int16)
    y2_val = np.asarray(y_all[i2.val], dtype=np.int16)
    y2_test = np.asarray(y_all[i2.test], dtype=np.int16)
    _log(
        f"  train={len(y2_train):,}  val={len(y2_val):,}  "
        f"test={len(y2_test):,}"
    )
    _log(
        "  NOTE: E2 preserves chronological order within each class only.  It "
        "does NOT provide strict global chronological isolation across classes "
        "and must not be described as leakage-free."
    )

    eligibility = e2.metadata["class_eligibility"]
    _log(
        f"  eligible classes ({eligibility['n_eligible']}/"
        f"{eligibility['n_eligible'] + eligibility['n_ineligible']}): "
        f"{eligibility['eligible_classes']}"
    )
    for cls in eligibility["ineligible_classes"]:
        row = eligibility["per_class"][cls]
        _log(
            f"    ineligible: {cls} — train={row['n_train']:,} "
            f"val={row['n_val']:,} test={row['n_test']:,} "
            f"({row['ineligibility_reason']})"
        )
    if e2.metadata.get("rare_classes"):
        _log(
            f"  rare classes (<{pc.rare_class_min_test_windows} test windows): "
            f"{e2.metadata['rare_classes']}"
        )

    p = art.write_class_eligibility(eligibility, out_dir)
    result.written["class_eligibility.json"] = p
    _log(f"  wrote {p.name}")

    # E2's training membership differs from E1's.  Reusing the E1 state here
    # would let some E2 validation/test windows influence learned replacements
    # and quantiles.  Fit an independent state on E2 train only.
    _log("  fitting E2-specific transforms on E2 train split only ...")
    transform_state_e2 = fit_transforms_indexed(X_all, i2.train)
    e2_transform_payload = transform_state_e2.to_dict()
    e2_transform_payload["fit_provenance"] = {
        "fit_experiment": "E2",
        "fit_split": "train",
        "fit_window_count": int(len(y2_train)),
        "fit_flow_value_count": int(len(y2_train) * T),
        "applies_to_experiments": ["E2"],
        "reason": (
            "E2 uses per-class chronological membership. Its finite "
            "replacements and scaling statistics are learned only from E2 "
            "training windows; no E2 validation or test window is used."
        ),
    }
    e2.metadata["preprocessing"] = {
        "state_artifact": "exp2/scaler_state.json",
        "fit_experiment": "E2",
        "fit_split": "train",
        "fit_window_count": int(len(y2_train)),
        "validation_or_test_used_for_fit": False,
    }

    exp2_dir = out_dir / "exp2"
    p = art.write_scaler_state(e2_transform_payload, exp2_dir)
    e2_written = art.write_split_streaming(
        exp2_dir,
        X_all,
        {"train": i2.train, "val": i2.val, "test": i2.test},
        {"train": y2_train, "val": y2_val, "test": y2_test},
        transform_state_e2,
        metadata=e2.metadata,
        verify=True,
    )
    e2_written["scaler_state"] = p
    result.written.update({f"exp2/{k}": v for k, v in e2_written.items()})
    _log(f"  wrote E2 artefacts to {exp2_dir}")

    # ------------------------------------------------------------------
    # 9. Global artefacts
    # ------------------------------------------------------------------
    p = art.write_feature_names(out_dir)
    result.written["feature_names.json"] = p

    p = art.write_class_mapping(out_dir)
    result.written["class_mapping.json"] = p

    p = art.write_binary_class_mapping(out_dir)
    result.written["binary_class_mapping.json"] = p

    result.summary = art.collect_artefact_summary(
        out_dir, e1_written, e2_written, e3_written
    )

    # Release every view of the disk-backed work arrays before deleting the
    # two exact temporary files.  These are pipeline-owned scratch artefacts,
    # never raw inputs.
    a1.y_train = a1.y_val = a1.y_test = None
    a3.y_train = a3.y_val = a3.y_test = None
    _close_memmap(X_all)
    _close_memmap(y_all)
    window_result.X = None
    window_result.y = None
    window_result.meta = []
    X_all = None
    y_all = None
    del y2_train, y2_val, y2_test, i2
    gc.collect()
    for work_name in ("unscaled_flow_features.npy", "flow_labels.npy"):
        work_path = work_dir / work_name
        if work_path.exists():
            work_path.unlink()
    if work_dir.exists():
        work_dir.rmdir()

    _log("\n" + "=" * 70)
    _log(f"Stage C complete.  Artefacts written to {out_dir}")
    _log("  exp1 = strict chronological multiclass DIAGNOSTIC (not a benchmark)")
    _log("  exp2 = multiclass separability (PRIMARY A, per-class chronological)")
    _log("  exp3 = strict chronological BINARY (PRIMARY B)")
    _log("=" * 70)

    return result
