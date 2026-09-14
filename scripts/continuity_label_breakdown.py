"""Supplementary diagnostic: split T=5 endpoint continuity by window label.

``02-20-2018.csv`` is 92.75% Benign and 7.25% DDoS-LOIC-HTTP, and the DDoS rows
sit in contiguous blocks. Aggregate continuity therefore mixes two populations.
This script re-measures metrics A, B, C and the heterogeneity rate separately
for label-pure Benign windows, label-pure attack windows, and mixed windows, so
the headline numbers can be read without that confound.

Analysis only. Reads one CSV read-only, writes one JSON under
``results/temporal_analysis``, touches nothing else. Uses the same ordering rule
and the same window construction as :mod:`src.analysis.window_continuity`.

Usage
-----
    python scripts/continuity_label_breakdown.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.window_continuity import (  # noqa: E402
    WINDOW_SIZE,
    _row_wise_unique_counts,
)
from src.config import Config  # noqa: E402

OUTPUT_NAME = "continuity_label_breakdown.json"


def main() -> int:
    cfg = Config.load(None).validate()
    path = cfg.paths.raw_data_dir / cfg.profiling.csv_subdir / "02-20-2018.csv"
    if not path.is_file():
        print(f"CSV file not found: {path}", file=sys.stderr)
        return 2

    src: List[np.ndarray] = []
    dst: List[np.ndarray] = []
    lab: List[np.ndarray] = []
    for chunk in pd.read_csv(
        path, usecols=["Src IP", "Dst IP", "Label"], dtype=str, chunksize=2_000_000
    ):
        src.append(chunk["Src IP"].to_numpy())
        dst.append(chunk["Dst IP"].to_numpy())
        lab.append(chunk["Label"].to_numpy())

    pair = np.char.add(np.char.add(np.concatenate(src), ">"), np.concatenate(dst))
    labels = np.concatenate(lab)
    n_windows = len(pair) // WINDOW_SIZE
    usable = n_windows * WINDOW_SIZE

    pair_codes = np.unique(pair[:usable], return_inverse=True)[1].reshape(
        n_windows, WINDOW_SIZE
    )
    label_codes_flat, label_inverse = np.unique(labels[:usable], return_inverse=True)
    label_codes = label_inverse.reshape(n_windows, WINDOW_SIZE)

    # Contiguous label runs in capture order. This is what explains the main
    # analysis's near-perfect label purity: if the labels sit in long blocks,
    # almost every window falls strictly inside one block.
    label_flat = labels[:usable]
    change = np.flatnonzero(label_flat[1:] != label_flat[:-1]) + 1
    bounds = np.concatenate(([0], change, [usable]))
    run_lengths = np.diff(bounds)

    n_pairs = _row_wise_unique_counts(pair_codes)
    adjacent = pair_codes[:, 1:] == pair_codes[:, :-1]
    all_same = n_pairs == 1
    any_adjacent = adjacent.any(axis=1)
    heterogeneous = n_pairs == WINDOW_SIZE
    n_labels = _row_wise_unique_counts(label_codes)

    groups: Dict[str, np.ndarray] = {"mixed_label": n_labels > 1}
    for code, name in enumerate(label_codes_flat):
        groups[f"pure_{name}"] = (n_labels == 1) & (label_codes[:, 0] == code)

    report: Dict[str, object] = {
        "source_file": path.name,
        "n_rows_read": int(len(pair)),
        "n_windows": int(n_windows),
        "n_trailing_rows_discarded": int(len(pair) - usable),
        "note": "Supplementary diagnostic. Endpoint continuity per window-label "
        "group; same ordering rule and window construction as the main analysis.",
        "label_runs_in_capture_order": {
            "definition": "Maximal blocks of consecutive rows carrying one label, "
            "in the deterministic capture-order proxy. Explains why label purity "
            "is high without any temporal mechanism.",
            "n_runs": int(len(run_lengths)),
            "n_distinct_labels": int(len(label_codes_flat)),
            "run_length_min": int(run_lengths.min()),
            "run_length_median": float(np.median(run_lengths)),
            "run_length_mean": float(run_lengths.mean()),
            "run_length_max": int(run_lengths.max()),
            "n_runs_shorter_than_window": int((run_lengths < WINDOW_SIZE).sum()),
            "n_rows_in_runs_shorter_than_window": int(
                run_lengths[run_lengths < WINDOW_SIZE].sum()
            ),
            "pct_rows_in_runs_shorter_than_window": float(
                run_lengths[run_lengths < WINDOW_SIZE].sum() / usable * 100
            ),
        },
        "groups": {},
    }
    for name, mask in groups.items():
        count = int(mask.sum())
        report["groups"][name] = {
            "n_windows": count,
            "pct_of_windows": float(count / n_windows * 100) if n_windows else 0.0,
            "A_pct_windows_all_five_same_src_dst_pair": (
                float(all_same[mask].mean() * 100) if count else 0.0
            ),
            "B_pct_windows_any_adjacent_same_src_dst_pair": (
                float(any_adjacent[mask].mean() * 100) if count else 0.0
            ),
            "C_mean_unique_pairs_per_window": (
                float(n_pairs[mask].mean()) if count else 0.0
            ),
            "pct_windows_fully_heterogeneous_five_distinct_pairs": (
                float(heterogeneous[mask].mean() * 100) if count else 0.0
            ),
        }

    out = cfg.paths.results_dir / "temporal_analysis" / OUTPUT_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
