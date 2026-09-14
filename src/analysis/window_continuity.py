"""Measure flow-to-flow continuity inside T=5 capture-order windows.

Analysis only. This module answers one question: when five *globally
consecutive* flows are grouped in the project's deterministic capture-order
proxy, do those five flows belong to the same communication context?

It does not build the project's real windows, does not touch the loader or the
preprocessing pipeline, does not modify raw data, and produces no model input.

Method
------
The primary dataset is the 84-column variant, the only file carrying the
endpoint identifiers (``Src IP``, ``Src Port``, ``Dst IP``) this analysis needs.
Only 8 of its 84 columns are read.

Rows are ordered by the project's contract - source-file capture date, then
``Original_Index`` (0-based position among the file's data rows). Within a
single file that is just the file's own row order, which is the order
``pd.read_csv`` yields, so no sort is performed and no sort is needed.
``Timestamp`` is read for diagnostics only and never orders anything.

Embedded header rows (``Label == "Label"``) are excluded before grouping, so
window membership is defined over *usable* rows. ``Original_Index`` is still the
pre-exclusion position, so every row remains traceable to its source line.

Windows are non-overlapping: rows ``[0,5)``, ``[5,10)``, ... Trailing rows that
cannot fill a window are discarded and counted.

The file is streamed in chunks snapped to a multiple of 5 so a window never
straddles a chunk boundary; per-window statistics are appended as small fixed
-width arrays. Peak memory is one chunk plus the per-window columns, never the
whole file.

Random baseline
---------------
The same usable rows are regrouped into windows of 5 by a seeded permutation,
then measured with the identical code. That isolates the effect of *adjacency*:
both arms have identical row content and identical marginal label and endpoint
distributions, so any difference is attributable to consecutive grouping alone.
The baseline is a measuring stick. It is never used for training or splitting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.preprocessing import schema
from src.preprocessing.profiling import ProfilingError, scout_file

#: Window length under investigation.
WINDOW_SIZE = 5

#: The endpoint identifier columns this analysis needs. Present only in the
#: 84-column variant - never assumed for any other file.
IDENTIFIER_COLUMNS: Tuple[str, ...] = ("Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol")

#: Every column read. 8 of 84; the flow features are irrelevant here.
READ_COLUMNS: Tuple[str, ...] = IDENTIFIER_COLUMNS + (
    schema.TIMESTAMP_COLUMN,
    schema.LABEL_COLUMN,
)

#: Percentiles reported for the timestamp-span distribution.
SPAN_PERCENTILES: Tuple[int, ...] = (50, 95)

#: Cumulative span thresholds in seconds, reported alongside the median. The
#: distribution turns out to be bimodal (tight bursts vs whole-day spans), so a
#: single median describes neither mode.
SPAN_BUCKETS: Tuple[int, ...] = (0, 1, 10, 60, 300, 3600, 21600)

#: Per-window CSV columns, in order.
WINDOW_CSV_COLUMNS: Tuple[str, ...] = (
    "window_id",
    "first_original_index",
    "last_original_index",
    "n_unique_src_ip",
    "n_unique_dst_ip",
    "n_unique_pairs",
    "n_unique_five_tuples",
    "n_adjacent_same_pair",
    "all_same_pair",
    "any_adjacent_same_pair",
    "label_pure",
    "fifth_is_majority",
    "n_unique_labels",
    "span_seconds",
    "same_second",
)


class AnalysisError(RuntimeError):
    """The analysis could not be completed on the given file."""


def _percent(numerator: int, denominator: int) -> float:
    """Percentage, or 0.0 for an empty denominator (never a ZeroDivisionError)."""
    return (100.0 * numerator / denominator) if denominator else 0.0


@dataclass
class WindowStats:
    """Aggregate continuity measurements over a set of T=5 windows.

    Counts are exact. Percentages are derived at serialisation time so the
    stored numbers and the reported numbers can never disagree.
    """

    arm: str = ""
    n_windows: int = 0

    # -- endpoint continuity ------------------------------------------------ #
    n_all_same_pair: int = 0
    n_any_adjacent_same_pair: int = 0
    #: Windows whose five flows have five distinct Src/Dst pairs.
    n_fully_heterogeneous: int = 0
    #: Histogram of unique Src/Dst pairs per window, index 1..5.
    pair_histogram: Dict[int, int] = field(default_factory=dict)
    sum_unique_pairs: int = 0
    median_unique_pairs: float = 0.0
    sum_unique_src_ip: int = 0
    sum_unique_dst_ip: int = 0
    sum_unique_five_tuples: int = 0
    n_all_same_five_tuple: int = 0
    #: Total adjacent flow pairs sharing a Src/Dst pair (4 possible per window).
    sum_adjacent_same_pair: int = 0

    # -- label continuity --------------------------------------------------- #
    n_label_pure: int = 0
    n_fifth_is_majority: int = 0
    n_fifth_not_majority: int = 0
    label_count_histogram: Dict[int, int] = field(default_factory=dict)

    # -- timestamp diagnostics ---------------------------------------------- #
    n_same_second: int = 0
    span_median_seconds: float = 0.0
    span_p95_seconds: float = 0.0
    span_max_seconds: float = 0.0
    span_mean_seconds: float = 0.0
    #: Cumulative counts of windows with span <= each :data:`SPAN_BUCKETS` value.
    #: The span distribution is strongly bimodal, so a median on its own is
    #: actively misleading; these buckets show both modes.
    span_buckets: Dict[int, int] = field(default_factory=dict)
    n_span_valid: int = 0

    def to_dict(self) -> Dict[str, Any]:
        n = self.n_windows
        return {
            "arm": self.arm,
            "n_windows": n,
            "endpoint_continuity": {
                "A_pct_windows_all_five_same_src_dst_pair": _percent(
                    self.n_all_same_pair, n
                ),
                "B_pct_windows_any_adjacent_same_src_dst_pair": _percent(
                    self.n_any_adjacent_same_pair, n
                ),
                "C_mean_unique_pairs_per_window": (self.sum_unique_pairs / n) if n else 0.0,
                "D_median_unique_pairs_per_window": self.median_unique_pairs,
                "E_pct_windows_by_unique_pair_count": {
                    str(k): _percent(self.pair_histogram.get(k, 0), n)
                    for k in range(1, WINDOW_SIZE + 1)
                },
                "n_windows_by_unique_pair_count": {
                    str(k): self.pair_histogram.get(k, 0)
                    for k in range(1, WINDOW_SIZE + 1)
                },
                "pct_windows_fully_heterogeneous_five_distinct_pairs": _percent(
                    self.n_fully_heterogeneous, n
                ),
                "mean_unique_src_ip_per_window": (self.sum_unique_src_ip / n) if n else 0.0,
                "mean_unique_dst_ip_per_window": (self.sum_unique_dst_ip / n) if n else 0.0,
                "mean_unique_five_tuples_per_window": (
                    (self.sum_unique_five_tuples / n) if n else 0.0
                ),
                "pct_windows_all_five_same_five_tuple": _percent(
                    self.n_all_same_five_tuple, n
                ),
                "mean_adjacent_same_pair_per_window_max_4": (
                    (self.sum_adjacent_same_pair / n) if n else 0.0
                ),
                "pct_adjacent_flow_pairs_sharing_src_dst": _percent(
                    self.sum_adjacent_same_pair, n * (WINDOW_SIZE - 1)
                ),
                "counts": {
                    "all_same_pair": self.n_all_same_pair,
                    "any_adjacent_same_pair": self.n_any_adjacent_same_pair,
                    "fully_heterogeneous": self.n_fully_heterogeneous,
                    "all_same_five_tuple": self.n_all_same_five_tuple,
                    "adjacent_flow_pairs_sharing_src_dst": self.sum_adjacent_same_pair,
                    "adjacent_flow_pairs_total": n * (WINDOW_SIZE - 1),
                },
            },
            "label_continuity": {
                "F_pct_windows_label_pure": _percent(self.n_label_pure, n),
                "pct_windows_mixed_label": _percent(n - self.n_label_pure, n),
                "G_pct_windows_fifth_label_ne_majority": _percent(
                    self.n_fifth_not_majority, n
                ),
                "pct_windows_fifth_label_eq_majority": _percent(
                    self.n_fifth_is_majority, n
                ),
                "pct_windows_by_unique_label_count": {
                    str(k): _percent(self.label_count_histogram.get(k, 0), n)
                    for k in sorted(self.label_count_histogram)
                },
                "n_windows_by_unique_label_count": {
                    str(k): self.label_count_histogram[k]
                    for k in sorted(self.label_count_histogram)
                },
                "counts": {
                    "label_pure": self.n_label_pure,
                    "mixed_label": n - self.n_label_pure,
                    "fifth_eq_majority": self.n_fifth_is_majority,
                    "fifth_ne_majority": self.n_fifth_not_majority,
                },
            },
            "timestamp_diagnostics": {
                "note": (
                    "Timestamp is diagnostic only and never orders rows. Spans are "
                    "computed from parsed timestamps within a window and can be "
                    "negative-free by construction (max - min), but the column "
                    "itself is non-monotonic in file order."
                ),
                "median_span_seconds": self.span_median_seconds,
                "p95_span_seconds": self.span_p95_seconds,
                "max_span_seconds": self.span_max_seconds,
                "mean_span_seconds": self.span_mean_seconds,
                "pct_windows_entirely_within_one_second": _percent(
                    self.n_same_second, n
                ),
                "n_windows_entirely_within_one_second": self.n_same_second,
                "n_windows_with_a_valid_span": self.n_span_valid,
                "pct_windows_with_span_at_most": {
                    str(threshold): _percent(count, self.n_span_valid)
                    for threshold, count in sorted(self.span_buckets.items())
                },
                "n_windows_with_span_at_most": {
                    str(threshold): count
                    for threshold, count in sorted(self.span_buckets.items())
                },
            },
        }


@dataclass
class ContinuityResult:
    """Everything one analysis run measured."""

    source_file: str = ""
    source_path: str = ""
    schema_variant: str = ""
    n_columns_in_file: int = 0
    n_raw_rows: int = 0
    n_embedded_header_rows: int = 0
    n_usable_rows: int = 0
    n_rows_analyzed: int = 0
    n_windows: int = 0
    n_trailing_rows_discarded: int = 0
    window_size: int = WINDOW_SIZE
    seed: int = 0
    ordering_rule: str = ""
    n_unparsed_timestamps: int = 0
    consecutive: WindowStats = field(default_factory=WindowStats)
    random_baseline: Optional[WindowStats] = None
    #: Per-window rows for the consecutive arm, as a frame.
    per_window: Optional[pd.DataFrame] = None
    generated_at_utc: str = ""
    elapsed_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "generated_at_utc": self.generated_at_utc,
            "elapsed_seconds": self.elapsed_seconds,
            "dataset": {
                "source_file": self.source_file,
                "source_path": self.source_path,
                "schema_variant": self.schema_variant,
                "n_columns_in_file": self.n_columns_in_file,
                "n_raw_data_rows": self.n_raw_rows,
                "n_embedded_header_rows_excluded": self.n_embedded_header_rows,
                "n_usable_rows": self.n_usable_rows,
                "n_rows_analyzed": self.n_rows_analyzed,
                "n_windows": self.n_windows,
                "n_trailing_rows_discarded": self.n_trailing_rows_discarded,
                "n_unparsed_timestamps": self.n_unparsed_timestamps,
            },
            "parameters": {
                "window_size": self.window_size,
                "seed": self.seed,
                "ordering_rule": self.ordering_rule,
                "columns_read": list(READ_COLUMNS),
            },
            "definitions": DEFINITIONS,
            "consecutive": self.consecutive.to_dict(),
            "notes": self.notes,
        }
        if self.random_baseline is not None:
            payload["random_baseline"] = self.random_baseline.to_dict()
            payload["consecutive_vs_random"] = self.comparison()
        return payload

    def comparison(self) -> Dict[str, Any]:
        """Consecutive minus random, for the headline continuity metrics."""
        if self.random_baseline is None:
            return {}
        left = self.consecutive.to_dict()
        right = self.random_baseline.to_dict()
        out: Dict[str, Any] = {}
        for section, keys in (
            (
                "endpoint_continuity",
                (
                    "A_pct_windows_all_five_same_src_dst_pair",
                    "B_pct_windows_any_adjacent_same_src_dst_pair",
                    "C_mean_unique_pairs_per_window",
                    "pct_windows_fully_heterogeneous_five_distinct_pairs",
                ),
            ),
            (
                "label_continuity",
                (
                    "F_pct_windows_label_pure",
                    "G_pct_windows_fifth_label_ne_majority",
                ),
            ),
        ):
            for key in keys:
                out[key] = {
                    "consecutive": left[section][key],
                    "random": right[section][key],
                    "difference": left[section][key] - right[section][key],
                }
        return out


#: Exact definition of every metric, embedded in the JSON so a number can never
#: be read without its formula.
DEFINITIONS: Dict[str, str] = {
    "window": (
        "Non-overlapping group of 5 consecutive rows in the ordering sequence: "
        "window i holds sequence positions [5i, 5i+5). Trailing rows that cannot "
        "fill a window are discarded."
    ),
    "src_dst_pair": "The tuple (Src IP, Dst IP), compared as an exact string pair.",
    "five_tuple": "The tuple (Src IP, Src Port, Dst IP, Dst Port, Protocol).",
    "n_unique_src_ip": "Count of distinct Src IP values among the window's 5 rows.",
    "n_unique_dst_ip": "Count of distinct Dst IP values among the window's 5 rows.",
    "n_unique_pairs": "Count of distinct (Src IP, Dst IP) pairs among the 5 rows; 1..5.",
    "n_unique_five_tuples": "Count of distinct 5-tuples among the 5 rows; 1..5.",
    "n_adjacent_same_pair": (
        "Number of adjacent positions j in {0,1,2,3} where row j and row j+1 have "
        "the same (Src IP, Dst IP) pair. Range 0..4."
    ),
    "all_same_pair": "n_unique_pairs == 1.",
    "any_adjacent_same_pair": "n_adjacent_same_pair >= 1.",
    "fully_heterogeneous": "n_unique_pairs == 5 (no two flows share a Src/Dst pair).",
    "label_pure": "n_unique_labels == 1.",
    "majority_label": (
        "The label with the highest count in the window. Ties are broken by "
        "ascending label string, so the rule is deterministic; 'fifth_is_majority' "
        "is therefore a conservative measure - under a tie the fifth label counts "
        "as the majority only if it wins that alphabetical tiebreak."
    ),
    "fifth_is_majority": "The 5th row's label equals the window's majority label.",
    "span_seconds": (
        "max(parsed Timestamp) - min(parsed Timestamp) over the window's 5 rows, in "
        "seconds. Diagnostic only; Timestamp never orders rows. A window with any "
        "unparseable timestamp is excluded from the span statistics only."
    ),
    "same_second": "span_seconds == 0 (all 5 rows fall in one clock second).",
    "pct_windows_with_span_at_most": (
        "Cumulative share of span-valid windows whose span is at or below each "
        "threshold in seconds. Reported because the span distribution is bimodal - "
        "tight bursts versus whole-day spans - so the median describes neither mode."
    ),
    "A": "Percentage of windows with all_same_pair.",
    "B": "Percentage of windows with any_adjacent_same_pair.",
    "C": "Mean of n_unique_pairs over all windows.",
    "D": "Median of n_unique_pairs over all windows.",
    "E": "Percentage of windows at each n_unique_pairs value 1..5.",
    "F": "Percentage of windows with label_pure.",
    "G": "Percentage of windows where the 5th row's label != the majority label.",
    "random_baseline": (
        "The identical usable rows regrouped into windows of 5 by a seeded "
        "permutation, measured with the same code. Isolates the effect of "
        "consecutive adjacency: row content and marginal distributions are held "
        "fixed. Never used for training or splitting."
    ),
}


def _encode(values: np.ndarray) -> np.ndarray:
    """Map an array of strings to dense int64 codes.

    Only equality matters for every metric here, so the codes need no ordering
    guarantee; ``np.unique`` gives dense codes in one pass.
    """
    _, codes = np.unique(values, return_inverse=True)
    return codes.astype(np.int64, copy=False)


def _row_wise_unique_counts(codes: np.ndarray) -> np.ndarray:
    """Distinct values per row of an ``(n, WINDOW_SIZE)`` code matrix.

    Sorting each row and counting value changes is O(n * 5 log 5) with no Python
    loop; a per-row ``set`` would be ~100x slower over 1.6M windows.
    """
    ordered = np.sort(codes, axis=1)
    return 1 + (np.diff(ordered, axis=1) != 0).sum(axis=1).astype(np.int64)


def _majority_is_fifth(codes: np.ndarray) -> np.ndarray:
    """Whether each row's last element is the row's majority value.

    Ties are broken by the smallest code. Codes come from ``np.unique``, which is
    sorted lexicographically, so "smallest code" is "alphabetically first label"
    exactly as documented in :data:`DEFINITIONS`.
    """
    ordered = np.sort(codes, axis=1)
    n, width = ordered.shape
    # Seeded with the first (smallest) value at count 1, so a window of five
    # distinct labels resolves to the smallest code rather than to whichever
    # run the loop happened to visit last.
    best = ordered[:, 0].copy()
    best_count = np.ones(n, dtype=np.int64)
    run_value = ordered[:, 0].copy()
    run_count = np.ones(n, dtype=np.int64)
    for column in range(1, width):
        current = ordered[:, column]
        same = current == run_value
        run_count = np.where(same, run_count + 1, 1)
        run_value = np.where(same, run_value, current)
        # Strict > keeps the earlier (smaller) code on a tie.
        better = run_count > best_count
        best_count = np.where(better, run_count, best_count)
        best = np.where(better, run_value, best)
    return codes[:, -1] == best


class _WindowAccumulator:
    """Collects per-window measurements across chunks in bounded memory."""

    def __init__(self, arm: str) -> None:
        self.arm = arm
        self.first_index: List[np.ndarray] = []
        self.last_index: List[np.ndarray] = []
        self.unique_src: List[np.ndarray] = []
        self.unique_dst: List[np.ndarray] = []
        self.unique_pairs: List[np.ndarray] = []
        self.unique_tuples: List[np.ndarray] = []
        self.adjacent_same: List[np.ndarray] = []
        self.label_counts: List[np.ndarray] = []
        self.fifth_majority: List[np.ndarray] = []
        self.span: List[np.ndarray] = []
        self.span_valid: List[np.ndarray] = []

    def add(
        self,
        *,
        first_index: np.ndarray,
        last_index: np.ndarray,
        src: np.ndarray,
        dst: np.ndarray,
        pair: np.ndarray,
        five_tuple: np.ndarray,
        label: np.ndarray,
        stamp: np.ndarray,
    ) -> None:
        """Measure one batch of windows. Each argument is ``(n, 5)``."""
        self.first_index.append(first_index)
        self.last_index.append(last_index)
        self.unique_src.append(_row_wise_unique_counts(src))
        self.unique_dst.append(_row_wise_unique_counts(dst))
        self.unique_pairs.append(_row_wise_unique_counts(pair))
        self.unique_tuples.append(_row_wise_unique_counts(five_tuple))
        self.adjacent_same.append(
            (pair[:, 1:] == pair[:, :-1]).sum(axis=1).astype(np.int64)
        )
        self.label_counts.append(_row_wise_unique_counts(label))
        self.fifth_majority.append(_majority_is_fifth(label))
        # NaT propagates as NaN through the subtraction, so an unparseable
        # timestamp anywhere in a window invalidates only that window's span.
        valid = np.isfinite(stamp).all(axis=1)
        span = np.where(valid, np.nanmax(stamp, axis=1) - np.nanmin(stamp, axis=1), np.nan)
        self.span.append(span)
        self.span_valid.append(valid)

    def finish(self) -> Tuple[WindowStats, pd.DataFrame]:
        if not self.unique_pairs:
            return WindowStats(arm=self.arm), pd.DataFrame(
                columns=list(WINDOW_CSV_COLUMNS)
            )

        first_index = np.concatenate(self.first_index)
        last_index = np.concatenate(self.last_index)
        unique_src = np.concatenate(self.unique_src)
        unique_dst = np.concatenate(self.unique_dst)
        unique_pairs = np.concatenate(self.unique_pairs)
        unique_tuples = np.concatenate(self.unique_tuples)
        adjacent_same = np.concatenate(self.adjacent_same)
        label_counts = np.concatenate(self.label_counts)
        fifth_majority = np.concatenate(self.fifth_majority)
        span = np.concatenate(self.span)
        span_valid = np.concatenate(self.span_valid)

        n = int(unique_pairs.size)
        all_same = unique_pairs == 1
        label_pure = label_counts == 1
        same_second = span_valid & (span == 0.0)

        stats = WindowStats(
            arm=self.arm,
            n_windows=n,
            n_all_same_pair=int(all_same.sum()),
            n_any_adjacent_same_pair=int((adjacent_same >= 1).sum()),
            n_fully_heterogeneous=int((unique_pairs == WINDOW_SIZE).sum()),
            pair_histogram={
                int(k): int(v)
                for k, v in zip(*np.unique(unique_pairs, return_counts=True))
            },
            sum_unique_pairs=int(unique_pairs.sum()),
            median_unique_pairs=float(np.median(unique_pairs)),
            sum_unique_src_ip=int(unique_src.sum()),
            sum_unique_dst_ip=int(unique_dst.sum()),
            sum_unique_five_tuples=int(unique_tuples.sum()),
            n_all_same_five_tuple=int((unique_tuples == 1).sum()),
            sum_adjacent_same_pair=int(adjacent_same.sum()),
            n_label_pure=int(label_pure.sum()),
            n_fifth_is_majority=int(fifth_majority.sum()),
            n_fifth_not_majority=int((~fifth_majority).sum()),
            label_count_histogram={
                int(k): int(v)
                for k, v in zip(*np.unique(label_counts, return_counts=True))
            },
            n_same_second=int(same_second.sum()),
        )
        stats.n_span_valid = int(span_valid.sum())
        if span_valid.any():
            good = span[span_valid]
            stats.span_median_seconds = float(np.percentile(good, 50))
            stats.span_p95_seconds = float(np.percentile(good, 95))
            stats.span_max_seconds = float(good.max())
            stats.span_mean_seconds = float(good.mean())
            stats.span_buckets = {
                int(threshold): int((good <= threshold).sum())
                for threshold in SPAN_BUCKETS
            }

        frame = pd.DataFrame(
            {
                "window_id": np.arange(n, dtype=np.int64),
                "first_original_index": first_index,
                "last_original_index": last_index,
                "n_unique_src_ip": unique_src,
                "n_unique_dst_ip": unique_dst,
                "n_unique_pairs": unique_pairs,
                "n_unique_five_tuples": unique_tuples,
                "n_adjacent_same_pair": adjacent_same,
                "all_same_pair": all_same,
                "any_adjacent_same_pair": adjacent_same >= 1,
                "label_pure": label_pure,
                "fifth_is_majority": fifth_majority,
                "n_unique_labels": label_counts,
                "span_seconds": span,
                "same_second": same_second,
            }
        )
        return stats, frame[list(WINDOW_CSV_COLUMNS)]


def _read_usable_columns(
    path: Path,
    *,
    excluded: np.ndarray,
    n_usable: int,
    chunk_size: int,
    row_limit: Optional[int] = None,
) -> Tuple[Dict[str, np.ndarray], int]:
    """Read the 8 needed columns for the usable rows, as dense code arrays.

    Returns codes for src/dst/pair/five-tuple/label plus float64 epoch seconds
    and each row's ``Original_Index``. Strings are hashed to codes per chunk and
    unified at the end, so the full string columns are never all resident.
    """
    wanted = min(n_usable, row_limit) if row_limit else n_usable
    if wanted <= 0:
        raise AnalysisError(f"{path.name}: no usable rows to analyze")

    src_parts: List[np.ndarray] = []
    dst_parts: List[np.ndarray] = []
    pair_parts: List[np.ndarray] = []
    tuple_parts: List[np.ndarray] = []
    label_parts: List[np.ndarray] = []
    stamp_parts: List[np.ndarray] = []
    index_parts: List[np.ndarray] = []

    excluded_set = set(excluded.tolist())
    n_unparsed = 0
    position = 0          # data-row position in the original file
    kept = 0              # usable rows accepted so far

    reader = pd.read_csv(
        path,
        usecols=list(READ_COLUMNS),
        dtype=str,
        chunksize=chunk_size,
        na_filter=False,
        engine="c",
        encoding="utf-8-sig",
        on_bad_lines="error",
    )
    for chunk in reader:
        n_rows = len(chunk)
        positions = np.arange(position, position + n_rows, dtype=np.int64)
        position += n_rows

        if excluded_set:
            keep = ~np.isin(positions, excluded)
            if not keep.all():
                chunk = chunk.loc[keep]
                positions = positions[keep]
        if chunk.empty:
            continue

        remaining = wanted - kept
        if remaining <= 0:
            break
        if len(chunk) > remaining:
            chunk = chunk.iloc[:remaining]
            positions = positions[:remaining]
        kept += len(chunk)

        src = chunk["Src IP"].to_numpy()
        dst = chunk["Dst IP"].to_numpy()
        # Separator that cannot occur inside an IP, port or protocol value, so
        # the joined key is unambiguous.
        pair = np.char.add(np.char.add(src.astype(str), "|"), dst.astype(str))
        five = pair
        for column in ("Src Port", "Dst Port", "Protocol"):
            five = np.char.add(
                np.char.add(five, "|"), chunk[column].to_numpy().astype(str)
            )

        parsed = pd.to_datetime(
            chunk[schema.TIMESTAMP_COLUMN],
            format=schema.TIMESTAMP_FORMAT,
            errors="coerce",
            exact=True,
        )
        n_unparsed += int(parsed.isna().sum())

        src_parts.append(src.astype(str))
        dst_parts.append(dst.astype(str))
        pair_parts.append(pair)
        tuple_parts.append(five)
        label_parts.append(chunk[schema.LABEL_COLUMN].to_numpy().astype(str))
        # NaT must become NaN, not a number. Casting datetime64 straight to
        # float64 turns NaT into the int64-min sentinel (~9.22e18), which would
        # sail through the isfinite() check and produce an absurd span instead of
        # excluding the window.
        seconds = parsed.to_numpy(dtype="datetime64[s]").astype(np.float64)
        seconds[parsed.isna().to_numpy()] = np.nan
        stamp_parts.append(seconds)
        index_parts.append(positions)

        if kept >= wanted:
            break

    if kept < wanted:
        raise AnalysisError(
            f"{path.name}: expected {wanted:,} usable rows but read {kept:,}"
        )

    columns = {
        "src": _encode(np.concatenate(src_parts)),
        "dst": _encode(np.concatenate(dst_parts)),
        "pair": _encode(np.concatenate(pair_parts)),
        "five_tuple": _encode(np.concatenate(tuple_parts)),
        "label": _encode(np.concatenate(label_parts)),
        "stamp": np.concatenate(stamp_parts),
        "original_index": np.concatenate(index_parts),
    }
    return columns, n_unparsed


def _measure(
    columns: Dict[str, np.ndarray],
    order: np.ndarray,
    *,
    arm: str,
    batch_windows: int = 200_000,
) -> Tuple[WindowStats, pd.DataFrame]:
    """Group ``order`` into windows of 5 and measure them in batches."""
    n_windows = order.size // WINDOW_SIZE
    accumulator = _WindowAccumulator(arm)
    for start in range(0, n_windows, batch_windows):
        stop = min(start + batch_windows, n_windows)
        rows = order[start * WINDOW_SIZE : stop * WINDOW_SIZE].reshape(-1, WINDOW_SIZE)
        accumulator.add(
            first_index=columns["original_index"][rows[:, 0]],
            last_index=columns["original_index"][rows[:, -1]],
            src=columns["src"][rows],
            dst=columns["dst"][rows],
            pair=columns["pair"][rows],
            five_tuple=columns["five_tuple"][rows],
            label=columns["label"][rows],
            stamp=columns["stamp"][rows],
        )
    return accumulator.finish()


def analyze_file(
    path: str | Path,
    *,
    seed: int = 42,
    chunk_size: int = 500_000,
    row_limit: Optional[int] = None,
    with_random_baseline: bool = True,
    require_identifiers: bool = True,
) -> ContinuityResult:
    """Measure T=5 window continuity for one CSV file. Opens it read-only."""
    from src.preprocessing.profiling import inspect_header

    started = datetime.now(timezone.utc)
    path = Path(path)
    if not path.is_file():
        raise AnalysisError(f"file not found: {path}")

    header = inspect_header(path)
    missing = [name for name in READ_COLUMNS if name not in header.columns]
    if missing and require_identifiers:
        raise AnalysisError(
            f"{path.name} lacks the identifier column(s) {missing}. This analysis "
            "needs the endpoint identifiers, which only the 84-column variant "
            "carries; it is not assumed for any other file."
        )

    try:
        scout = scout_file(path, chunk_size=chunk_size, verify_line_count=True)
    except ProfilingError as exc:  # pragma: no cover - propagated with context
        raise AnalysisError(f"{path.name}: row accounting failed: {exc}") from exc

    excluded = np.asarray(sorted(scout.embedded_header_positions), dtype=np.int64)
    columns, n_unparsed = _read_usable_columns(
        path,
        excluded=excluded,
        n_usable=scout.n_usable_rows,
        chunk_size=chunk_size,
        row_limit=row_limit,
    )

    n_analyzed = int(columns["label"].size)
    n_windows = n_analyzed // WINDOW_SIZE
    order = np.arange(n_analyzed, dtype=np.int64)

    consecutive, per_window = _measure(columns, order, arm="consecutive")

    baseline: Optional[WindowStats] = None
    if with_random_baseline:
        # Seeded generator, not the global numpy state: reproducible and it does
        # not disturb any other RNG in the process.
        shuffled = np.random.default_rng(seed).permutation(n_analyzed)
        baseline, _ = _measure(columns, shuffled, arm="random_baseline")

    result = ContinuityResult(
        source_file=path.name,
        source_path=str(path),
        schema_variant=header.variant.variant,
        n_columns_in_file=header.n_columns,
        n_raw_rows=scout.n_raw_rows,
        n_embedded_header_rows=len(scout.embedded_header_positions),
        n_usable_rows=scout.n_usable_rows,
        n_rows_analyzed=n_analyzed,
        n_windows=n_windows,
        n_trailing_rows_discarded=n_analyzed - n_windows * WINDOW_SIZE,
        seed=seed,
        ordering_rule=(
            "deterministic capture-order proxy: (source file capture date, "
            "Original_Index). Within one file this is the file's own data-row "
            "order; no sort is applied and Timestamp is never an ordering key."
        ),
        n_unparsed_timestamps=n_unparsed,
        consecutive=consecutive,
        random_baseline=baseline,
        per_window=per_window,
        generated_at_utc=started.isoformat(),
    )
    if row_limit:
        result.notes.append(
            f"row_limit={row_limit:,} was applied: this is a truncated debug run, "
            "not a measurement of the whole file."
        )
    if n_unparsed:
        result.notes.append(
            f"{n_unparsed:,} timestamp value(s) did not parse and are excluded from "
            "the span statistics only; their rows still take part in every "
            "endpoint and label metric."
        )
    result.elapsed_seconds = (
        datetime.now(timezone.utc) - started
    ).total_seconds()
    return result
