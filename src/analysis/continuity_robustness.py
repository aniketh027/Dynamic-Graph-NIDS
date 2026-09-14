"""Robustness of the T=5 continuity result: window size sweep and cross-file check.

The first continuity experiment measured one file at one window size. Two
questions were left open, and this module answers both by measurement:

1. **Is T=5 a sensible window size?** Sweep T in {3, 5, 10} over the same rows
   with the same ordering rule, seed and metric definitions.
2. **Is the 02-20 result representative?** Only ``02-20-2018.csv`` carries
   endpoint identifiers, so the full metric set cannot be computed elsewhere.
   Instead a deliberately weaker identity - ``(Dst Port, Protocol)`` - is
   measured on every file *and* on 02-20, where the true ``(Src IP, Dst IP)``
   identity is also available. The 02-20 row calibrates the proxy: it shows how
   far the proxy over- or under-states real endpoint continuity, which is what
   makes the other nine files interpretable at all.

Definitions generalise the T=5 set by replacing "5" with T and "fifth flow" with
"final flow". At T=5 with the true identity this module must reproduce the
published T=5 numbers exactly; :func:`equivalence_check` asserts that against the
frozen summary rather than trusting it.

**Analysis only.** Reads CSVs read-only, writes under ``results/`` only, changes
no preprocessing, loader or windowing code, and produces no model input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.analysis.window_continuity import (
    SPAN_BUCKETS,
    AnalysisError,
    _encode,
    _majority_is_fifth,
    _percent,
    _row_wise_unique_counts,
)
from src.preprocessing import schema
from src.preprocessing.profiling import ProfilingError, inspect_header, scout_file

#: Window sizes compared. 5 is the project's current proposal; 3 and 10 bracket
#: it so the trend either side is visible rather than assumed.
WINDOW_SIZES: Tuple[int, ...] = (3, 5, 10)

#: The full endpoint identity. Only the 84-column variant carries these.
TRUE_IDENTITY_COLUMNS: Tuple[str, ...] = ("Src IP", "Dst IP")

#: The weaker identity available in every file. A service-level stand-in, not a
#: conversation: many hosts talk to port 80/TCP, so it collapses distinct
#: conversations together and therefore *overstates* continuity by construction.
PROXY_IDENTITY_COLUMNS: Tuple[str, ...] = ("Dst Port", "Protocol")

IDENTITY_TRUE = "src_dst_ip_pair"
IDENTITY_PROXY = "dst_port_protocol"

#: Columns needed for the five-tuple, measured only where available.
FIVE_TUPLE_COLUMNS: Tuple[str, ...] = (
    "Src IP",
    "Src Port",
    "Dst IP",
    "Dst Port",
    "Protocol",
)

ROBUSTNESS_DEFINITIONS: Dict[str, str] = {
    "T": "Window size: the number of consecutive rows per non-overlapping window.",
    "window": (
        "Non-overlapping group of T consecutive rows in the deterministic "
        "capture-order proxy: window i holds sequence positions [T*i, T*i+T). "
        "Trailing rows that cannot fill a window are discarded."
    ),
    "identity": (
        f"'{IDENTITY_TRUE}' = the exact (Src IP, Dst IP) pair, available only in "
        f"the 84-column variant. '{IDENTITY_PROXY}' = the (Dst Port, Protocol) "
        "pair, available in every file. The proxy is a service-level key, not a "
        "conversation key: distinct host pairs talking to the same port collapse "
        "onto one proxy value, so proxy continuity is an upper bound on real "
        "endpoint continuity, never a substitute for it."
    ),
    "A_pct_windows_all_same_key": (
        "Percentage of windows whose T rows all share one identity value "
        "(n_unique_keys == 1). Generalises metric A from the T=5 experiment."
    ),
    "B_pct_windows_any_adjacent_same_key": (
        "Percentage of windows with at least one adjacent position j in "
        "{0..T-2} where row j and row j+1 share the identity value. Generalises "
        "metric B. Note B is structurally easier to satisfy as T grows, because "
        "it asks whether any 1 of T-1 links collides."
    ),
    "C_mean_unique_keys_per_window": "Mean of n_unique_keys over all windows; range 1..T.",
    "D_median_unique_keys_per_window": "Median of n_unique_keys over all windows.",
    "pct_windows_fully_heterogeneous": (
        "Percentage of windows with n_unique_keys == T, i.e. no two rows in the "
        "window share the identity value. This is the T-generalisation of the "
        "'5 distinct pairs' figure."
    ),
    "pct_adjacent_links_sharing_key": (
        "Share of all n_windows*(T-1) adjacent links whose two rows share the "
        "identity value. Unlike B this does not saturate as T grows, so it is the "
        "adjacency measure that can be compared across window sizes."
    ),
    "F_pct_windows_label_pure": "Percentage of windows whose T labels are identical.",
    "G_pct_windows_final_label_ne_majority": (
        "Percentage of windows where the final (T-th) row's label differs from the "
        "window's majority label. Ties are broken by ascending label string, so "
        "the measure is deterministic and conservative."
    ),
    "span_seconds": (
        "max(parsed Timestamp) - min(parsed Timestamp) over the window's T rows, "
        "in seconds. Diagnostic only: Timestamp never orders rows. A window "
        "containing an unparseable timestamp is excluded from span statistics only."
    ),
    "pct_windows_with_span_at_most": (
        "Cumulative share of span-valid windows at or below each threshold in "
        "seconds. Reported because the span distribution is bimodal, so a median "
        "describes neither mode."
    ),
    "random_baseline": (
        "The identical rows regrouped into windows of T by a seeded permutation "
        "and measured with the same code. Row content and marginal distributions "
        "are held fixed, so a difference isolates the effect of consecutive "
        "adjacency. Never used for training or splitting."
    ),
}


@dataclass
class ArmMetrics:
    """Measurements for one (file, T, identity, arm) combination."""

    arm: str = ""
    n_windows: int = 0
    n_all_same_key: int = 0
    n_any_adjacent_same_key: int = 0
    n_fully_heterogeneous: int = 0
    sum_unique_keys: int = 0
    median_unique_keys: float = 0.0
    key_histogram: Dict[int, int] = field(default_factory=dict)
    sum_adjacent_same_key: int = 0
    n_label_pure: int = 0
    n_final_ne_majority: int = 0
    label_histogram: Dict[int, int] = field(default_factory=dict)
    n_span_valid: int = 0
    n_same_second: int = 0
    span_median_seconds: float = 0.0
    span_p95_seconds: float = 0.0
    span_max_seconds: float = 0.0
    span_mean_seconds: float = 0.0
    span_buckets: Dict[int, int] = field(default_factory=dict)
    #: Distinct 5-tuples per window; only measured where the columns exist.
    sum_unique_five_tuples: Optional[int] = None
    n_all_same_five_tuple: Optional[int] = None

    def to_dict(self, window_size: int) -> Dict[str, Any]:
        n = self.n_windows
        payload: Dict[str, Any] = {
            "arm": self.arm,
            "n_windows": n,
            "A_pct_windows_all_same_key": _percent(self.n_all_same_key, n),
            "B_pct_windows_any_adjacent_same_key": _percent(
                self.n_any_adjacent_same_key, n
            ),
            "C_mean_unique_keys_per_window": (self.sum_unique_keys / n) if n else 0.0,
            "D_median_unique_keys_per_window": self.median_unique_keys,
            "pct_windows_fully_heterogeneous": _percent(self.n_fully_heterogeneous, n),
            "pct_adjacent_links_sharing_key": _percent(
                self.sum_adjacent_same_key, n * (window_size - 1)
            ),
            "mean_adjacent_same_key_per_window": (
                (self.sum_adjacent_same_key / n) if n else 0.0
            ),
            "F_pct_windows_label_pure": _percent(self.n_label_pure, n),
            "pct_windows_mixed_label": _percent(n - self.n_label_pure, n),
            "G_pct_windows_final_label_ne_majority": _percent(
                self.n_final_ne_majority, n
            ),
            "counts": {
                "all_same_key": self.n_all_same_key,
                "any_adjacent_same_key": self.n_any_adjacent_same_key,
                "fully_heterogeneous": self.n_fully_heterogeneous,
                "label_pure": self.n_label_pure,
                "mixed_label": n - self.n_label_pure,
                "final_ne_majority": self.n_final_ne_majority,
            },
            "n_windows_by_unique_key_count": {
                str(k): int(self.key_histogram.get(k, 0))
                for k in range(1, window_size + 1)
            },
            "pct_windows_by_unique_key_count": {
                str(k): _percent(int(self.key_histogram.get(k, 0)), n)
                for k in range(1, window_size + 1)
            },
            "n_windows_by_unique_label_count": {
                str(k): int(v) for k, v in sorted(self.label_histogram.items())
            },
            "timestamp_diagnostics": {
                "median_span_seconds": self.span_median_seconds,
                "p95_span_seconds": self.span_p95_seconds,
                "max_span_seconds": self.span_max_seconds,
                "mean_span_seconds": self.span_mean_seconds,
                "n_windows_with_a_valid_span": self.n_span_valid,
                "n_windows_entirely_within_one_second": self.n_same_second,
                "pct_windows_entirely_within_one_second": _percent(
                    self.n_same_second, n
                ),
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
        if self.sum_unique_five_tuples is not None:
            payload["mean_unique_five_tuples_per_window"] = (
                (self.sum_unique_five_tuples / n) if n else 0.0
            )
            payload["pct_windows_all_same_five_tuple"] = _percent(
                int(self.n_all_same_five_tuple or 0), n
            )
        return payload


@dataclass
class SweepCell:
    """One (file, T, identity) result: the consecutive arm plus its baseline."""

    source_file: str = ""
    window_size: int = 0
    identity: str = ""
    identity_columns: Tuple[str, ...] = ()
    n_rows_analyzed: int = 0
    n_windows: int = 0
    n_trailing_rows_discarded: int = 0
    consecutive: ArmMetrics = field(default_factory=ArmMetrics)
    random_baseline: Optional[ArmMetrics] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "source_file": self.source_file,
            "window_size": self.window_size,
            "identity": self.identity,
            "identity_columns": list(self.identity_columns),
            "n_rows_analyzed": self.n_rows_analyzed,
            "n_windows": self.n_windows,
            "n_trailing_rows_discarded": self.n_trailing_rows_discarded,
            "consecutive": self.consecutive.to_dict(self.window_size),
        }
        if self.random_baseline is not None:
            payload["random_baseline"] = self.random_baseline.to_dict(self.window_size)
        return payload


@dataclass
class FileColumns:
    """Decoded arrays for one file, read once and reused for every T."""

    source_file: str = ""
    source_path: str = ""
    schema_variant: str = ""
    n_columns_in_file: int = 0
    n_raw_rows: int = 0
    n_embedded_header_rows: int = 0
    n_usable_rows: int = 0
    n_rows_read: int = 0
    n_unparsed_timestamps: int = 0
    has_true_identity: bool = False
    missing_identifier_columns: Tuple[str, ...] = ()
    keys: Dict[str, np.ndarray] = field(default_factory=dict)
    label: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    stamp: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    five_tuple: Optional[np.ndarray] = None

    def provenance(self) -> Dict[str, Any]:
        return {
            "source_file": self.source_file,
            "source_path": self.source_path,
            "schema_variant": self.schema_variant,
            "n_columns_in_file": self.n_columns_in_file,
            "n_raw_data_rows": self.n_raw_rows,
            "n_embedded_header_rows_excluded": self.n_embedded_header_rows,
            "n_usable_rows": self.n_usable_rows,
            "n_rows_analyzed": self.n_rows_read,
            "n_unparsed_timestamps": self.n_unparsed_timestamps,
            "has_true_endpoint_identity": self.has_true_identity,
            "missing_identifier_columns": list(self.missing_identifier_columns),
        }


def identifier_inventory(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    """Which files carry which of the five identifier columns. Header read only.

    Task 1: this is measured from each file's own header, not assumed from the
    file name or from the 02-20 layout.
    """
    inventory: List[Dict[str, Any]] = []
    for path in sorted(paths, key=lambda p: p.name):
        header = inspect_header(path)
        present = [c for c in FIVE_TUPLE_COLUMNS if c in header.columns]
        missing = [c for c in FIVE_TUPLE_COLUMNS if c not in header.columns]
        inventory.append(
            {
                "source_file": path.name,
                "n_columns": header.n_columns,
                "schema_variant": header.variant.variant,
                "size_bytes": path.stat().st_size,
                "identifier_columns_present": present,
                "identifier_columns_missing": missing,
                "has_full_five_tuple": not missing,
                "has_true_endpoint_identity": all(
                    c in header.columns for c in TRUE_IDENTITY_COLUMNS
                ),
                "has_proxy_identity": all(
                    c in header.columns for c in PROXY_IDENTITY_COLUMNS
                ),
            }
        )
    return inventory


def read_file_columns(
    path: Path, *, chunk_size: int = 500_000, row_limit: Optional[int] = None
) -> FileColumns:
    """Read one file's identity, label and timestamp columns as dense codes.

    Reads only the columns that exist in that file. Embedded header rows are
    excluded using the same row accounting as the T=5 experiment, and no sort is
    applied: the array order *is* the file's data-row order, which within one
    file is the project's deterministic capture-order proxy.
    """
    header = inspect_header(path)
    available = set(header.columns)
    has_true = all(c in available for c in TRUE_IDENTITY_COLUMNS)
    missing_proxy = [c for c in PROXY_IDENTITY_COLUMNS if c not in available]
    if missing_proxy:
        raise AnalysisError(
            f"{path.name} lacks {missing_proxy}, so not even the proxy identity "
            "can be measured."
        )
    for required in (schema.TIMESTAMP_COLUMN, schema.LABEL_COLUMN):
        if required not in available:
            raise AnalysisError(f"{path.name} lacks the {required!r} column.")

    try:
        scout = scout_file(path, chunk_size=chunk_size, verify_line_count=True)
    except ProfilingError as exc:
        raise AnalysisError(f"{path.name}: row accounting failed: {exc}") from exc

    excluded = np.asarray(sorted(scout.embedded_header_positions), dtype=np.int64)
    excluded_set = set(excluded.tolist())
    wanted = min(scout.n_usable_rows, row_limit) if row_limit else scout.n_usable_rows
    if wanted <= 0:
        raise AnalysisError(f"{path.name}: no usable rows to analyze")

    usecols = [
        c
        for c in FIVE_TUPLE_COLUMNS
        if c in available
    ] + [schema.TIMESTAMP_COLUMN, schema.LABEL_COLUMN]
    has_five_tuple = all(c in available for c in FIVE_TUPLE_COLUMNS)

    parts: Dict[str, List[np.ndarray]] = {
        "true": [],
        "proxy": [],
        "five_tuple": [],
        "label": [],
        "stamp": [],
    }
    n_unparsed = 0
    position = 0
    kept = 0

    reader = pd.read_csv(
        path,
        usecols=usecols,
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
        if chunk.empty:
            continue

        remaining = wanted - kept
        if remaining <= 0:
            break
        if len(chunk) > remaining:
            chunk = chunk.iloc[:remaining]
        kept += len(chunk)

        def joined(columns: Sequence[str]) -> np.ndarray:
            # "|" cannot occur inside an IP, port or protocol value, so the
            # composite key is unambiguous.
            out = chunk[columns[0]].to_numpy().astype(str)
            for name in columns[1:]:
                out = np.char.add(
                    np.char.add(out, "|"), chunk[name].to_numpy().astype(str)
                )
            return out

        if has_true:
            parts["true"].append(joined(TRUE_IDENTITY_COLUMNS))
        parts["proxy"].append(joined(PROXY_IDENTITY_COLUMNS))
        if has_five_tuple:
            parts["five_tuple"].append(joined(FIVE_TUPLE_COLUMNS))
        parts["label"].append(chunk[schema.LABEL_COLUMN].to_numpy().astype(str))

        parsed = pd.to_datetime(
            chunk[schema.TIMESTAMP_COLUMN],
            format=schema.TIMESTAMP_FORMAT,
            errors="coerce",
            exact=True,
        )
        n_unparsed += int(parsed.isna().sum())
        # NaT through datetime64 -> float64 becomes the int64-min sentinel, which
        # is finite and would produce an absurd span. Re-mask it explicitly.
        seconds = parsed.to_numpy(dtype="datetime64[s]").astype(np.float64)
        seconds[parsed.isna().to_numpy()] = np.nan
        parts["stamp"].append(seconds)

        if kept >= wanted:
            break

    if kept < wanted:
        raise AnalysisError(
            f"{path.name}: expected {wanted:,} usable rows but read {kept:,}"
        )

    keys: Dict[str, np.ndarray] = {
        IDENTITY_PROXY: _encode(np.concatenate(parts["proxy"]))
    }
    if has_true:
        keys[IDENTITY_TRUE] = _encode(np.concatenate(parts["true"]))

    return FileColumns(
        source_file=path.name,
        source_path=str(path),
        schema_variant=header.variant.variant,
        n_columns_in_file=header.n_columns,
        n_raw_rows=scout.n_raw_rows,
        n_embedded_header_rows=len(scout.embedded_header_positions),
        n_usable_rows=scout.n_usable_rows,
        n_rows_read=kept,
        n_unparsed_timestamps=n_unparsed,
        has_true_identity=has_true,
        missing_identifier_columns=tuple(
            c for c in FIVE_TUPLE_COLUMNS if c not in available
        ),
        keys=keys,
        label=_encode(np.concatenate(parts["label"])),
        stamp=np.concatenate(parts["stamp"]),
        five_tuple=(
            _encode(np.concatenate(parts["five_tuple"])) if has_five_tuple else None
        ),
    )


def _measure_arm(
    *,
    arm: str,
    key: np.ndarray,
    label: np.ndarray,
    stamp: np.ndarray,
    five_tuple: Optional[np.ndarray],
    order: np.ndarray,
    window_size: int,
    batch_windows: int = 200_000,
) -> ArmMetrics:
    """Group ``order`` into windows of ``window_size`` and measure them."""
    n_windows = order.size // window_size
    stats = ArmMetrics(arm=arm, n_windows=n_windows)
    if n_windows == 0:
        return stats

    unique_chunks: List[np.ndarray] = []
    span_chunks: List[np.ndarray] = []
    span_valid_chunks: List[np.ndarray] = []
    key_hist: Dict[int, int] = {}
    label_hist: Dict[int, int] = {}
    sum_tuples = 0
    n_all_tuple = 0

    for start in range(0, n_windows, batch_windows):
        stop = min(start + batch_windows, n_windows)
        rows = order[start * window_size : stop * window_size].reshape(-1, window_size)

        window_keys = key[rows]
        n_unique = _row_wise_unique_counts(window_keys)
        adjacent = (window_keys[:, 1:] == window_keys[:, :-1]).sum(axis=1)

        stats.n_all_same_key += int((n_unique == 1).sum())
        stats.n_any_adjacent_same_key += int((adjacent >= 1).sum())
        stats.n_fully_heterogeneous += int((n_unique == window_size).sum())
        stats.sum_unique_keys += int(n_unique.sum())
        stats.sum_adjacent_same_key += int(adjacent.sum())
        for value, count in zip(*np.unique(n_unique, return_counts=True)):
            key_hist[int(value)] = key_hist.get(int(value), 0) + int(count)
        unique_chunks.append(n_unique)

        window_labels = label[rows]
        n_labels = _row_wise_unique_counts(window_labels)
        stats.n_label_pure += int((n_labels == 1).sum())
        stats.n_final_ne_majority += int((~_majority_is_fifth(window_labels)).sum())
        for value, count in zip(*np.unique(n_labels, return_counts=True)):
            label_hist[int(value)] = label_hist.get(int(value), 0) + int(count)

        if five_tuple is not None:
            window_tuples = five_tuple[rows]
            n_tuples = _row_wise_unique_counts(window_tuples)
            sum_tuples += int(n_tuples.sum())
            n_all_tuple += int((n_tuples == 1).sum())

        window_stamp = stamp[rows]
        valid = np.isfinite(window_stamp).all(axis=1)
        span = np.where(
            valid,
            np.nanmax(window_stamp, axis=1) - np.nanmin(window_stamp, axis=1),
            np.nan,
        )
        span_chunks.append(span)
        span_valid_chunks.append(valid)

    stats.key_histogram = key_hist
    stats.label_histogram = label_hist
    stats.median_unique_keys = float(np.median(np.concatenate(unique_chunks)))
    if five_tuple is not None:
        stats.sum_unique_five_tuples = sum_tuples
        stats.n_all_same_five_tuple = n_all_tuple

    span = np.concatenate(span_chunks)
    span_valid = np.concatenate(span_valid_chunks)
    stats.n_span_valid = int(span_valid.sum())
    stats.n_same_second = int((span_valid & (span == 0.0)).sum())
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
    return stats


def sweep_file(
    columns: FileColumns,
    *,
    window_sizes: Sequence[int] = WINDOW_SIZES,
    identities: Optional[Sequence[str]] = None,
    seed: int = 42,
    with_random_baseline: bool = True,
) -> List[SweepCell]:
    """Measure every (window size, identity) combination for one file."""
    if identities is None:
        identities = (
            [IDENTITY_TRUE, IDENTITY_PROXY]
            if columns.has_true_identity
            else [IDENTITY_PROXY]
        )
    n = int(columns.label.size)
    consecutive_order = np.arange(n, dtype=np.int64)
    # One permutation per file, derived from the seed alone, so every T and every
    # identity sees the same shuffle and the baselines stay comparable.
    shuffled = np.random.default_rng(seed).permutation(n) if with_random_baseline else None

    cells: List[SweepCell] = []
    for window_size in window_sizes:
        n_windows = n // window_size
        for identity in identities:
            key = columns.keys[identity]
            # The five-tuple detail only means something for the true identity.
            five_tuple = (
                columns.five_tuple if identity == IDENTITY_TRUE else None
            )
            cell = SweepCell(
                source_file=columns.source_file,
                window_size=window_size,
                identity=identity,
                identity_columns=(
                    TRUE_IDENTITY_COLUMNS
                    if identity == IDENTITY_TRUE
                    else PROXY_IDENTITY_COLUMNS
                ),
                n_rows_analyzed=n,
                n_windows=n_windows,
                n_trailing_rows_discarded=n - n_windows * window_size,
                consecutive=_measure_arm(
                    arm="consecutive",
                    key=key,
                    label=columns.label,
                    stamp=columns.stamp,
                    five_tuple=five_tuple,
                    order=consecutive_order,
                    window_size=window_size,
                ),
            )
            if shuffled is not None:
                cell.random_baseline = _measure_arm(
                    arm="random_baseline",
                    key=key,
                    label=columns.label,
                    stamp=columns.stamp,
                    five_tuple=five_tuple,
                    order=shuffled,
                    window_size=window_size,
                )
            cells.append(cell)
    return cells


#: Metrics checked against the published T=5 summary, as
#: (robustness key, published key, tolerance).
EQUIVALENCE_KEYS: Tuple[Tuple[str, str, float], ...] = (
    ("A_pct_windows_all_same_key", "A_pct_windows_all_five_same_src_dst_pair", 1e-9),
    (
        "B_pct_windows_any_adjacent_same_key",
        "B_pct_windows_any_adjacent_same_src_dst_pair",
        1e-9,
    ),
    ("C_mean_unique_keys_per_window", "C_mean_unique_pairs_per_window", 1e-9),
    ("D_median_unique_keys_per_window", "D_median_unique_pairs_per_window", 1e-9),
    (
        "pct_windows_fully_heterogeneous",
        "pct_windows_fully_heterogeneous_five_distinct_pairs",
        1e-9,
    ),
    (
        "pct_adjacent_links_sharing_key",
        "pct_adjacent_flow_pairs_sharing_src_dst",
        1e-9,
    ),
)


def equivalence_check(cell: SweepCell, published: Dict[str, Any]) -> Dict[str, Any]:
    """Prove this module reproduces the frozen T=5 result rather than assuming it.

    ``published`` is the ``continuity_summary.json`` payload from the first
    experiment. A mismatch means the generalised code changed a definition, which
    would invalidate every cross-T comparison built on it.
    """
    if cell.window_size != 5 or cell.identity != IDENTITY_TRUE:
        raise AnalysisError(
            "equivalence_check needs the T=5 true-identity cell, got "
            f"T={cell.window_size} identity={cell.identity}"
        )
    measured = cell.consecutive.to_dict(cell.window_size)
    reference = published["consecutive"]["endpoint_continuity"]
    ref_label = published["consecutive"]["label_continuity"]

    comparisons: List[Dict[str, Any]] = []
    for own_key, published_key, tolerance in EQUIVALENCE_KEYS:
        left = float(measured[own_key])
        right = float(reference[published_key])
        comparisons.append(
            {
                "metric": published_key,
                "robustness_value": left,
                "published_value": right,
                "abs_difference": abs(left - right),
                "match": abs(left - right) <= tolerance,
            }
        )
    for own_key, right in (
        ("F_pct_windows_label_pure", ref_label["F_pct_windows_label_pure"]),
        (
            "G_pct_windows_final_label_ne_majority",
            ref_label["G_pct_windows_fifth_label_ne_majority"],
        ),
    ):
        left = float(measured[own_key])
        comparisons.append(
            {
                "metric": own_key,
                "robustness_value": left,
                "published_value": float(right),
                "abs_difference": abs(left - float(right)),
                "match": abs(left - float(right)) <= 1e-9,
            }
        )
    structural = [
        {
            "metric": "n_windows",
            "robustness_value": cell.n_windows,
            "published_value": published["dataset"]["n_windows"],
            "abs_difference": abs(cell.n_windows - published["dataset"]["n_windows"]),
            "match": cell.n_windows == published["dataset"]["n_windows"],
        },
        {
            "metric": "n_rows_analyzed",
            "robustness_value": cell.n_rows_analyzed,
            "published_value": published["dataset"]["n_rows_analyzed"],
            "abs_difference": abs(
                cell.n_rows_analyzed - published["dataset"]["n_rows_analyzed"]
            ),
            "match": cell.n_rows_analyzed == published["dataset"]["n_rows_analyzed"],
        },
    ]
    comparisons = structural + comparisons
    return {
        "note": "The generalised T-sweep code must reproduce the frozen T=5 "
        "measurement exactly. Any mismatch invalidates the cross-T comparison.",
        "all_match": all(item["match"] for item in comparisons),
        "comparisons": comparisons,
    }


@dataclass
class RobustnessResult:
    """Everything one robustness run measured."""

    generated_at_utc: str = ""
    elapsed_seconds: float = 0.0
    seed: int = 0
    window_sizes: Tuple[int, ...] = WINDOW_SIZES
    ordering_rule: str = ""
    inventory: List[Dict[str, Any]] = field(default_factory=list)
    files: List[FileColumns] = field(default_factory=list)
    cells: List[SweepCell] = field(default_factory=list)
    equivalence: Optional[Dict[str, Any]] = None
    notes: List[str] = field(default_factory=list)

    def cell(
        self, *, source_file: str, window_size: int, identity: str
    ) -> Optional[SweepCell]:
        for item in self.cells:
            if (
                item.source_file == source_file
                and item.window_size == window_size
                and item.identity == identity
            ):
                return item
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at_utc": self.generated_at_utc,
            "elapsed_seconds": self.elapsed_seconds,
            "parameters": {
                "seed": self.seed,
                "window_sizes": list(self.window_sizes),
                "ordering_rule": self.ordering_rule,
                "true_identity_columns": list(TRUE_IDENTITY_COLUMNS),
                "proxy_identity_columns": list(PROXY_IDENTITY_COLUMNS),
            },
            "identifier_inventory": self.inventory,
            "files": [item.provenance() for item in self.files],
            "equivalence_with_published_T5": self.equivalence,
            "cells": [item.to_dict() for item in self.cells],
            "definitions": ROBUSTNESS_DEFINITIONS,
            "notes": self.notes,
        }

    def comparison_frame(self) -> pd.DataFrame:
        """The Task 4 comparison table, one row per (T, file, identity)."""
        rows: List[Dict[str, Any]] = []
        for cell in self.cells:
            arm = cell.consecutive.to_dict(cell.window_size)
            baseline = (
                cell.random_baseline.to_dict(cell.window_size)
                if cell.random_baseline
                else None
            )
            rows.append(
                {
                    "T": cell.window_size,
                    "file": cell.source_file,
                    "identity": cell.identity,
                    "windows": cell.n_windows,
                    "trailing_rows_discarded": cell.n_trailing_rows_discarded,
                    "same_pair_pct": arm["A_pct_windows_all_same_key"],
                    "adjacent_pair_pct": arm["B_pct_windows_any_adjacent_same_key"],
                    "adjacent_pair_pct_random": (
                        baseline["B_pct_windows_any_adjacent_same_key"]
                        if baseline
                        else None
                    ),
                    "adjacent_links_pct": arm["pct_adjacent_links_sharing_key"],
                    "adjacent_links_pct_random": (
                        baseline["pct_adjacent_links_sharing_key"]
                        if baseline
                        else None
                    ),
                    "mean_unique_pairs": arm["C_mean_unique_keys_per_window"],
                    "median_unique_pairs": arm["D_median_unique_keys_per_window"],
                    "fully_heterogeneous_pct": arm["pct_windows_fully_heterogeneous"],
                    "label_pure_pct": arm["F_pct_windows_label_pure"],
                    "final_ne_majority_pct": arm[
                        "G_pct_windows_final_label_ne_majority"
                    ],
                    "label_pure_pct_random": (
                        baseline["F_pct_windows_label_pure"] if baseline else None
                    ),
                    "median_span_seconds": arm["timestamp_diagnostics"][
                        "median_span_seconds"
                    ],
                    "p95_span_seconds": arm["timestamp_diagnostics"][
                        "p95_span_seconds"
                    ],
                    "max_span_seconds": arm["timestamp_diagnostics"][
                        "max_span_seconds"
                    ],
                    "same_second_pct": arm["timestamp_diagnostics"][
                        "pct_windows_entirely_within_one_second"
                    ],
                }
            )
        frame = pd.DataFrame(rows)
        return frame.sort_values(["identity", "T", "file"]).reset_index(drop=True)


ORDERING_RULE = (
    "deterministic capture-order proxy: (source file capture date, "
    "Original_Index). Within one file this is the file's own data-row order; no "
    "sort is applied and Timestamp is never an ordering key."
)


def run_robustness(
    paths: Sequence[Path],
    *,
    seed: int = 42,
    window_sizes: Sequence[int] = WINDOW_SIZES,
    chunk_size: int = 500_000,
    row_limit: Optional[int] = None,
    with_random_baseline: bool = True,
    published_summary: Optional[Dict[str, Any]] = None,
    progress: Optional[Any] = None,
) -> RobustnessResult:
    """Measure the window-size sweep across every given file."""
    started = datetime.now(timezone.utc)
    result = RobustnessResult(
        generated_at_utc=started.isoformat(),
        seed=seed,
        window_sizes=tuple(window_sizes),
        ordering_rule=ORDERING_RULE,
        inventory=identifier_inventory(paths),
    )

    for path in sorted(paths, key=lambda p: p.name):
        if progress:
            progress(f"reading {path.name} ...")
        columns = read_file_columns(path, chunk_size=chunk_size, row_limit=row_limit)
        result.files.append(columns)
        if progress:
            progress(
                f"  {columns.n_rows_read:,} rows | "
                f"identity: {'true + proxy' if columns.has_true_identity else 'proxy only'}"
            )
        result.cells.extend(
            sweep_file(
                columns,
                window_sizes=window_sizes,
                seed=seed,
                with_random_baseline=with_random_baseline,
            )
        )
        # Arrays for one file can be hundreds of MB; drop them once measured so
        # ten files do not accumulate.
        columns.keys = {}
        columns.label = np.empty(0, dtype=np.int64)
        columns.stamp = np.empty(0, dtype=np.float64)
        columns.five_tuple = None

    if published_summary is not None and row_limit:
        # A truncated run measures different rows, so a mismatch here would say
        # nothing about definition drift. Skip rather than report a false alarm.
        result.notes.append(
            "The equivalence check against the published T=5 summary was skipped "
            f"because row_limit={row_limit:,} truncates the input; it only means "
            "anything on a whole-file run."
        )
    elif published_summary is not None:
        reference_file = published_summary["dataset"]["source_file"]
        cell = result.cell(
            source_file=reference_file, window_size=5, identity=IDENTITY_TRUE
        )
        if cell is not None:
            result.equivalence = equivalence_check(cell, published_summary)
        else:
            result.notes.append(
                f"No T=5 true-identity cell for {reference_file} in this run, so the "
                "equivalence check against the published T=5 summary was skipped."
            )

    if row_limit:
        result.notes.append(
            f"row_limit={row_limit:,} was applied: truncated debug run, not a "
            "measurement of the whole files."
        )
    for item in result.files:
        if item.n_unparsed_timestamps:
            result.notes.append(
                f"{item.source_file}: {item.n_unparsed_timestamps:,} timestamp "
                "value(s) did not parse and are excluded from span statistics only."
            )
    result.elapsed_seconds = (datetime.now(timezone.utc) - started).total_seconds()
    return result
