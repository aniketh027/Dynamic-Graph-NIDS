"""Stage B: multi-day intake and global corpus profiling.

Establishes the factual foundation for every later stage by measuring the whole
CIC-IDS2018 CSV corpus. **Measure-only**: no value is replaced, no row is
removed, no file is written into ``data/raw``, and no label is encoded.

What this module measures
-------------------------
1. **Discovery** - every CSV under the corpus root, with an explicitly derived
   capture date, ordered deterministically by that date.
2. **Schema** - each file's header classified against the two known layouts
   (80-column standard, 84-column extended). Extra columns are reported, never
   silently discarded.
3. **Intake** - raw row count, embedded header rows (``Label == "Label"``),
   usable row count, and whether the file sits on the Excel row cap.
4. **Labels** - the exact label strings, per file, per capture date and globally.
5. **Class-by-time** - each class's first/last capture date and whether a naive
   chronological 70/10/20 split would leave a class unseen during training.
6. **Non-finite values** - ``NaN`` / ``+inf`` / ``-inf`` per column per file.
7. **Negatives** - per column, with the ``-1`` CICFlowMeter sentinel identified.
8. **Duplicates** - exact / temporal / label-conflicting, per file and globally.
9. **Timestamps** - parse rate, range, tie groups, AM/PM markers. Diagnostic only.

The ordering contract
---------------------
The project's deterministic sequencing order is::

    source file capture date  ->  original row position within that file

This is a **deterministic capture-order proxy**, not physical chronology: the
raw ``Timestamp`` is on a 12-hour clock with no AM/PM marker and is tied up to
2,050 rows deep, so it cannot order rows. Every row stays addressable as
``(source_file, source_file_date, original_index)``.

Memory strategy
---------------
Each file is read in two passes. Pass 1 reads only ``Label`` (cheap: ~1.3 s for
358 MB) to locate the embedded header rows. Pass 2 then reads the 80 standard
columns in chunks with the feature dtypes forced to ``float64`` and the header
lines skipped, so no chunk can ever degrade to Python-object strings. Peak
memory is one chunk (~125 MB at the default 200k rows) plus the duplicate hash
accumulators - never a concatenated corpus frame.
"""

from __future__ import annotations

import hashlib
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.preprocessing import schema
from src.preprocessing.loader import (
    LoaderError,
    RowCountMismatchError,
    count_physical_lines,
    read_header,
)

#: ``MM-DD-YYYY`` anywhere in a CSV file name (the CIC-IDS2018 CSV convention).
CSV_DATE_PATTERN = re.compile(r"(?<!\d)(\d{2})-(\d{2})-(\d{4})(?!\d)")

#: ``DD-MM-YYYY`` in a Parquet file name (that copy uses the opposite order).
PARQUET_DATE_PATTERN = re.compile(r"-(\d{2})-(\d{2})-(\d{4})_")

#: Odd 64-bit constant used to mix two row hashes into one (as in the loader).
_HASH_MIX = np.uint64(0x9E3779B97F4A7C15)

#: Sentinel used by the vectorised finite-extremum accumulators.
_NEG_INF = float("-inf")
_POS_INF = float("inf")


class ProfilingError(LoaderError):
    """Raised when the corpus cannot be profiled as configured."""


class CorpusNotFoundError(ProfilingError):
    """The configured CSV corpus directory is missing or holds no CSV files."""


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def parse_capture_date(name: str, *, pattern: re.Pattern[str] = CSV_DATE_PATTERN,
                       order: str = "MDY") -> Optional[date]:
    """Capture date encoded in a file name, or ``None``.

    Deterministic and independent of the filesystem: the date comes from the
    name's own digits. ``order`` is ``"MDY"`` for the CSV convention
    (``02-14-2018`` = 14 Feb 2018) or ``"DMY"`` for the Parquet convention
    (``...-14-02-2018_...``).
    """
    match = pattern.search(name)
    if match is None:
        return None
    first, second, year = (int(part) for part in match.groups())
    month, day = (first, second) if order == "MDY" else (second, first)
    try:
        return date(year, month, day)
    except ValueError:
        return None


@dataclass
class DiscoveredFile:
    """One discovered corpus file, before any of its content is read."""

    source_file: str
    path: Path
    size_bytes: int
    #: Capture date used for sequencing.
    capture_date: Optional[date] = None
    #: How :attr:`capture_date` was derived (``"filename"`` / ``"unknown"``).
    capture_date_source: str = "unknown"
    #: Date implied by the file's own ``Timestamp`` values, filled in by the
    #: data pass and cross-checked against :attr:`capture_date`.
    observed_date: Optional[date] = None

    @property
    def size_gib(self) -> float:
        return self.size_bytes / (1 << 30)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_file": self.source_file,
            "path": str(self.path),
            "size_bytes": self.size_bytes,
            "size_gib": round(self.size_gib, 4),
            "capture_date": self.capture_date.isoformat() if self.capture_date else None,
            "capture_date_source": self.capture_date_source,
            "observed_date": self.observed_date.isoformat() if self.observed_date else None,
        }


def discover_csv_files(
    csv_dir: str | Path, patterns: Sequence[str] = ("*.csv",)
) -> List[DiscoveredFile]:
    """Recursively discover corpus CSVs, ordered deterministically.

    Filenames are never hard-coded. The returned order is by
    ``(capture_date, source_file)`` with undated files last, so it does not
    depend on the order the filesystem happens to return.
    """
    root = Path(csv_dir)
    if not root.is_dir():
        raise CorpusNotFoundError(
            f"CSV corpus directory not found: {root}\n"
            "Place the CIC-IDS2018 CSV files under the configured raw data dir."
        )

    seen: Dict[Path, DiscoveredFile] = {}
    for pattern in patterns:
        for path in root.rglob(pattern):
            if not path.is_file():
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            capture_date = parse_capture_date(path.name)
            seen[resolved] = DiscoveredFile(
                source_file=path.name,
                path=resolved,
                size_bytes=path.stat().st_size,
                capture_date=capture_date,
                capture_date_source="filename" if capture_date else "unknown",
            )

    if not seen:
        raise CorpusNotFoundError(
            f"No CSV files matching {list(patterns)} under {root}."
        )
    return sequence_files(seen.values())


def sequence_files(files: Iterable[DiscoveredFile]) -> List[DiscoveredFile]:
    """Order files by the deterministic capture-order proxy.

    Sort key is ``(capture_date, source_file)``; undated files sort last but
    still deterministically, by name. ``source_file`` breaks ties so two files
    sharing a capture date can never swap places between runs.
    """
    return sorted(
        files,
        key=lambda item: (
            item.capture_date is None,
            item.capture_date or date.min,
            item.source_file,
        ),
    )


def discover_parquet_files(
    parquet_dir: str | Path, patterns: Sequence[str] = ("*.parquet",)
) -> List[DiscoveredFile]:
    """Discover the Parquet copy (for the metadata comparison only)."""
    root = Path(parquet_dir)
    if not root.is_dir():
        return []
    found: Dict[Path, DiscoveredFile] = {}
    for pattern in patterns:
        for path in root.rglob(pattern):
            if not path.is_file():
                continue
            resolved = path.resolve()
            if resolved in found:
                continue
            capture_date = parse_capture_date(
                path.name, pattern=PARQUET_DATE_PATTERN, order="DMY"
            )
            found[resolved] = DiscoveredFile(
                source_file=path.name,
                path=resolved,
                size_bytes=path.stat().st_size,
                capture_date=capture_date,
                capture_date_source="filename" if capture_date else "unknown",
            )
    return sequence_files(found.values())


# --------------------------------------------------------------------------- #
# Row identity
# --------------------------------------------------------------------------- #
def kth_kept_positions(ranks: np.ndarray, excluded: np.ndarray) -> np.ndarray:
    """Original row positions of the usable rows at the given ranks.

    ``ranks`` are 0-based positions among the *kept* rows; ``excluded`` is the
    sorted array of data-row positions that were removed (the embedded header
    rows). Returns each rank's position in the original, unfiltered row
    sequence — i.e. its ``Original_Index``.

    Uses the "k-th missing integer" identity: exactly ``excluded[j] - j`` kept
    rows precede ``excluded[j]``, so the number of excluded rows at or below the
    answer is ``searchsorted(excluded - arange, rank, "right")``. This avoids
    materialising a keep-mask over the whole file (64 MB for the largest file).
    """
    ranks = np.asarray(ranks, dtype=np.int64)
    if excluded.size == 0:
        return ranks
    shifted = np.asarray(excluded, dtype=np.int64) - np.arange(
        excluded.size, dtype=np.int64
    )
    return ranks + np.searchsorted(shifted, ranks, side="right")


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #
@dataclass
class HeaderInfo:
    """The file's header line, exactly as measured."""

    n_columns: int = 0
    md5_stripped: str = ""
    md5_raw_bytes: str = ""
    line_ending: str = ""
    columns: List[str] = field(default_factory=list)
    variant: schema.SchemaVariant = field(default_factory=schema.SchemaVariant)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_columns": self.n_columns,
            "md5_stripped": self.md5_stripped,
            "md5_raw_bytes": self.md5_raw_bytes,
            "line_ending": self.line_ending,
            "schema": self.variant.to_dict(),
        }


@dataclass
class IntakeStats:
    """Row accounting for one file. Nothing is removed from the raw file."""

    n_raw_rows: int = 0
    n_physical_lines: Optional[int] = None
    line_count_verified: bool = False
    n_embedded_header_rows: int = 0
    embedded_header_positions: List[int] = field(default_factory=list)
    embedded_header_positions_truncated: bool = False
    n_usable_rows: int = 0
    n_rows_profiled: int = 0
    row_limit: Optional[int] = None
    at_excel_row_cap: bool = False
    first_original_index: Optional[int] = None
    last_original_index: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_raw_rows": self.n_raw_rows,
            "n_physical_lines": self.n_physical_lines,
            "line_count_verified": self.line_count_verified,
            "n_embedded_header_rows": self.n_embedded_header_rows,
            "embedded_header_positions": self.embedded_header_positions,
            "embedded_header_positions_truncated": self.embedded_header_positions_truncated,
            "n_usable_rows": self.n_usable_rows,
            "n_rows_profiled": self.n_rows_profiled,
            "row_limit": self.row_limit,
            "at_excel_row_cap": self.at_excel_row_cap,
            "excel_row_cap": schema.EXCEL_ROW_CAP,
            "first_original_index": self.first_original_index,
            "last_original_index": self.last_original_index,
        }


@dataclass
class NonFiniteStats:
    """Non-finite value counts. Measured only — Stage B replaces nothing."""

    nan_counts: Dict[str, int] = field(default_factory=dict)
    posinf_counts: Dict[str, int] = field(default_factory=dict)
    neginf_counts: Dict[str, int] = field(default_factory=dict)
    total_nan: int = 0
    total_posinf: int = 0
    total_neginf: int = 0
    #: Columns holding a non-finite value that are not the two known rate columns.
    unexpected_columns: List[str] = field(default_factory=list)
    policy_note: str = (
        "Stage B measures only. Finite replacement values will be derived from "
        "the TRAIN split alone in Stage C and then applied to train/val/test."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nan_counts": self.nan_counts,
            "posinf_counts": self.posinf_counts,
            "neginf_counts": self.neginf_counts,
            "total_nan": self.total_nan,
            "total_posinf": self.total_posinf,
            "total_neginf": self.total_neginf,
            "total_non_finite": self.total_nan + self.total_posinf + self.total_neginf,
            "unexpected_columns": self.unexpected_columns,
            "policy_note": self.policy_note,
        }


@dataclass
class NegativeStats:
    """Negative-value counts. Nothing is clipped or replaced."""

    negative_counts: Dict[str, int] = field(default_factory=dict)
    #: Per column, the range of the negative values found (max, min).
    negative_extremes: Dict[str, Dict[str, float]] = field(default_factory=dict)
    #: Count of the exact ``-1`` CICFlowMeter "not observed" sentinel.
    sentinel_minus_one_counts: Dict[str, int] = field(default_factory=dict)
    #: Columns whose negatives are all exactly ``-1``.
    only_minus_one_columns: List[str] = field(default_factory=list)
    #: Negatives outside the two known sentinel columns (overflow artefacts).
    non_sentinel_columns: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "negative_counts": self.negative_counts,
            "negative_extremes": self.negative_extremes,
            "sentinel_minus_one_counts": self.sentinel_minus_one_counts,
            "only_minus_one_columns": self.only_minus_one_columns,
            "non_sentinel_columns": self.non_sentinel_columns,
            "policy_note": (
                "-1 is the CICFlowMeter 'not observed' sentinel and is preserved. "
                "No clipping or replacement happens in Stage B."
            ),
        }


@dataclass
class TimestampProfile:
    """Timestamp diagnostics. Never used for ordering."""

    column: str = schema.TIMESTAMP_COLUMN
    format: str = schema.TIMESTAMP_FORMAT
    n_parsed: int = 0
    n_unparsed: int = 0
    unparsed_examples: List[str] = field(default_factory=list)
    n_distinct: int = 0
    min_parsed: Optional[str] = None
    max_parsed: Optional[str] = None
    hour_histogram: Dict[str, int] = field(default_factory=dict)
    date_histogram: Dict[str, int] = field(default_factory=dict)
    n_rows_with_am_pm_marker: int = 0
    n_rows_off_capture_date: int = 0
    off_capture_date_examples: List[str] = field(default_factory=list)
    #: Same-second tie groups (a "group" is one distinct timestamp value).
    n_tie_groups: int = 0
    n_singleton_groups: int = 0
    tie_group_max: int = 0
    tie_group_mean: float = 0.0
    tie_group_median: float = 0.0
    tie_group_p95: float = 0.0
    n_rows_in_tie_groups: int = 0
    monotonic_in_file_order: Optional[bool] = None
    n_decreasing_steps: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column": self.column,
            "format": self.format,
            "n_parsed": self.n_parsed,
            "n_unparsed": self.n_unparsed,
            "unparsed_examples": self.unparsed_examples,
            "n_distinct": self.n_distinct,
            "min_parsed": self.min_parsed,
            "max_parsed": self.max_parsed,
            "hour_histogram": self.hour_histogram,
            "date_histogram": self.date_histogram,
            "n_rows_with_am_pm_marker": self.n_rows_with_am_pm_marker,
            "n_rows_off_capture_date": self.n_rows_off_capture_date,
            "off_capture_date_examples": self.off_capture_date_examples,
            "tie_groups": {
                "n_groups": self.n_tie_groups,
                "n_singleton_groups": self.n_singleton_groups,
                "max": self.tie_group_max,
                "mean": round(self.tie_group_mean, 3),
                "median": self.tie_group_median,
                "p95": self.tie_group_p95,
                "n_rows_in_multi_row_groups": self.n_rows_in_tie_groups,
            },
            "monotonic_in_file_order": self.monotonic_in_file_order,
            "n_decreasing_steps": self.n_decreasing_steps,
            "ordering_note": (
                "Diagnostic only. Ordering uses capture date + original_index; the "
                "raw hour field holds 01-12 with no AM/PM marker and is heavily tied."
            ),
        }


@dataclass
class DuplicateProfile:
    """The three duplicate categories. Nothing is ever removed."""

    n_rows: int = 0
    n_distinct_feature_vectors: int = 0
    #: A. same 78 features + same Timestamp + same Label.
    n_exact: int = 0
    #: B. same 78 features + same Label, Timestamp differs.
    n_temporal: int = 0
    #: A + B.
    n_feature_label_duplicates: int = 0
    #: C. feature vectors carrying more than one distinct label.
    n_conflicting_groups: int = 0
    n_conflicting_rows: int = 0
    conflicting_label_combinations: Dict[str, Dict[str, int]] = field(default_factory=dict)
    n_conflicting_combinations: int = 0
    basis: str = ""
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_distinct_feature_vectors": self.n_distinct_feature_vectors,
            "A_exact": self.n_exact,
            "B_temporal": self.n_temporal,
            "AB_feature_label_duplicates": self.n_feature_label_duplicates,
            "C_conflicting_groups": self.n_conflicting_groups,
            "C_conflicting_rows": self.n_conflicting_rows,
            "C_n_label_combinations": self.n_conflicting_combinations,
            "C_label_combinations": self.conflicting_label_combinations,
            "basis": self.basis,
            "note": self.note,
        }


@dataclass
class FileProfile:
    """Everything Stage B measured about one CSV file."""

    file: DiscoveredFile
    header: HeaderInfo = field(default_factory=HeaderInfo)
    intake: IntakeStats = field(default_factory=IntakeStats)
    label_counts: Dict[str, int] = field(default_factory=dict)
    non_finite: NonFiniteStats = field(default_factory=NonFiniteStats)
    negatives: NegativeStats = field(default_factory=NegativeStats)
    timestamps: TimestampProfile = field(default_factory=TimestampProfile)
    duplicates: Optional[DuplicateProfile] = None
    #: Finite min/max per model feature column (Stage C needs these for scaling).
    feature_min: Dict[str, float] = field(default_factory=dict)
    feature_max: Dict[str, float] = field(default_factory=dict)
    constant_columns: List[str] = field(default_factory=list)
    all_zero_columns: List[str] = field(default_factory=list)
    #: Set when the file could not be profiled; the run continues regardless.
    error: Optional[str] = None
    profile_seconds: float = 0.0

    @property
    def source_file(self) -> str:
        return self.file.source_file

    @property
    def capture_date(self) -> Optional[date]:
        return self.file.capture_date

    @property
    def usable(self) -> bool:
        return self.error is None and self.header.variant.usable

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.file.to_dict(),
            "header": self.header.to_dict(),
            "intake": self.intake.to_dict(),
            "labels": {
                "n_labels": len(self.label_counts),
                "counts": self.label_counts,
            },
            "non_finite": self.non_finite.to_dict(),
            "negatives": self.negatives.to_dict(),
            "timestamps": self.timestamps.to_dict(),
            "duplicates": self.duplicates.to_dict() if self.duplicates else None,
            "feature_ranges": {
                "min": self.feature_min,
                "max": self.feature_max,
                "constant_columns": self.constant_columns,
                "all_zero_columns": self.all_zero_columns,
            },
            "error": self.error,
            "profile_seconds": round(self.profile_seconds, 2),
        }


@dataclass
class ScoutResult:
    """Pass 1 output for one file: row accounting and the label column only."""

    source_file: str
    n_raw_rows: int = 0
    embedded_header_positions: List[int] = field(default_factory=list)
    n_usable_rows: int = 0
    #: Label counts over the usable rows (the ``"Label"`` pseudo-label removed).
    label_counts: Dict[str, int] = field(default_factory=dict)
    n_physical_lines: Optional[int] = None
    error: Optional[str] = None


@dataclass
class SplitCoverage:
    """Whether a naive chronological split leaves classes out of training.

    Analysis only. Stage B never changes the split strategy and never uses
    random stratification.
    """

    ratios: Tuple[float, float, float] = (0.70, 0.10, 0.20)
    n_rows: int = 0
    train_end: int = 0
    val_end: int = 0
    train_counts: Dict[str, int] = field(default_factory=dict)
    val_counts: Dict[str, int] = field(default_factory=dict)
    test_counts: Dict[str, int] = field(default_factory=dict)
    absent_from_train: List[str] = field(default_factory=list)
    absent_from_val: List[str] = field(default_factory=list)
    absent_from_test: List[str] = field(default_factory=list)
    train_only: List[str] = field(default_factory=list)
    #: Classes with fewer than this many training rows are flagged as starved.
    starved_threshold: int = 100
    starved_in_train: Dict[str, int] = field(default_factory=dict)
    #: The capture date each split boundary falls on.
    boundary_files: Dict[str, Optional[str]] = field(default_factory=dict)
    exact: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ratios": {
                "train": self.ratios[0],
                "validation": self.ratios[1],
                "test": self.ratios[2],
            },
            "n_rows": self.n_rows,
            "boundaries": {
                "train": [0, self.train_end],
                "validation": [self.train_end, self.val_end],
                "test": [self.val_end, self.n_rows],
            },
            "boundary_files": self.boundary_files,
            "counts": {
                "train": self.train_counts,
                "validation": self.val_counts,
                "test": self.test_counts,
            },
            "absent_from_train": self.absent_from_train,
            "absent_from_validation": self.absent_from_val,
            "absent_from_test": self.absent_from_test,
            "train_only": self.train_only,
            "starved_in_train": self.starved_in_train,
            "starved_threshold": self.starved_threshold,
            "exact_row_level": self.exact,
            "note": (
                "Diagnostic only. Stage B does not change the split strategy and "
                "does not use random stratification."
            ),
        }


@dataclass
class ClassTimeline:
    """Per-class first/last appearance across capture dates."""

    #: label -> {ISO date -> count}
    per_date: Dict[str, Dict[str, int]] = field(default_factory=dict)
    first_date: Dict[str, str] = field(default_factory=dict)
    last_date: Dict[str, str] = field(default_factory=dict)
    n_dates_present: Dict[str, int] = field(default_factory=dict)
    #: Classes appearing on exactly one capture date.
    single_date_classes: List[str] = field(default_factory=list)
    #: Classes whose first appearance is in the final third of the date sequence.
    late_only_classes: List[str] = field(default_factory=list)
    dates: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "capture_dates": self.dates,
            "per_class_per_date": self.per_date,
            "first_date": self.first_date,
            "last_date": self.last_date,
            "n_dates_present": self.n_dates_present,
            "single_date_classes": self.single_date_classes,
            "late_only_classes": self.late_only_classes,
        }


@dataclass
class ParquetComparison:
    """Metadata-only comparison against the Parquet copy.

    The CSV corpus is the primary temporal source: the Parquet copy has no
    ``Timestamp`` column, so it cannot support the temporal HTGraph pipeline.
    """

    n_files: int = 0
    total_size_bytes: int = 0
    total_rows: int = 0
    files: List[Dict[str, Any]] = field(default_factory=list)
    has_timestamp: Dict[str, bool] = field(default_factory=dict)
    has_label: Dict[str, bool] = field(default_factory=dict)
    #: CSV capture date -> matched Parquet file name.
    date_correspondence: Dict[str, Optional[str]] = field(default_factory=dict)
    unmatched_csv_dates: List[str] = field(default_factory=list)
    unmatched_parquet_files: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_temporal_source": "CSV",
            "reason": (
                "The Parquet copy does not carry the Timestamp column, so it cannot "
                "support the temporal HTGraph pipeline. It is compared here on "
                "metadata only and is never read by the pipeline."
            ),
            "n_files": self.n_files,
            "total_size_bytes": self.total_size_bytes,
            "total_rows": self.total_rows,
            "files": self.files,
            "has_timestamp": self.has_timestamp,
            "has_label": self.has_label,
            "date_correspondence": self.date_correspondence,
            "unmatched_csv_dates": self.unmatched_csv_dates,
            "unmatched_parquet_files": self.unmatched_parquet_files,
            "error": self.error,
        }


@dataclass
class CorpusProfile:
    """The complete Stage B measurement of the CSV corpus."""

    csv_dir: Path
    files: List[FileProfile] = field(default_factory=list)
    #: Deterministic sequence: capture date, then original row order within a file.
    sequence: List[str] = field(default_factory=list)
    ordering_contract: str = (
        "deterministic capture-order proxy = source-file capture date, then "
        "Original_Index (original row position within that file)"
    )
    total_raw_rows: int = 0
    total_embedded_header_rows: int = 0
    total_usable_rows: int = 0
    total_rows_profiled: int = 0
    total_size_bytes: int = 0
    label_counts: Dict[str, int] = field(default_factory=dict)
    label_vocabulary: List[str] = field(default_factory=list)
    timeline: ClassTimeline = field(default_factory=ClassTimeline)
    split_coverage: SplitCoverage = field(default_factory=SplitCoverage)
    global_duplicates: Optional[DuplicateProfile] = None
    parquet: Optional[ParquetComparison] = None
    schema_variants: Dict[str, List[str]] = field(default_factory=dict)
    truncated_files: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    row_limit: Optional[int] = None
    elapsed_seconds: float = 0.0

    @property
    def dates(self) -> List[str]:
        seen: List[str] = []
        for profile in self.files:
            iso = profile.capture_date.isoformat() if profile.capture_date else "unknown"
            if iso not in seen:
                seen.append(iso)
        return seen

    def by_name(self, source_file: str) -> Optional[FileProfile]:
        for profile in self.files:
            if profile.source_file == source_file:
                return profile
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "csv_dir": str(self.csv_dir),
            "ordering_contract": self.ordering_contract,
            "sequence": self.sequence,
            "capture_dates": self.dates,
            "n_files": len(self.files),
            "totals": {
                "size_bytes": self.total_size_bytes,
                "size_gib": round(self.total_size_bytes / (1 << 30), 4),
                "raw_rows": self.total_raw_rows,
                "embedded_header_rows": self.total_embedded_header_rows,
                "usable_rows": self.total_usable_rows,
                "rows_profiled": self.total_rows_profiled,
            },
            "row_limit": self.row_limit,
            "schema_variants": self.schema_variants,
            "labels": {
                "n_labels": len(self.label_vocabulary),
                "vocabulary": self.label_vocabulary,
                "counts": self.label_counts,
                "note": (
                    "Original dataset spelling and casing preserved exactly. No "
                    "label encoding or class mapping is created in Stage B."
                ),
            },
            "class_timeline": self.timeline.to_dict(),
            "naive_chronological_split": self.split_coverage.to_dict(),
            "global_duplicates": (
                self.global_duplicates.to_dict() if self.global_duplicates else None
            ),
            "parquet_comparison": self.parquet.to_dict() if self.parquet else None,
            "truncated_files": self.truncated_files,
            "warnings": self.warnings,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }

    def summary(self) -> str:
        lines = [
            f"csv dir              : {self.csv_dir}",
            f"files profiled       : {len(self.files)}"
            f" ({self.total_size_bytes / (1 << 30):.2f} GiB)",
            f"capture dates        : {', '.join(self.dates)}",
            f"raw data rows        : {self.total_raw_rows:,}",
            f"embedded header rows : {self.total_embedded_header_rows:,}",
            f"usable rows          : {self.total_usable_rows:,}",
            f"rows profiled        : {self.total_rows_profiled:,}"
            + (f" (row limit {self.row_limit:,})" if self.row_limit else ""),
            f"label vocabulary     : {len(self.label_vocabulary)} classes",
            f"schema variants      : "
            + ", ".join(f"{k}={len(v)}" for k, v in sorted(self.schema_variants.items())),
            f"truncated files      : {len(self.truncated_files)}"
            f" (at the {schema.EXCEL_ROW_CAP:,}-row Excel cap)",
            f"elapsed              : {self.elapsed_seconds:.1f}s",
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Header inspection
# --------------------------------------------------------------------------- #
def inspect_header(path: Path) -> HeaderInfo:
    """Read and classify one file's header line. Opens the file read-only.

    Records both md5 variants because they answer different questions: the
    stripped md5 says "are the column *names* identical" (it is, for all nine
    80-column files), while the raw-bytes md5 also captures the line ending
    (``03-02-2018.csv`` and ``02-20-2018.csv`` use CRLF, the rest LF).
    """
    with path.open("rb") as handle:
        raw = handle.readline()
    if raw.endswith(b"\r\n"):
        line_ending = "CRLF"
    elif raw.endswith(b"\n"):
        line_ending = "LF"
    elif raw.endswith(b"\r"):
        line_ending = "CR"
    else:
        line_ending = "none"

    text = raw.decode("utf-8-sig", errors="replace").rstrip("\r\n")
    names = [name.strip() for name in text.split(",")] if text else []
    return HeaderInfo(
        n_columns=len(names),
        md5_stripped=hashlib.md5(text.encode("utf-8")).hexdigest(),
        md5_raw_bytes=hashlib.md5(raw).hexdigest(),
        line_ending=line_ending,
        columns=names,
        variant=schema.detect_schema_variant(names),
    )


# --------------------------------------------------------------------------- #
# Pass 1: the label column only
# --------------------------------------------------------------------------- #
def scout_file(
    path: Path,
    *,
    chunk_size: int = 500_000,
    verify_line_count: bool = True,
) -> ScoutResult:
    """Read only ``Label`` to get row accounting and the label vocabulary.

    Cheap enough to run over the whole corpus first (~1.3 s for a 358 MB file)
    and it answers three things the data pass needs up front: how many data rows
    the file has, exactly which data-row positions hold an embedded header line,
    and which labels occur. Knowing the header positions in advance is what lets
    the data pass force ``float64`` dtypes — a chunk that accidentally contained
    the literal string ``"Dst Port"`` would otherwise silently become 78 columns
    of Python objects (~1 GB per chunk).

    Nothing is written; ``path`` is opened read-only.
    """
    result = ScoutResult(source_file=path.name)
    counts: Counter[str] = Counter()
    positions: List[int] = []
    offset = 0

    reader = pd.read_csv(
        path,
        usecols=[schema.LABEL_COLUMN],
        dtype=str,
        chunksize=chunk_size,
        na_filter=False,
        engine="c",
        encoding="utf-8-sig",
        on_bad_lines="error",
    )
    for chunk in reader:
        values = chunk[schema.LABEL_COLUMN].to_numpy()
        counts.update(pd.Series(values).value_counts().to_dict())
        local = np.flatnonzero(values == schema.EMBEDDED_HEADER_LABEL)
        if local.size:
            positions.extend((local + offset).tolist())
        offset += len(chunk)

    result.n_raw_rows = offset
    result.embedded_header_positions = positions
    n_header = len(positions)
    result.n_usable_rows = offset - n_header
    counts.pop(schema.EMBEDDED_HEADER_LABEL, None)
    result.label_counts = {str(k): int(v) for k, v in counts.items()}

    if verify_line_count:
        physical = count_physical_lines(path)
        result.n_physical_lines = physical
        # 1 header line + one line per data row. A mismatch means embedded
        # newlines inside quoted fields (or a truncated final line) and would
        # invalidate every position we just recorded.
        if physical != offset + 1:
            raise RowCountMismatchError(
                f"{path.name}: parsed {offset:,} data rows but the file holds "
                f"{physical:,} physical lines (expected {offset + 1:,} = 1 header "
                "+ 1 per data row). Row positions cannot be trusted; refusing to "
                "profile this file."
            )
    return result


# --------------------------------------------------------------------------- #
# Pass 2 accumulators
# --------------------------------------------------------------------------- #
class _FeatureAccumulator:
    """Streaming per-column statistics over the 78 model feature columns.

    Everything is a length-78 vector updated once per chunk, so memory does not
    grow with the number of rows. Deliberately measure-only: no value read here
    is ever written back.
    """

    def __init__(self, columns: Sequence[str]) -> None:
        self.columns = list(columns)
        width = len(self.columns)
        self.n_rows = 0
        self.nan = np.zeros(width, dtype=np.int64)
        self.posinf = np.zeros(width, dtype=np.int64)
        self.neginf = np.zeros(width, dtype=np.int64)
        self.negative = np.zeros(width, dtype=np.int64)
        self.minus_one = np.zeros(width, dtype=np.int64)
        self.finite_min = np.full(width, _POS_INF, dtype=np.float64)
        self.finite_max = np.full(width, _NEG_INF, dtype=np.float64)
        #: Most-negative and least-negative value seen, among negatives only.
        self.negative_min = np.full(width, _POS_INF, dtype=np.float64)
        self.negative_max = np.full(width, _NEG_INF, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        if values.size == 0:
            return
        self.n_rows += values.shape[0]

        finite = np.isfinite(values)
        if finite.all():
            self.finite_min = np.minimum(self.finite_min, values.min(axis=0))
            self.finite_max = np.maximum(self.finite_max, values.max(axis=0))
        else:
            self.nan += np.isnan(values).sum(axis=0)
            self.posinf += np.isposinf(values).sum(axis=0)
            self.neginf += np.isneginf(values).sum(axis=0)
            # Substituting the neutral extremum keeps the reduction vectorised
            # without touching ``values`` itself.
            self.finite_min = np.minimum(
                self.finite_min, np.where(finite, values, _POS_INF).min(axis=0)
            )
            self.finite_max = np.maximum(
                self.finite_max, np.where(finite, values, _NEG_INF).max(axis=0)
            )

        # ``values < 0`` is False for NaN, which is what we want.
        negative = values < 0
        if negative.any():
            self.negative += negative.sum(axis=0)
            self.minus_one += (values == schema.NOT_OBSERVED_SENTINEL).sum(axis=0)
            self.negative_min = np.minimum(
                self.negative_min, np.where(negative, values, _POS_INF).min(axis=0)
            )
            self.negative_max = np.maximum(
                self.negative_max, np.where(negative, values, _NEG_INF).max(axis=0)
            )

    # -- reporting ---------------------------------------------------------- #
    def _named(self, counts: np.ndarray) -> Dict[str, int]:
        return {
            column: int(count)
            for column, count in zip(self.columns, counts)
            if count > 0
        }

    def non_finite_stats(self) -> NonFiniteStats:
        stats = NonFiniteStats(
            nan_counts=self._named(self.nan),
            posinf_counts=self._named(self.posinf),
            neginf_counts=self._named(self.neginf),
            total_nan=int(self.nan.sum()),
            total_posinf=int(self.posinf.sum()),
            total_neginf=int(self.neginf.sum()),
        )
        touched = set(stats.nan_counts) | set(stats.posinf_counts) | set(stats.neginf_counts)
        stats.unexpected_columns = sorted(touched - set(schema.RATE_COLUMNS))
        return stats

    def negative_stats(self) -> NegativeStats:
        counts = self._named(self.negative)
        minus_one = self._named(self.minus_one)
        extremes: Dict[str, Dict[str, float]] = {}
        only_minus_one: List[str] = []
        for index, column in enumerate(self.columns):
            if self.negative[index] == 0:
                continue
            extremes[column] = {
                "min": float(self.negative_min[index]),
                "max": float(self.negative_max[index]),
            }
            if self.minus_one[index] == self.negative[index]:
                only_minus_one.append(column)
        return NegativeStats(
            negative_counts=counts,
            negative_extremes=extremes,
            sentinel_minus_one_counts=minus_one,
            only_minus_one_columns=only_minus_one,
            non_sentinel_columns={
                column: count
                for column, count in counts.items()
                if column not in schema.SENTINEL_NEGATIVE_COLUMNS
            },
        )

    def ranges(self) -> Tuple[Dict[str, float], Dict[str, float], List[str], List[str]]:
        minimums: Dict[str, float] = {}
        maximums: Dict[str, float] = {}
        constant: List[str] = []
        all_zero: List[str] = []
        for index, column in enumerate(self.columns):
            low = float(self.finite_min[index])
            high = float(self.finite_max[index])
            minimums[column] = low
            maximums[column] = high
            if low == high:
                constant.append(column)
                if low == 0.0:
                    all_zero.append(column)
        return minimums, maximums, constant, all_zero


class _TimestampAccumulator:
    """Streaming timestamp diagnostics. Never used to order anything."""

    #: Refuse to track distinct values beyond this many (guards against a file
    #: where the timestamp is effectively unique per row).
    DISTINCT_CAP = 4_000_000

    def __init__(self, *, max_examples: int = 64, capture_date: Optional[date] = None) -> None:
        self.max_examples = max_examples
        self.capture_date = capture_date
        self.value_counts: Counter[str] = Counter()
        self.hours: Counter[str] = Counter()
        self.dates: Counter[str] = Counter()
        self.n_parsed = 0
        self.n_unparsed = 0
        self.n_am_pm = 0
        self.unparsed_examples: List[str] = []
        self.off_date_examples: List[str] = []
        self.n_off_date = 0
        self.min_parsed: Optional[pd.Timestamp] = None
        self.max_parsed: Optional[pd.Timestamp] = None
        self.distinct_overflow = False
        self.n_decreasing = 0
        self._previous: Optional[pd.Timestamp] = None

    def update(self, raw: pd.Series) -> None:
        if raw.empty:
            return
        parsed = pd.to_datetime(
            raw, format=schema.TIMESTAMP_FORMAT, errors="coerce", exact=True
        )
        valid = parsed.dropna()
        self.n_parsed += int(len(valid))
        n_bad = int(len(parsed) - len(valid))
        self.n_unparsed += n_bad
        if n_bad and len(self.unparsed_examples) < self.max_examples:
            bad = raw[parsed.isna()].head(self.max_examples - len(self.unparsed_examples))
            self.unparsed_examples.extend(str(value) for value in bad)

        if len(valid):
            low, high = valid.min(), valid.max()
            self.min_parsed = low if self.min_parsed is None else min(self.min_parsed, low)
            self.max_parsed = high if self.max_parsed is None else max(self.max_parsed, high)
            self.hours.update(valid.dt.hour.value_counts().to_dict())
            first, last = valid.iloc[0], valid.iloc[-1]
            if self._previous is not None and first < self._previous:
                self.n_decreasing += 1
            # Count descents inside the chunk too, in file order.
            self.n_decreasing += int((valid.to_numpy()[1:] < valid.to_numpy()[:-1]).sum())
            self._previous = last

        # Date part of the raw string: DD/MM/YYYY, no parsing needed.
        date_part = raw.str.slice(0, 10)
        self.dates.update(date_part.value_counts().to_dict())
        if self.capture_date is not None:
            expected = self.capture_date.strftime("%d/%m/%Y")
            off = date_part != expected
            n_off = int(off.sum())
            if n_off:
                self.n_off_date += n_off
                if len(self.off_date_examples) < self.max_examples:
                    room = self.max_examples - len(self.off_date_examples)
                    self.off_date_examples.extend(
                        str(value) for value in raw[off].head(room)
                    )

        self.n_am_pm += int(
            raw.str.contains(r"\b[AaPp]\.?[Mm]\.?\b", regex=True, na=False).sum()
        )

        if not self.distinct_overflow:
            self.value_counts.update(raw.value_counts().to_dict())
            if len(self.value_counts) > self.DISTINCT_CAP:
                self.distinct_overflow = True
                self.value_counts.clear()

    def result(self) -> TimestampProfile:
        profile = TimestampProfile(
            n_parsed=self.n_parsed,
            n_unparsed=self.n_unparsed,
            unparsed_examples=self.unparsed_examples[: self.max_examples],
            n_distinct=-1 if self.distinct_overflow else len(self.value_counts),
            min_parsed=self.min_parsed.isoformat() if self.min_parsed is not None else None,
            max_parsed=self.max_parsed.isoformat() if self.max_parsed is not None else None,
            hour_histogram={
                f"{int(hour):02d}": int(count) for hour, count in sorted(self.hours.items())
            },
            date_histogram={
                str(key): int(value)
                for key, value in sorted(self.dates.items(), key=lambda kv: -kv[1])
            },
            n_rows_with_am_pm_marker=self.n_am_pm,
            n_rows_off_capture_date=self.n_off_date,
            off_capture_date_examples=self.off_date_examples[: self.max_examples],
            n_decreasing_steps=self.n_decreasing,
            monotonic_in_file_order=(self.n_decreasing == 0) if self.n_parsed else None,
        )
        if not self.distinct_overflow and self.value_counts:
            sizes = np.fromiter(self.value_counts.values(), dtype=np.int64)
            profile.n_tie_groups = int(sizes.size)
            profile.n_singleton_groups = int((sizes == 1).sum())
            profile.tie_group_max = int(sizes.max())
            profile.tie_group_mean = float(sizes.mean())
            profile.tie_group_median = float(np.median(sizes))
            profile.tie_group_p95 = float(np.percentile(sizes, 95))
            profile.n_rows_in_tie_groups = int(sizes[sizes > 1].sum())
        return profile


# --------------------------------------------------------------------------- #
# Duplicate measurement
# --------------------------------------------------------------------------- #
def combine_hashes(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Order-dependent 64-bit hash combine (xor-multiply), as in the loader."""
    with np.errstate(over="ignore"):
        return np.bitwise_xor(left, np.multiply(right, _HASH_MIX, dtype=np.uint64))


def measure_duplicates(
    feature_hash: np.ndarray,
    feature_label_hash: np.ndarray,
    full_hash: np.ndarray,
    label_codes: np.ndarray,
    vocabulary: Sequence[str],
    *,
    basis: str,
    max_combinations: int = 64,
) -> DuplicateProfile:
    """Measure the three duplicate categories. Removes and rewrites nothing.

    * **A. exact** - identical 78 features *and* raw ``Timestamp`` *and* ``Label``.
    * **B. temporal** - identical features and label, different ``Timestamp``.
    * **C. conflicting** - one feature vector carrying more than one label.

    A/B come from distinct-hash counts. C needs grouping, done with a single
    ``lexsort`` on ``(label_code, feature_hash)`` plus ``add.reduceat``, so it
    stays vectorised at corpus scale. Sorting by that key also yields the
    distinct-feature and distinct-(feature, label) counts for free.
    """
    n_rows = int(feature_hash.size)
    profile = DuplicateProfile(n_rows=n_rows, basis=basis)
    profile.note = (
        "Counts are measurements only - Stage B deletes nothing and modifies no "
        "raw data. 64-bit row hashes; expected false-collision count at n="
        f"{n_rows:,} is about {n_rows * n_rows / 2 / 2**64:.2e}."
    )
    if n_rows == 0:
        return profile

    # Sort once by (feature vector, label). Groups of equal feature_hash are
    # contiguous, and within a group equal labels are contiguous too.
    order = np.lexsort((label_codes, feature_hash))
    sorted_features = feature_hash[order]
    sorted_labels = label_codes[order]

    starts_group = np.empty(n_rows, dtype=bool)
    starts_group[0] = True
    np.not_equal(sorted_features[1:], sorted_features[:-1], out=starts_group[1:])
    group_starts = np.flatnonzero(starts_group)

    starts_pair = np.empty(n_rows, dtype=bool)
    starts_pair[0] = True
    np.not_equal(sorted_labels[1:], sorted_labels[:-1], out=starts_pair[1:])
    np.logical_or(starts_pair[1:], starts_group[1:], out=starts_pair[1:])

    profile.n_distinct_feature_vectors = int(group_starts.size)
    n_distinct_feature_label = int(starts_pair.sum())
    profile.n_feature_label_duplicates = n_rows - n_distinct_feature_label
    profile.n_exact = n_rows - int(np.unique(full_hash).size)
    profile.n_temporal = profile.n_feature_label_duplicates - profile.n_exact

    # C. groups whose distinct-label count exceeds one.
    labels_per_group = np.add.reduceat(starts_pair, group_starts)
    group_sizes = np.diff(np.append(group_starts, n_rows))
    conflicting = labels_per_group > 1
    profile.n_conflicting_groups = int(conflicting.sum())
    profile.n_conflicting_rows = int(group_sizes[conflicting].sum())

    if profile.n_conflicting_groups and len(vocabulary) <= 64:
        # One bit per label: each group's label set becomes a uint64 bitmask, so
        # the combinations can be counted with bincount instead of a Python loop
        # over every conflicting group.
        group_index = np.repeat(
            np.arange(group_starts.size, dtype=np.int64), group_sizes
        )
        pair_rows = np.flatnonzero(starts_pair)
        masks = np.zeros(group_starts.size, dtype=np.uint64)
        bits = np.left_shift(
            np.uint64(1), sorted_labels[pair_rows].astype(np.uint64)
        )
        np.bitwise_or.at(masks, group_index[pair_rows], bits)

        unique_masks, inverse = np.unique(masks[conflicting], return_inverse=True)
        rows_per = np.bincount(
            inverse, weights=group_sizes[conflicting].astype(np.float64)
        )
        groups_per = np.bincount(inverse)
        profile.n_conflicting_combinations = int(unique_masks.size)
        ranking = np.argsort(-rows_per, kind="stable")[:max_combinations]
        for position in ranking:
            mask = int(unique_masks[position])
            names = [
                vocabulary[bit]
                for bit in range(len(vocabulary))
                if mask & (1 << bit)
            ]
            profile.conflicting_label_combinations[" + ".join(names)] = {
                "groups": int(groups_per[position]),
                "rows": int(rows_per[position]),
            }
    elif profile.n_conflicting_groups:
        profile.note += (
            f" Label combinations not enumerated: {len(vocabulary)} labels exceeds "
            "the 64-bit bitmask limit."
        )
    return profile


# --------------------------------------------------------------------------- #
# Corpus-wide row store
# --------------------------------------------------------------------------- #
class _CorpusStore:
    """Pre-allocated per-row arrays covering the whole corpus, in sequence order.

    Sized exactly from pass 1 and filled in place during pass 2, so the corpus is
    never concatenated and no array is ever reallocated. Holds at most 26 bytes
    per row (three uint64 hashes plus an int16 label code) - about 421 MB for
    16.2M rows - versus roughly 10 GB for a full in-memory feature frame.

    The int16 label codes are an internal measurement device, not a class
    mapping: they index the pass-1 vocabulary sorted alphabetically and are
    never persisted. Fitting the real label encoder is a later stage's job.
    """

    def __init__(
        self, capacity: int, vocabulary: Sequence[str], *, with_hashes: bool
    ) -> None:
        self.capacity = int(capacity)
        self.vocabulary = list(vocabulary)
        self.label_codes = np.zeros(self.capacity, dtype=np.int16)
        self.with_hashes = with_hashes
        self.feature_hash: Optional[np.ndarray] = None
        self.feature_label_hash: Optional[np.ndarray] = None
        self.full_hash: Optional[np.ndarray] = None
        if with_hashes:
            self.feature_hash = np.zeros(self.capacity, dtype=np.uint64)
            self.feature_label_hash = np.zeros(self.capacity, dtype=np.uint64)
            self.full_hash = np.zeros(self.capacity, dtype=np.uint64)
        self.filled = 0
        #: source_file -> (start, stop) slice of the rows contributed by that file.
        self.spans: Dict[str, Tuple[int, int]] = {}

    @property
    def hash_arrays(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if not self.with_hashes:
            return None
        assert self.feature_hash is not None  # for type checkers
        assert self.feature_label_hash is not None
        assert self.full_hash is not None
        return (self.feature_hash, self.feature_label_hash, self.full_hash)

    def write_labels(self, start: int, codes: np.ndarray) -> None:
        stop = start + codes.size
        if stop > self.capacity:
            raise ProfilingError(
                f"corpus store overflow: writing rows [{start}, {stop}) into a "
                f"store sized {self.capacity}. Pass 1 and pass 2 disagree about "
                "the row count."
            )
        self.label_codes[start:stop] = codes

    def truncate(self, total: int) -> None:
        """Shrink the arrays to the number of rows actually profiled."""
        self.filled = int(total)
        if self.filled == self.capacity:
            return
        self.label_codes = self.label_codes[: self.filled]
        if self.with_hashes:
            self.feature_hash = self.feature_hash[: self.filled]  # type: ignore[index]
            self.feature_label_hash = self.feature_label_hash[: self.filled]  # type: ignore[index]
            self.full_hash = self.full_hash[: self.filled]  # type: ignore[index]


class _HashBuffer:
    """Collects one file's row hashes, into the corpus arrays when they exist."""

    def __init__(
        self,
        arrays: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]],
        start: int,
    ) -> None:
        self._arrays = arrays
        self._start = start
        self._position = start
        self._chunks: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def append(
        self, feature: np.ndarray, feature_label: np.ndarray, full: np.ndarray
    ) -> None:
        count = feature.size
        if self._arrays is None:
            self._chunks.append((feature, feature_label, full))
        else:
            stop = self._position + count
            for array, values in zip(self._arrays, (feature, feature_label, full)):
                array[self._position : stop] = values
        self._position += count

    def result(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The file's three hash arrays (views into the corpus store if pooled)."""
        if self._arrays is not None:
            span = slice(self._start, self._position)
            return (
                self._arrays[0][span],
                self._arrays[1][span],
                self._arrays[2][span],
            )
        if not self._chunks:
            empty = np.zeros(0, dtype=np.uint64)
            return (empty, empty.copy(), empty.copy())
        return (
            np.concatenate([part[0] for part in self._chunks]),
            np.concatenate([part[1] for part in self._chunks]),
            np.concatenate([part[2] for part in self._chunks]),
        )


# --------------------------------------------------------------------------- #
# Pass 2: the data pass
# --------------------------------------------------------------------------- #
def profile_file(
    discovered: DiscoveredFile,
    scout: ScoutResult,
    header: HeaderInfo,
    *,
    chunk_size: int = 200_000,
    max_examples: int = 64,
    compute_duplicates: bool = True,
    row_limit: Optional[int] = None,
    store: Optional[_CorpusStore] = None,
    store_offset: int = 0,
    warnings: Optional[List[str]] = None,
    logger: Optional[Any] = None,
) -> FileProfile:
    """Measure one CSV file in chunks. Opens the file read-only.

    Reads exactly the 80 standard columns by name — which also works for the
    84-column file, whose four extra identifier columns are a prefix and are
    reported rather than silently dropped — with the 78 feature dtypes forced to
    ``float64`` and the embedded header lines excluded via ``skiprows``. Forcing
    the dtypes matters for more than memory: with inference, the same integer
    column can come back as ``int64`` in one chunk and ``float64`` in another,
    and ``hash(5) != hash(5.0)``, which would silently corrupt the duplicate
    counts at chunk boundaries.

    ``row_limit`` caps the number of usable rows read (debug mode).
    """
    started = time.perf_counter()
    warnings = warnings if warnings is not None else []
    profile = FileProfile(file=discovered, header=header)
    profile.intake = IntakeStats(
        n_raw_rows=scout.n_raw_rows,
        n_physical_lines=scout.n_physical_lines,
        line_count_verified=scout.n_physical_lines is not None,
        n_embedded_header_rows=len(scout.embedded_header_positions),
        embedded_header_positions=scout.embedded_header_positions[:max_examples],
        embedded_header_positions_truncated=(
            len(scout.embedded_header_positions) > max_examples
        ),
        n_usable_rows=scout.n_usable_rows,
        row_limit=row_limit,
        at_excel_row_cap=scout.n_raw_rows == schema.EXCEL_ROW_CAP,
    )

    if not header.variant.usable:
        profile.error = (
            f"unusable schema ({header.variant.variant}, {header.n_columns} columns): "
            + "; ".join(header.variant.notes)
        )
        profile.profile_seconds = time.perf_counter() - started
        return profile

    feature_columns = list(schema.MODEL_FEATURE_COLUMNS)
    features = _FeatureAccumulator(feature_columns)
    timestamps = _TimestampAccumulator(
        max_examples=max_examples, capture_date=discovered.capture_date
    )
    excluded = np.asarray(scout.embedded_header_positions, dtype=np.int64)
    vocabulary = store.vocabulary if store is not None else sorted(scout.label_counts)
    hashes = _HashBuffer(
        store.hash_arrays if (store is not None and compute_duplicates) else None,
        store_offset,
    )
    local_codes: List[np.ndarray] = []
    label_totals = np.zeros(len(vocabulary), dtype=np.int64)
    n_profiled = 0
    n_blank_labels = 0
    n_blank_timestamps = 0

    reader = pd.read_csv(
        discovered.path,
        usecols=list(schema.CSV_COLUMNS),
        dtype=schema.read_dtypes(),
        chunksize=chunk_size,
        skiprows=[position + 1 for position in scout.embedded_header_positions] or None,
        nrows=row_limit,
        engine="c",
        encoding="utf-8-sig",
        on_bad_lines="error",
    )
    for chunk in reader:
        count = len(chunk)
        if count == 0:
            continue

        # -- row identity: position in the ORIGINAL, unfiltered row sequence --- #
        ranks = np.arange(n_profiled, n_profiled + count, dtype=np.int64)
        original = kth_kept_positions(ranks, excluded)
        if excluded.size:
            nearest = np.minimum(
                np.searchsorted(excluded, original), excluded.size - 1
            )
            if np.any(excluded[nearest] == original):
                raise ProfilingError(
                    f"{discovered.source_file}: derived Original_Index collided with "
                    "an embedded header row position. Row identity is unreliable."
                )
        if profile.intake.first_original_index is None:
            profile.intake.first_original_index = int(original[0])
        profile.intake.last_original_index = int(original[-1])

        # -- labels ------------------------------------------------------------ #
        label_series = chunk[schema.LABEL_COLUMN]
        n_blank = int(label_series.isna().sum())
        if n_blank:
            n_blank_labels += n_blank
            label_series = label_series.fillna("")
        label_values = label_series.to_numpy()
        codes = pd.Categorical(label_values, categories=vocabulary).codes
        if np.any(codes < 0):
            unknown = sorted(set(label_values[codes < 0]))[:5]
            raise ProfilingError(
                f"{discovered.source_file}: label(s) {unknown} appeared in the data "
                "pass but not in the pass-1 vocabulary. The file changed between "
                "passes, or its rows are not being read consistently."
            )
        codes = codes.astype(np.int16, copy=False)
        label_totals += np.bincount(codes, minlength=len(vocabulary))
        if store is not None:
            store.write_labels(store_offset + n_profiled, codes)
        else:
            local_codes.append(codes)

        # -- timestamps (diagnostic only) -------------------------------------- #
        timestamp_series = chunk[schema.TIMESTAMP_COLUMN]
        n_blank_ts = int(timestamp_series.isna().sum())
        if n_blank_ts:
            n_blank_timestamps += n_blank_ts
            timestamp_series = timestamp_series.fillna("")
        timestamps.update(timestamp_series)

        # -- numeric features -------------------------------------------------- #
        feature_frame = chunk[feature_columns]
        features.update(feature_frame.to_numpy(dtype=np.float64, copy=False))

        # -- duplicate hashes -------------------------------------------------- #
        if compute_duplicates:
            feature_hash = pd.util.hash_pandas_object(
                feature_frame, index=False
            ).to_numpy()
            feature_label = combine_hashes(
                feature_hash, pd.util.hash_array(label_values)
            )
            hashes.append(
                feature_hash,
                feature_label,
                combine_hashes(
                    feature_label, pd.util.hash_array(timestamp_series.to_numpy())
                ),
            )

        n_profiled += count

    profile.intake.n_rows_profiled = n_profiled
    profile.non_finite = features.non_finite_stats()
    profile.negatives = features.negative_stats()
    (
        profile.feature_min,
        profile.feature_max,
        profile.constant_columns,
        profile.all_zero_columns,
    ) = features.ranges()
    profile.timestamps = timestamps.result()
    profile.label_counts = {
        vocabulary[index]: int(total)
        for index, total in enumerate(label_totals)
        if total > 0
    }

    # Cross-check the two passes instead of trusting either one.
    expected = scout.n_usable_rows if row_limit is None else min(
        row_limit, scout.n_usable_rows
    )
    if n_profiled != expected:
        raise RowCountMismatchError(
            f"{discovered.source_file}: the data pass read {n_profiled:,} rows but "
            f"pass 1 predicted {expected:,}."
        )
    if row_limit is None and profile.label_counts != scout.label_counts:
        raise ProfilingError(
            f"{discovered.source_file}: label counts differ between the two passes. "
            f"pass 1: {scout.label_counts}; pass 2: {profile.label_counts}"
        )
    if n_blank_labels:
        warnings.append(
            f"{discovered.source_file}: {n_blank_labels:,} row(s) have an empty Label "
            "field (counted under the empty-string label)."
        )
    if n_blank_timestamps:
        warnings.append(
            f"{discovered.source_file}: {n_blank_timestamps:,} row(s) have an empty "
            "Timestamp field."
        )
    if (
        row_limit is None
        and n_profiled
        and profile.intake.last_original_index != scout.n_raw_rows - 1
    ):
        warnings.append(
            f"{discovered.source_file}: the last usable row sits at Original_Index "
            f"{profile.intake.last_original_index}, not {scout.n_raw_rows - 1} - the "
            "final data row is an embedded header line."
        )

    if compute_duplicates and n_profiled:
        feature_hash, feature_label_hash, full_hash = hashes.result()
        codes = (
            store.label_codes[store_offset : store_offset + n_profiled]
            if store is not None
            else np.concatenate(local_codes)
        )
        profile.duplicates = measure_duplicates(
            feature_hash,
            feature_label_hash,
            full_hash,
            codes,
            vocabulary,
            basis=(
                f"{len(feature_columns)} model feature columns (raw float64 values, "
                "unmodified) + raw Timestamp string + raw Label string"
            ),
            max_combinations=max_examples,
        )

    # Cross-check the filename-derived capture date against the file's own dates.
    if profile.timestamps.date_histogram:
        modal = next(iter(profile.timestamps.date_histogram))
        try:
            day, month, year = (int(part) for part in modal.split("/"))
            discovered.observed_date = date(year, month, day)
        except ValueError:
            discovered.observed_date = None

    profile.profile_seconds = time.perf_counter() - started
    if logger is not None:
        logger.info(
            "%s: %s usable rows profiled in %.1fs (%s labels, %s non-finite)",
            discovered.source_file,
            f"{n_profiled:,}",
            profile.profile_seconds,
            len(profile.label_counts),
            f"{profile.non_finite.to_dict()['total_non_finite']:,}",
        )
    return profile


# --------------------------------------------------------------------------- #
# Class-by-time analysis (analysis only - the split strategy is not changed)
# --------------------------------------------------------------------------- #
def build_class_timeline(profiles: Sequence[FileProfile]) -> ClassTimeline:
    """When each class appears, across the corpus's capture dates.

    ``profiles`` must already be in capture-date order.
    """
    timeline = ClassTimeline()
    for profile in profiles:
        iso = profile.capture_date.isoformat() if profile.capture_date else "unknown"
        if iso not in timeline.dates:
            timeline.dates.append(iso)
        for label, count in profile.label_counts.items():
            per_date = timeline.per_date.setdefault(label, {})
            per_date[iso] = per_date.get(iso, 0) + count

    n_dates = len(timeline.dates)
    late_cutoff = timeline.dates[(n_dates * 2) // 3] if n_dates >= 3 else None
    for label in sorted(timeline.per_date):
        present = [iso for iso in timeline.dates if timeline.per_date[label].get(iso)]
        if not present:
            continue
        timeline.first_date[label] = present[0]
        timeline.last_date[label] = present[-1]
        timeline.n_dates_present[label] = len(present)
        if len(present) == 1:
            timeline.single_date_classes.append(label)
        if late_cutoff is not None and present[0] >= late_cutoff:
            timeline.late_only_classes.append(label)
    return timeline


def analyse_split_coverage(
    label_codes: np.ndarray,
    vocabulary: Sequence[str],
    profiles: Sequence[FileProfile],
    ratios: Tuple[float, float, float] = (0.70, 0.10, 0.20),
    *,
    starved_threshold: int = 100,
) -> SplitCoverage:
    """Would a naive chronological split leave classes unseen during training?

    Exact, at row level: ``label_codes`` holds one entry per usable row in the
    deterministic capture-order sequence, so the split boundaries are cut at the
    same row indices a real chronological split would use.

    Diagnostic only. Stage B does not change the split strategy and does not use
    random stratification.
    """
    n_rows = int(label_codes.size)
    coverage = SplitCoverage(
        ratios=ratios, n_rows=n_rows, starved_threshold=starved_threshold
    )
    if n_rows == 0:
        return coverage

    coverage.train_end = int(n_rows * ratios[0])
    coverage.val_end = coverage.train_end + int(n_rows * ratios[1])
    n_labels = len(vocabulary)

    def _counts(segment: np.ndarray) -> Dict[str, int]:
        if segment.size == 0:
            return {}
        totals = np.bincount(segment, minlength=n_labels)
        return {
            vocabulary[index]: int(total)
            for index, total in enumerate(totals)
            if total > 0
        }

    coverage.train_counts = _counts(label_codes[: coverage.train_end])
    coverage.val_counts = _counts(label_codes[coverage.train_end : coverage.val_end])
    coverage.test_counts = _counts(label_codes[coverage.val_end :])

    present = {
        label
        for counts in (coverage.train_counts, coverage.val_counts, coverage.test_counts)
        for label in counts
    }
    coverage.absent_from_train = sorted(present - set(coverage.train_counts))
    coverage.absent_from_val = sorted(present - set(coverage.val_counts))
    coverage.absent_from_test = sorted(present - set(coverage.test_counts))
    coverage.train_only = sorted(
        set(coverage.train_counts) - set(coverage.val_counts) - set(coverage.test_counts)
    )
    coverage.starved_in_train = {
        label: count
        for label, count in sorted(coverage.train_counts.items())
        if count < starved_threshold
    }

    # Which file each boundary lands in, so the report can name the cut point.
    offset = 0
    boundaries = {"train_end": coverage.train_end, "validation_end": coverage.val_end}
    coverage.boundary_files = {name: None for name in boundaries}
    for profile in profiles:
        rows = profile.intake.n_rows_profiled
        for name, index in boundaries.items():
            if offset <= index < offset + rows:
                coverage.boundary_files[name] = (
                    f"{profile.source_file} @ row {index - offset:,}"
                )
        offset += rows
    return coverage


# --------------------------------------------------------------------------- #
# Parquet comparison (metadata only)
# --------------------------------------------------------------------------- #
def compare_parquet_metadata(
    parquet_dir: str | Path,
    csv_profiles: Sequence[FileProfile],
    *,
    patterns: Sequence[str] = ("*.parquet",),
) -> ParquetComparison:
    """Compare the Parquet copy on **metadata only** - no data is read.

    Uses the Parquet footer (row count, schema) rather than loading any column,
    because the comparison exists only to confirm what the CSV corpus provides
    that the Parquet copy does not: the ``Timestamp`` column the temporal
    pipeline depends on.
    """
    comparison = ParquetComparison()
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - pyarrow is a project dependency
        comparison.error = f"pyarrow not available: {exc}"
        return comparison

    files = discover_parquet_files(parquet_dir, patterns)
    comparison.n_files = len(files)
    if not files:
        comparison.error = f"no Parquet files found under {parquet_dir}"
        return comparison

    by_date: Dict[str, str] = {}
    for item in files:
        entry: Dict[str, Any] = {
            "source_file": item.source_file,
            "size_bytes": item.size_bytes,
            "capture_date": item.capture_date.isoformat() if item.capture_date else None,
        }
        try:
            metadata = pq.ParquetFile(item.path)
            names = list(metadata.schema_arrow.names)
            entry["n_rows"] = int(metadata.metadata.num_rows)
            entry["n_columns"] = len(names)
            entry["n_row_groups"] = int(metadata.metadata.num_row_groups)
            entry["has_timestamp"] = schema.TIMESTAMP_COLUMN in names
            entry["has_label"] = schema.LABEL_COLUMN in names
            entry["missing_vs_csv"] = [
                name for name in schema.CSV_COLUMNS if name not in names
            ]
            comparison.total_rows += entry["n_rows"]
            comparison.has_timestamp[item.source_file] = entry["has_timestamp"]
            comparison.has_label[item.source_file] = entry["has_label"]
        except Exception as exc:  # pragma: no cover - unreadable Parquet footer
            entry["error"] = f"{type(exc).__name__}: {exc}"
        comparison.total_size_bytes += item.size_bytes
        comparison.files.append(entry)
        if item.capture_date:
            by_date[item.capture_date.isoformat()] = item.source_file

    csv_dates = [
        profile.capture_date.isoformat()
        for profile in csv_profiles
        if profile.capture_date
    ]
    comparison.date_correspondence = {iso: by_date.get(iso) for iso in csv_dates}
    comparison.unmatched_csv_dates = [
        iso for iso in csv_dates if iso not in by_date
    ]
    comparison.unmatched_parquet_files = sorted(
        item.source_file
        for item in files
        if not item.capture_date or item.capture_date.isoformat() not in set(csv_dates)
    )
    return comparison


# --------------------------------------------------------------------------- #
# Corpus orchestration
# --------------------------------------------------------------------------- #
def profile_corpus(
    csv_dir: str | Path,
    *,
    chunk_size: int = 200_000,
    csv_patterns: Sequence[str] = ("*.csv",),
    only_files: Optional[Sequence[str]] = None,
    compute_duplicates: bool = True,
    compute_global_duplicates: bool = True,
    compare_parquet: bool = True,
    parquet_dir: str | Path | None = None,
    parquet_patterns: Sequence[str] = ("*.parquet",),
    split_ratios: Tuple[float, float, float] = (0.70, 0.10, 0.20),
    row_limit: Optional[int] = None,
    max_examples: int = 64,
    logger: Optional[Any] = None,
) -> CorpusProfile:
    """Profile every corpus CSV and assemble the global measurement.

    Order of operations
    -------------------
    1. Discover and deterministically sequence the CSV files.
    2. Pass 1 (scout) over each file's ``Label`` column: row accounting, embedded
       header positions, per-file and global label vocabulary.
    3. Pass 2 over each file in chunks: statistics, timestamps, duplicate hashes.
    4. Corpus-wide reduction: label totals, timeline, split coverage, global
       duplicates, Parquet metadata comparison.

    If one file fails, the run records its ``error`` and continues with the rest;
    only a discovery failure or a pass-1 mismatch aborts the whole run.
    """
    started = time.perf_counter()
    discovered = discover_csv_files(csv_dir, csv_patterns)
    if only_files is not None:
        wanted = set(only_files)
        discovered = [item for item in discovered if item.source_file in wanted]
        missing = wanted - {item.source_file for item in discovered}
        if missing:
            raise CorpusNotFoundError(
                f"requested file(s) not found under {csv_dir}: {sorted(missing)}"
            )
    corpus = CorpusProfile(csv_dir=Path(csv_dir), row_limit=row_limit)
    # The corpus-wide reduction reads the per-row hashes the data pass writes, so
    # it cannot run when duplicate measurement is off: it would report an
    # all-zero store as real measurements.
    compute_global_duplicates = compute_global_duplicates and compute_duplicates

    # --- pass 0: headers (one line per file) --------------------------------- #
    headers: Dict[str, HeaderInfo] = {}
    for item in discovered:
        headers[item.source_file] = inspect_header(item.path)

    # --- pass 1: scout every file ------------------------------------------- #
    scouts: Dict[str, ScoutResult] = {}
    for item in discovered:
        header = headers[item.source_file]
        if not header.variant.usable:
            # Without a ``Label`` column under its expected name the file cannot
            # be scouted at all. Recorded and skipped rather than fatal: one odd
            # file must not abort the whole corpus run.
            scouts[item.source_file] = ScoutResult(
                source_file=item.source_file,
                error=(
                    f"unusable schema ({header.variant.variant}, "
                    f"{header.n_columns} columns)"
                ),
            )
            continue
        try:
            scouts[item.source_file] = scout_file(item.path, verify_line_count=True)
        except RowCountMismatchError:
            raise
        except Exception as exc:  # pragma: no cover - read failures propagate
            raise ProfilingError(
                f"pass 1 failed on {item.source_file}: {type(exc).__name__}: {exc}"
            ) from exc

    # Deterministic global vocabulary: pass-1 labels, sorted alphabetically so
    # the int16 codes (an internal measurement device) are stable across runs.
    global_counts: Counter[str] = Counter()
    for item in discovered:
        global_counts.update(scouts[item.source_file].label_counts)
    vocabulary = sorted(global_counts)
    corpus.label_vocabulary = vocabulary
    corpus.label_counts = {label: global_counts[label] for label in vocabulary}

    # Exact sequence positions, so the split analysis cuts at real row indices.
    total_usable = sum(
        scouts[item.source_file].n_usable_rows for item in discovered
    )
    profiled = total_usable if row_limit is None else min(total_usable, row_limit)
    store = _CorpusStore(
        profiled,
        vocabulary,
        with_hashes=compute_global_duplicates,
    )

    # --- pass 2: profile each file in chunks --------------------------------- #
    offset = 0
    for item in discovered:
        scout = scouts[item.source_file]
        file_profiled = profile_file(
            item,
            scout,
            headers[item.source_file],
            chunk_size=chunk_size,
            max_examples=max_examples,
            compute_duplicates=compute_duplicates,
            row_limit=row_limit,
            store=store,
            store_offset=offset,
            warnings=corpus.warnings,
            logger=logger,
        )
        corpus.files.append(file_profiled)
        if file_profiled.error:
            corpus.warnings.append(f"{item.source_file}: {file_profiled.error}")
            continue
        offset += file_profiled.intake.n_rows_profiled

    store.truncate(offset)

    # --- corpus-wide reduction ---------------------------------------------- #
    for profile in corpus.files:
        if profile.error:
            continue
        corpus.total_raw_rows += profile.intake.n_raw_rows
        corpus.total_embedded_header_rows += profile.intake.n_embedded_header_rows
        corpus.total_usable_rows += profile.intake.n_usable_rows
        corpus.total_rows_profiled += profile.intake.n_rows_profiled
        corpus.total_size_bytes += profile.file.size_bytes
        if profile.intake.at_excel_row_cap:
            corpus.truncated_files.append(profile.source_file)

    for profile in corpus.files:
        if profile.error:
            continue
        iso = (
            profile.capture_date.isoformat()
            if profile.capture_date
            else "unknown"
        )
        sequence_key = f"{iso}/{profile.source_file}"
        corpus.sequence.append(sequence_key)
        variant = profile.header.variant.variant
        corpus.schema_variants.setdefault(variant, []).append(profile.source_file)

        # Cross-check the capture date against the observed dates.
        if (
            profile.capture_date
            and profile.timestamps.date_histogram
            and profile.file.observed_date
            and profile.file.observed_date != profile.capture_date
        ):
            corpus.warnings.append(
                f"{profile.source_file}: filename capture date {profile.capture_date} "
                f"differs from the file's own modal Timestamp date "
                f"{profile.file.observed_date}."
            )

    corpus.timeline = build_class_timeline(corpus.files)
    if offset > 0:
        corpus.split_coverage = analyse_split_coverage(
            store.label_codes,
            vocabulary,
            corpus.files,
            ratios=split_ratios,
        )

    if compute_global_duplicates and offset > 0:
        feature_hash, feature_label_hash, full_hash = store.hash_arrays  # type: ignore[misc]
        corpus.global_duplicates = measure_duplicates(
            feature_hash,
            feature_label_hash,
            full_hash,
            store.label_codes,
            vocabulary,
            basis=(
                "whole-corpus concatenation in deterministic capture-order sequence; "
                "78 model feature columns (raw float64, unmodified) + raw Timestamp "
                "string + raw Label string"
            ),
            max_combinations=max_examples,
        )

    if compare_parquet:
        target = (
            Path(parquet_dir)
            if parquet_dir is not None
            else Path(csv_dir).parent / "parquet"
        )
        corpus.parquet = compare_parquet_metadata(
            target, corpus.files, patterns=parquet_patterns
        )

    corpus.elapsed_seconds = time.perf_counter() - started
    if logger is not None:
        logger.info("corpus profiling complete:\n%s", corpus.summary())
    return corpus


# --------------------------------------------------------------------------- #
# Config entry point
# --------------------------------------------------------------------------- #
def profile_from_config(cfg: Any, *, logger: Optional[Any] = None, **overrides: Any) -> CorpusProfile:
    """Profile the corpus using the project configuration.

    Reuses ``cfg.profiling`` (paths, chunk size, which measurements to run),
    ``cfg.split`` (the ratios for the coverage analysis) and ``cfg.debug`` (the
    row cap). Keyword ``overrides`` win over the config, for CLI flags.
    """
    profiling = cfg.profiling
    settings: Dict[str, Any] = {
        "chunk_size": profiling.chunk_size,
        "csv_patterns": tuple(profiling.csv_patterns),
        "compute_duplicates": profiling.compute_duplicates,
        "compute_global_duplicates": (
            profiling.compute_duplicates and profiling.compute_global_duplicates
        ),
        "compare_parquet": profiling.compare_parquet,
        "parquet_dir": cfg.paths.raw_data_dir / profiling.parquet_subdir,
        "split_ratios": cfg.split.as_tuple(),
        "row_limit": cfg.debug.row_limit_or_none,
        "max_examples": profiling.max_examples,
        "logger": logger,
    }
    settings.update(overrides)
    csv_dir = settings.pop("csv_dir", cfg.paths.raw_data_dir / profiling.csv_subdir)
    return profile_corpus(csv_dir, **settings)
