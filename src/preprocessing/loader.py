"""Raw CIC-IDS2018 day-CSV loader.

Turns one raw day CSV into a validated, fully finite numeric feature frame plus
the string labels, the parsed timestamps and the original row positions. This is
the single clean-data entry point every later stage depends on.

What this module does
---------------------
1. Verifies the file's header against the confirmed 80-column schema
   (:mod:`src.preprocessing.schema`) and refuses to continue on a mismatch.
2. Counts the raw data rows and (optionally) cross-checks that count against the
   file's physical line count.
3. Removes **only** the repeated header rows that were concatenated into the
   data — rows whose ``Label`` field is exactly ``"Label"``. No other row is
   ever dropped.
4. Records every row's original position in the source CSV as ``Original_Index``
   *before* any removal, so the source order survives untouched.
5. Reads the 78 model feature columns straight into ``float64``, counts the
   ``NaN`` / ``+inf`` / ``-inf`` values the raw file contains, then replaces them
   according to the configured cleaning policy. The returned feature frame is
   guaranteed finite.
6. Parses ``Timestamp`` for validation and logging only, keeping the raw string
   representation untouched alongside it.
7. Measures — and never removes — exact / temporal / label-conflicting duplicate
   rows.

What this module deliberately does NOT do
-----------------------------------------
* It does not sort by timestamp. The raw timestamps use a 12-hour clock with no
  AM/PM marker and are massively tied (up to 2,050 flows per second), so they
  are unusable as an ordering key. Rows stay in source-file order and
  ``Original_Index`` is the deterministic capture-order proxy.
* It does not clip negative values. In ``Init Fwd Win Byts`` /
  ``Init Bwd Win Byts`` the only negative value present is ``-1``, the
  CICFlowMeter "not observed" sentinel, and it is preserved.
* It does not encode labels, scale features, deduplicate, split, window, or
  touch the source file in any way. The file is opened read-only.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.preprocessing import schema

#: Bytes read per chunk when counting the file's physical lines.
LINE_COUNT_CHUNK = 8 << 20

#: Odd 64-bit constant (golden-ratio derived, as in ``boost::hash_combine``) used
#: to mix two row hashes into one.
_HASH_MIX = np.uint64(0x9E3779B97F4A7C15)


class LoaderError(RuntimeError):
    """Base class for every failure this loader raises deliberately."""


class SchemaMismatchError(LoaderError):
    """The file's header does not match the confirmed CIC-IDS2018 schema."""


class RowCountMismatchError(LoaderError):
    """Measured row counts disagree — the file is not what we think it is."""


class DataQualityError(LoaderError):
    """The loaded data violates a guarantee this stage must provide."""


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #
@dataclass
class DuplicateStats:
    """Duplicate-row measurements. Nothing is removed on the basis of these."""

    n_rows: int = 0
    #: Distinct 78-column feature vectors.
    n_distinct_feature_vectors: int = 0
    #: A. same features + same timestamp + same label.
    n_exact_duplicate_rows: int = 0
    #: B. same features + same label, timestamp differs.
    n_temporal_duplicate_rows: int = 0
    #: A + B, i.e. every row whose (features, label) pair is not unique.
    n_feature_label_duplicate_rows: int = 0
    #: C. feature vectors that carry more than one distinct label.
    n_conflicting_feature_groups: int = 0
    #: Rows belonging to a conflicting feature group.
    n_conflicting_rows: int = 0
    #: Labels seen on the conflicting groups.
    conflicting_label_pairs: Dict[str, int] = field(default_factory=dict)
    basis: str = ""
    hash_note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_distinct_feature_vectors": self.n_distinct_feature_vectors,
            "A_exact_duplicate_rows": self.n_exact_duplicate_rows,
            "B_temporal_duplicate_rows": self.n_temporal_duplicate_rows,
            "AB_feature_label_duplicate_rows": self.n_feature_label_duplicate_rows,
            "C_conflicting_feature_groups": self.n_conflicting_feature_groups,
            "C_conflicting_rows": self.n_conflicting_rows,
            "C_conflicting_label_pairs": self.conflicting_label_pairs,
            "basis": self.basis,
            "hash_note": self.hash_note,
        }


@dataclass
class CleaningStats:
    """Everything that was measured or changed while cleaning."""

    n_raw_rows: int = 0
    n_physical_lines: Optional[int] = None
    line_count_verified: bool = False
    n_embedded_header_rows: int = 0
    embedded_header_original_indices: List[int] = field(default_factory=list)
    n_rows_after_header_removal: int = 0
    truncated: bool = False
    row_limit: Optional[int] = None

    nan_counts_before: Dict[str, int] = field(default_factory=dict)
    posinf_counts_before: Dict[str, int] = field(default_factory=dict)
    neginf_counts_before: Dict[str, int] = field(default_factory=dict)
    total_nan_before: int = 0
    total_infinities_before: int = 0

    nan_replacement: float = 0.0
    infinity_policy: str = ""
    infinity_substitutions: Dict[str, Dict[str, float]] = field(default_factory=dict)

    total_nan_after: int = 0
    total_infinities_after: int = 0

    negative_value_counts: Dict[str, int] = field(default_factory=dict)
    sentinel_negative_counts: Dict[str, int] = field(default_factory=dict)
    non_sentinel_negative_columns: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_raw_rows": self.n_raw_rows,
            "n_physical_lines": self.n_physical_lines,
            "line_count_verified": self.line_count_verified,
            "n_embedded_header_rows": self.n_embedded_header_rows,
            "embedded_header_original_indices": self.embedded_header_original_indices,
            "n_rows_after_header_removal": self.n_rows_after_header_removal,
            "truncated": self.truncated,
            "row_limit": self.row_limit,
            "nan_counts_before": self.nan_counts_before,
            "posinf_counts_before": self.posinf_counts_before,
            "neginf_counts_before": self.neginf_counts_before,
            "total_nan_before": self.total_nan_before,
            "total_infinities_before": self.total_infinities_before,
            "nan_replacement": self.nan_replacement,
            "infinity_policy": self.infinity_policy,
            "infinity_substitutions": self.infinity_substitutions,
            "total_nan_after": self.total_nan_after,
            "total_infinities_after": self.total_infinities_after,
            "negative_value_counts": self.negative_value_counts,
            "sentinel_negative_counts": self.sentinel_negative_counts,
            "non_sentinel_negative_columns": self.non_sentinel_negative_columns,
        }


@dataclass
class TimestampStats:
    """Timestamp parse result. For validation and logging only."""

    column: str = schema.TIMESTAMP_COLUMN
    format: str = schema.TIMESTAMP_FORMAT
    n_parsed: int = 0
    n_unparsed: int = 0
    n_distinct_raw: int = 0
    min_parsed: Optional[str] = None
    max_parsed: Optional[str] = None
    hour_histogram: Dict[str, int] = field(default_factory=dict)
    date_histogram: Dict[str, int] = field(default_factory=dict)
    n_rows_with_am_pm_marker: int = 0
    ordering_note: str = (
        "Timestamp is NOT used for ordering: the raw hour field only holds 01-12 "
        "with no AM/PM marker, and values are heavily tied. Rows keep their "
        "source-file order; Original_Index is the capture-order proxy."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column": self.column,
            "format": self.format,
            "n_parsed": self.n_parsed,
            "n_unparsed": self.n_unparsed,
            "n_distinct_raw": self.n_distinct_raw,
            "min_parsed": self.min_parsed,
            "max_parsed": self.max_parsed,
            "hour_histogram": self.hour_histogram,
            "date_histogram": self.date_histogram,
            "n_rows_with_am_pm_marker": self.n_rows_with_am_pm_marker,
            "ordering_note": self.ordering_note,
        }


@dataclass
class LoadStats:
    """Complete measurement record for one loaded day file."""

    source_file: str = ""
    source_path: str = ""
    source_size_bytes: int = 0
    header_md5: str = ""
    n_csv_columns: int = 0
    n_flow_features: int = 0
    n_identifier_features: int = 0
    n_model_features: int = 0
    feature_dtype: str = ""
    cleaning: CleaningStats = field(default_factory=CleaningStats)
    timestamps: TimestampStats = field(default_factory=TimestampStats)
    label_distribution: Dict[str, int] = field(default_factory=dict)
    n_labels: int = 0
    duplicates: Optional[DuplicateStats] = None
    load_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_file": self.source_file,
            "source_path": self.source_path,
            "source_size_bytes": self.source_size_bytes,
            "header_md5": self.header_md5,
            "columns": {
                "n_csv_columns": self.n_csv_columns,
                "n_flow_features": self.n_flow_features,
                "n_identifier_features": self.n_identifier_features,
                "n_model_features": self.n_model_features,
                "feature_dtype": self.feature_dtype,
            },
            "cleaning": self.cleaning.to_dict(),
            "timestamps": self.timestamps.to_dict(),
            "labels": {
                "n_labels": self.n_labels,
                "distribution": self.label_distribution,
            },
            "duplicates": self.duplicates.to_dict() if self.duplicates else None,
            "load_seconds": self.load_seconds,
        }


@dataclass
class LoadedDay:
    """One cleaned day of CIC-IDS2018 flows.

    ``features``, ``labels``, ``timestamps``, ``timestamps_raw`` and
    ``original_index`` all share an identical ``RangeIndex``, so they can be
    combined positionally without any realignment.
    """

    features: pd.DataFrame
    labels: pd.Series
    timestamps: pd.Series
    timestamps_raw: pd.Series
    original_index: pd.Series
    source_file: str
    source_path: str
    feature_columns: List[str]
    stats: LoadStats

    # -- shape -------------------------------------------------------------- #
    @property
    def n_rows(self) -> int:
        return int(len(self.features))

    @property
    def n_features(self) -> int:
        return int(self.features.shape[1])

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_rows, self.n_features)

    @property
    def label_distribution(self) -> Dict[str, int]:
        counts = self.labels.value_counts()
        return {str(k): int(v) for k, v in counts.items()}

    # -- integrity ---------------------------------------------------------- #
    def validate(self) -> "LoadedDay":
        """Re-check every guarantee this stage promises. Raises on violation."""
        parts = {
            "labels": self.labels,
            "timestamps": self.timestamps,
            "timestamps_raw": self.timestamps_raw,
            "original_index": self.original_index,
        }
        for name, series in parts.items():
            if len(series) != self.n_rows:
                raise DataQualityError(
                    f"{name} has {len(series)} rows but features has {self.n_rows}"
                )
            if not series.index.equals(self.features.index):
                raise DataQualityError(f"{name} index does not match the feature index")

        if list(self.features.columns) != list(self.feature_columns):
            raise DataQualityError("feature_columns does not match the frame's columns")

        non_float = [
            column
            for column, dtype in self.features.dtypes.items()
            if not pd.api.types.is_float_dtype(dtype)
        ]
        if non_float:
            raise DataQualityError(f"non-float feature column(s): {non_float[:5]}")

        for column in self.features.columns:
            values = self.features[column].to_numpy(copy=False)
            if not np.isfinite(values).all():
                raise DataQualityError(f"feature column {column!r} still holds NaN/inf")

        if self.original_index.duplicated().any():
            raise DataQualityError("Original_Index is not unique")
        if not self.original_index.is_monotonic_increasing:
            raise DataQualityError("Original_Index is not increasing - order was lost")

        if self.labels.isna().any():
            raise DataQualityError("Label column contains missing values")
        if (self.labels == schema.EMBEDDED_HEADER_LABEL).any():
            raise DataQualityError(
                f"Label still contains the embedded-header value "
                f"{schema.EMBEDDED_HEADER_LABEL!r}"
            )
        return self

    # -- reporting ---------------------------------------------------------- #
    def to_metadata_dict(self) -> Dict[str, Any]:
        """Small JSON-serialisable record of this load (no feature values)."""
        return {
            "shape": {"rows": self.n_rows, "features": self.n_features},
            "feature_columns": list(self.feature_columns),
            "stats": self.stats.to_dict(),
        }

    def summary(self) -> str:
        """Multi-line human-readable summary."""
        stats = self.stats
        clean = stats.cleaning
        lines = [
            f"source file            : {self.source_file}",
            f"source path            : {self.source_path}",
            f"raw data rows          : {clean.n_raw_rows:,}",
            f"embedded header rows   : {clean.n_embedded_header_rows} (removed)",
            f"rows after cleaning    : {clean.n_rows_after_header_removal:,}",
            f"feature frame shape    : {self.n_rows:,} x {self.n_features} "
            f"({stats.feature_dtype})",
            f"NaN before / after     : {clean.total_nan_before:,} / {clean.total_nan_after}",
            f"+/-inf before / after  : "
            f"{clean.total_infinities_before:,} / {clean.total_infinities_after}",
            f"timestamps parsed      : {stats.timestamps.n_parsed:,} "
            f"(unparsed {stats.timestamps.n_unparsed})",
            f"timestamp range        : {stats.timestamps.min_parsed} .. "
            f"{stats.timestamps.max_parsed}",
            f"distinct labels        : {stats.n_labels}",
        ]
        for label, count in stats.label_distribution.items():
            share = count / self.n_rows if self.n_rows else 0.0
            lines.append(f"    {label:<24} {count:>12,}  ({share:.4%})")
        if stats.duplicates is not None:
            duplicates = stats.duplicates
            lines.extend(
                [
                    f"duplicates (measured, nothing removed):",
                    f"    A exact (feat+ts+label)      : "
                    f"{duplicates.n_exact_duplicate_rows:,}",
                    f"    B temporal (ts differs)      : "
                    f"{duplicates.n_temporal_duplicate_rows:,}",
                    f"    C conflicting feature groups : "
                    f"{duplicates.n_conflicting_feature_groups:,} "
                    f"({duplicates.n_conflicting_rows:,} rows)",
                ]
            )
        lines.append(f"load time              : {stats.load_seconds:.1f}s")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def read_header(path: Path) -> tuple[List[str], str]:
    """Return the CSV's header names and the md5 of the raw header line."""
    with path.open("rb") as handle:
        raw = handle.readline()
    line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
    names = [name.strip() for name in line.split(",")]
    return names, hashlib.md5(line.encode("utf-8")).hexdigest()


def verify_header(path: Path) -> str:
    """Check the file header against :data:`schema.CSV_COLUMNS`.

    Returns the header md5. Raises :class:`SchemaMismatchError` with the exact
    difference — a silently-accepted schema drift would corrupt everything
    downstream.
    """
    names, md5 = read_header(path)
    expected = list(schema.CSV_COLUMNS)
    if names == expected:
        return md5

    details = [
        f"{path.name} header does not match the confirmed CIC-IDS2018 schema.",
        f"  expected {len(expected)} columns, found {len(names)}",
        f"  expected header md5 {schema.HEADER_MD5}, found {md5}",
    ]
    missing = [name for name in expected if name not in names]
    unexpected = [name for name in names if name not in expected]
    if missing:
        details.append(f"  missing columns   : {missing[:10]}")
    if unexpected:
        details.append(f"  unexpected columns: {unexpected[:10]}")
    if not missing and not unexpected:
        reordered = [
            (index, found, want)
            for index, (found, want) in enumerate(zip(names, expected))
            if found != want
        ]
        details.append(f"  same names, different order, first mismatch: {reordered[:3]}")
    raise SchemaMismatchError("\n".join(details))


def count_physical_lines(path: Path) -> int:
    """Number of physical lines in ``path`` (counts a final unterminated line)."""
    total = 0
    trailing_newline = True
    with path.open("rb") as handle:
        while chunk := handle.read(LINE_COUNT_CHUNK):
            total += chunk.count(b"\n")
            trailing_newline = chunk.endswith(b"\n")
    if total and not trailing_newline:
        total += 1
    return total


def _combine_hashes(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Order-dependent 64-bit hash combine (xor-multiply)."""
    with np.errstate(over="ignore"):
        return np.bitwise_xor(left, np.multiply(right, _HASH_MIX, dtype=np.uint64))


def _nonfinite_counts(values: np.ndarray) -> tuple[int, int, int]:
    """``(n_nan, n_posinf, n_neginf)`` for a float array. Fast path: all finite."""
    if np.isfinite(values).all():
        return 0, 0, 0
    return (
        int(np.isnan(values).sum()),
        int(np.isposinf(values).sum()),
        int(np.isneginf(values).sum()),
    )


def _duplicate_stats(
    features: pd.DataFrame, labels: pd.Series, timestamps_raw: pd.Series
) -> DuplicateStats:
    """Measure the three duplicate categories. Removes nothing."""
    n_rows = len(features)
    feature_hash = pd.util.hash_pandas_object(features, index=False).to_numpy()
    label_hash = pd.util.hash_array(labels.to_numpy())
    timestamp_hash = pd.util.hash_array(timestamps_raw.to_numpy())

    feature_label_hash = _combine_hashes(feature_hash, label_hash)
    full_hash = _combine_hashes(feature_label_hash, timestamp_hash)

    n_distinct_features = int(np.unique(feature_hash).size)
    n_distinct_feature_label = int(np.unique(feature_label_hash).size)
    n_distinct_full = int(np.unique(full_hash).size)

    n_exact = n_rows - n_distinct_full
    n_feature_label_duplicates = n_rows - n_distinct_feature_label

    # C. one feature vector carrying more than one label.
    grouped = pd.DataFrame({"h": feature_hash, "label": labels.to_numpy()})
    labels_per_group = grouped.groupby("h", sort=False)["label"].nunique()
    conflicting_groups = labels_per_group.index[labels_per_group.to_numpy() > 1]
    conflicting_mask = grouped["h"].isin(conflicting_groups).to_numpy()
    conflicting_label_counts = (
        grouped.loc[conflicting_mask, "label"].value_counts() if conflicting_mask.any()
        else pd.Series(dtype="int64")
    )

    return DuplicateStats(
        n_rows=n_rows,
        n_distinct_feature_vectors=n_distinct_features,
        n_exact_duplicate_rows=n_exact,
        n_temporal_duplicate_rows=n_feature_label_duplicates - n_exact,
        n_feature_label_duplicate_rows=n_feature_label_duplicates,
        n_conflicting_feature_groups=int(conflicting_groups.size),
        n_conflicting_rows=int(conflicting_mask.sum()),
        conflicting_label_pairs={
            str(k): int(v) for k, v in conflicting_label_counts.items()
        },
        basis=(
            "cleaned float64 feature values + raw Timestamp string + raw Label string"
        ),
        hash_note=(
            "64-bit hashes; collision probability at n="
            f"{n_rows:,} is on the order of 1e-6"
        ),
    )


def _timestamp_stats(raw: pd.Series, parsed: pd.Series) -> TimestampStats:
    """Summarise the timestamp column (validation / logging only)."""
    valid = parsed.dropna()
    hours = valid.dt.strftime("%H").value_counts().sort_index() if len(valid) else None
    dates = raw.str.slice(0, 10).value_counts() if len(raw) else None
    return TimestampStats(
        n_parsed=int(len(valid)),
        n_unparsed=int(parsed.isna().sum()),
        n_distinct_raw=int(raw.nunique()),
        min_parsed=valid.min().isoformat() if len(valid) else None,
        max_parsed=valid.max().isoformat() if len(valid) else None,
        hour_histogram={str(k): int(v) for k, v in hours.items()} if hours is not None else {},
        date_histogram=(
            {str(k): int(v) for k, v in dates.sort_values(ascending=False).items()}
            if dates is not None
            else {}
        ),
        n_rows_with_am_pm_marker=int(
            raw.str.contains(r"\b[AaPp]\.?[Mm]\.?\b", regex=True, na=False).sum()
        ),
    )


# --------------------------------------------------------------------------- #
# The loader
# --------------------------------------------------------------------------- #
def load_day_csv(
    path: str | Path,
    *,
    nan_replacement: float = 0.0,
    infinity_policy: str = "column_finite_extremum",
    infinity_replacement: float = 0.0,
    compute_duplicate_stats: bool = True,
    verify_raw_line_count: bool = True,
    require_all_timestamps_parse: bool = True,
    row_limit: Optional[int] = None,
    expected_raw_rows: Optional[int] = None,
    expected_embedded_header_rows: Optional[int] = None,
    logger: Any = None,
) -> LoadedDay:
    """Load and clean one raw CIC-IDS2018 day CSV.

    Parameters
    ----------
    path:
        The raw day CSV. Opened read-only; never modified.
    nan_replacement:
        Value substituted for ``NaN`` in the model feature columns.
    infinity_policy:
        ``"column_finite_extremum"`` replaces ``+inf`` / ``-inf`` with the
        largest / smallest **finite** value observed in that column;
        ``"constant"`` uses ``infinity_replacement`` for both signs.
    compute_duplicate_stats:
        Measure the duplicate categories (nothing is ever removed).
    verify_raw_line_count:
        Cross-check the parsed row count against the file's physical line count.
        Also proves no field contains an embedded newline, which the
        header-row removal relies on.
    require_all_timestamps_parse:
        Raise if any surviving row's ``Timestamp`` fails to parse.
    row_limit:
        Debug mode: read at most this many raw data rows.
    expected_raw_rows, expected_embedded_header_rows:
        Optional hard expectations. When given and not met, the loader raises
        :class:`RowCountMismatchError` rather than continuing silently.
    logger:
        Optional ``logging.Logger``.

    Returns
    -------
    LoadedDay
        Validated features / labels / timestamps / original positions + stats.
    """
    started = time.time()
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Day CSV not found: {path}")
    if infinity_policy not in ("column_finite_extremum", "constant"):
        raise ValueError(f"unknown infinity_policy: {infinity_policy!r}")

    def log(level: str, message: str, *args: Any) -> None:
        if logger is not None:
            getattr(logger, level)(message, *args)

    stats = LoadStats(
        source_file=path.name,
        source_path=str(path),
        source_size_bytes=path.stat().st_size,
        n_csv_columns=len(schema.CSV_COLUMNS),
        n_flow_features=len(schema.FLOW_FEATURE_COLUMNS),
        n_identifier_features=len(schema.IDENTIFIER_FEATURE_COLUMNS),
        n_model_features=len(schema.MODEL_FEATURE_COLUMNS),
        feature_dtype="float64",
    )
    clean = stats.cleaning
    clean.row_limit = row_limit
    clean.nan_replacement = float(nan_replacement)
    clean.infinity_policy = infinity_policy

    log("info", "Loading %s (%.1f MB)", path.name, stats.source_size_bytes / 1e6)

    # -- 1. schema ---------------------------------------------------------- #
    stats.header_md5 = verify_header(path)
    log(
        "info",
        "Header verified: %d columns, md5 %s",
        len(schema.CSV_COLUMNS),
        stats.header_md5,
    )

    # -- 2. locate the embedded header rows (Label column only) ------------- #
    label_only = pd.read_csv(
        path,
        usecols=[schema.LABEL_COLUMN],
        dtype={schema.LABEL_COLUMN: str},
        na_filter=False,
        nrows=row_limit,
    )
    raw_labels = label_only[schema.LABEL_COLUMN].to_numpy()
    clean.n_raw_rows = int(raw_labels.size)
    header_positions = np.flatnonzero(raw_labels == schema.EMBEDDED_HEADER_LABEL)
    clean.n_embedded_header_rows = int(header_positions.size)
    clean.embedded_header_original_indices = [int(p) for p in header_positions]
    clean.truncated = row_limit is not None and clean.n_raw_rows >= row_limit
    del label_only, raw_labels

    log("info", "Raw data rows detected: %d", clean.n_raw_rows)
    log(
        "info",
        "Embedded header rows (Label == %r): %d",
        schema.EMBEDDED_HEADER_LABEL,
        clean.n_embedded_header_rows,
    )

    # -- 3. cross-check against the physical line count --------------------- #
    if verify_raw_line_count and not clean.truncated:
        clean.n_physical_lines = count_physical_lines(path)
        expected_lines = clean.n_raw_rows + 1  # + the real header line
        if clean.n_physical_lines != expected_lines:
            raise RowCountMismatchError(
                f"{path.name}: parsed {clean.n_raw_rows:,} data rows, which implies "
                f"{expected_lines:,} physical lines, but the file has "
                f"{clean.n_physical_lines:,}. A field probably contains an embedded "
                "newline, so row positions cannot be trusted."
            )
        clean.line_count_verified = True
        log("info", "Physical line count agrees: %d lines", clean.n_physical_lines)

    if expected_raw_rows is not None and clean.n_raw_rows != expected_raw_rows:
        raise RowCountMismatchError(
            f"{path.name}: expected {expected_raw_rows:,} raw data rows, "
            f"measured {clean.n_raw_rows:,}."
        )
    if (
        expected_embedded_header_rows is not None
        and clean.n_embedded_header_rows != expected_embedded_header_rows
    ):
        raise RowCountMismatchError(
            f"{path.name}: expected {expected_embedded_header_rows} embedded header "
            f"rows, measured {clean.n_embedded_header_rows}."
        )

    # -- 4. read the data, skipping only the embedded header rows ----------- #
    # ``skiprows`` takes physical line indices: line 0 is the real header, so
    # data row i sits on line i + 1. Verified above via the line count.
    n_expected_rows = clean.n_raw_rows - clean.n_embedded_header_rows
    frame = pd.read_csv(
        path,
        dtype=schema.read_dtypes(),
        skiprows=[int(position) + 1 for position in header_positions],
        nrows=n_expected_rows if row_limit is not None else None,
    )
    if len(frame) != n_expected_rows:
        raise RowCountMismatchError(
            f"{path.name}: after removing {clean.n_embedded_header_rows} embedded "
            f"header rows, expected {n_expected_rows:,} rows but read {len(frame):,}."
        )
    clean.n_rows_after_header_removal = int(len(frame))
    log("info", "Rows after embedded-header removal: %d", clean.n_rows_after_header_removal)

    # -- 5. original row positions (assigned before any removal) ------------ #
    keep_mask = np.ones(clean.n_raw_rows, dtype=bool)
    keep_mask[header_positions] = False
    original_index = pd.Series(
        np.flatnonzero(keep_mask), name=schema.ORIGINAL_INDEX_COLUMN, dtype="int64"
    )

    # -- 6. split label / timestamp out of the frame (no copy) -------------- #
    labels = frame.pop(schema.LABEL_COLUMN).rename(schema.LABEL_COLUMN)
    timestamps_raw = frame.pop(schema.TIMESTAMP_COLUMN).rename(schema.TIMESTAMP_COLUMN)
    if list(frame.columns) != list(schema.MODEL_FEATURE_COLUMNS):
        raise SchemaMismatchError(
            "feature column order changed while reading; expected the 78 model "
            "feature columns in CSV order"
        )
    log(
        "info",
        "Feature columns: %d (%d numeric flow features + %d identifier features)",
        stats.n_model_features,
        stats.n_flow_features,
        stats.n_identifier_features,
    )

    # -- 7. count and replace NaN / +-inf, column by column ----------------- #
    for column in frame.columns:
        values = frame[column].to_numpy(copy=False)
        n_nan, n_posinf, n_neginf = _nonfinite_counts(values)
        if not (n_nan or n_posinf or n_neginf):
            continue

        if n_nan:
            clean.nan_counts_before[column] = n_nan
        if n_posinf:
            clean.posinf_counts_before[column] = n_posinf
        if n_neginf:
            clean.neginf_counts_before[column] = n_neginf

        if infinity_policy == "column_finite_extremum":
            finite = values[np.isfinite(values)]
            positive_substitute = (
                float(finite.max()) if finite.size else float(nan_replacement)
            )
            negative_substitute = (
                float(finite.min()) if finite.size else float(nan_replacement)
            )
        else:
            positive_substitute = negative_substitute = float(infinity_replacement)

        cleaned = np.where(np.isnan(values), float(nan_replacement), values)
        if n_posinf:
            cleaned = np.where(np.isposinf(cleaned), positive_substitute, cleaned)
        if n_neginf:
            cleaned = np.where(np.isneginf(cleaned), negative_substitute, cleaned)
        frame[column] = cleaned

        substitutions: Dict[str, float] = {}
        if n_posinf:
            substitutions["posinf"] = positive_substitute
        if n_neginf:
            substitutions["neginf"] = negative_substitute
        if substitutions:
            clean.infinity_substitutions[column] = substitutions

    clean.total_nan_before = sum(clean.nan_counts_before.values())
    clean.total_infinities_before = sum(clean.posinf_counts_before.values()) + sum(
        clean.neginf_counts_before.values()
    )
    log(
        "info",
        "Non-finite values in raw features: NaN=%d +inf=%d -inf=%d",
        clean.total_nan_before,
        sum(clean.posinf_counts_before.values()),
        sum(clean.neginf_counts_before.values()),
    )
    if clean.infinity_substitutions:
        log("info", "Infinity substitutions (%s): %s",
            infinity_policy, clean.infinity_substitutions)

    # -- 8. prove the result is finite -------------------------------------- #
    remaining_nan = 0
    remaining_inf = 0
    for column in frame.columns:
        values = frame[column].to_numpy(copy=False)
        if np.isfinite(values).all():
            continue
        remaining_nan += int(np.isnan(values).sum())
        remaining_inf += int(np.isinf(values).sum())
    clean.total_nan_after = remaining_nan
    clean.total_infinities_after = remaining_inf
    if remaining_nan or remaining_inf:
        raise DataQualityError(
            f"{path.name}: cleaning left {remaining_nan} NaN and {remaining_inf} "
            "infinite values in the feature frame."
        )
    log("info", "Feature frame is fully finite (NaN=0, inf=0)")

    # -- 9. negative values: report, never clip ----------------------------- #
    for column in frame.columns:
        n_negative = int((frame[column].to_numpy(copy=False) < 0).sum())
        if not n_negative:
            continue
        clean.negative_value_counts[column] = n_negative
        if column in schema.SENTINEL_NEGATIVE_COLUMNS:
            clean.sentinel_negative_counts[column] = int(
                (frame[column].to_numpy(copy=False) == schema.NOT_OBSERVED_SENTINEL).sum()
            )
        else:
            clean.non_sentinel_negative_columns[column] = n_negative
    log("info", "Negative values per column (preserved): %s", clean.negative_value_counts)
    if clean.non_sentinel_negative_columns:
        log(
            "warning",
            "Negative values outside the known -1 sentinel columns: %s "
            "(left untouched; decide per column in a later stage)",
            clean.non_sentinel_negative_columns,
        )

    # -- 10. timestamps: parse for validation, keep the raw strings --------- #
    timestamps = pd.to_datetime(
        timestamps_raw, format=schema.TIMESTAMP_FORMAT, errors="coerce"
    ).rename(schema.TIMESTAMP_COLUMN)
    stats.timestamps = _timestamp_stats(timestamps_raw, timestamps)
    log(
        "info",
        "Timestamps parsed: %d ok, %d failed, %d distinct raw values, range %s .. %s",
        stats.timestamps.n_parsed,
        stats.timestamps.n_unparsed,
        stats.timestamps.n_distinct_raw,
        stats.timestamps.min_parsed,
        stats.timestamps.max_parsed,
    )
    if require_all_timestamps_parse and stats.timestamps.n_unparsed:
        unparsed_examples = timestamps_raw[timestamps.isna()].head(5).tolist()
        raise DataQualityError(
            f"{path.name}: {stats.timestamps.n_unparsed} Timestamp value(s) do not "
            f"match {schema.TIMESTAMP_FORMAT!r}, e.g. {unparsed_examples}"
        )

    # -- 11. labels: preserve the original strings -------------------------- #
    label_counts = labels.value_counts()
    stats.label_distribution = {str(k): int(v) for k, v in label_counts.items()}
    stats.n_labels = int(label_counts.size)
    log("info", "Label distribution (%d classes): %s", stats.n_labels, stats.label_distribution)

    # -- 12. align indices -------------------------------------------------- #
    index = pd.RangeIndex(len(frame))
    frame.index = index
    labels.index = index
    timestamps_raw.index = index
    timestamps.index = index
    original_index.index = index

    # -- 13. duplicates: measure only ---------------------------------------- #
    if compute_duplicate_stats:
        stats.duplicates = _duplicate_stats(frame, labels, timestamps_raw)
        log(
            "info",
            "Duplicates measured (none removed): exact=%d temporal=%d "
            "conflicting_groups=%d conflicting_rows=%d",
            stats.duplicates.n_exact_duplicate_rows,
            stats.duplicates.n_temporal_duplicate_rows,
            stats.duplicates.n_conflicting_feature_groups,
            stats.duplicates.n_conflicting_rows,
        )

    stats.load_seconds = round(time.time() - started, 2)
    log("info", "Final feature frame: %d rows x %d columns", len(frame), frame.shape[1])

    result = LoadedDay(
        features=frame,
        labels=labels,
        timestamps=timestamps,
        timestamps_raw=timestamps_raw,
        original_index=original_index,
        source_file=path.name,
        source_path=str(path),
        feature_columns=list(schema.MODEL_FEATURE_COLUMNS),
        stats=stats,
    )
    return result.validate()


def resolve_primary_csv(cfg: Any) -> Path:
    """Absolute path of ``cfg.loader.primary_csv`` inside the raw data dir."""
    candidate = Path(cfg.loader.primary_csv).expanduser()
    if not candidate.is_absolute():
        candidate = Path(cfg.paths.raw_data_dir) / candidate
    return candidate


def load_from_config(
    cfg: Any, *, path: Optional[str | Path] = None, logger: Any = None, **overrides: Any
) -> LoadedDay:
    """Load a day CSV using the project configuration's cleaning policy."""
    target = Path(path) if path is not None else resolve_primary_csv(cfg)
    kwargs: Dict[str, Any] = {
        "nan_replacement": cfg.loader.nan_replacement,
        "infinity_policy": cfg.loader.infinity_policy,
        "infinity_replacement": cfg.loader.infinity_replacement,
        "compute_duplicate_stats": cfg.loader.compute_duplicate_stats,
        "verify_raw_line_count": cfg.loader.verify_raw_line_count,
        "row_limit": cfg.debug.row_limit_or_none,
        "logger": logger,
    }
    kwargs.update(overrides)
    return load_day_csv(target, **kwargs)


__all__: Sequence[str] = (
    "DataQualityError",
    "DuplicateStats",
    "CleaningStats",
    "LoadStats",
    "LoadedDay",
    "LoaderError",
    "RowCountMismatchError",
    "SchemaMismatchError",
    "TimestampStats",
    "count_physical_lines",
    "load_day_csv",
    "load_from_config",
    "read_header",
    "resolve_primary_csv",
    "verify_header",
)
