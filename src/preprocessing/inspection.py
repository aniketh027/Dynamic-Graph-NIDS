"""Schema-agnostic dataset inspection.

Reports what a CSV *actually contains* — no dataset-specific schema is assumed
anywhere in this module. Column roles (timestamp / label) are reported as
**candidates** derived from generic heuristics, never as facts, so that the real
CIC-IDS2018 schema can be confirmed by a human before any preprocessing code is
written.

For each discovered file the report contains:

* filename (and size on disk)
* shape (rows x columns)
* column names (plus a whitespace-in-name diagnostic)
* candidate timestamp columns (name hints + parse test)
* candidate label columns (name hints + low-cardinality hints)
* numeric vs non-numeric columns
* missing-value counts (per column and total)
* infinity counts (per numeric column and total)
* class distribution of the best candidate label column

Large files are read in chunks so memory stays bounded; in debug mode only the
configured row limit is read.
"""

from __future__ import annotations

import math
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Heuristics (generic, not dataset specific)
# --------------------------------------------------------------------------- #
#: Substrings that hint at a time column. Matched case-insensitively against
#: normalised column names (lower-cased, non-alphanumerics -> "_").
TIMESTAMP_NAME_HINTS: Tuple[str, ...] = (
    "timestamp",
    "time",
    "date",
    "datetime",
    "epoch",
    "ts",
    "first_seen",
    "last_seen",
    "start",
    "end",
)

#: Substrings that hint at a label / target column.
LABEL_NAME_HINTS: Tuple[str, ...] = (
    "label",
    "class",
    "target",
    "attack",
    "category",
    "attack_cat",
    "is_attack",
    "malicious",
    "anomaly",
    "intrusion",
    "verdict",
    "y",
)

#: Name hints that should NOT be treated as label columns even if they contain a
#: hint substring (duration-like / counter-like names).
LABEL_NAME_ANTI_HINTS: Tuple[str, ...] = ("classifier", "classification_time")


def _normalise(name: str) -> str:
    """Lower-case a column name and collapse separators to ``_``."""
    return "".join(ch if ch.isalnum() else "_" for ch in str(name).strip().lower())


def _name_matches(name: str, hints: Sequence[str]) -> List[str]:
    """Return the hints present in ``name``.

    A hint matches when it is a whole ``_``-separated token; hints of four or
    more characters may also match as a substring (so ``time`` matches
    ``timestamp``). Short hints such as ``ts`` or ``y`` stay token-only to avoid
    firing on unrelated names.
    """
    normalised = _normalise(name)
    tokens = {token for token in normalised.split("_") if token}
    matched = []
    for hint in hints:
        if hint in tokens or (len(hint) >= 4 and hint in normalised):
            matched.append(hint)
    return matched


def _datetime_parse_ratio(sample: pd.Series) -> float:
    """Fraction of ``sample`` values that parse as datetimes (never raises)."""
    for kwargs in ({"format": "mixed"}, {}):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(sample, errors="coerce", **kwargs)
            return float(parsed.notna().mean())
        except (ValueError, TypeError, OverflowError):
            continue
    return 0.0


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #
@dataclass
class ColumnCandidate:
    """A column flagged as a possible timestamp/label, with the reason why."""

    column: str
    reasons: List[str] = field(default_factory=list)
    #: Higher = stronger evidence. Used only for ordering the report.
    score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column": self.column,
            "reasons": self.reasons,
            "score": round(self.score, 3),
            "details": self.details,
        }


@dataclass
class FileReport:
    """Inspection result for a single file."""

    filename: str
    path: str
    size_mb: float
    n_rows: int
    n_columns: int
    truncated: bool
    row_limit: Optional[int]
    encoding: str
    columns: List[str] = field(default_factory=list)
    columns_with_whitespace: List[str] = field(default_factory=list)
    duplicate_columns: List[str] = field(default_factory=list)
    numeric_columns: List[str] = field(default_factory=list)
    non_numeric_columns: List[str] = field(default_factory=list)
    dtypes: Dict[str, str] = field(default_factory=dict)
    missing_counts: Dict[str, int] = field(default_factory=dict)
    total_missing: int = 0
    infinity_counts: Dict[str, int] = field(default_factory=dict)
    total_infinities: int = 0
    timestamp_candidates: List[ColumnCandidate] = field(default_factory=list)
    label_candidates: List[ColumnCandidate] = field(default_factory=list)
    selected_label_column: Optional[str] = None
    class_distribution: Dict[str, int] = field(default_factory=dict)
    class_distribution_truncated: bool = False
    errors: List[str] = field(default_factory=list)

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.n_rows, self.n_columns)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "path": self.path,
            "size_mb": self.size_mb,
            "shape": {"rows": self.n_rows, "columns": self.n_columns},
            "truncated": self.truncated,
            "row_limit": self.row_limit,
            "encoding": self.encoding,
            "columns": self.columns,
            "columns_with_whitespace": self.columns_with_whitespace,
            "duplicate_columns": self.duplicate_columns,
            "numeric_columns": self.numeric_columns,
            "non_numeric_columns": self.non_numeric_columns,
            "dtypes": self.dtypes,
            "missing_counts": self.missing_counts,
            "total_missing": self.total_missing,
            "infinity_counts": self.infinity_counts,
            "total_infinities": self.total_infinities,
            "timestamp_candidates": [c.to_dict() for c in self.timestamp_candidates],
            "label_candidates": [c.to_dict() for c in self.label_candidates],
            "selected_label_column": self.selected_label_column,
            "class_distribution": self.class_distribution,
            "class_distribution_truncated": self.class_distribution_truncated,
            "errors": self.errors,
        }


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the configured dataset directory contains no candidate file."""


# --------------------------------------------------------------------------- #
# File discovery
# --------------------------------------------------------------------------- #
def discover_files(
    data_dir: Path,
    patterns: Iterable[str] = ("*.csv", "*.csv.gz"),
    dataset_file: Optional[str] = None,
) -> List[Path]:
    """Find candidate dataset files in ``data_dir`` (recursively).

    ``dataset_file`` restricts the search to a single file name/relative path.
    Returns a sorted, de-duplicated list. Never raises when nothing is found —
    the caller decides how to report an empty result.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        return []

    if dataset_file:
        candidate = Path(dataset_file)
        candidate = candidate if candidate.is_absolute() else data_dir / candidate
        return [candidate] if candidate.is_file() else []

    found: List[Path] = []
    for pattern in patterns:
        found.extend(p for p in data_dir.rglob(pattern) if p.is_file())
    return sorted(set(found))


# --------------------------------------------------------------------------- #
# Per-file inspection
# --------------------------------------------------------------------------- #
def _read_chunks(
    path: Path, chunk_size: int, row_limit: Optional[int], encoding: str
) -> Iterable[pd.DataFrame]:
    """Yield chunks of the CSV, honouring an optional overall row limit."""
    effective_chunk = chunk_size if row_limit is None else min(chunk_size, row_limit)
    reader = pd.read_csv(
        path,
        chunksize=effective_chunk,
        encoding=encoding,
        skipinitialspace=False,
        nrows=row_limit,
        on_bad_lines="warn",
    )
    for chunk in reader:
        yield chunk


def _detect_encoding_and_header(path: Path) -> Tuple[str, List[str]]:
    """Return a working encoding and the header row for ``path``."""
    last_error: Optional[Exception] = None
    for encoding in ("utf-8", "latin-1"):
        try:
            header = pd.read_csv(path, nrows=0, encoding=encoding)
            return encoding, [str(c) for c in header.columns]
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
    raise UnicodeDecodeError(  # pragma: no cover - both encodings failing is rare
        "utf-8/latin-1", b"", 0, 1, f"could not decode {path.name}: {last_error}"
    )


def _timestamp_candidates(
    df_sample: pd.DataFrame, columns: Sequence[str], sample_size: int
) -> List[ColumnCandidate]:
    """Flag columns that *may* encode time, with the evidence for each."""
    candidates: List[ColumnCandidate] = []
    for column in columns:
        reasons: List[str] = []
        score = 0.0
        details: Dict[str, Any] = {}
        series = df_sample[column] if column in df_sample.columns else None

        hits = _name_matches(column, TIMESTAMP_NAME_HINTS)
        if hits:
            reasons.append(f"name hint: {', '.join(hits)}")
            score += 1.0

        if series is not None:
            if pd.api.types.is_datetime64_any_dtype(series):
                reasons.append("dtype is datetime64")
                score += 2.0
            elif series.dtype == object:
                sample = series.dropna().astype(str).head(sample_size)
                if len(sample) > 0:
                    ratio = _datetime_parse_ratio(sample)
                    details["parse_success_ratio"] = round(ratio, 3)
                    details["example_values"] = sample.head(3).tolist()
                    if ratio >= 0.9:
                        reasons.append(f"{ratio:.0%} of sampled values parse as datetime")
                        score += 2.0
                    elif ratio >= 0.5:
                        reasons.append(f"only {ratio:.0%} of sampled values parse as datetime")
                        score += 0.5

        if reasons:
            candidates.append(ColumnCandidate(column, reasons, score, details))

    candidates.sort(key=lambda c: (-c.score, c.column))
    return candidates


def _label_candidates(
    df_sample: pd.DataFrame,
    columns: Sequence[str],
    max_cardinality: int,
    timestamp_columns: Sequence[str],
) -> List[ColumnCandidate]:
    """Flag columns that *may* be the label, with the evidence for each."""
    candidates: List[ColumnCandidate] = []
    for column in columns:
        if column in timestamp_columns:
            continue
        normalised = _normalise(column)
        if any(anti in normalised for anti in LABEL_NAME_ANTI_HINTS):
            continue

        reasons: List[str] = []
        score = 0.0
        details: Dict[str, Any] = {}

        hits = _name_matches(column, LABEL_NAME_HINTS)
        if hits:
            reasons.append(f"name hint: {', '.join(hits)}")
            score += 2.0

        if column in df_sample.columns:
            series = df_sample[column].dropna()
            n_unique = int(series.nunique(dropna=True))
            details["n_unique_in_sample"] = n_unique
            if 1 < n_unique <= max_cardinality:
                is_text = series.dtype == object or isinstance(
                    series.dtype, pd.CategoricalDtype
                )
                if is_text:
                    reasons.append(f"low-cardinality text column ({n_unique} distinct)")
                    score += 1.5
                elif pd.api.types.is_numeric_dtype(series) and n_unique <= 2:
                    reasons.append(f"binary numeric column ({n_unique} distinct)")
                    score += 1.0
                else:
                    reasons.append(f"low-cardinality column ({n_unique} distinct)")
                    score += 0.25
                details["example_values"] = [
                    _jsonable(v) for v in series.unique()[:10].tolist()
                ]

        if reasons:
            candidates.append(ColumnCandidate(column, reasons, score, details))

    candidates.sort(key=lambda c: (-c.score, c.column))
    return candidates


def _jsonable(value: Any) -> Any:
    """Convert numpy/pandas scalars to plain Python for JSON serialisation."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else str(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def inspect_file(
    path: Path,
    *,
    chunk_size: int = 500_000,
    row_limit: Optional[int] = None,
    max_label_cardinality: int = 50,
    timestamp_sample_size: int = 200,
) -> FileReport:
    """Inspect one CSV file and return a :class:`FileReport`.

    No schema is assumed. Statistics are accumulated chunk-by-chunk so files far
    larger than RAM can be inspected; ``row_limit`` (debug mode) caps the number
    of rows read.
    """
    path = Path(path)
    encoding, header = _detect_encoding_and_header(path)

    report = FileReport(
        filename=path.name,
        path=str(path),
        size_mb=round(path.stat().st_size / (1024 * 1024), 2),
        n_rows=0,
        n_columns=len(header),
        truncated=False,
        row_limit=row_limit,
        encoding=encoding,
        columns=header,
    )
    report.columns_with_whitespace = [c for c in header if c != c.strip()]
    # Names that collide once surrounding whitespace/case differences are removed
    # (pandas already renames exact duplicates to ``name.1``).
    collisions = Counter(_normalise(c) for c in header)
    report.duplicate_columns = sorted(
        c for c in header if collisions[_normalise(c)] > 1
    )

    missing = pd.Series(0, index=pd.Index(header), dtype="int64")
    infinities = pd.Series(0, index=pd.Index(header), dtype="int64")
    numeric_flags: Dict[str, bool] = {}
    dtypes: Dict[str, str] = {}
    first_chunk: Optional[pd.DataFrame] = None
    label_counts: Dict[Any, int] = {}
    selected_label: Optional[str] = None

    for chunk in _read_chunks(path, chunk_size, row_limit, encoding):
        chunk.columns = [str(c) for c in chunk.columns]
        if first_chunk is None:
            first_chunk = chunk.head(max(timestamp_sample_size, 1_000)).copy()
            report.timestamp_candidates = _timestamp_candidates(
                first_chunk, chunk.columns, timestamp_sample_size
            )
            report.label_candidates = _label_candidates(
                first_chunk,
                chunk.columns,
                max_label_cardinality,
                [c.column for c in report.timestamp_candidates],
            )
            if report.label_candidates:
                selected_label = report.label_candidates[0].column

        report.n_rows += len(chunk)
        missing = missing.add(chunk.isna().sum(), fill_value=0)

        for column in chunk.columns:
            series = chunk[column]
            is_numeric = pd.api.types.is_numeric_dtype(series)
            # A column is numeric only if it is numeric in every chunk.
            numeric_flags[column] = numeric_flags.get(column, True) and is_numeric
            dtypes[column] = str(series.dtype)
            if is_numeric:
                n_inf = int(np.isinf(series.to_numpy(dtype="float64", na_value=np.nan)).sum())
                if n_inf:
                    infinities[column] = infinities.get(column, 0) + n_inf

        if selected_label is not None and selected_label in chunk.columns:
            for value, count in chunk[selected_label].value_counts(dropna=False).items():
                key = _jsonable(value)
                label_counts[key] = label_counts.get(key, 0) + int(count)

    if first_chunk is None:
        report.errors.append("file contains a header but no data rows")

    if row_limit is not None and report.n_rows >= row_limit:
        report.truncated = True

    report.dtypes = dtypes
    report.numeric_columns = [c for c, flag in numeric_flags.items() if flag]
    report.non_numeric_columns = [c for c, flag in numeric_flags.items() if not flag]
    report.missing_counts = {c: int(v) for c, v in missing.items() if int(v) > 0}
    report.total_missing = int(missing.sum())
    report.infinity_counts = {c: int(v) for c, v in infinities.items() if int(v) > 0}
    report.total_infinities = int(infinities.sum())
    report.selected_label_column = selected_label
    report.class_distribution = dict(
        sorted(label_counts.items(), key=lambda kv: -kv[1])
    )
    report.class_distribution_truncated = report.truncated
    return report


def inspect_directory(
    data_dir: Path,
    *,
    patterns: Iterable[str] = ("*.csv", "*.csv.gz"),
    dataset_file: Optional[str] = None,
    chunk_size: int = 500_000,
    row_limit: Optional[int] = None,
    max_label_cardinality: int = 50,
    timestamp_sample_size: int = 200,
    logger: Any = None,
) -> List[FileReport]:
    """Inspect every candidate file in ``data_dir``.

    Raises :class:`DatasetNotFoundError` when no candidate file exists, so the
    caller can print the "dataset not found" guidance and exit gracefully.
    """
    files = discover_files(data_dir, patterns, dataset_file)
    if not files:
        raise DatasetNotFoundError(str(data_dir))

    reports: List[FileReport] = []
    for path in files:
        if logger is not None:
            logger.info("Inspecting %s ...", path.name)
        try:
            reports.append(
                inspect_file(
                    path,
                    chunk_size=chunk_size,
                    row_limit=row_limit,
                    max_label_cardinality=max_label_cardinality,
                    timestamp_sample_size=timestamp_sample_size,
                )
            )
        except Exception as exc:  # keep going: one broken file must not stop the scan
            if logger is not None:
                logger.error("Failed to inspect %s: %s", path.name, exc)
            broken = FileReport(
                filename=path.name,
                path=str(path),
                size_mb=round(path.stat().st_size / (1024 * 1024), 2),
                n_rows=0,
                n_columns=0,
                truncated=False,
                row_limit=row_limit,
                encoding="unknown",
            )
            broken.errors.append(f"{type(exc).__name__}: {exc}")
            reports.append(broken)
    return reports
