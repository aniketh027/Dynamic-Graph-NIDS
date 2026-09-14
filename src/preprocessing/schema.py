"""Confirmed CSE-CIC-IDS2018 CSV schema.

Unlike :mod:`src.preprocessing.inspection` — which is deliberately
schema-agnostic and only reports *candidate* column roles — this module records
the schema that was **measured** from the raw files during the dataset
inspection phase. It is the single source of truth for column names and column
roles for every stage that follows.

Measured facts (see ``results/inspection/INSPECTION_REPORT.md`` and
``results/data_profile/STAGE_B_REPORT.md``)
------------------------------------------------------------------------
* Nine of the ten day files carry the same 80 column **names**
  (md5 of the stripped header line ``d8434b53de72c6a7f4b3ee19ba28381d``).
* The 80 columns decompose as::

      Dst Port + Protocol + Timestamp + 76 numeric flow features + Label

* ``Timestamp`` and ``Label`` are **not** model features.
* ``Dst Port`` and ``Protocol`` are numeric identifiers/categoricals that are
  used as model features, giving 78 model feature columns in total.
* ``02-20-2018.csv`` carries **84** columns: the same 80, with four extra
  identifier columns prepended (:data:`EXTENDED_IDENTIFIER_COLUMNS`). It is a
  superset, so the 78 model features are available there unchanged.

Terminology used across the project
-----------------------------------
``FLOW_FEATURE_COLUMNS``
    The 76 numeric flow features (the CICFlowMeter measurements).
``IDENTIFIER_FEATURE_COLUMNS``
    ``Dst Port`` and ``Protocol``.
``MODEL_FEATURE_COLUMNS``
    The 78 columns actually fed to a model = the 76 flow features plus the 2
    identifier features, kept in original CSV order.

Nothing here encodes labels, scales values, or assumes a class vocabulary; the
class mapping is fixed in a later preprocessing stage because some CIC-IDS2018
classes are time-segregated across day files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Raw CSV layout
# --------------------------------------------------------------------------- #
#: The 80 CSV columns, in the exact order they appear in the raw day files.
CSV_COLUMNS: Tuple[str, ...] = (
    "Dst Port",
    "Protocol",
    "Timestamp",
    "Flow Duration",
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
    "Flow Byts/s",
    "Flow Pkts/s",
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
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd Header Len",
    "Bwd Header Len",
    "Fwd Pkts/s",
    "Bwd Pkts/s",
    "Pkt Len Min",
    "Pkt Len Max",
    "Pkt Len Mean",
    "Pkt Len Std",
    "Pkt Len Var",
    "FIN Flag Cnt",
    "SYN Flag Cnt",
    "RST Flag Cnt",
    "PSH Flag Cnt",
    "ACK Flag Cnt",
    "URG Flag Cnt",
    "CWE Flag Count",
    "ECE Flag Cnt",
    "Down/Up Ratio",
    "Pkt Size Avg",
    "Fwd Seg Size Avg",
    "Bwd Seg Size Avg",
    "Fwd Byts/b Avg",
    "Fwd Pkts/b Avg",
    "Fwd Blk Rate Avg",
    "Bwd Byts/b Avg",
    "Bwd Pkts/b Avg",
    "Bwd Blk Rate Avg",
    "Subflow Fwd Pkts",
    "Subflow Fwd Byts",
    "Subflow Bwd Pkts",
    "Subflow Bwd Byts",
    "Init Fwd Win Byts",
    "Init Bwd Win Byts",
    "Fwd Act Data Pkts",
    "Fwd Seg Size Min",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
    "Label",
)

#: md5 of the raw header line **with the trailing newline stripped**, measured on
#: all nine 80-column day CSVs. Stripping makes the value independent of the
#: line ending, which matters: eight files use LF and ``03-02-2018.csv`` uses
#: CRLF, so the *raw bytes* of their header lines are NOT identical even though
#: the 80 names are. Used only as a diagnostic in error messages.
#: The dataset inspection report quotes ``64a4bae24658fa55bc9f55889af725da`` for
#: the same header — that is the md5 of the LF-terminated line including ``\n``.
HEADER_MD5 = "d8434b53de72c6a7f4b3ee19ba28381d"

# --------------------------------------------------------------------------- #
# The 84-column variant (02-20-2018.csv only)
# --------------------------------------------------------------------------- #
#: Extra identifier columns prepended to the standard 80 in the one 84-column
#: file. Measured: ``CSV_COLUMNS_EXTENDED[4:] == CSV_COLUMNS`` exactly, so the
#: file is a strict superset and every model feature is available unchanged.
#: They are NOT model features - a flow ID and raw IP addresses would leak
#: identity and cannot generalise - but they are never removed from the raw file
#: and their presence is reported explicitly.
EXTENDED_IDENTIFIER_COLUMNS: Tuple[str, ...] = ("Flow ID", "Src IP", "Src Port", "Dst IP")

#: The 84 columns of the extended variant, in file order.
CSV_COLUMNS_EXTENDED: Tuple[str, ...] = EXTENDED_IDENTIFIER_COLUMNS + CSV_COLUMNS

#: md5 of the extended header line with the trailing newline stripped.
#: (That file uses CRLF; its raw-bytes md5 is ``8062b6f49c245d6f3049dd77af3dc2ea``.)
HEADER_MD5_EXTENDED = "7a20eb9a113f6e512c09397ff997537a"

#: Names of the two known layouts.
VARIANT_STANDARD = "standard_80"
VARIANT_EXTENDED = "extended_84"
VARIANT_UNKNOWN = "unknown"

#: Seven of the ten CSVs hold exactly this many data rows: 2**20 - 1, the Excel
#: worksheet row limit. A file sitting exactly on it is very likely truncated.
EXCEL_ROW_CAP = 1_048_575


# --------------------------------------------------------------------------- #
# Column roles
# --------------------------------------------------------------------------- #
TIMESTAMP_COLUMN = "Timestamp"
LABEL_COLUMN = "Label"

#: Numeric identifier/categorical columns that are still used as model features.
IDENTIFIER_FEATURE_COLUMNS: Tuple[str, ...] = ("Dst Port", "Protocol")

#: Columns that are never model features.
NON_FEATURE_COLUMNS: Tuple[str, ...] = (TIMESTAMP_COLUMN, LABEL_COLUMN)

#: The 76 numeric flow features (CSV order).
FLOW_FEATURE_COLUMNS: Tuple[str, ...] = tuple(
    column
    for column in CSV_COLUMNS
    if column not in IDENTIFIER_FEATURE_COLUMNS and column not in NON_FEATURE_COLUMNS
)

#: The 78 model feature columns = flow features + identifier features (CSV order).
MODEL_FEATURE_COLUMNS: Tuple[str, ...] = tuple(
    column for column in CSV_COLUMNS if column not in NON_FEATURE_COLUMNS
)

# --------------------------------------------------------------------------- #
# Measured data quirks (see the inspection report for the per-file counts)
# --------------------------------------------------------------------------- #
#: Rate columns that contain ``+Infinity`` / ``NaN`` tokens in the raw CSV
#: (division by a zero flow duration).
RATE_COLUMNS: Tuple[str, ...] = ("Flow Byts/s", "Flow Pkts/s")

#: Columns where a negative value is the CICFlowMeter "not observed" sentinel
#: (measured: the only negative value present in these columns is ``-1``).
#: These values are legitimate and must NOT be clipped or imputed.
SENTINEL_NEGATIVE_COLUMNS: Tuple[str, ...] = ("Init Fwd Win Byts", "Init Bwd Win Byts")

#: The sentinel value itself.
NOT_OBSERVED_SENTINEL = -1

#: Rows where ``Label`` holds this exact string are repeated header lines that
#: were concatenated into the data (every one of their 80 fields equals its own
#: column name). They are the only rows this stage removes.
EMBEDDED_HEADER_LABEL = "Label"

#: Raw timestamp format. Measured: the hour field only ever holds 01-12 and no
#: AM/PM marker is present anywhere, so the wall-clock hour is ambiguous. That
#: ambiguity is NOT resolved here; see the inspection report.
TIMESTAMP_FORMAT = "%d/%m/%Y %H:%M:%S"

#: Name of the metadata column carrying each row's original position in the
#: source CSV (0-based, counted over data rows only, assigned before any row is
#: removed). This — combined with the source file's capture date — is the
#: project's deterministic capture-order proxy, because the raw Timestamp is
#: ambiguous and heavily tied.
ORIGINAL_INDEX_COLUMN = "Original_Index"


# --------------------------------------------------------------------------- #
# Self-consistency (cheap, runs at import; a typo above must not go unnoticed)
# --------------------------------------------------------------------------- #
def _validate_schema() -> None:
    if len(CSV_COLUMNS) != 80:
        raise AssertionError(f"CSV_COLUMNS must hold 80 names, got {len(CSV_COLUMNS)}")
    if len(set(CSV_COLUMNS)) != len(CSV_COLUMNS):
        raise AssertionError("CSV_COLUMNS contains duplicate names")
    if len(FLOW_FEATURE_COLUMNS) != 76:
        raise AssertionError(
            f"expected 76 numeric flow features, got {len(FLOW_FEATURE_COLUMNS)}"
        )
    if len(MODEL_FEATURE_COLUMNS) != 78:
        raise AssertionError(
            f"expected 78 model feature columns, got {len(MODEL_FEATURE_COLUMNS)}"
        )
    if len(CSV_COLUMNS_EXTENDED) != 84:
        raise AssertionError(
            f"CSV_COLUMNS_EXTENDED must hold 84 names, got {len(CSV_COLUMNS_EXTENDED)}"
        )
    if CSV_COLUMNS_EXTENDED[len(EXTENDED_IDENTIFIER_COLUMNS) :] != CSV_COLUMNS:
        raise AssertionError("the extended variant must be a superset of CSV_COLUMNS")
    if set(EXTENDED_IDENTIFIER_COLUMNS) & set(CSV_COLUMNS):
        raise AssertionError("extra identifier columns overlap the standard schema")
    missing = [
        column
        for group in (RATE_COLUMNS, SENTINEL_NEGATIVE_COLUMNS, IDENTIFIER_FEATURE_COLUMNS)
        for column in group
        if column not in CSV_COLUMNS
    ]
    if missing:
        raise AssertionError(f"unknown column name(s) referenced: {missing}")


_validate_schema()


# --------------------------------------------------------------------------- #
# Schema-variant detection
# --------------------------------------------------------------------------- #
@dataclass
class SchemaVariant:
    """Result of matching one file's header against the known layouts.

    ``usable`` is the only thing later stages need: it says the 78 model feature
    columns plus ``Timestamp`` and ``Label`` are all present under their exact
    expected names, so the file can be read with the standard dtype map. Extra
    columns are reported in :attr:`extra_columns`, never silently dropped —
    excluding them from the model feature set is an explicit decision recorded
    in :attr:`notes`.
    """

    variant: str = VARIANT_UNKNOWN
    n_columns: int = 0
    usable: bool = False
    #: Columns present in the file but not part of the standard 80.
    extra_columns: List[str] = field(default_factory=list)
    #: Standard-80 columns the file does not have.
    missing_columns: List[str] = field(default_factory=list)
    #: Standard-80 columns whose position differs from the canonical order.
    reordered: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variant": self.variant,
            "n_columns": self.n_columns,
            "usable": self.usable,
            "extra_columns": self.extra_columns,
            "missing_columns": self.missing_columns,
            "reordered": self.reordered,
            "notes": self.notes,
        }


def detect_schema_variant(names: Sequence[str]) -> SchemaVariant:
    """Classify a CSV header against the two known CIC-IDS2018 layouts.

    Never raises: an unrecognised header is reported as ``usable=False`` with the
    exact difference, so a multi-file run can report the odd file instead of
    dying on it.
    """
    names = [str(name) for name in names]
    result = SchemaVariant(n_columns=len(names))
    standard = list(CSV_COLUMNS)

    if names == standard:
        result.variant = VARIANT_STANDARD
        result.usable = True
        result.notes.append("exact match for the standard 80-column layout")
        return result

    if names == list(CSV_COLUMNS_EXTENDED):
        result.variant = VARIANT_EXTENDED
        result.usable = True
        result.extra_columns = list(EXTENDED_IDENTIFIER_COLUMNS)
        result.notes.append(
            "known 84-column layout: the standard 80 columns with "
            f"{', '.join(EXTENDED_IDENTIFIER_COLUMNS)} prepended"
        )
        result.notes.append(
            "the 4 extra columns are read-excluded from the 78 model features by "
            "explicit decision (flow ID and raw endpoint identity do not "
            "generalise); they remain untouched in the raw file"
        )
        return result

    present = set(names)
    result.missing_columns = [name for name in standard if name not in present]
    result.extra_columns = [name for name in names if name not in set(standard)]
    if not result.missing_columns:
        # All 80 are there; check whether their relative order still matches.
        subset = [name for name in names if name in set(standard)]
        result.reordered = [
            found for found, want in zip(subset, standard) if found != want
        ]
        result.usable = not result.reordered
        result.variant = VARIANT_UNKNOWN
        if result.usable:
            result.notes.append(
                "unrecognised layout, but all 80 standard columns are present in "
                "canonical relative order, so the file is readable"
            )
        else:
            result.notes.append(
                "all 80 standard columns are present but their order differs from "
                "the canonical order"
            )
    else:
        result.notes.append(
            f"{len(result.missing_columns)} standard column(s) are missing; the "
            "file cannot be read with the project schema"
        )
    return result


def read_dtypes() -> Dict[str, object]:
    """``dtype`` mapping for :func:`pandas.read_csv`.

    The 78 model feature columns are read directly as ``float64`` (the C parser
    turns ``Infinity`` / ``NaN`` tokens into ``inf`` / ``nan``, which the loader
    then counts and replaces). ``Timestamp`` and ``Label`` are read as text so
    the raw representation is preserved byte-for-byte.

    Only valid once the embedded header rows have been excluded from the read —
    otherwise the literal string ``"Dst Port"`` would hit a float column.
    """
    import numpy as np

    dtypes: Dict[str, object] = {column: np.float64 for column in MODEL_FEATURE_COLUMNS}
    dtypes[TIMESTAMP_COLUMN] = str
    dtypes[LABEL_COLUMN] = str
    return dtypes


def describe() -> str:
    """Human-readable summary of the schema (used in logs and reports)."""
    return "\n".join(
        [
            f"raw CSV columns          : {len(CSV_COLUMNS)} (standard) / "
            f"{len(CSV_COLUMNS_EXTENDED)} (extended, 02-20 only)",
            f"  timestamp column       : {TIMESTAMP_COLUMN} (format {TIMESTAMP_FORMAT!r})",
            f"  label column           : {LABEL_COLUMN}",
            f"  identifier features    : {len(IDENTIFIER_FEATURE_COLUMNS)} "
            f"({', '.join(IDENTIFIER_FEATURE_COLUMNS)})",
            f"  numeric flow features  : {len(FLOW_FEATURE_COLUMNS)}",
            f"  model feature columns  : {len(MODEL_FEATURE_COLUMNS)} "
            "(flow features + identifier features)",
            f"  extended-only columns  : {len(EXTENDED_IDENTIFIER_COLUMNS)} "
            f"({', '.join(EXTENDED_IDENTIFIER_COLUMNS)}) - not model features",
            f"  rate columns (inf/NaN) : {', '.join(RATE_COLUMNS)}",
            f"  sentinel -1 columns    : {', '.join(SENTINEL_NEGATIVE_COLUMNS)}",
        ]
    )
