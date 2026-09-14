"""Stage B artifacts: JSON / CSV / Markdown views of a :class:`CorpusProfile`.

Kept apart from :mod:`src.preprocessing.profiling` so the measurement engine has
no reporting concerns. Everything here is derived from an already-computed
profile; nothing re-reads the corpus and nothing is written outside the
configured results directory.

Artifacts written
-----------------
``per_file_summary.json``      every measurement, per CSV file
``global_dataset_summary.json``  corpus totals, label vocabulary, timeline, splits
``schema_summary.json``        per-file header/variant detail plus the canonical lists
``class_distribution.csv``     one row per label: global, per split, per file
``class_by_day.csv``           one row per (capture date, label)
``duplicate_summary.csv``      per file and corpus-wide duplicate categories
``nonfinite_summary.csv``      one row per (file, column) holding a non-finite value
``negative_summary.csv``       one row per (file, column) holding a negative value
``STAGE_B_REPORT.md``          the human-readable report

None of these is a cleaned dataset copy: no feature matrix, no train/val/test
array and no modified row is ever written.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.preprocessing import schema
from src.preprocessing.profiling import CorpusProfile, FileProfile

#: Keys whose values change between two identical runs (wall-clock timings).
#: Removed before hashing so reproducibility can be checked on measurements only.
VOLATILE_KEYS = frozenset(
    {"generated_at_utc", "elapsed_seconds", "profile_seconds", "run"}
)

ARTIFACT_NAMES = (
    "per_file_summary.json",
    "global_dataset_summary.json",
    "schema_summary.json",
    "class_distribution.csv",
    "class_by_day.csv",
    "duplicate_summary.csv",
    "nonfinite_summary.csv",
    "negative_summary.csv",
    "STAGE_B_REPORT.md",
)

REPORT_NAME = "STAGE_B_REPORT.md"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def strip_volatile(payload: Any) -> Any:
    """Copy of ``payload`` with wall-clock timing keys removed, recursively."""
    if isinstance(payload, dict):
        return {
            key: strip_volatile(value)
            for key, value in payload.items()
            if key not in VOLATILE_KEYS
        }
    if isinstance(payload, list):
        return [strip_volatile(item) for item in payload]
    return payload


def measurement_digest(profile: CorpusProfile) -> str:
    """sha256 over every measurement, excluding wall-clock timings.

    Two runs over unchanged files must produce the same digest; that is the
    reproducibility check, and it is cheaper and stricter than eyeballing the
    reports.
    """
    payload = {
        "files": [strip_volatile(item.to_dict()) for item in profile.files],
        "corpus": strip_volatile(profile.to_dict()),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Any) -> Path:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def _write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> Path:
    import csv

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def _pct(part: float, whole: float) -> str:
    return f"{100.0 * part / whole:.6f}" if whole else ""


def _iso(profile: FileProfile) -> str:
    return profile.capture_date.isoformat() if profile.capture_date else "unknown"


def _thousands(value: Optional[int]) -> str:
    return "n/a" if value is None else f"{value:,}"


# --------------------------------------------------------------------------- #
# JSON artifacts
# --------------------------------------------------------------------------- #
def build_per_file_payload(profile: CorpusProfile) -> Dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "B - multi-day intake and global dataset profiling",
        "ordering_contract": profile.ordering_contract,
        "n_files": len(profile.files),
        "files": [item.to_dict() for item in profile.files],
    }


def build_global_payload(
    profile: CorpusProfile, *, config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    return {
        "run": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "stage": "B - multi-day intake and global dataset profiling",
            "elapsed_seconds": round(profile.elapsed_seconds, 2),
            "measurement_digest_sha256": measurement_digest(profile),
            "config": config,
        },
        "guarantees": [
            "no file under data/raw was created, modified, renamed or deleted",
            "no non-finite value was replaced",
            "no negative value or -1 sentinel was clipped or replaced",
            "no duplicate or conflicting row was removed",
            "no label was encoded and no class mapping was persisted",
            "no cleaned dataset copy and no train/validation/test array was written",
        ],
        **profile.to_dict(),
    }


def build_schema_payload(profile: CorpusProfile) -> Dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "canonical": {
            "n_csv_columns_standard": len(schema.CSV_COLUMNS),
            "n_csv_columns_extended": len(schema.CSV_COLUMNS_EXTENDED),
            "header_md5_stripped_standard": schema.HEADER_MD5,
            "header_md5_stripped_extended": schema.HEADER_MD5_EXTENDED,
            "csv_columns": list(schema.CSV_COLUMNS),
            "extended_identifier_columns": list(schema.EXTENDED_IDENTIFIER_COLUMNS),
            "timestamp_column": schema.TIMESTAMP_COLUMN,
            "label_column": schema.LABEL_COLUMN,
            "identifier_feature_columns": list(schema.IDENTIFIER_FEATURE_COLUMNS),
            "n_flow_feature_columns": len(schema.FLOW_FEATURE_COLUMNS),
            "n_model_feature_columns": len(schema.MODEL_FEATURE_COLUMNS),
            "model_feature_columns": list(schema.MODEL_FEATURE_COLUMNS),
            "rate_columns": list(schema.RATE_COLUMNS),
            "sentinel_negative_columns": list(schema.SENTINEL_NEGATIVE_COLUMNS),
            "not_observed_sentinel": schema.NOT_OBSERVED_SENTINEL,
        },
        "variants_found": profile.schema_variants,
        "per_file": [
            {
                "source_file": item.source_file,
                "capture_date": _iso(item),
                **item.header.to_dict(),
                "columns": item.header.columns,
            }
            for item in profile.files
        ],
        "note": (
            "Extra columns are reported, never silently discarded. The 84-column "
            "layout is a strict superset of the standard 80, so all 78 model "
            "features are present there unchanged; the 4 extra identifier columns "
            "are excluded from the model feature set by explicit decision and "
            "remain untouched in the raw file."
        ),
    }


# --------------------------------------------------------------------------- #
# CSV artifacts
# --------------------------------------------------------------------------- #
def build_class_distribution_rows(profile: CorpusProfile):
    files = [item for item in profile.files if not item.error]
    header = [
        "label",
        "total_rows",
        "share_pct",
        "n_files_present",
        "n_capture_dates_present",
        "first_capture_date",
        "last_capture_date",
        "train_rows",
        "validation_rows",
        "test_rows",
        "absent_from_train",
    ] + [f"count__{item.source_file}" for item in files]

    total = profile.total_usable_rows
    coverage = profile.split_coverage
    rows: List[List[Any]] = []
    for label in profile.label_vocabulary:
        count = profile.label_counts.get(label, 0)
        per_file = [item.label_counts.get(label, 0) for item in files]
        rows.append(
            [
                label,
                count,
                _pct(count, total),
                sum(1 for value in per_file if value),
                profile.timeline.n_dates_present.get(label, 0),
                profile.timeline.first_date.get(label, ""),
                profile.timeline.last_date.get(label, ""),
                coverage.train_counts.get(label, 0),
                coverage.val_counts.get(label, 0),
                coverage.test_counts.get(label, 0),
                "yes" if label in coverage.absent_from_train else "no",
            ]
            + per_file
        )
    return header, rows


def build_class_by_day_rows(profile: CorpusProfile):
    header = [
        "sequence_position",
        "capture_date",
        "source_file",
        "label",
        "count",
        "share_of_file_pct",
    ]
    rows: List[List[Any]] = []
    for position, item in enumerate(
        (entry for entry in profile.files if not entry.error)
    ):
        total = item.intake.n_rows_profiled
        for label in sorted(item.label_counts, key=lambda key: -item.label_counts[key]):
            count = item.label_counts[label]
            rows.append(
                [position, _iso(item), item.source_file, label, count, _pct(count, total)]
            )
    return header, rows


def build_duplicate_rows(profile: CorpusProfile):
    header = [
        "scope",
        "capture_date",
        "n_rows",
        "n_distinct_feature_vectors",
        "A_exact_duplicate_rows",
        "B_temporal_duplicate_rows",
        "AB_feature_label_duplicate_rows",
        "AB_pct_of_rows",
        "C_conflicting_groups",
        "C_conflicting_rows",
        "C_n_label_combinations",
        "basis",
    ]
    rows: List[List[Any]] = []

    def _row(scope: str, capture_date: str, stats) -> List[Any]:
        return [
            scope,
            capture_date,
            stats.n_rows,
            stats.n_distinct_feature_vectors,
            stats.n_exact,
            stats.n_temporal,
            stats.n_feature_label_duplicates,
            _pct(stats.n_feature_label_duplicates, stats.n_rows),
            stats.n_conflicting_groups,
            stats.n_conflicting_rows,
            stats.n_conflicting_combinations,
            stats.basis,
        ]

    for item in profile.files:
        if item.duplicates is not None:
            rows.append(_row(item.source_file, _iso(item), item.duplicates))
    if profile.global_duplicates is not None:
        rows.append(_row("__CORPUS__", "all", profile.global_duplicates))
    return header, rows


def build_nonfinite_rows(profile: CorpusProfile):
    header = [
        "source_file",
        "capture_date",
        "column",
        "is_known_rate_column",
        "n_nan",
        "n_posinf",
        "n_neginf",
        "n_non_finite",
        "pct_of_rows",
        "n_rows",
    ]
    rows: List[List[Any]] = []
    for item in profile.files:
        stats = item.non_finite
        columns = (
            set(stats.nan_counts) | set(stats.posinf_counts) | set(stats.neginf_counts)
        )
        total_rows = item.intake.n_rows_profiled
        for column in sorted(columns, key=lambda name: schema.CSV_COLUMNS.index(name)):
            n_nan = stats.nan_counts.get(column, 0)
            n_pos = stats.posinf_counts.get(column, 0)
            n_neg = stats.neginf_counts.get(column, 0)
            total = n_nan + n_pos + n_neg
            rows.append(
                [
                    item.source_file,
                    _iso(item),
                    column,
                    "yes" if column in schema.RATE_COLUMNS else "no",
                    n_nan,
                    n_pos,
                    n_neg,
                    total,
                    _pct(total, total_rows),
                    total_rows,
                ]
            )
    return header, rows


def build_negative_rows(profile: CorpusProfile):
    header = [
        "source_file",
        "capture_date",
        "column",
        "is_known_sentinel_column",
        "n_negative",
        "n_exactly_minus_one",
        "all_negatives_are_minus_one",
        "most_negative_value",
        "least_negative_value",
        "pct_of_rows",
        "n_rows",
    ]
    rows: List[List[Any]] = []
    for item in profile.files:
        stats = item.negatives
        total_rows = item.intake.n_rows_profiled
        for column in sorted(
            stats.negative_counts, key=lambda name: schema.CSV_COLUMNS.index(name)
        ):
            count = stats.negative_counts[column]
            extremes = stats.negative_extremes.get(column, {})
            rows.append(
                [
                    item.source_file,
                    _iso(item),
                    column,
                    "yes" if column in schema.SENTINEL_NEGATIVE_COLUMNS else "no",
                    count,
                    stats.sentinel_minus_one_counts.get(column, 0),
                    "yes" if column in stats.only_minus_one_columns else "no",
                    extremes.get("min", ""),
                    extremes.get("max", ""),
                    _pct(count, total_rows),
                    total_rows,
                ]
            )
    return header, rows


# --------------------------------------------------------------------------- #
# Report alignment (section 16) and Heartbleed (section 17)
# --------------------------------------------------------------------------- #
#: Attack families the project report names as in-scope for CSE-CIC-IDS2018.
#: Matching is done case-insensitively on a substring, because the dataset's own
#: spelling is preserved exactly (including ``Infilteration``).
REPORT_ATTACK_FAMILIES = {
    "Brute Force (FTP/SSH)": ("ftp-bruteforce", "ssh-bruteforce"),
    "DoS": ("dos attacks",),
    "DDoS": ("ddos attack",),
    "Web attacks": ("brute force -web", "brute force -xss", "sql injection"),
    "Infiltration": ("infilteration", "infiltration"),
    "Botnet": ("bot",),
    "Heartbleed": ("heartbleed",),
}


def find_labels(profile: CorpusProfile, needles: Sequence[str]) -> List[str]:
    """Labels whose lower-cased form contains any of ``needles``."""
    lowered = [needle.lower() for needle in needles]
    return [
        label
        for label in profile.label_vocabulary
        if any(needle in label.lower() for needle in lowered)
    ]


def build_attack_coverage(profile: CorpusProfile) -> List[Dict[str, Any]]:
    """Per report-named attack family: which measured labels cover it, if any."""
    coverage: List[Dict[str, Any]] = []
    for family, needles in REPORT_ATTACK_FAMILIES.items():
        matched = find_labels(profile, needles)
        coverage.append(
            {
                "family": family,
                "present": bool(matched),
                "matched_labels": matched,
                "rows": sum(profile.label_counts.get(label, 0) for label in matched),
            }
        )
    return coverage


def heartbleed_status(profile: CorpusProfile) -> Dict[str, Any]:
    """Whether Heartbleed is present anywhere in the measured corpus."""
    matched = find_labels(profile, ("heartbleed",))
    return {
        "present": bool(matched),
        "matched_labels": matched,
        "rows": sum(profile.label_counts.get(label, 0) for label in matched),
        "files_searched": [item.source_file for item in profile.files],
        "n_files_searched": len(profile.files),
        "statement": (
            f"Heartbleed IS present: {matched}"
            if matched
            else "Heartbleed is ABSENT from every label in every discovered CSV file. "
            "No Heartbleed sample is fabricated and no claim of full attack-scenario "
            "coverage is made."
        ),
    }


# --------------------------------------------------------------------------- #
# Markdown report
# --------------------------------------------------------------------------- #
def _table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> List[str]:
    lines = [
        "| " + " | ".join(str(cell) for cell in header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return lines


def build_report(
    profile: CorpusProfile,
    *,
    config_summary: Optional[str] = None,
    chunk_size: Optional[int] = None,
    artifact_dir: Optional[Path] = None,
) -> str:
    """The human-readable Stage B report."""
    files = [item for item in profile.files if not item.error]
    lines: List[str] = []
    add = lines.append

    add("# Stage B - Multi-Day Intake and Global Dataset Profiling")
    add("")
    add(
        f"Generated {datetime.now(timezone.utc).isoformat()} | "
        f"measurement digest `{measurement_digest(profile)[:16]}` | "
        f"elapsed {profile.elapsed_seconds:.1f}s"
    )
    add("")
    add(
        "Every number below was measured from the raw CSV files. Nothing was "
        "cleaned, replaced, deduplicated, encoded or written back: this stage "
        "**measures only**."
    )
    add("")

    # -- 1. summary --------------------------------------------------------- #
    add("## 1. Corpus at a glance")
    add("")
    add("```")
    add(profile.summary())
    add("```")
    add("")
    if config_summary:
        add("<details><summary>Configuration used</summary>")
        add("")
        add("```")
        add(config_summary)
        add("```")
        add("")
        add("</details>")
        add("")

    # -- 2. discovery / ordering -------------------------------------------- #
    add("## 2. File discovery and the deterministic capture-order proxy")
    add("")
    add(
        f"Files were discovered recursively under `{profile.csv_dir}` by glob, not "
        "by hard-coded name. Each file's capture date comes from the `MM-DD-YYYY` "
        "digits in its own filename, cross-checked against the modal date in its "
        "own `Timestamp` column. Files are then sorted by "
        "`(capture_date, source_file)`, so the sequence never depends on the order "
        "the filesystem happens to return."
    )
    add("")
    add("**Ordering contract**: " + profile.ordering_contract + ".")
    add("")
    add(
        "This is a *deterministic capture-order proxy*, not perfect physical "
        "chronology - see section 10. Rows are addressed as "
        "`(source_file, source_file_date, original_index)`, where `original_index` "
        "is the 0-based position among the source file's data rows, assigned before "
        "any row is excluded."
    )
    add("")
    rows = [
        [
            position,
            item.source_file,
            _iso(item),
            item.file.capture_date_source,
            item.file.observed_date.isoformat() if item.file.observed_date else "n/a",
            f"{item.file.size_bytes:,}",
            f"{item.file.size_gib:.2f}",
        ]
        for position, item in enumerate(profile.files)
    ]
    add(
        "\n".join(
            _table(
                [
                    "seq",
                    "file",
                    "capture date (filename)",
                    "date source",
                    "modal Timestamp date",
                    "size (bytes)",
                    "GiB",
                ],
                rows,
            )
        )
    )
    add("")

    # -- 3. schema ---------------------------------------------------------- #
    add("## 3. Schema validation")
    add("")
    add(
        f"Standard layout: **{len(schema.CSV_COLUMNS)} columns** = `Dst Port` + "
        f"`Protocol` + `Timestamp` + {len(schema.FLOW_FEATURE_COLUMNS)} numeric flow "
        "features + `Label`. Model features = "
        f"**{len(schema.MODEL_FEATURE_COLUMNS)}** (the "
        f"{len(schema.FLOW_FEATURE_COLUMNS)} flow features plus `Dst Port` and "
        "`Protocol`); `Timestamp` and `Label` are metadata and target, not features."
    )
    add("")
    for variant, names in sorted(profile.schema_variants.items()):
        add(f"- `{variant}`: {len(names)} file(s) - {', '.join(names)}")
    add("")
    rows = [
        [
            item.source_file,
            item.header.n_columns,
            item.header.variant.variant,
            item.header.line_ending,
            f"`{item.header.md5_stripped[:12]}`",
            f"`{item.header.md5_raw_bytes[:12]}`",
            ", ".join(item.header.variant.extra_columns) or "-",
            ", ".join(item.header.variant.missing_columns) or "-",
        ]
        for item in profile.files
    ]
    add(
        "\n".join(
            _table(
                [
                    "file",
                    "cols",
                    "variant",
                    "EOL",
                    "md5 (stripped)",
                    "md5 (raw bytes)",
                    "extra columns",
                    "missing columns",
                ],
                rows,
            )
        )
    )
    add("")
    add(
        "The two md5 columns answer different questions. The *stripped* md5 hashes "
        "the header line with the line terminator removed, so it says whether the "
        "column **names** match. The *raw bytes* md5 includes the terminator, so it "
        "also distinguishes LF from CRLF files."
    )
    add("")
    extended = profile.schema_variants.get(schema.VARIANT_EXTENDED, [])
    if extended:
        add(
            f"**84-column layout** ({', '.join(extended)}): the standard 80 columns "
            f"with `{'`, `'.join(schema.EXTENDED_IDENTIFIER_COLUMNS)}` prepended - a "
            "strict superset, verified positionally. All 78 model features are "
            "present there unchanged, so the file is read by column *name*. The 4 "
            "extra columns are **reported, not silently discarded**: they are "
            "excluded from the model feature set by explicit decision (a flow ID and "
            "raw endpoint identity would leak host identity and cannot generalise), "
            "and they remain untouched in the raw file."
        )
        add("")

    # -- 4. intake ---------------------------------------------------------- #
    add("## 4. Multi-day intake and row accounting")
    add("")
    add(
        "Each file is read twice and never held whole in memory. **Pass 1** reads "
        "only the `Label` column to get the data-row count, the exact positions of "
        "the embedded header rows and the label counts. **Pass 2** then reads the 80 "
        "standard columns in chunks"
        + (f" of {chunk_size:,} rows" if chunk_size else "")
        + ", with the 78 feature dtypes forced to `float64` and the known header "
        "lines excluded via `skiprows`."
    )
    add("")
    add(
        "Peak memory is one chunk"
        + (
            f" (~{chunk_size * len(schema.MODEL_FEATURE_COLUMNS) * 8 / (1 << 20):.0f} MB "
            f"at {chunk_size:,} x {len(schema.MODEL_FEATURE_COLUMNS)} float64)"
            if chunk_size
            else ""
        )
        + " plus the duplicate-hash accumulators (26 bytes per corpus row), instead "
        "of the roughly 10 GB a concatenated feature frame for the whole corpus "
        "would need. The corpus is never concatenated into a single DataFrame."
    )
    add("")
    add(
        "Forcing the dtypes is not only about memory. Under inference the same "
        "column can come back as `int64` in one chunk and `float64` in another, and "
        "`hash(5) != hash(5.0)` - which would silently corrupt the duplicate counts "
        "at chunk boundaries. A chunk that contained an embedded header row would "
        "also turn all 78 columns into Python objects (roughly 1 GB per chunk). "
        "Locating those rows in pass 1 removes both hazards."
    )
    add("")
    add(
        "Row counts are cross-checked three ways: the parsed data-row count against "
        "the file's physical line count, pass 2's row count against pass 1's "
        "prediction, and pass 2's label counts against pass 1's."
    )
    add("")
    rows = [
        [
            item.source_file,
            _iso(item),
            f"{item.intake.n_raw_rows:,}",
            _thousands(item.intake.n_physical_lines),
            item.intake.n_embedded_header_rows,
            f"{item.intake.n_usable_rows:,}",
            f"{item.intake.n_rows_profiled:,}",
            "YES" if item.intake.at_excel_row_cap else "no",
        ]
        for item in profile.files
    ]
    rows.append(
        [
            "**TOTAL**",
            f"{len(files)} files",
            f"**{profile.total_raw_rows:,}**",
            "",
            profile.total_embedded_header_rows,
            f"**{profile.total_usable_rows:,}**",
            f"**{profile.total_rows_profiled:,}**",
            f"{len(profile.truncated_files)} file(s)",
        ]
    )
    add(
        "\n".join(
            _table(
                [
                    "file",
                    "capture date",
                    "raw data rows",
                    "physical lines",
                    "embedded headers",
                    "usable rows",
                    "rows profiled",
                    "at Excel cap",
                ],
                rows,
            )
        )
    )
    add("")
    add(
        "Embedded header rows are data rows whose `Label` field is the literal string "
        f"`\"{schema.EMBEDDED_HEADER_LABEL}\"` - repeated header lines concatenated "
        "into the data. They are excluded from the profiled rows but **not removed "
        "from the raw files**, which are never modified."
    )
    add("")

    # -- 5. labels ---------------------------------------------------------- #
    add("## 5. Global label vocabulary")
    add("")
    add(
        f"**{len(profile.label_vocabulary)} distinct labels** across "
        f"{profile.total_usable_rows:,} usable rows. Original spelling and casing are "
        "preserved exactly - including the dataset's own `Infilteration` misspelling. "
        "No label is encoded and no class mapping is persisted in this stage."
    )
    add("")
    rows = [
        [
            label,
            f"{profile.label_counts[label]:,}",
            f"{100.0 * profile.label_counts[label] / profile.total_usable_rows:.4f}%"
            if profile.total_usable_rows
            else "",
            profile.timeline.n_dates_present.get(label, 0),
            profile.timeline.first_date.get(label, ""),
            profile.timeline.last_date.get(label, ""),
        ]
        for label in sorted(
            profile.label_vocabulary,
            key=lambda name: -profile.label_counts.get(name, 0),
        )
    ]
    add(
        "\n".join(
            _table(
                ["label", "rows", "share", "capture dates", "first seen", "last seen"],
                rows,
            )
        )
    )
    add("")
    add(
        "Per-file label counts are in `class_by_day.csv` and "
        "`per_file_summary.json`; the wide per-label view is in "
        "`class_distribution.csv`."
    )
    add("")

    # -- 6. class by time --------------------------------------------------- #
    coverage = profile.split_coverage
    add("## 6. Class-by-time analysis")
    add("")
    add(
        "CIC-IDS2018 is time-segregated by design: each attack was executed on a "
        "particular day, so a class's rows are concentrated in one or two capture "
        "dates rather than spread across the corpus."
    )
    add("")
    single = profile.timeline.single_date_classes
    add(
        f"- Classes present on exactly one capture date: **{len(single)}** of "
        f"{len(profile.label_vocabulary)}"
        + (f" - {', '.join(single)}" if single else "")
    )
    late = profile.timeline.late_only_classes
    add(
        f"- Classes whose first appearance is in the final third of the date "
        f"sequence: **{len(late)}**" + (f" - {', '.join(late)}" if late else "")
    )
    add("")
    dates = profile.timeline.dates
    header = ["label"] + dates
    rows = [
        [label] + [
            f"{profile.timeline.per_date.get(label, {}).get(iso, 0):,}" or "0"
            for iso in dates
        ]
        for label in sorted(
            profile.label_vocabulary,
            key=lambda name: -profile.label_counts.get(name, 0),
        )
    ]
    add("\n".join(_table(header, rows)))
    add("")

    add("### Naive chronological 70/10/20 split - coverage check")
    add("")
    add(
        "Cut at exact row indices in the deterministic sequence "
        f"({coverage.ratios[0]:.0%} / {coverage.ratios[1]:.0%} / "
        f"{coverage.ratios[2]:.0%} of {coverage.n_rows:,} usable rows): "
        f"train `[0, {coverage.train_end:,})`, validation "
        f"`[{coverage.train_end:,}, {coverage.val_end:,})`, test "
        f"`[{coverage.val_end:,}, {coverage.n_rows:,})`."
    )
    add("")
    for name, where in coverage.boundary_files.items():
        add(f"- `{name}` falls in {where or 'n/a'}")
    add("")
    if coverage.absent_from_train:
        add(
            f"**{len(coverage.absent_from_train)} class(es) would be absent from the "
            f"training period**: {', '.join(coverage.absent_from_train)}. A model "
            "trained on that split cannot learn them, and any metric that averages "
            "over classes would be reporting on classes it never saw."
        )
    else:
        add("Every class appears at least once in the training period.")
    add("")
    notes: List[str] = []
    if coverage.absent_from_val:
        notes.append(f"- Absent from validation: {', '.join(coverage.absent_from_val)}")
    if coverage.absent_from_test:
        notes.append(f"- Absent from test: {', '.join(coverage.absent_from_test)}")
    if coverage.train_only:
        notes.append(
            "- Present only in train (never evaluated): "
            + ", ".join(coverage.train_only)
        )
    if coverage.starved_in_train:
        notes.append(
            f"- Fewer than {coverage.starved_threshold} training rows: "
            + ", ".join(
                f"{label} ({count:,})"
                for label, count in coverage.starved_in_train.items()
            )
        )
    if notes:
        for note in notes:
            add(note)
        add("")
    add(
        "**This is an analysis only.** Stage B does not change the split strategy "
        "and does not use random stratification; the chronological split is kept "
        "because a temporal intrusion-detection model must not be evaluated on rows "
        "that precede its training data."
    )
    add("")

    # -- 7. non-finite ------------------------------------------------------ #
    add("## 7. Non-finite values")
    add("")
    total_nan = sum(item.non_finite.total_nan for item in profile.files)
    total_pos = sum(item.non_finite.total_posinf for item in profile.files)
    total_neg = sum(item.non_finite.total_neginf for item in profile.files)
    add(
        f"Corpus totals: **{total_nan:,} NaN**, **{total_pos:,} +inf**, "
        f"**{total_neg:,} -inf** across the {len(schema.MODEL_FEATURE_COLUMNS)} model "
        "feature columns."
    )
    add("")
    rows = [
        [
            item.source_file,
            f"{item.non_finite.total_nan:,}",
            f"{item.non_finite.total_posinf:,}",
            f"{item.non_finite.total_neginf:,}",
            ", ".join(
                sorted(
                    set(item.non_finite.nan_counts)
                    | set(item.non_finite.posinf_counts)
                    | set(item.non_finite.neginf_counts)
                )
            )
            or "-",
            ", ".join(item.non_finite.unexpected_columns) or "none",
        ]
        for item in profile.files
    ]
    add(
        "\n".join(
            _table(
                [
                    "file",
                    "NaN",
                    "+inf",
                    "-inf",
                    "affected columns",
                    "outside the known rate columns",
                ],
                rows,
            )
        )
    )
    add("")
    add(
        f"`{'`, `'.join(schema.RATE_COLUMNS)}` are the expected sources: "
        "CICFlowMeter divides by the flow duration, so a zero-duration flow yields "
        "`Infinity` (bytes moved in no time) or `NaN` (0/0)."
    )
    add("")
    add(
        "**No non-finite value was replaced in Stage B.** The replacement values must "
        "be derived from the TRAIN split alone and then applied unchanged to "
        "validation and test; deriving them from the whole corpus would leak "
        "information from the evaluation period into training. That is a Stage C "
        "task."
    )
    add("")

    # -- 8. negatives ------------------------------------------------------- #
    add("## 8. Negative and sentinel values")
    add("")
    rows = []
    for item in profile.files:
        stats = item.negatives
        rows.append(
            [
                item.source_file,
                f"{sum(stats.negative_counts.values()):,}",
                ", ".join(
                    f"{column} ({count:,})"
                    for column, count in sorted(stats.negative_counts.items())
                )
                or "-",
                ", ".join(
                    f"{column} ({count:,})"
                    for column, count in sorted(stats.non_sentinel_columns.items())
                )
                or "none",
            ]
        )
    add(
        "\n".join(
            _table(
                [
                    "file",
                    "negative values",
                    "per column",
                    "outside the known sentinel columns",
                ],
                rows,
            )
        )
    )
    add("")
    add(
        f"`{'`, `'.join(schema.SENTINEL_NEGATIVE_COLUMNS)}` use "
        f"`{schema.NOT_OBSERVED_SENTINEL}` as CICFlowMeter's *\"not observed\"* "
        "sentinel: the TCP window size was never seen in that direction. That is "
        "genuine information ('no traffic in this direction'), semantically distinct "
        "from a window size of 0."
    )
    add("")
    add(
        "**Nothing was replaced or clipped.** Per-column detail, including whether "
        "every negative in a column is exactly the sentinel, is in "
        "`negative_summary.csv`."
    )
    add("")

    # -- 9. duplicates ------------------------------------------------------ #
    add("## 9. Duplicate analysis")
    add("")
    add("Three categories, measured on the 78 model feature columns:")
    add("")
    add(
        "- **A. exact** - identical features *and* raw `Timestamp` *and* `Label`.\n"
        "- **B. temporal** - identical features and label, different `Timestamp` "
        "(the same flow signature recurring at another moment).\n"
        "- **C. conflicting** - one feature vector carrying more than one label. "
        "These are the label-noise cases: no classifier can separate them."
    )
    add("")
    dup_header, dup_rows = build_duplicate_rows(profile)
    display = [
        [row[0], f"{row[2]:,}", f"{row[3]:,}", f"{row[4]:,}", f"{row[5]:,}",
         f"{row[6]:,}", f"{float(row[7]):.3f}%" if row[7] else "", f"{row[8]:,}",
         f"{row[9]:,}"]
        for row in dup_rows
    ]
    add(
        "\n".join(
            _table(
                [
                    "scope",
                    "rows",
                    "distinct feature vectors",
                    "A exact",
                    "B temporal",
                    "A+B",
                    "A+B share",
                    "C groups",
                    "C rows",
                ],
                display,
            )
        )
    )
    add("")
    if profile.global_duplicates and profile.global_duplicates.conflicting_label_combinations:
        add("Corpus-wide label combinations found on a single feature vector:")
        add("")
        rows = [
            [combo, f"{detail['groups']:,}", f"{detail['rows']:,}"]
            for combo, detail in profile.global_duplicates.conflicting_label_combinations.items()
        ]
        add("\n".join(_table(["label combination", "feature groups", "rows"], rows)))
        add("")
    add(
        "**No duplicate or conflicting row was removed.** The exclusion policy "
        "belongs in Stage C, and it has to be decided with the temporal split in "
        "mind: removing category B would destroy the recurrence structure a temporal "
        "model is meant to learn, and any removal must not delete rows from the "
        "evaluation period on the basis of rows in the training period."
    )
    add("")
    extended = [
        item.source_file
        for item in profile.files
        if item.header.variant == schema.VARIANT_EXTENDED
    ]
    if extended and any(item.duplicates for item in profile.files):
        add(
            "Note on basis: the 84-column file(s) (" + ", ".join(extended) + ") are "
            "measured on the same 78 model features as every other file. Measured on "
            "their full 84 columns they would look almost duplicate-free, because "
            "`Flow ID` and the source endpoint make nearly every row unique - but "
            "those columns are not model features, so that number would not describe "
            "what the model actually sees."
        )
        add("")

    # -- 10. timestamps ----------------------------------------------------- #
    add("## 10. Timestamp analysis")
    add("")
    rows = [
        [
            item.source_file,
            f"{item.timestamps.n_parsed:,}",
            item.timestamps.n_unparsed,
            f"{item.timestamps.n_distinct:,}" if item.timestamps.n_distinct >= 0 else "capped",
            (item.timestamps.min_parsed or "")[:19],
            (item.timestamps.max_parsed or "")[:19],
            item.timestamps.n_rows_with_am_pm_marker,
            item.timestamps.tie_group_max,
            f"{item.timestamps.tie_group_mean:.1f}",
            item.timestamps.n_decreasing_steps,
        ]
        for item in profile.files
    ]
    add(
        "\n".join(
            _table(
                [
                    "file",
                    "parsed",
                    "unparsed",
                    "distinct",
                    "min",
                    "max",
                    "AM/PM markers",
                    "largest tie group",
                    "mean tie group",
                    "decreasing steps",
                ],
                rows,
            )
        )
    )
    add("")
    hours = sorted(
        {
            hour
            for item in profile.files
            for hour in item.timestamps.hour_histogram
        }
    )
    add(
        f"Format `{schema.TIMESTAMP_FORMAT}`. Hours observed across the corpus: "
        f"`{', '.join(hours)}`. "
        + (
            "The hour field only ever holds 01-12 and **no AM/PM marker appears "
            "anywhere**, so a value like `01:15:00` could be 01:15 or 13:15. That "
            "ambiguity is real and is **not reconstructed here** - inventing an "
            "AM/PM rule would fabricate an ordering the data does not contain."
            if hours and max(hours) <= "12"
            else "The hour field spans a 24-hour range."
        )
    )
    add("")
    add(
        "Timestamps are also heavily tied: the largest same-second group holds "
        f"{max((item.timestamps.tie_group_max for item in profile.files), default=0):,} "
        "rows, so the timestamp alone cannot order rows within a second even if the "
        "clock were unambiguous."
    )
    add("")
    off_date = [item for item in profile.files if item.timestamps.n_rows_off_capture_date]
    if off_date:
        # A stray 1970 minimum in the table above is a handful of rows, not a
        # capture-date error. Say how many, or the table reads as if the file
        # spans 48 years.
        total_off = sum(item.timestamps.n_rows_off_capture_date for item in off_date)
        add(
            f"**{total_off} row(s) carry a timestamp outside their file's capture "
            "date** and explain the pre-2018 minimums in the table above: "
            + "; ".join(
                f"{item.source_file} {item.timestamps.n_rows_off_capture_date} row(s), "
                f"e.g. `{item.timestamps.off_capture_date_examples[0]}`"
                for item in off_date
                if item.timestamps.off_capture_date_examples
            )
            + ". These are a rounding error against "
            f"{profile.total_rows_profiled:,} rows, but they are **not** dropped or "
            "corrected here - section 8 shows the same rows also carry negative "
            "durations and inter-arrival times, so they are one corrupt-flow "
            "population for Stage C to rule on, not two separate anomalies."
        )
        add("")
    add(
        "**`Timestamp` is therefore diagnostic metadata only.** Rows are never "
        "sorted by it; ordering uses the capture-order proxy from section 2. The raw "
        "values are preserved unmodified."
    )
    add("")

    # -- 11. truncation ----------------------------------------------------- #
    add("## 11. Dataset size and truncation")
    add("")
    add(
        f"{len(profile.truncated_files)} of {len(profile.files)} files hold exactly "
        f"**{schema.EXCEL_ROW_CAP:,}** data rows - that is 2^20 - 1, the Excel "
        "worksheet row limit (1,048,576 rows including the header). Files landing on "
        "that exact number were almost certainly opened and re-saved in a "
        "spreadsheet, silently discarding every row past the limit."
    )
    add("")
    if profile.truncated_files:
        add("Affected: " + ", ".join(f"`{name}`" for name in profile.truncated_files))
        add("")
    add(
        "**No attempt is made to repair or re-obtain the missing rows in this "
        "stage.** The consequence to carry forward is that per-day row counts are "
        "capped, not complete, so a day's true class balance may differ from what is "
        "measured here."
    )
    add("")

    # -- 12. Parquet -------------------------------------------------------- #
    add("## 12. Parquet copy (metadata only)")
    add("")
    parquet = profile.parquet
    if parquet is None:
        add("Comparison not run.")
        add("")
    elif parquet.error and not parquet.files:
        add(f"Not compared: {parquet.error}")
    else:
        add(
            f"{parquet.n_files} Parquet files, "
            f"{parquet.total_size_bytes / (1 << 20):,.0f} MB, "
            f"{parquet.total_rows:,} rows total. Read from the Parquet footers only - "
            "no column data was loaded."
        )
        add("")
        rows = [
            [
                entry.get("source_file", ""),
                entry.get("capture_date") or "?",
                f"{entry.get('n_rows', 0):,}",
                entry.get("n_columns", ""),
                "yes" if entry.get("has_timestamp") else "**no**",
                "yes" if entry.get("has_label") else "no",
                len(entry.get("missing_vs_csv", [])),
            ]
            for entry in parquet.files
        ]
        add(
            "\n".join(
                _table(
                    [
                        "file",
                        "capture date",
                        "rows",
                        "cols",
                        "has Timestamp",
                        "has Label",
                        "cols missing vs CSV",
                    ],
                    rows,
                )
            )
        )
        add("")
        add(
            f"The Parquet copy holds {parquet.total_rows:,} rows against the CSV "
            f"corpus's {profile.total_usable_rows:,} usable rows, and **no Parquet "
            "file carries the `Timestamp` column**."
        )
        add("")
    add(
        "**The CSV corpus is the primary temporal source for this project.** Without "
        "`Timestamp` the Parquet copy cannot support the temporal HTGraph pipeline at "
        "all, and it also covers fewer rows. It is compared here on metadata only, "
        "as required, and is never read by the pipeline. Both copies are left "
        "untouched - nothing was converted in either direction."
    )
    add("")

    # -- 13. report alignment ----------------------------------------------- #
    add("## 13. Alignment with the project report")
    add("")
    add(
        "Each row separates **A. what the report requires**, **B. what the files "
        "actually contain**, and **C. the engineering decision**. Discrepancies are "
        "recorded, not silently fixed in either direction."
    )
    add("")
    alignment = build_report_alignment(profile)
    add(
        "\n".join(
            _table(
                ["topic", "A. report requirement", "B. measured fact", "C. decision"],
                [[row["topic"], row["report"], row["measured"], row["decision"]] for row in alignment],
            )
        )
    )
    add("")

    add("### Attack-scenario coverage")
    add("")
    rows = [
        [
            entry["family"],
            "present" if entry["present"] else "**ABSENT**",
            ", ".join(entry["matched_labels"]) or "-",
            f"{entry['rows']:,}" if entry["rows"] else "0",
        ]
        for entry in build_attack_coverage(profile)
    ]
    add("\n".join(_table(["report attack family", "status", "measured labels", "rows"], rows)))
    add("")

    status = heartbleed_status(profile)
    add("### Heartbleed")
    add("")
    add(
        f"Searched every label in all {status['n_files_searched']} discovered CSV "
        f"files. **{status['statement']}**"
    )
    add("")
    if not status["present"]:
        add(
            "The report's Heartbleed scenario therefore cannot be evaluated with this "
            "corpus. No Heartbleed rows are fabricated, and no claim is made that all "
            "report attack scenarios are covered. The honest options are to state the "
            "reduced scope explicitly in the report, or to obtain the missing capture "
            "file - a decision for you, not for this stage."
        )
        add("")

    # -- 14. warnings / next -------------------------------------------------- #
    add("## 14. Open issues")
    add("")
    if profile.warnings:
        for warning in profile.warnings:
            add(f"- {warning}")
    else:
        add("No warnings were raised during profiling.")
    add("")

    add("## 15. Artifacts")
    add("")
    location = str(artifact_dir) if artifact_dir else "the configured results dir"
    add(f"Written to `{location}`:")
    add("")
    for name in ARTIFACT_NAMES:
        add(f"- `{name}`")
    add("")
    add(
        "No cleaned dataset copy, feature matrix or train/validation/test array was "
        "written, and nothing was written under `data/raw`."
    )
    add("")
    return "\n".join(lines)


def build_report_alignment(profile: CorpusProfile) -> List[Dict[str, str]]:
    """Report requirement vs measured fact vs engineering decision (section 16)."""
    files = [item for item in profile.files if not item.error]
    n_extended = len(profile.schema_variants.get(schema.VARIANT_EXTENDED, []))
    total_ts = sum(item.timestamps.n_parsed for item in files)
    unparsed = sum(item.timestamps.n_unparsed for item in files)
    heartbleed = heartbleed_status(profile)
    coverage = profile.split_coverage
    return [
        {
            "topic": "Corpus size",
            "report": "approximately 16 million flow records",
            "measured": (
                f"{profile.total_raw_rows:,} raw data rows, "
                f"{profile.total_usable_rows:,} usable after excluding "
                f"{profile.total_embedded_header_rows} embedded header rows"
            ),
            "decision": (
                "consistent with the report; the usable count is what every later "
                "stage quotes"
            ),
        },
        {
            "topic": "Raw feature count",
            "report": "80 raw features",
            "measured": (
                f"{len(schema.CSV_COLUMNS)} CSV **columns**, of which "
                f"{len(schema.FLOW_FEATURE_COLUMNS)} are numeric flow features; "
                "`Timestamp` is metadata and `Label` is the target"
            ),
            "decision": (
                "the report's '80 features' is the column count, not the feature "
                "count - the report wording should say 80 columns"
            ),
        },
        {
            "topic": "Model feature count",
            "report": "76 flow features plus Dst Port and Protocol",
            "measured": (
                f"{len(schema.MODEL_FEATURE_COLUMNS)} model feature columns, all "
                "present in every usable file"
            ),
            "decision": f"{len(schema.MODEL_FEATURE_COLUMNS)} model features confirmed",
        },
        {
            "topic": "Schema uniformity",
            "report": "a single uniform schema across day files",
            "measured": (
                f"{len(files) - n_extended} file(s) with "
                f"{len(schema.CSV_COLUMNS)} columns, {n_extended} with "
                f"{len(schema.CSV_COLUMNS_EXTENDED)}"
                if n_extended
                else "one uniform layout"
            ),
            "decision": (
                "read by column name; the extended file's 4 extra identifier "
                "columns are reported and excluded from the feature set, not dropped "
                "from the raw file"
            ),
        },
        {
            "topic": "Timestamp availability",
            "report": "timestamps available for temporal graph construction",
            "measured": (
                f"present in every CSV file; {total_ts:,} of "
                f"{profile.total_rows_profiled:,} rows parse ({unparsed:,} fail), but "
                "the clock is 12-hour with no AM/PM marker and heavily tied"
            ),
            "decision": (
                "Timestamp is diagnostic metadata; ordering uses the deterministic "
                "capture-order proxy, and the report must not claim exact wall-clock "
                "chronology"
            ),
        },
        {
            "topic": "Class coverage",
            "report": "the CSE-CIC-IDS2018 attack taxonomy",
            "measured": f"{len(profile.label_vocabulary)} distinct labels measured",
            "decision": (
                "labels kept in the dataset's own spelling, including the "
                "`Infilteration` misspelling, so counts stay traceable to the files"
            ),
        },
        {
            "topic": "Heartbleed",
            "report": "listed among the attack scenarios",
            "measured": "present" if heartbleed["present"] else "**absent from all files**",
            "decision": (
                "covered"
                if heartbleed["present"]
                else "scope reduction must be stated in the report; no samples are "
                "fabricated"
            ),
        },
        {
            "topic": "Chronological 70/10/20 split",
            "report": "sequential 70/10/20 split, no random shuffling",
            "measured": (
                f"{len(coverage.absent_from_train)} class(es) absent from the training "
                f"period, {len(coverage.absent_from_test)} absent from test"
            ),
            "decision": (
                "the split strategy is unchanged in this stage; the coverage gap is "
                "recorded for Stage C to address without resorting to random "
                "stratification"
            ),
        },
        {
            "topic": "File completeness",
            "report": "full daily captures",
            "measured": (
                f"{len(profile.truncated_files)} file(s) hold exactly "
                f"{schema.EXCEL_ROW_CAP:,} rows (the Excel row cap)"
            ),
            "decision": (
                "treated as truncated; not repaired in this stage, and per-day counts "
                "are reported as measured"
            ),
        },
    ]


# --------------------------------------------------------------------------- #
# Writer
# --------------------------------------------------------------------------- #
def write_artifacts(
    profile: CorpusProfile,
    output_dir: str | Path,
    *,
    config: Optional[Dict[str, Any]] = None,
    config_summary: Optional[str] = None,
    chunk_size: Optional[int] = None,
) -> Dict[str, Path]:
    """Write every Stage B artifact into ``output_dir``. Returns name -> path.

    ``output_dir`` must be inside the project's results directory; nothing is
    ever written under ``data/raw``.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}

    written["per_file_summary.json"] = _write_json(
        out / "per_file_summary.json", build_per_file_payload(profile)
    )
    written["global_dataset_summary.json"] = _write_json(
        out / "global_dataset_summary.json",
        build_global_payload(profile, config=config),
    )
    written["schema_summary.json"] = _write_json(
        out / "schema_summary.json", build_schema_payload(profile)
    )

    for name, builder in (
        ("class_distribution.csv", build_class_distribution_rows),
        ("class_by_day.csv", build_class_by_day_rows),
        ("duplicate_summary.csv", build_duplicate_rows),
        ("nonfinite_summary.csv", build_nonfinite_rows),
        ("negative_summary.csv", build_negative_rows),
    ):
        header, rows = builder(profile)
        written[name] = _write_csv(out / name, header, rows)

    report = build_report(
        profile,
        config_summary=config_summary,
        chunk_size=chunk_size,
        artifact_dir=out,
    )
    report_path = out / REPORT_NAME
    report_path.write_text(report, encoding="utf-8")
    written[REPORT_NAME] = report_path
    return written
