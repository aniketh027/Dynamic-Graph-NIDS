"""Inspect the raw dataset that has been placed in the configured data directory.

This script makes NO assumptions about the dataset schema. It reports what the
files actually contain so the preprocessing pipeline can be written against the
real columns.

Usage
-----
    python scripts/inspect_dataset.py
    python scripts/inspect_dataset.py --debug                 # cap rows read
    python scripts/inspect_dataset.py --data-dir path/to/csvs
    python scripts/inspect_dataset.py --file Thursday.csv --max-columns 0

Exit codes
----------
    0  a report was produced
    2  no dataset file found in the configured directory
    1  unexpected error
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

# Allow running the script directly (``python scripts/inspect_dataset.py``).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config, ConfigError  # noqa: E402
from src.preprocessing.inspection import (  # noqa: E402
    DatasetNotFoundError,
    FileReport,
    inspect_directory,
)
from src.utils.logging_utils import get_logger  # noqa: E402

DATASET_MISSING_MESSAGE = (
    "Dataset not found. Add CIC-IDS2018 CSV to the configured data/raw directory."
)
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_NO_DATASET = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Schema-agnostic inspection of the raw dataset CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="Path to a YAML config file.")
    parser.add_argument(
        "--data-dir", default=None, help="Override the configured raw data directory."
    )
    parser.add_argument(
        "--file",
        default=None,
        help="Inspect only this file (name or path relative to the data dir).",
    )
    debug_group = parser.add_mutually_exclusive_group()
    debug_group.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        default=None,
        help="Enable debug mode (read at most the configured row limit).",
    )
    debug_group.add_argument(
        "--no-debug",
        dest="debug",
        action="store_false",
        help="Disable debug mode (read every row).",
    )
    parser.add_argument(
        "--debug-rows", type=int, default=None, help="Override the debug row limit."
    )
    parser.add_argument(
        "--max-columns",
        type=int,
        default=25,
        help="Column names to print per file (0 = print all).",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Do not write the JSON report."
    )
    return parser


def _resolve_config(args: argparse.Namespace) -> Config:
    cfg = Config.load(args.config)
    if args.data_dir:
        path = Path(args.data_dir).expanduser()
        cfg.paths.raw_data_dir = path if path.is_absolute() else PROJECT_ROOT / path
    if args.file:
        cfg.paths.dataset_file = args.file
    if args.debug is not None:
        cfg.debug.enabled = args.debug
    if args.debug_rows is not None:
        cfg.debug.row_limit = args.debug_rows
    return cfg.validate()


def _fmt_list(values: List[str], limit: int, indent: str = "    ") -> str:
    """Format a list of names, optionally truncated to ``limit`` entries."""
    if not values:
        return f"{indent}(none)"
    shown = values if limit <= 0 else values[:limit]
    lines = [f"{indent}{i:>3}. {name}" for i, name in enumerate(shown, start=1)]
    if len(shown) < len(values):
        lines.append(f"{indent}... and {len(values) - len(shown)} more")
    return "\n".join(lines)


def print_report(report: FileReport, max_columns: int) -> None:
    """Print one file's inspection result in a readable form."""
    print("=" * 78)
    print(f"FILE: {report.filename}")
    print("=" * 78)
    print(f"  path                 : {report.path}")
    print(f"  size                 : {report.size_mb} MB   encoding: {report.encoding}")

    if report.errors:
        for error in report.errors:
            print(f"  ERROR                : {error}")
        if report.n_columns == 0:
            return

    rows_note = ""
    if report.truncated:
        rows_note = f"  (TRUNCATED to debug row limit = {report.row_limit:,})"
    print(f"  shape (rows, cols)   : ({report.n_rows:,}, {report.n_columns}){rows_note}")
    print(f"  numeric columns      : {len(report.numeric_columns)}")
    print(f"  non-numeric columns  : {len(report.non_numeric_columns)}")
    print(f"  missing values total : {report.total_missing:,}")
    print(f"  infinity values total: {report.total_infinities:,}")

    print("\n  COLUMN NAMES")
    print(_fmt_list(report.columns, max_columns))
    if report.columns_with_whitespace:
        print("\n  ! column names with leading/trailing whitespace:")
        print(_fmt_list([repr(c) for c in report.columns_with_whitespace], max_columns))
    if report.duplicate_columns:
        print("\n  ! column names that collide after normalisation:")
        print(_fmt_list(report.duplicate_columns, max_columns))

    print("\n  CANDIDATE TIMESTAMP COLUMNS (heuristic - confirm manually)")
    if report.timestamp_candidates:
        for candidate in report.timestamp_candidates:
            print(f"    - {candidate.column}  [{'; '.join(candidate.reasons)}]")
            examples = candidate.details.get("example_values")
            if examples:
                print(f"        examples: {examples}")
    else:
        print("    (none detected)")

    print("\n  CANDIDATE LABEL COLUMNS (heuristic - confirm manually)")
    if report.label_candidates:
        for candidate in report.label_candidates:
            print(f"    - {candidate.column}  [{'; '.join(candidate.reasons)}]")
    else:
        print("    (none detected)")

    print("\n  NUMERIC COLUMNS")
    print(_fmt_list(report.numeric_columns, max_columns))
    print("\n  NON-NUMERIC COLUMNS")
    print(_fmt_list(report.non_numeric_columns, max_columns))

    print("\n  MISSING VALUES (columns with count > 0)")
    if report.missing_counts:
        for column, count in sorted(
            report.missing_counts.items(), key=lambda kv: -kv[1]
        )[: max_columns if max_columns > 0 else None]:
            share = count / report.n_rows if report.n_rows else 0.0
            print(f"    {column}: {count:,} ({share:.4%})")
    else:
        print("    (no missing values)")

    print("\n  INFINITY VALUES (numeric columns with count > 0)")
    if report.infinity_counts:
        for column, count in sorted(
            report.infinity_counts.items(), key=lambda kv: -kv[1]
        )[: max_columns if max_columns > 0 else None]:
            print(f"    {column}: {count:,}")
    else:
        print("    (no +/-inf values)")

    print("\n  CLASS DISTRIBUTION")
    if report.selected_label_column and report.class_distribution:
        note = " (over the truncated slice only)" if report.class_distribution_truncated else ""
        print(f"    detected label column: {report.selected_label_column}{note}")
        total = sum(report.class_distribution.values()) or 1
        for value, count in report.class_distribution.items():
            print(f"    {str(value):<40} {count:>12,}  ({count / total:.4%})")
    elif report.selected_label_column:
        print(f"    detected label column: {report.selected_label_column} (no values counted)")
    else:
        print("    no label column detected - class distribution unavailable")
    print()


def save_json_report(cfg: Config, reports: List[FileReport]) -> Path:
    """Write the machine-readable report under ``results/inspection``."""
    out_dir = cfg.paths.results_dir / "inspection"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"dataset_inspection_{timestamp}.json"
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": cfg.to_dict(),
        "n_files": len(reports),
        "files": [report.to_dict() for report in reports],
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        cfg = _resolve_config(args)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    cfg.ensure_directories()
    logger = get_logger("inspect_dataset", logs_dir=cfg.paths.logs_dir)

    print("=" * 78)
    print("DATASET INSPECTION (schema-agnostic)")
    print("=" * 78)
    print(cfg.summary())
    print()

    try:
        reports = inspect_directory(
            cfg.paths.raw_data_dir,
            patterns=cfg.inspection.file_patterns,
            dataset_file=cfg.paths.dataset_file,
            chunk_size=cfg.inspection.chunk_size,
            row_limit=cfg.debug.row_limit_or_none,
            max_label_cardinality=cfg.inspection.max_label_cardinality,
            timestamp_sample_size=cfg.inspection.timestamp_sample_size,
            logger=logger,
        )
    except DatasetNotFoundError:
        print(DATASET_MISSING_MESSAGE)
        print(f"Configured directory: {cfg.paths.raw_data_dir}")
        print(f"Patterns searched   : {', '.join(cfg.inspection.file_patterns)} (recursive)")
        logger.warning("%s (looked in %s)", DATASET_MISSING_MESSAGE, cfg.paths.raw_data_dir)
        return EXIT_NO_DATASET
    except Exception as exc:  # pragma: no cover - unexpected failure
        print(f"Inspection failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        logger.exception("Inspection failed")
        return EXIT_ERROR

    print(f"Found {len(reports)} candidate file(s) in {cfg.paths.raw_data_dir}\n")
    for report in reports:
        print_report(report, args.max_columns)

    if not args.no_save:
        out_path = save_json_report(cfg, reports)
        print(f"JSON report written to: {out_path}")
        logger.info("JSON report written to %s", out_path)

    print(
        "\nNEXT STEP: confirm the timestamp column, the label column and the feature\n"
        "columns above before any preprocessing code is written."
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
