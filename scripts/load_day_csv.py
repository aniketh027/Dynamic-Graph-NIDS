"""Load and clean one raw CIC-IDS2018 day CSV, then report what was measured.

Reads the file read-only, removes only the embedded header rows, guarantees a
finite numeric feature frame, and prints/saves the measurement record. No
train/val/test artefacts, graphs or checkpoints are produced.

Usage
-----
    python scripts/load_day_csv.py                     # cfg.loader.primary_csv
    python scripts/load_day_csv.py --file csv/03-01-2018.csv
    python scripts/load_day_csv.py --debug             # cap rows read
    python scripts/load_day_csv.py --no-duplicates     # skip duplicate scan
    python scripts/load_day_csv.py --no-save           # do not write the JSON

Exit codes
----------
    0  the file loaded and every guarantee held
    1  unexpected error
    2  the requested file does not exist
    3  the file failed a schema / row-count / data-quality check
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

# Allow running the script directly (``python scripts/load_day_csv.py``).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config, ConfigError  # noqa: E402
from src.preprocessing import schema  # noqa: E402
from src.preprocessing.loader import (  # noqa: E402
    LoaderError,
    load_from_config,
    resolve_primary_csv,
)
from src.utils.logging_utils import get_logger, log_section  # noqa: E402

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_NO_FILE = 2
EXIT_BAD_DATA = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load and clean one raw CIC-IDS2018 day CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="Path to a YAML config file.")
    parser.add_argument(
        "--file",
        default=None,
        help="Day CSV to load (path relative to the raw data dir, or absolute).",
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
        "--no-debug", dest="debug", action="store_false", help="Read every row."
    )
    parser.add_argument(
        "--debug-rows", type=int, default=None, help="Override the debug row limit."
    )
    parser.add_argument(
        "--no-duplicates",
        action="store_true",
        help="Skip the duplicate measurement pass (faster).",
    )
    parser.add_argument(
        "--expect-raw-rows",
        type=int,
        default=None,
        help="Fail unless the file has exactly this many raw data rows.",
    )
    parser.add_argument(
        "--expect-header-rows",
        type=int,
        default=None,
        help="Fail unless exactly this many embedded header rows are found.",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Do not write the metadata JSON."
    )
    return parser


def _resolve_config(args: argparse.Namespace) -> Config:
    cfg = Config.load(args.config)
    if args.file:
        cfg.loader.primary_csv = args.file
    if args.debug is not None:
        cfg.debug.enabled = args.debug
    if args.debug_rows is not None:
        cfg.debug.row_limit = args.debug_rows
    if args.no_duplicates:
        cfg.loader.compute_duplicate_stats = False
    return cfg.validate()


def save_metadata(cfg: Config, payload: dict, stem: str) -> Path:
    """Write the small measurement record under ``results/loader``."""
    out_dir = cfg.paths.results_dir / "loader"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"load_{stem}.json"
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
    logger = get_logger("load_day_csv", logs_dir=cfg.paths.logs_dir)

    target = resolve_primary_csv(cfg)
    print("=" * 78)
    print("RAW DAY-CSV LOADER")
    print("=" * 78)
    print(cfg.summary())
    print()
    print(schema.describe())
    print()

    if not target.is_file():
        message = f"Day CSV not found: {target}"
        print(message, file=sys.stderr)
        print(f"Configured raw data dir: {cfg.paths.raw_data_dir}", file=sys.stderr)
        logger.error(message)
        return EXIT_NO_FILE

    log_section(logger, f"Loading {target.name}")
    try:
        day = load_from_config(
            cfg,
            path=target,
            logger=logger,
            expected_raw_rows=args.expect_raw_rows,
            expected_embedded_header_rows=args.expect_header_rows,
        )
    except LoaderError as exc:
        print(f"\nDATA CHECK FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        logger.error("Load failed: %s: %s", type(exc).__name__, exc)
        return EXIT_BAD_DATA
    except Exception as exc:  # pragma: no cover - unexpected failure
        print(f"\nLoad failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        logger.exception("Load failed")
        return EXIT_ERROR

    print("-" * 78)
    print("RESULT")
    print("-" * 78)
    print(day.summary())
    print()

    clean = day.stats.cleaning
    print("CLEANING DETAIL")
    print(f"  NaN counts (raw)        : {clean.nan_counts_before or '(none)'}")
    print(f"  +inf counts (raw)       : {clean.posinf_counts_before or '(none)'}")
    print(f"  -inf counts (raw)       : {clean.neginf_counts_before or '(none)'}")
    print(f"  NaN replaced with       : {clean.nan_replacement}")
    print(f"  infinity policy         : {clean.infinity_policy}")
    print(f"  infinity substitutions  : {clean.infinity_substitutions or '(none)'}")
    print(f"  negatives kept as-is    : {clean.negative_value_counts or '(none)'}")
    print(f"  of which -1 sentinels   : {clean.sentinel_negative_counts or '(none)'}")
    if clean.non_sentinel_negative_columns:
        print(
            "  ! negatives outside the known sentinel columns: "
            f"{clean.non_sentinel_negative_columns}"
        )
    if clean.embedded_header_original_indices:
        shown = clean.embedded_header_original_indices[:10]
        more = len(clean.embedded_header_original_indices) - len(shown)
        suffix = f" ... and {more} more" if more > 0 else ""
        print(f"  removed header rows at Original_Index: {shown}{suffix}")
    print()

    if not args.no_save:
        payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": cfg.to_dict(),
            **day.to_metadata_dict(),
        }
        out_path = save_metadata(cfg, payload, target.stem)
        print(f"Metadata JSON written to: {out_path}")
        logger.info("Metadata JSON written to %s", out_path)

    print(
        "\nNOTE: nothing was written to the raw dataset directory, and no duplicate\n"
        "rows were removed - the duplicate counts above are measurements only."
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
