"""Profile the whole raw CIC-IDS2018 CSV corpus (Stage B) and write the report.

Reads every CSV under the configured corpus directory read-only, measures it,
and writes the measurement artifacts under ``results/data_profile``. Nothing is
cleaned, replaced, deduplicated or encoded; nothing is written under
``data/raw``; no train/validation/test arrays are produced.

Usage
-----
    python scripts/profile_dataset.py                     # full corpus
    python scripts/profile_dataset.py --debug             # cap rows per file
    python scripts/profile_dataset.py --files 03-01-2018.csv 02-28-2018.csv
    python scripts/profile_dataset.py --no-duplicates     # skip the hash passes
    python scripts/profile_dataset.py --no-global-duplicates
    python scripts/profile_dataset.py --no-parquet
    python scripts/profile_dataset.py --verify-unchanged  # hash raw files after
    python scripts/profile_dataset.py --digest-only       # print digest, no write

Exit codes
----------
    0  the corpus was profiled and every cross-check held
    1  unexpected error
    2  the corpus directory does not exist or holds no CSV files
    3  a file failed a schema / row-count / consistency check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Allow running the script directly (``python scripts/profile_dataset.py``).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config, ConfigError  # noqa: E402
from src.preprocessing import profile_artifacts, schema  # noqa: E402
from src.preprocessing.loader import LoaderError  # noqa: E402
from src.preprocessing.profiling import (  # noqa: E402
    CorpusNotFoundError,
    ProfilingError,
    discover_csv_files,
    profile_from_config,
)
from src.utils.logging_utils import get_logger, log_section  # noqa: E402

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_NO_FILE = 2
EXIT_BAD_DATA = 3

#: Read size for the raw-file fingerprint (``--verify-unchanged``).
FINGERPRINT_CHUNK = 8 << 20


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile every raw CIC-IDS2018 CSV file (measure only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="Path to a YAML config file.")
    parser.add_argument(
        "--csv-dir",
        default=None,
        help="Corpus directory (default: <raw_data_dir>/<profiling.csv_subdir>).",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Profile only these file names (still ordered by capture date).",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=None, help="Rows per chunk in the data pass."
    )
    debug_group = parser.add_mutually_exclusive_group()
    debug_group.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        default=None,
        help="Enable debug mode (cap rows read per file).",
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
        help="Skip duplicate measurement entirely (per file and corpus-wide).",
    )
    parser.add_argument(
        "--no-global-duplicates",
        action="store_true",
        help="Measure duplicates per file only (lower peak memory).",
    )
    parser.add_argument(
        "--no-parquet", action="store_true", help="Skip the Parquet metadata comparison."
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Artifact directory (default: <results_dir>/<profiling.output_subdir>).",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Do not write any artifact."
    )
    parser.add_argument(
        "--digest-only",
        action="store_true",
        help="Print the measurement digest and exit without writing artifacts.",
    )
    parser.add_argument(
        "--verify-unchanged",
        action="store_true",
        help="Fingerprint every raw CSV before and after profiling and prove it "
        "was not modified.",
    )
    return parser


def _resolve_config(args: argparse.Namespace) -> Config:
    cfg = Config.load(args.config)
    if args.chunk_size is not None:
        cfg.profiling.chunk_size = args.chunk_size
    if args.debug is not None:
        cfg.debug.enabled = args.debug
    if args.debug_rows is not None:
        cfg.debug.row_limit = args.debug_rows
    if args.no_duplicates:
        cfg.profiling.compute_duplicates = False
    if args.no_global_duplicates:
        cfg.profiling.compute_global_duplicates = False
    if args.no_parquet:
        cfg.profiling.compare_parquet = False
    return cfg.validate()


def fingerprint_raw_files(paths: List[Path]) -> Dict[str, Dict[str, object]]:
    """Size, mtime and sha256 of each raw file, for the not-modified proof."""
    prints: Dict[str, Dict[str, object]] = {}
    for path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(FINGERPRINT_CHUNK):
                digest.update(chunk)
        stat = path.stat()
        prints[path.name] = {
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": digest.hexdigest(),
        }
    return prints


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        cfg = _resolve_config(args)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    cfg.ensure_directories()
    logger = get_logger("profile_dataset", logs_dir=cfg.paths.logs_dir)

    csv_dir = (
        Path(args.csv_dir)
        if args.csv_dir
        else cfg.paths.raw_data_dir / cfg.profiling.csv_subdir
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else cfg.paths.results_dir / cfg.profiling.output_subdir
    )

    print("=" * 78)
    print("STAGE B - MULTI-DAY INTAKE AND GLOBAL DATASET PROFILING")
    print("=" * 78)
    print(cfg.summary())
    print()
    print(schema.describe())
    print()

    try:
        discovered = discover_csv_files(csv_dir, cfg.profiling.csv_patterns)
    except CorpusNotFoundError as exc:
        print(f"{exc}", file=sys.stderr)
        logger.error("Corpus discovery failed: %s", exc)
        return EXIT_NO_FILE

    if args.files:
        wanted = set(args.files)
        selected = [item for item in discovered if item.source_file in wanted]
        missing = wanted - {item.source_file for item in selected}
        if missing:
            print(f"Requested file(s) not found: {sorted(missing)}", file=sys.stderr)
            return EXIT_NO_FILE
        discovered = selected

    print(f"Discovered {len(discovered)} CSV file(s) under {csv_dir}")
    print("Deterministic capture-order sequence:")
    for position, item in enumerate(discovered):
        # str() first: date.__format__ passes a non-empty spec to strftime, so
        # a bare "<12" would print itself instead of padding the date.
        print(
            f"  {position:>2}. {item.source_file:<18} "
            f"{str(item.capture_date or '?'):<12} "
            f"{item.size_bytes / (1 << 20):>9,.0f} MB"
        )
    print()

    before = None
    after = None
    if args.verify_unchanged:
        print("Fingerprinting raw files (sha256) before profiling ...")
        before = fingerprint_raw_files([item.path for item in discovered])
        print(f"  {len(before)} file(s) fingerprinted\n")

    log_section(logger, f"Profiling {len(discovered)} CSV file(s)")
    try:
        profile = profile_from_config(
            cfg,
            logger=logger,
            csv_dir=csv_dir,
            csv_patterns=tuple(cfg.profiling.csv_patterns),
            only_files=args.files,
        )
    except (ProfilingError, LoaderError) as exc:
        print(f"\nDATA CHECK FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        logger.error("Profiling failed: %s: %s", type(exc).__name__, exc)
        return EXIT_BAD_DATA
    except Exception as exc:  # pragma: no cover - unexpected failure
        print(f"\nProfiling failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        logger.exception("Profiling failed")
        return EXIT_ERROR

    print("-" * 78)
    print("RESULT")
    print("-" * 78)
    print(profile.summary())
    print()

    digest = profile_artifacts.measurement_digest(profile)
    print(f"measurement digest (sha256) : {digest}")
    print(
        "  Two runs over unchanged files must print the same digest; it covers every "
        "measurement and excludes wall-clock timings."
    )
    print()

    coverage = profile.split_coverage
    print("CLASS COVERAGE UNDER A NAIVE CHRONOLOGICAL 70/10/20 SPLIT (analysis only)")
    print(f"  train [0, {coverage.train_end:,})  val [{coverage.train_end:,}, "
          f"{coverage.val_end:,})  test [{coverage.val_end:,}, {coverage.n_rows:,})")
    print(f"  absent from train      : {coverage.absent_from_train or '(none)'}")
    print(f"  absent from validation : {coverage.absent_from_val or '(none)'}")
    print(f"  absent from test       : {coverage.absent_from_test or '(none)'}")
    print(f"  under {coverage.starved_threshold} train rows : "
          f"{coverage.starved_in_train or '(none)'}")
    print()

    status = profile_artifacts.heartbleed_status(profile)
    print(f"HEARTBLEED: {status['statement']}")
    print()

    if profile.warnings:
        print(f"WARNINGS ({len(profile.warnings)})")
        for warning in profile.warnings:
            print(f"  ! {warning}")
        print()

    if args.verify_unchanged and before is not None:
        print("Re-fingerprinting raw files after profiling ...")
        after = fingerprint_raw_files([item.path for item in discovered])
        changed = [
            name for name, values in before.items() if after.get(name) != values
        ]
        if changed:
            print(f"  RAW FILES CHANGED: {changed}", file=sys.stderr)
            logger.error("Raw files changed during profiling: %s", changed)
            return EXIT_BAD_DATA
        print(
            f"  verified: all {len(before)} raw CSV file(s) byte-identical "
            "(size, mtime and sha256 unchanged)\n"
        )

    if args.digest_only:
        return EXIT_OK

    if not args.no_save:
        written = profile_artifacts.write_artifacts(
            profile,
            output_dir,
            config=cfg.to_dict(),
            config_summary=cfg.summary(),
            chunk_size=cfg.profiling.chunk_size,
        )
        print(f"Artifacts written to {output_dir}:")
        for name in sorted(written):
            size = written[name].stat().st_size
            print(f"  {name:<28} {size:>10,} bytes")
        logger.info("Artifacts written to %s", output_dir)
        if after is not None and before is not None:
            (output_dir / "raw_file_fingerprints.json").write_text(
                json.dumps({"before": before, "after": after}, indent=2),
                encoding="utf-8",
            )
            print(f"  {'raw_file_fingerprints.json':<28} (proof of no modification)")
        print()

    print(
        "NOTE: this stage measured only. No non-finite value was replaced, no negative\n"
        "value clipped, no duplicate removed, no label encoded, and no file under\n"
        "data/raw was created, modified, renamed or deleted."
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
