"""Measure T=5 flow-to-flow continuity in the raw corpus (analysis only).

Reads one CSV read-only, groups its usable rows into non-overlapping windows of
5 in the project's deterministic capture-order proxy, and measures how much
communication context those five flows actually share. Writes its artifacts under
``results/temporal_analysis``.

This script changes no pipeline behaviour. It does not build the project's real
windows, does not touch the loader or preprocessing, does not write under
``data/raw``, and produces no model input.

Usage
-----
    python scripts/analyze_window_continuity.py                  # full file
    python scripts/analyze_window_continuity.py --rows 500000    # quick check
    python scripts/analyze_window_continuity.py --seed 7
    python scripts/analyze_window_continuity.py --digest-only    # no artifacts
    python scripts/analyze_window_continuity.py --verify-unchanged

Exit codes
----------
    0  the analysis ran and every artifact was written
    1  unexpected error
    2  the requested CSV file does not exist
    3  the file cannot support this analysis (missing identifier columns, or its
       row accounting could not be trusted)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis import continuity_artifacts  # noqa: E402
from src.analysis.window_continuity import (  # noqa: E402
    WINDOW_SIZE,
    AnalysisError,
    analyze_file,
)
from src.config import Config, ConfigError  # noqa: E402
from src.utils.logging_utils import get_logger, log_section  # noqa: E402

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_NO_FILE = 2
EXIT_BAD_DATA = 3

#: The 84-column variant: the only file carrying endpoint identifiers.
DEFAULT_FILE = "02-20-2018.csv"

#: Results sub-directory for this analysis.
OUTPUT_SUBDIR = "temporal_analysis"

FINGERPRINT_CHUNK = 8 << 20


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure T=5 window continuity (analysis only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="Path to a YAML config file.")
    parser.add_argument(
        "--file",
        default=DEFAULT_FILE,
        help="CSV file name under the corpus directory, or a full path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for the random-grouping baseline (default: the project seed).",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=None,
        help="Analyze only the first N usable rows (debug runs).",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=500_000, help="Rows per read chunk."
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip the random-grouping comparison baseline.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"Artifact directory (default: <results_dir>/{OUTPUT_SUBDIR}).",
    )
    parser.add_argument(
        "--digest-only",
        action="store_true",
        help="Print the measurement digest and exit without writing artifacts.",
    )
    parser.add_argument(
        "--verify-unchanged",
        action="store_true",
        help="Fingerprint the source CSV before and after and prove it was not "
        "modified.",
    )
    return parser


def fingerprint(path: Path) -> Dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(FINGERPRINT_CHUNK):
            digest.update(chunk)
    stat = path.stat()
    return {
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        cfg = Config.load(args.config).validate()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    cfg.ensure_directories()
    logger = get_logger("analyze_window_continuity", logs_dir=cfg.paths.logs_dir)

    seed = args.seed if args.seed is not None else cfg.seed
    candidate = Path(args.file)
    path = (
        candidate
        if candidate.is_absolute() or candidate.exists()
        else cfg.paths.raw_data_dir / cfg.profiling.csv_subdir / args.file
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else cfg.paths.results_dir / OUTPUT_SUBDIR
    )

    print("=" * 78)
    print(f"T={WINDOW_SIZE} WINDOW CONTINUITY - EMPIRICAL ANALYSIS (measure only)")
    print("=" * 78)
    print(f"dataset      : {path}")
    print(f"seed         : {seed}")
    print(f"row limit    : {args.rows or '(none - whole file)'}")
    print(f"baseline     : {'off' if args.no_baseline else 'random grouping, seeded'}")
    print()

    if not path.is_file():
        print(f"CSV file not found: {path}", file=sys.stderr)
        logger.error("CSV file not found: %s", path)
        return EXIT_NO_FILE

    before = None
    if args.verify_unchanged:
        print("Fingerprinting the source file (sha256) before analysis ...")
        before = fingerprint(path)
        print(f"  {before['sha256'][:16]}...  {before['size_bytes']:,} bytes\n")

    log_section(logger, f"Analyzing T={WINDOW_SIZE} continuity in {path.name}")
    try:
        result = analyze_file(
            path,
            seed=seed,
            chunk_size=args.chunk_size,
            row_limit=args.rows,
            with_random_baseline=not args.no_baseline,
        )
    except AnalysisError as exc:
        print(f"\nANALYSIS FAILED: {exc}", file=sys.stderr)
        logger.error("Analysis failed: %s", exc)
        return EXIT_BAD_DATA
    except Exception as exc:  # pragma: no cover - unexpected failure
        print(f"\nAnalysis failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        logger.exception("Analysis failed")
        return EXIT_ERROR

    payload = result.to_dict()
    endpoint = payload["consecutive"]["endpoint_continuity"]
    label = payload["consecutive"]["label_continuity"]
    time_stats = payload["consecutive"]["timestamp_diagnostics"]

    print("-" * 78)
    print("RESULT")
    print("-" * 78)
    print(f"rows analyzed            : {result.n_rows_analyzed:,}")
    print(f"windows (T={WINDOW_SIZE})            : {result.n_windows:,}")
    print(f"trailing rows discarded  : {result.n_trailing_rows_discarded}")
    print()
    print("ENDPOINT CONTINUITY (consecutive capture order)")
    print(f"  A  all 5 same Src/Dst pair     : "
          f"{endpoint['A_pct_windows_all_five_same_src_dst_pair']:.4f}%")
    print(f"  B  any adjacent same pair      : "
          f"{endpoint['B_pct_windows_any_adjacent_same_src_dst_pair']:.4f}%")
    print(f"  C  mean unique pairs / window  : "
          f"{endpoint['C_mean_unique_pairs_per_window']:.4f}")
    print(f"  D  median unique pairs         : "
          f"{endpoint['D_median_unique_pairs_per_window']:.1f}")
    print(f"     fully heterogeneous (5)     : "
          f"{endpoint['pct_windows_fully_heterogeneous_five_distinct_pairs']:.4f}%")
    print("  E  windows by unique pair count :")
    for k in range(1, WINDOW_SIZE + 1):
        key = str(k)
        print(f"       {k} pair(s) : "
              f"{endpoint['E_pct_windows_by_unique_pair_count'][key]:>9.4f}%  "
              f"({endpoint['n_windows_by_unique_pair_count'][key]:,} windows)")
    print()
    print("LABEL CONTINUITY")
    print(f"  F  label-pure windows          : {label['F_pct_windows_label_pure']:.4f}%")
    print(f"     mixed-label windows         : {label['pct_windows_mixed_label']:.4f}%")
    print(f"  G  5th label != majority       : "
          f"{label['G_pct_windows_fifth_label_ne_majority']:.4f}%")
    print()
    print("TEMPORAL DIAGNOSTICS (Timestamp is diagnostic only)")
    print(f"  median span (s)                : {time_stats['median_span_seconds']:,.1f}")
    print(f"  p95 span (s)                   : {time_stats['p95_span_seconds']:,.1f}")
    print(f"  max span (s)                   : {time_stats['max_span_seconds']:,.1f}")
    print(f"  entirely within one second     : "
          f"{time_stats['pct_windows_entirely_within_one_second']:.4f}%")
    print()

    if result.random_baseline is not None:
        print("CONSECUTIVE vs RANDOM GROUPING (baseline is a measuring stick only)")
        for key, values in result.comparison().items():
            print(f"  {key:<52} {values['consecutive']:>12.4f} "
                  f"{values['random']:>12.4f} {values['difference']:>+12.4f}")
        print()

    digest = continuity_artifacts.measurement_digest(result)
    print(f"measurement digest (sha256) : {digest}")
    print("  Two runs over an unchanged file with the same seed must print the same "
          "digest.")
    print()

    if result.notes:
        print(f"NOTES ({len(result.notes)})")
        for note in result.notes:
            print(f"  ! {note}")
        print()

    if args.verify_unchanged and before is not None:
        print("Re-fingerprinting the source file after analysis ...")
        after = fingerprint(path)
        if after != before:
            print(f"  SOURCE FILE CHANGED: {path}", file=sys.stderr)
            logger.error("Source file changed during analysis: %s", path)
            return EXIT_BAD_DATA
        print("  verified: source CSV byte-identical (size, mtime and sha256 "
              "unchanged)\n")

    if args.digest_only:
        return EXIT_OK

    written = continuity_artifacts.write_artifacts(result, output_dir)
    print(f"Artifacts written to {output_dir}:")
    for name in sorted(written):
        print(f"  {name:<30} {written[name].stat().st_size:>12,} bytes")
    logger.info("Artifacts written to %s", output_dir)
    if args.verify_unchanged and before is not None:
        (output_dir / "source_file_fingerprint.json").write_text(
            json.dumps({"before": before, "after": fingerprint(path)}, indent=2),
            encoding="utf-8",
        )
        print(f"  {'source_file_fingerprint.json':<30} (proof of no modification)")
    print()

    print(
        "NOTE: analysis only. The preprocessing pipeline, loader and windowing code\n"
        "are unchanged, no file under data/raw was touched, no model input was\n"
        "produced, and no architecture decision is made here."
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
