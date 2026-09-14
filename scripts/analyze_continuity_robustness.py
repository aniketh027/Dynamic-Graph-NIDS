"""Window-size robustness sweep: T=3 / T=5 / T=10 across the whole corpus.

Second analysis-only experiment. Answers two questions the first experiment left
open: whether the 02-20 T=5 continuity profile is representative of the corpus,
and whether T=5 is a sensible window size at all.

Reads every CSV read-only, groups each file's usable rows into non-overlapping
windows at each T in the project's deterministic capture-order proxy, and
measures the same metric set as the T=5 experiment - same ordering rule, same
seed, same definitions generalised from 5 to T.

This script changes no pipeline behaviour. It does not build the project's real
windows, does not touch the loader, preprocessing or architecture, does not write
under ``data/raw``, and produces no model input.

Usage
-----
    python scripts/analyze_continuity_robustness.py                    # all files
    python scripts/analyze_continuity_robustness.py --rows 200000      # quick check
    python scripts/analyze_continuity_robustness.py --files 02-20-2018.csv
    python scripts/analyze_continuity_robustness.py --window-sizes 3 5 10
    python scripts/analyze_continuity_robustness.py --digest-only      # no artifacts
    python scripts/analyze_continuity_robustness.py --verify-unchanged

Exit codes
----------
    0  the sweep ran and every artifact was written
    1  unexpected error
    2  no CSV files were found to analyze
    3  a file cannot support this analysis, or a source file changed on disk, or
       the generalised code failed to reproduce the published T=5 measurement
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis import robustness_artifacts  # noqa: E402
from src.analysis.continuity_robustness import (  # noqa: E402
    IDENTITY_PROXY,
    IDENTITY_TRUE,
    WINDOW_SIZES,
    AnalysisError,
    run_robustness,
)
from src.config import Config, ConfigError  # noqa: E402
from src.utils.logging_utils import get_logger, log_section  # noqa: E402

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_NO_FILE = 2
EXIT_BAD_DATA = 3

#: Results sub-directory for this experiment.
OUTPUT_SUBDIR = Path("temporal_analysis") / "robustness"

#: The frozen first-experiment summary the generalised code is checked against.
PUBLISHED_SUMMARY = Path("temporal_analysis") / "continuity_summary.json"

FINGERPRINT_CHUNK = 8 << 20


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Window-size robustness sweep (analysis only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="Path to a YAML config file.")
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="CSV file names to sweep (default: every CSV in the corpus directory).",
    )
    parser.add_argument(
        "--window-sizes",
        nargs="+",
        type=int,
        default=list(WINDOW_SIZES),
        help="Window sizes to compare.",
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
        help="Analyze only the first N usable rows per file (debug runs).",
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
        help=f"Artifact directory (default: <results_dir>/{OUTPUT_SUBDIR.as_posix()}).",
    )
    parser.add_argument(
        "--digest-only",
        action="store_true",
        help="Print the measurement digest and exit without writing artifacts.",
    )
    parser.add_argument(
        "--verify-unchanged",
        action="store_true",
        help="Fingerprint every source CSV before and after and prove none was "
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


def discover(cfg: Config, names: Optional[List[str]]) -> List[Path]:
    """Resolve the CSVs to sweep. Sorted by name so the run order is deterministic."""
    corpus = cfg.paths.raw_data_dir / cfg.profiling.csv_subdir
    if names:
        return sorted(
            (
                Path(name)
                if Path(name).is_absolute() or Path(name).exists()
                else corpus / name
            )
            for name in names
        )
    # Never rely on the filesystem's own listing order.
    return sorted(corpus.glob("*.csv"), key=lambda p: p.name)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        cfg = Config.load(args.config).validate()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    cfg.ensure_directories()
    logger = get_logger("analyze_continuity_robustness", logs_dir=cfg.paths.logs_dir)

    seed = args.seed if args.seed is not None else cfg.seed
    window_sizes = sorted(set(args.window_sizes))
    if any(t < 2 for t in window_sizes):
        print("Window sizes must all be >= 2.", file=sys.stderr)
        return EXIT_ERROR
    paths = discover(cfg, args.files)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else cfg.paths.results_dir / OUTPUT_SUBDIR
    )

    print("=" * 78)
    print("WINDOW-SIZE ROBUSTNESS SWEEP (measure only)")
    print("=" * 78)
    print(f"window sizes : {window_sizes}")
    print(f"seed         : {seed}")
    print(f"row limit    : {args.rows or '(none - whole files)'}")
    print(f"baseline     : {'off' if args.no_baseline else 'random grouping, seeded'}")
    print(f"files        : {len(paths)}")
    print()

    missing = [p for p in paths if not p.is_file()]
    if missing or not paths:
        for path in missing:
            print(f"CSV file not found: {path}", file=sys.stderr)
        if not paths:
            print("No CSV files found to analyze.", file=sys.stderr)
        logger.error("No usable CSV inputs (%d missing)", len(missing))
        return EXIT_NO_FILE

    before: Dict[str, Dict[str, object]] = {}
    if args.verify_unchanged:
        print("Fingerprinting every source file (sha256) before the sweep ...")
        for path in paths:
            before[path.name] = fingerprint(path)
            print(f"  {path.name:<18} {before[path.name]['sha256'][:16]}...  "
                  f"{before[path.name]['size_bytes']:>14,} bytes")
        print()

    published: Optional[Dict[str, Any]] = None
    summary_path = cfg.paths.results_dir / PUBLISHED_SUMMARY
    if summary_path.is_file():
        published = json.loads(summary_path.read_text(encoding="utf-8"))
        print(f"Equivalence reference: {summary_path.name} (frozen T=5 measurement)\n")
    else:
        print(f"NOTE: {summary_path} not found - the equivalence check against the "
              "published T=5 result will be skipped.\n")

    log_section(logger, f"Robustness sweep T={window_sizes} over {len(paths)} file(s)")
    try:
        result = run_robustness(
            paths,
            seed=seed,
            window_sizes=window_sizes,
            chunk_size=args.chunk_size,
            row_limit=args.rows,
            with_random_baseline=not args.no_baseline,
            published_summary=published,
            progress=print,
        )
    except AnalysisError as exc:
        print(f"\nANALYSIS FAILED: {exc}", file=sys.stderr)
        logger.error("Analysis failed: %s", exc)
        return EXIT_BAD_DATA
    except Exception as exc:  # pragma: no cover - unexpected failure
        print(f"\nAnalysis failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        logger.exception("Analysis failed")
        return EXIT_ERROR

    print()
    print("-" * 78)
    print("IDENTIFIER INVENTORY (measured from each file's own header)")
    print("-" * 78)
    for item in result.inventory:
        state = (
            "full 5-tuple"
            if item["has_full_five_tuple"]
            else f"missing {item['identifier_columns_missing']}"
        )
        print(f"  {item['source_file']:<18} {item['n_columns']:>3} cols  "
              f"{item['schema_variant']:<14} {state}")
    n_true = sum(1 for i in result.inventory if i["has_true_endpoint_identity"])
    print(f"\n  {n_true} of {len(result.inventory)} file(s) carry (Src IP, Dst IP); "
          f"all {len(result.inventory)} carry the (Dst Port, Protocol) proxy.")
    print()

    if result.equivalence is not None:
        print("-" * 78)
        print("EQUIVALENCE WITH THE PUBLISHED T=5 MEASUREMENT")
        print("-" * 78)
        for item in result.equivalence["comparisons"]:
            flag = "ok " if item["match"] else "FAIL"
            print(f"  [{flag}] {item['metric']:<52} "
                  f"{item['robustness_value']!s:>18} vs {item['published_value']!s:>18}")
        print(f"  all match: {result.equivalence['all_match']}")
        if not result.equivalence["all_match"]:
            print("\nThe generalised sweep does not reproduce the frozen T=5 result. "
                  "The cross-T comparison would be invalid, so no artifacts were "
                  "written.", file=sys.stderr)
            logger.error("Equivalence check against the published T=5 summary failed")
            return EXIT_BAD_DATA
        print()

    print("-" * 78)
    print("SWEEP (consecutive arm)")
    print("-" * 78)
    header = (f"  {'T':>3} {'file':<18} {'identity':<18} {'windows':>12} "
              f"{'A%':>9} {'B%':>9} {'link%':>9} {'C':>8} {'F%':>9} {'G%':>8}")
    print(header)
    for cell in sorted(
        result.cells, key=lambda c: (c.identity, c.window_size, c.source_file)
    ):
        arm = cell.consecutive.to_dict(cell.window_size)
        print(f"  {cell.window_size:>3} {cell.source_file:<18} {cell.identity:<18} "
              f"{cell.n_windows:>12,} "
              f"{arm['A_pct_windows_all_same_key']:>9.4f} "
              f"{arm['B_pct_windows_any_adjacent_same_key']:>9.4f} "
              f"{arm['pct_adjacent_links_sharing_key']:>9.4f} "
              f"{arm['C_mean_unique_keys_per_window']:>8.4f} "
              f"{arm['F_pct_windows_label_pure']:>9.4f} "
              f"{arm['G_pct_windows_final_label_ne_majority']:>8.4f}")
    print()

    reference = next(
        (c.source_file for c in result.cells if c.identity == IDENTITY_TRUE), None
    )
    position = robustness_artifacts.representativeness(result)
    if position.get("available"):
        print("-" * 78)
        print(f"POSITION OF {position['reference_file']} IN THE {IDENTITY_PROXY} "
              f"DISTRIBUTION (T=5)")
        print("-" * 78)
        for metric, item in position["metrics"].items():
            z = item["z_score_vs_other_files"]
            print(f"  {metric:<42} {item['reference_value']:>10.4f}  "
                  f"z={'n/a' if z is None else f'{z:+6.2f}'}  "
                  f"rank {item['rank_of_reference_high_to_low']}/{item['n_ranked']}  "
                  f"{item['position']}")
        print()

    digest = robustness_artifacts.measurement_digest(result)
    print(f"measurement digest (sha256) : {digest}")
    print("  Two runs over unchanged files with the same seed must print the same "
          "digest.")
    print()

    if result.notes:
        print(f"NOTES ({len(result.notes)})")
        for note in result.notes:
            print(f"  ! {note}")
        print()

    after: Dict[str, Dict[str, object]] = {}
    if args.verify_unchanged and before:
        print("Re-fingerprinting every source file after the sweep ...")
        changed = []
        for path in paths:
            after[path.name] = fingerprint(path)
            if after[path.name] != before[path.name]:
                changed.append(path.name)
        if changed:
            print(f"  SOURCE FILES CHANGED: {changed}", file=sys.stderr)
            logger.error("Source files changed during the sweep: %s", changed)
            return EXIT_BAD_DATA
        print(f"  verified: all {len(paths)} source CSV(s) byte-identical "
              "(size, mtime and sha256 unchanged)\n")

    if args.digest_only:
        return EXIT_OK

    written = robustness_artifacts.write_artifacts(result, output_dir)
    print(f"Artifacts written to {output_dir}:")
    for name in sorted(written):
        print(f"  {name:<34} {written[name].stat().st_size:>12,} bytes")
    logger.info("Artifacts written to %s", output_dir)
    if args.verify_unchanged and before:
        (output_dir / "source_file_fingerprints.json").write_text(
            json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8"
        )
        print(f"  {'source_file_fingerprints.json':<34} (proof of no modification)")
    print()

    print(
        "NOTE: analysis only. The preprocessing pipeline, loader, windowing code and\n"
        "architecture are unchanged, no file under data/raw was touched, no model\n"
        "input was produced, and no representation is recommended here."
    )
    if reference is None:
        print("\nNOTE: no file in this run carried (Src IP, Dst IP), so every number\n"
              "above is the weaker proxy identity and overstates endpoint continuity.")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
