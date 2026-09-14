"""Stage C preprocessing pipeline — command-line entry point.

Runs the full cleaning → windowing → split → scaling → artefact pipeline and
writes all outputs under ``<processed_data_dir>/<output_subdir>/``.

Usage
-----
    python scripts/run_preprocessing.py                   # full corpus
    python scripts/run_preprocessing.py --debug           # 100k rows/file
    python scripts/run_preprocessing.py --rows 50000      # custom cap
    python scripts/run_preprocessing.py --config configs/custom.yaml

Exit codes
----------
    0  pipeline completed and all artefacts were written
    1  unexpected error
    2  a required corpus file was not found
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config, ConfigError  # noqa: E402
from src.preprocessing.pipeline import run_pipeline  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_NO_FILE = 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage C: final preprocessing, T=5 windowing, and split construction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=None, help="Path to a YAML config file.")
    p.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (reads cfg.debug.row_limit rows per file).",
    )
    p.add_argument(
        "--rows",
        type=int,
        default=None,
        help="Cap rows read per file regardless of debug mode (overrides --debug cap).",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        cfg = Config.load(args.config).validate()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    if args.debug:
        cfg.debug.enabled = True

    row_limit: Optional[int] = args.rows
    if row_limit is None and cfg.debug.enabled:
        row_limit = cfg.debug.row_limit

    cfg.ensure_directories()
    logger = get_logger("run_preprocessing", logs_dir=cfg.paths.logs_dir)

    logger.info("Starting Stage C preprocessing pipeline")
    logger.info(
        "row_limit=%s  window_size=%d  seed=%d",
        row_limit or "none",
        cfg.window_size,
        cfg.seed,
    )

    try:
        result = run_pipeline(cfg, row_limit=row_limit, progress=print)
    except FileNotFoundError as exc:
        print(f"\nMissing corpus file: {exc}", file=sys.stderr)
        logger.error("Corpus file not found: %s", exc)
        return EXIT_NO_FILE
    except Exception as exc:
        print(f"\nPipeline failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        logger.exception("Pipeline failed")
        return EXIT_ERROR

    print("\nArtefact summary:")
    for section_name, entries in result.summary.items():
        if isinstance(entries, dict) and entries:
            print(f"  [{section_name}]")
            for name, info in entries.items():
                if isinstance(info, dict):
                    size = info.get("size_bytes", 0)
                    print(f"    {name:<40} {size:>14,} bytes")

    logger.info(
        "Stage C complete — %d artefacts under %s",
        len(result.written),
        result.out_dir,
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
