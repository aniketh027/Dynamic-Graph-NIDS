"""Read-only failure audit or OWNER-only reviewed E.11 suffix continuation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import e11_failure_recovery as recovery  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("audit", help="Verify the reviewed failure without predictive arrays or writes.")
    run = commands.add_parser("recover", help="OWNER ONLY: continue checkpoint 7 onward, reuse six immutable results.")
    run.add_argument("--owner-confirmed-reviewed-failure-recovery", action="store_true", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "audit":
        result = recovery.audit_failure(PROJECT_ROOT)
        result.pop("preparation")
        result.pop("completed_results")
    else:
        result = recovery.run_owner_recovery(PROJECT_ROOT,
                owner_confirmed_reviewed_failure_recovery=args.owner_confirmed_reviewed_failure_recovery)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
