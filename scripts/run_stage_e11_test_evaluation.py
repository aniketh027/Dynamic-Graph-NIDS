"""Read-only E.10 audit or explicitly owner-launched E.11 test evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import official_test as e11  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="Verify E.10 and checkpoint provenance; never open predictive arrays.")
    audit.add_argument("--official-study-id", required=True, choices=(e11.STUDY,))
    evaluate = commands.add_parser("evaluate", help="OWNER ONLY: consumes the official test evaluation reservation.")
    evaluate.add_argument("--official-study-id", required=True, choices=(e11.STUDY,))
    evaluate.add_argument("--run-id", required=True)
    evaluate.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    evaluate.add_argument("--owner-confirmed-manual-test-evaluation", action="store_true", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "audit":
        report = e11.audit_e10(PROJECT_ROOT, args.official_study_id)
        report.pop("evaluation_records")
        report.pop("e10_artifact_sha256")
        report.pop("frozen_metadata_sha256")
    else:
        report = e11.run_owner_evaluation(PROJECT_ROOT, args.run_id, study=args.official_study_id,
                device=args.device, owner_confirmed_manual_test_evaluation=args.owner_confirmed_manual_test_evaluation)
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
