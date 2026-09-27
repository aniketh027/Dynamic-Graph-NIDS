"""Run or safely aggregate the bounded Stage E.8 optimization pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training import optimization_pilot as e8  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--run-id", help="Fresh run ID, or existing ID to aggregate")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/stage_e8/htgnn_optimization_pilot_observations.json"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--executor", default="Codex")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    output = args.output if args.output.is_absolute() else project_root / args.output
    run_id = args.run_id or e8.new_run_id()
    if args.aggregate_only:
        if not args.run_id:
            parser.error("--aggregate-only requires --run-id")
        report = e8.aggregate_existing_run(
            project_root=project_root,
            run_id=run_id,
            output_path=output,
            executor=args.executor,
        )
    else:
        report = e8.run_pilot(
            project_root=project_root,
            run_id=run_id,
            output_path=output,
            device=args.device,
            executor=args.executor,
        )
    print(
        json.dumps(
            {
                "overall_status": report["overall_status"],
                "run_id": report["run_id"],
                "candidate_count": len(report["candidate_results"]),
                "output": str(output),
                "production_configuration_finalized": False,
            },
            sort_keys=True,
        )
    )
    return 0 if report["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
