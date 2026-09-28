"""Run the frozen bounded Stage E.9 HTGNN E2 batch-size pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training import batch_size_pilot as e9  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/stage_e9/htgnn_batch_size_pilot_observations.json"),
    )
    parser.add_argument("--executor", default="Codex")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    output = args.output if args.output.is_absolute() else project_root / args.output
    report = e9.run_pilot(project_root, executor=args.executor)
    observation_sha256 = e9.write_observations(
        report, output, project_root=PROJECT_ROOT
    )
    print(
        json.dumps(
            {
                "overall_status": report["overall_status"],
                "execution_id": report["execution_id"],
                "selected_htgnn_training_batch_size": report["selection"][
                    "selected_batch_size"
                ],
                "observations_physical_sha256": observation_sha256,
                "output": str(output),
                "learning_rate_finalized": False,
                "test_accessed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
