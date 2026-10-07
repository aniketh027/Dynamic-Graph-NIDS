"""E.12B metadata preflight or explicitly acknowledged owner execution."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pruning import stage_e12b as b


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="test-locked provenance preflight; no output writes")
    execute = commands.add_parser("execute", help="OWNER ONLY: reserve and evaluate nine fixed-mask checkpoints once")
    execute.add_argument("--run-id", required=True)
    execute.add_argument("--device", choices=("cpu", "cuda"), required=True)
    execute.add_argument("--policy-sha256", required=True)
    execute.add_argument("--freeze-sha256", required=True)
    execute.add_argument("--implementation-sha256", required=True)
    execute.add_argument("--owner-confirmed-one-time-pruned-test-execution", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.command == "prepare":
        result = b.prepare(root)
    else:
        result = b.run_owner(root, run_id=args.run_id, device=args.device, policy_sha256=args.policy_sha256,
                             freeze_sha256=args.freeze_sha256, implementation_sha256=args.implementation_sha256,
                             owner_confirmed_one_time_pruned_test_execution=args.owner_confirmed_one_time_pruned_test_execution)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
