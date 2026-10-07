"""Owner-only E.12A entry point. Dry-run never opens data/checkpoints."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Must precede torch import/CUDA initialization; recorded in the E.12A manifest.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from src.pruning import stage_e12 as e12


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("dry-run")
    phase_a = commands.add_parser("select-validation")
    phase_a.add_argument("--run-id", required=True)
    phase_a.add_argument("--device", choices=("cpu", "cuda"), required=True)
    phase_a.add_argument("--max-relative-f1-drop", type=float, required=True)
    phase_a.add_argument("--tolerance-rationale", required=True)
    phase_a.add_argument("--owner-confirmed-validation-contract", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "dry-run":
        print(json.dumps({"stage": "E.12A", "ratios": e12.RATIOS,
                          "graphs": [e12.make_policy([1.] * 78, r).record()["pruned_graph"] for r in e12.RATIOS],
                          "tolerance": "owner approval required; no default", "real_execution": False, "phase_b_enabled": False}, indent=2))
        return
    report = e12.run_phase_a(ROOT, run_id=args.run_id, device=args.device,
                            max_relative_f1_drop=args.max_relative_f1_drop, tolerance_rationale=args.tolerance_rationale,
                            owner_confirmed_validation_contract=args.owner_confirmed_validation_contract)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
