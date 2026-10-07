"""Owner-controlled Stage E.10 single-job and result-only commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training import full_training as e10  # noqa: E402


def _add_study_id(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--official-study-id",
        required=True,
        help="Fresh identity matching e10-official-YYYYMMDD-vN.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)

    dry = commands.add_parser(
        "dry-run", help="Print exact jobs without loading data or creating models."
    )
    _add_study_id(dry)
    dry.add_argument(
        "--phase", choices=("lr-selection", "final-seed"), required=True
    )
    dry.add_argument("--selection-plan", type=Path)
    dry.add_argument("--format", choices=("json", "powershell"), default="json")

    run = commands.add_parser(
        "run-job", help="Run exactly one explicit owner-confirmed ML job."
    )
    _add_study_id(run)
    run.add_argument(
        "--phase", choices=("lr-selection", "final-seed"), required=True
    )
    run.add_argument("--experiment", choices=e10.TASKS, required=True)
    run.add_argument("--family", choices=e10.MODEL_FAMILIES, required=True)
    run.add_argument(
        "--learning-rate", type=float, choices=e10.LEARNING_RATES, required=True
    )
    run.add_argument("--seed", type=int, choices=e10.FINAL_SEEDS, required=True)
    run.add_argument("--selection-plan", type=Path)
    run.add_argument("--device", choices=("cuda",), default="cuda")
    run.add_argument(
        "--owner-confirmed-manual-execution",
        action="store_true",
        required=True,
        help="Required acknowledgement that the owner launched this one ML job.",
    )

    select = commands.add_parser(
        "aggregate-lr", help="Freeze LRs from exactly 27 COMPLETE official results."
    )
    _add_study_id(select)

    final = commands.add_parser(
        "aggregate-final", help="Build the result-only report after all final seeds."
    )
    _add_study_id(final)
    final.add_argument("--selection-plan", type=Path)
    return parser


def _phase(value: str) -> str:
    return value.replace("-", "_")


def _powershell_quote(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _powershell_job_command(job: dict[str, object]) -> str:
    parts = [
        r".\.venv\Scripts\python.exe",
        r"scripts\run_stage_e10_full_training.py",
        "run-job",
        "--official-study-id",
        _powershell_quote(job["official_study_id"]),
        "--phase",
        _powershell_quote(str(job["phase"]).replace("_", "-")),
        "--experiment",
        _powershell_quote(job["experiment"]),
        "--family",
        _powershell_quote(job["family"]),
        "--learning-rate",
        _powershell_quote(job["learning_rate"]),
        "--seed",
        str(job["seed"]),
        "--device",
        "cuda",
        "--owner-confirmed-manual-execution",
    ]
    if job["phase"] == "final_seed":
        parts.extend(
            [
                "--selection-plan",
                _powershell_quote(
                    Path("results")
                    / "stage_e10"
                    / str(job["official_study_id"])
                    / "selected_learning_rates.json"
                ),
            ]
        )
    command = " ".join(parts)
    return (
        f"{command} # batch_size={job['batch_size']} "
        f"result={_powershell_quote(job['result'])} "
        f"checkpoint={_powershell_quote(job['checkpoint'])}"
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.project_root.resolve()
    if args.command == "dry-run":
        jobs = e10.dry_run_jobs(
            root,
            args.official_study_id,
            phase=_phase(args.phase),
            selection_plan_path=args.selection_plan,
        )
        if args.format == "powershell":
            for job in jobs:
                print(_powershell_job_command(job))
        else:
            print(json.dumps(jobs, indent=2, sort_keys=True))
        return 0
    if args.command == "run-job":
        candidate = e10.resolve_single_job(
            root,
            args.official_study_id,
            phase=_phase(args.phase),
            experiment=args.experiment,
            family=args.family,
            learning_rate=args.learning_rate,
            seed=args.seed,
            selection_plan_path=args.selection_plan,
        )
        preflight = e10.preflight(root)
        result = e10.execute_candidate(
            root,
            args.official_study_id,
            candidate,
            preflight_result=preflight,
            device=args.device,
            owner_confirmed_manual_execution=args.owner_confirmed_manual_execution,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "aggregate-lr":
        result = e10.aggregate_lr_selection(root, args.official_study_id)
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "aggregate-final":
        result = e10.aggregate_final_report(
            root, args.official_study_id, args.selection_plan
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
