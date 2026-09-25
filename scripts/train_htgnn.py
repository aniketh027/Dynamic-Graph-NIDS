"""Launch one explicit Stage E.7 HTGNN training run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training import harness as e7  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train one frozen HTGNN task using train-only optimization and "
            "validation-only checkpoint selection"
        )
    )
    parser.add_argument("--task", required=True, choices=e7.TASKS)
    parser.add_argument(
        "--learning-rate",
        required=True,
        type=float,
        choices=e7.LEARNING_RATE_CANDIDATES,
    )
    parser.add_argument(
        "--seed", required=True, type=int, choices=e7.APPROVED_SEEDS
    )
    parser.add_argument("--batch-size", type=int, default=e7.DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=e7.MAX_EPOCHS)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument(
        "--max-train-batches",
        type=int,
        help="Development/smoke bound; omit for the complete train split.",
    )
    parser.add_argument(
        "--max-validation-batches",
        type=int,
        help="Development/smoke bound; omit for the complete validation split.",
    )
    args = parser.parse_args()

    config = e7.TrainingConfig(
        task=args.task,
        learning_rate=args.learning_rate,
        seed=args.seed,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        max_train_batches=args.max_train_batches,
        max_validation_batches=args.max_validation_batches,
    )
    result = e7.run_training(
        config,
        project_root=PROJECT_ROOT,
        checkpoint_path=args.checkpoint,
        log_path=args.log,
        device=args.device,
    )
    print(json.dumps({
        "task": result.task,
        "device": result.device,
        "epochs_completed": result.epochs_completed,
        "stopped_early": result.stopped_early,
        "best_epoch": result.best_epoch,
        "selection_name": result.selection_name,
        "best_selection_value": result.best_selection_value,
        "parameter_update_observed": result.parameter_update_observed,
        "checkpoint": result.checkpoint_path,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
