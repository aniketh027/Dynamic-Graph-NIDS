"""Stage E.7 task-aware training and validation subsystem."""

from src.training.harness import (
    APPROVED_SEEDS,
    HARNESS_VERSION,
    LEARNING_RATE_CANDIDATES,
    TrainingConfig,
    TrainingContractError,
    evaluate_saved_validation_checkpoint,
    run_training,
)

__all__ = [
    "APPROVED_SEEDS",
    "HARNESS_VERSION",
    "LEARNING_RATE_CANDIDATES",
    "TrainingConfig",
    "TrainingContractError",
    "evaluate_saved_validation_checkpoint",
    "run_training",
]
