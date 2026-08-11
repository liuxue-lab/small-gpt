"""Stable public API for the small-gpt training system."""

from .config import (
    ResolvedTrainingPlan,
    TrainingConfig,
    TrainingConfigError,
    training_field_names,
)
from .optimizer import (
    OptimizerConfigError,
    OptimizerParameterGroups,
    build_optimizer,
    partition_parameters,
)
from .scheduler import (
    SchedulerConfigError,
    SchedulerStateError,
    WarmupCosineScheduler,
    build_scheduler,
    warmup_cosine_lr,
)
from .state import (
    TRAINER_STATE_SCHEMA_VERSION,
    TrainerState,
    TrainerStateError,
)

__all__ = [
    "OptimizerConfigError",
    "OptimizerParameterGroups",
    "ResolvedTrainingPlan",
    "SchedulerConfigError",
    "SchedulerStateError",
    "TRAINER_STATE_SCHEMA_VERSION",
    "TrainerState",
    "TrainerStateError",
    "TrainingConfig",
    "TrainingConfigError",
    "WarmupCosineScheduler",
    "build_optimizer",
    "build_scheduler",
    "partition_parameters",
    "training_field_names",
    "warmup_cosine_lr",
]
