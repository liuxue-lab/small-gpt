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
from .precision import (
    DeviceResolutionError,
    PrecisionConfigError,
    PrecisionPolicy,
    resolve_device,
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
from .trainer import (
    BatchContractError,
    NonFiniteTrainingError,
    Trainer,
    TrainingStepError,
    UpdateMetrics,
)

__all__ = [
    "BatchContractError",
    "DeviceResolutionError",
    "NonFiniteTrainingError",
    "OptimizerConfigError",
    "OptimizerParameterGroups",
    "PrecisionConfigError",
    "PrecisionPolicy",
    "ResolvedTrainingPlan",
    "SchedulerConfigError",
    "SchedulerStateError",
    "TRAINER_STATE_SCHEMA_VERSION",
    "Trainer",
    "TrainerState",
    "TrainerStateError",
    "TrainingConfig",
    "TrainingConfigError",
    "TrainingStepError",
    "UpdateMetrics",
    "WarmupCosineScheduler",
    "build_optimizer",
    "build_scheduler",
    "partition_parameters",
    "resolve_device",
    "training_field_names",
    "warmup_cosine_lr",
]
