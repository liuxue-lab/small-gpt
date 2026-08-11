"""Stable public API for the small-gpt training system."""

from .config import (
    ResolvedTrainingPlan,
    TrainingConfig,
    TrainingConfigError,
    training_field_names,
)
from .data_stream import (
    DataStreamError,
    OffsetSampler,
    TrainingDataStream,
    ValidationDataStream,
)
from .evaluation import (
    EvaluationError,
    EvaluationMetrics,
    evaluate_model,
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
from .run_logging import (
    JsonlMetricLogger,
    MetricLoggingError,
    RunDirectoryError,
    RunPaths,
    initialize_run_directory,
    validate_run_id,
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
    "DataStreamError",
    "DeviceResolutionError",
    "EvaluationError",
    "EvaluationMetrics",
    "JsonlMetricLogger",
    "MetricLoggingError",
    "NonFiniteTrainingError",
    "OffsetSampler",
    "OptimizerConfigError",
    "OptimizerParameterGroups",
    "PrecisionConfigError",
    "PrecisionPolicy",
    "ResolvedTrainingPlan",
    "RunDirectoryError",
    "RunPaths",
    "SchedulerConfigError",
    "SchedulerStateError",
    "TRAINER_STATE_SCHEMA_VERSION",
    "Trainer",
    "TrainerState",
    "TrainerStateError",
    "TrainingConfig",
    "TrainingConfigError",
    "TrainingDataStream",
    "TrainingStepError",
    "UpdateMetrics",
    "ValidationDataStream",
    "WarmupCosineScheduler",
    "build_optimizer",
    "build_scheduler",
    "evaluate_model",
    "initialize_run_directory",
    "partition_parameters",
    "resolve_device",
    "training_field_names",
    "validate_run_id",
    "warmup_cosine_lr",
]
