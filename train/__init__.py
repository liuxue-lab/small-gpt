"""Stable public API for the small-gpt training system."""

from .config import (
    ResolvedTrainingPlan,
    TrainingConfig,
    TrainingConfigError,
    training_field_names,
)

__all__ = [
    "ResolvedTrainingPlan",
    "TrainingConfig",
    "TrainingConfigError",
    "training_field_names",
]
