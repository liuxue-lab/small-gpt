from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any

from .config import ResolvedTrainingPlan


TRAINER_STATE_SCHEMA_VERSION = 1


class TrainerStateError(ValueError):
    """Raised when mutable trainer state violates the frozen contract."""


def _is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_non_negative_int(value: object, field: str) -> int:
    if not _is_plain_int(value) or int(value) < 0:
        raise TrainerStateError(
            f"{field} must be a non-negative integer, got {value!r}"
        )
    return int(value)


def _require_positive_int(value: object, field: str) -> int:
    if not _is_plain_int(value) or int(value) <= 0:
        raise TrainerStateError(f"{field} must be a positive integer, got {value!r}")
    return int(value)


def _validate_loss(value: object, field: str) -> None:
    if value is None:
        return
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise TrainerStateError(
            f"{field} must be a finite non-negative number or null, got {value!r}"
        )


@dataclass(slots=True)
class TrainerState:
    """Checkpointable counters committed only after complete optimizer updates."""

    run_id: str
    schema_version: int = TRAINER_STATE_SCHEMA_VERSION
    global_step: int = 0
    micro_steps_seen: int = 0
    tokens_seen: int = 0
    samples_consumed: int = 0
    data_epoch: int = 0
    batches_consumed_in_epoch: int = 0
    best_validation_loss: float | None = None
    last_validation_loss: float | None = None
    last_eval_step: int = 0
    last_save_step: int = 0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise TrainerStateError(
                f"run_id must be a non-empty string, got {self.run_id!r}"
            )
        if not _is_plain_int(self.schema_version):
            raise TrainerStateError(
                "schema_version must be an integer, "
                f"got {self.schema_version!r}"
            )
        if self.schema_version != TRAINER_STATE_SCHEMA_VERSION:
            raise TrainerStateError(
                "unsupported trainer state schema version: "
                f"{self.schema_version!r}"
            )

        for field_name in (
            "global_step",
            "micro_steps_seen",
            "tokens_seen",
            "samples_consumed",
            "data_epoch",
            "batches_consumed_in_epoch",
            "last_eval_step",
            "last_save_step",
        ):
            _require_non_negative_int(getattr(self, field_name), field_name)

        if self.last_eval_step > self.global_step:
            raise TrainerStateError(
                "last_eval_step cannot exceed global_step: "
                f"{self.last_eval_step} > {self.global_step}"
            )
        if self.last_save_step > self.global_step:
            raise TrainerStateError(
                "last_save_step cannot exceed global_step: "
                f"{self.last_save_step} > {self.global_step}"
            )

        _validate_loss(self.best_validation_loss, "best_validation_loss")
        _validate_loss(self.last_validation_loss, "last_validation_loss")
        if (self.best_validation_loss is None) != (
            self.last_validation_loss is None
        ):
            raise TrainerStateError(
                "best_validation_loss and last_validation_loss must either both "
                "be null or both contain values"
            )
        if (
            self.best_validation_loss is not None
            and self.last_validation_loss is not None
            and self.best_validation_loss > self.last_validation_loss
        ):
            raise TrainerStateError(
                "best_validation_loss cannot exceed last_validation_loss"
            )

    def validate_for_plan(self, plan: ResolvedTrainingPlan) -> None:
        if not isinstance(plan, ResolvedTrainingPlan):
            raise TypeError(
                "plan must be a ResolvedTrainingPlan, "
                f"got {type(plan)!r}"
            )
        self.validate()

        if self.global_step > plan.total_updates:
            raise TrainerStateError(
                "global_step cannot exceed the resolved total updates: "
                f"{self.global_step} > {plan.total_updates}"
            )

        expected_micro_steps = (
            self.global_step * plan.gradient_accumulation_steps
        )
        if self.micro_steps_seen != expected_micro_steps:
            raise TrainerStateError(
                "micro_steps_seen is inconsistent with global_step and the "
                "resolved accumulation plan: "
                f"expected {expected_micro_steps}, got {self.micro_steps_seen}"
            )

        expected_tokens = self.global_step * plan.tokens_per_update
        if self.tokens_seen != expected_tokens:
            raise TrainerStateError(
                "tokens_seen is inconsistent with global_step and the resolved "
                f"token plan: expected {expected_tokens}, got {self.tokens_seen}"
            )

        samples_per_update = (
            plan.micro_batch_size * plan.gradient_accumulation_steps
        )
        expected_samples = self.global_step * samples_per_update
        if self.samples_consumed != expected_samples:
            raise TrainerStateError(
                "samples_consumed is inconsistent with global_step and the "
                f"resolved batch plan: expected {expected_samples}, "
                f"got {self.samples_consumed}"
            )

    def record_update(
        self,
        *,
        micro_steps: int,
        tokens: int,
        samples: int,
    ) -> None:
        """Atomically commit counters for one successfully completed update."""

        micro_steps = _require_positive_int(micro_steps, "micro_steps")
        tokens = _require_positive_int(tokens, "tokens")
        samples = _require_positive_int(samples, "samples")
        self.validate()

        self.global_step += 1
        self.micro_steps_seen += micro_steps
        self.tokens_seen += tokens
        self.samples_consumed += samples
        self.batches_consumed_in_epoch += micro_steps
        self.validate()

    def record_evaluation(self, loss: float) -> None:
        if loss is None:
            raise TrainerStateError("loss must be a finite non-negative number")
        _validate_loss(loss, "loss")
        loss = float(loss)
        self.validate()

        self.last_validation_loss = loss
        if self.best_validation_loss is None:
            self.best_validation_loss = loss
        else:
            self.best_validation_loss = min(self.best_validation_loss, loss)
        self.last_eval_step = self.global_step
        self.validate()

    def record_checkpoint(self) -> None:
        self.validate()
        self.last_save_step = self.global_step

    def state_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> TrainerState:
        if not isinstance(state, Mapping):
            raise TrainerStateError(
                f"trainer state must be a mapping, got {type(state)!r}"
            )

        expected_fields = {field.name for field in fields(cls)}
        provided_fields = set(state)
        missing_fields = expected_fields - provided_fields
        unknown_fields = provided_fields - expected_fields
        if missing_fields:
            raise TrainerStateError(
                f"trainer state is missing fields: {sorted(missing_fields)}"
            )
        if unknown_fields:
            raise TrainerStateError(
                f"trainer state has unknown fields: {sorted(unknown_fields)}"
            )

        try:
            return cls(**dict(state))
        except TypeError as error:
            raise TrainerStateError(
                f"could not construct TrainerState: {error}"
            ) from error
