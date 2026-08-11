from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

from .config import ResolvedTrainingPlan, TrainingConfig


SCHEDULER_STATE_SCHEMA_VERSION = 1


class SchedulerConfigError(ValueError):
    """Raised when the warmup/cosine schedule definition is invalid."""


class SchedulerStateError(ValueError):
    """Raised when scheduler application or restoration is inconsistent."""


def _is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_non_negative_update(value: object, field: str) -> int:
    if not _is_plain_int(value) or int(value) < 0:
        raise SchedulerConfigError(
            f"{field} must be a non-negative integer, got {value!r}"
        )
    return int(value)


def _validate_schedule(
    *,
    total_updates: object,
    warmup_updates: object,
    max_learning_rate: object,
    min_learning_rate: object,
) -> tuple[int, int, float, float]:
    if not _is_plain_int(total_updates) or int(total_updates) <= 0:
        raise SchedulerConfigError(
            "total_updates must be a positive integer, "
            f"got {total_updates!r}"
        )
    total_updates = int(total_updates)
    warmup_updates = _require_non_negative_update(
        warmup_updates,
        "warmup_updates",
    )
    if warmup_updates >= total_updates:
        raise SchedulerConfigError(
            "warmup_updates must be smaller than total_updates, "
            f"got warmup={warmup_updates}, total={total_updates}"
        )

    for value, field in (
        (max_learning_rate, "max_learning_rate"),
        (min_learning_rate, "min_learning_rate"),
    ):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise SchedulerConfigError(f"{field} must be finite, got {value!r}")

    max_learning_rate = float(max_learning_rate)
    min_learning_rate = float(min_learning_rate)
    if max_learning_rate <= 0.0:
        raise SchedulerConfigError("max_learning_rate must be positive")
    if min_learning_rate < 0.0 or min_learning_rate > max_learning_rate:
        raise SchedulerConfigError(
            "min_learning_rate must be non-negative and no greater than "
            "max_learning_rate"
        )

    return (
        total_updates,
        warmup_updates,
        max_learning_rate,
        min_learning_rate,
    )


def warmup_cosine_lr(
    update_index: int,
    *,
    total_updates: int,
    warmup_updates: int,
    max_learning_rate: float,
    min_learning_rate: float,
) -> float:
    """Return LR for the 0-based optimizer update that is about to run."""

    update_index = _require_non_negative_update(update_index, "update_index")
    (
        total_updates,
        warmup_updates,
        max_learning_rate,
        min_learning_rate,
    ) = _validate_schedule(
        total_updates=total_updates,
        warmup_updates=warmup_updates,
        max_learning_rate=max_learning_rate,
        min_learning_rate=min_learning_rate,
    )

    if update_index >= total_updates:
        return min_learning_rate

    if warmup_updates > 0 and update_index < warmup_updates:
        return max_learning_rate * (update_index + 1) / warmup_updates

    cosine_updates = total_updates - warmup_updates
    if cosine_updates == 1:
        # The sole post-warmup update is also the final planned update. The
        # terminal minimum takes precedence because both endpoints cannot be
        # represented by one update.
        return min_learning_rate

    progress = (update_index - warmup_updates) / (cosine_updates - 1)
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    learning_rate = min_learning_rate + (
        max_learning_rate - min_learning_rate
    ) * cosine_factor
    return min(max_learning_rate, max(min_learning_rate, learning_rate))


class WarmupCosineScheduler:
    """Explicit update-index scheduler without PyTorch `_LRScheduler` offsets."""

    STATE_FIELDS = frozenset(
        {
            "schema_version",
            "total_updates",
            "warmup_updates",
            "max_learning_rate",
            "min_learning_rate",
            "last_applied_update",
            "last_learning_rate",
        }
    )

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        total_updates: int,
        warmup_updates: int,
        max_learning_rate: float,
        min_learning_rate: float,
    ) -> None:
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(
                "optimizer must be a torch.optim.Optimizer, "
                f"got {type(optimizer)!r}"
            )
        if not optimizer.param_groups:
            raise SchedulerConfigError("optimizer must contain parameter groups")

        (
            self.total_updates,
            self.warmup_updates,
            self.max_learning_rate,
            self.min_learning_rate,
        ) = _validate_schedule(
            total_updates=total_updates,
            warmup_updates=warmup_updates,
            max_learning_rate=max_learning_rate,
            min_learning_rate=min_learning_rate,
        )
        self.optimizer = optimizer
        self.last_applied_update: int | None = None
        self.last_learning_rate: float | None = None

    @property
    def next_update_index(self) -> int:
        if self.last_applied_update is None:
            return 0
        return self.last_applied_update + 1

    def lr_for_update(self, update_index: int) -> float:
        return warmup_cosine_lr(
            update_index,
            total_updates=self.total_updates,
            warmup_updates=self.warmup_updates,
            max_learning_rate=self.max_learning_rate,
            min_learning_rate=self.min_learning_rate,
        )

    def _set_optimizer_lr(self, learning_rate: float) -> None:
        for parameter_group in self.optimizer.param_groups:
            parameter_group["lr"] = learning_rate

    def apply_for_update(self, update_index: int) -> float:
        update_index = _require_non_negative_update(update_index, "update_index")
        if update_index >= self.total_updates:
            raise SchedulerStateError(
                "cannot apply a learning rate beyond the planned update horizon: "
                f"update={update_index}, total={self.total_updates}"
            )
        if update_index != self.next_update_index:
            raise SchedulerStateError(
                "scheduler updates must be applied exactly once and in order: "
                f"expected {self.next_update_index}, got {update_index}"
            )

        learning_rate = self.lr_for_update(update_index)
        self._set_optimizer_lr(learning_rate)
        self.last_applied_update = update_index
        self.last_learning_rate = learning_rate
        return learning_rate

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEDULER_STATE_SCHEMA_VERSION,
            "total_updates": self.total_updates,
            "warmup_updates": self.warmup_updates,
            "max_learning_rate": self.max_learning_rate,
            "min_learning_rate": self.min_learning_rate,
            "last_applied_update": self.last_applied_update,
            "last_learning_rate": self.last_learning_rate,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise SchedulerStateError(
                f"scheduler state must be a mapping, got {type(state)!r}"
            )

        provided_fields = set(state)
        missing_fields = self.STATE_FIELDS - provided_fields
        unknown_fields = provided_fields - self.STATE_FIELDS
        if missing_fields:
            raise SchedulerStateError(
                f"scheduler state is missing fields: {sorted(missing_fields)}"
            )
        if unknown_fields:
            raise SchedulerStateError(
                f"scheduler state has unknown fields: {sorted(unknown_fields)}"
            )
        if (
            not _is_plain_int(state["schema_version"])
            or state["schema_version"] != SCHEDULER_STATE_SCHEMA_VERSION
        ):
            raise SchedulerStateError(
                "unsupported scheduler state schema version: "
                f"{state['schema_version']!r}"
            )

        try:
            restored_identity_values = _validate_schedule(
                total_updates=state["total_updates"],
                warmup_updates=state["warmup_updates"],
                max_learning_rate=state["max_learning_rate"],
                min_learning_rate=state["min_learning_rate"],
            )
        except SchedulerConfigError as error:
            raise SchedulerStateError(
                f"scheduler state contains an invalid schedule: {error}"
            ) from error
        restored_identity = dict(
            zip(
                (
                    "total_updates",
                    "warmup_updates",
                    "max_learning_rate",
                    "min_learning_rate",
                ),
                restored_identity_values,
                strict=True,
            )
        )
        expected_identity = {
            "total_updates": self.total_updates,
            "warmup_updates": self.warmup_updates,
            "max_learning_rate": self.max_learning_rate,
            "min_learning_rate": self.min_learning_rate,
        }
        for field, expected in expected_identity.items():
            if restored_identity[field] != expected:
                raise SchedulerStateError(
                    f"scheduler state {field} does not match the active plan: "
                    f"expected {expected!r}, got {restored_identity[field]!r}"
                )

        last_applied_update = state["last_applied_update"]
        last_learning_rate = state["last_learning_rate"]
        if last_applied_update is None:
            if last_learning_rate is not None:
                raise SchedulerStateError(
                    "last_learning_rate must be null before the first update"
                )
            self.last_applied_update = None
            self.last_learning_rate = None
            return

        if (
            not _is_plain_int(last_applied_update)
            or int(last_applied_update) < 0
            or int(last_applied_update) >= self.total_updates
        ):
            raise SchedulerStateError(
                "last_applied_update must identify a completed planned update, "
                f"got {last_applied_update!r}"
            )
        last_applied_update = int(last_applied_update)
        expected_learning_rate = self.lr_for_update(last_applied_update)
        if (
            not isinstance(last_learning_rate, (int, float))
            or isinstance(last_learning_rate, bool)
            or not math.isfinite(float(last_learning_rate))
            or not math.isclose(
                float(last_learning_rate),
                expected_learning_rate,
                rel_tol=1.0e-12,
                abs_tol=0.0,
            )
        ):
            raise SchedulerStateError(
                "last_learning_rate is inconsistent with last_applied_update"
            )

        self.last_applied_update = last_applied_update
        self.last_learning_rate = expected_learning_rate
        self._set_optimizer_lr(expected_learning_rate)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    plan: ResolvedTrainingPlan,
) -> WarmupCosineScheduler:
    if not isinstance(config, TrainingConfig):
        raise TypeError(
            f"config must be a TrainingConfig, got {type(config)!r}"
        )
    if not isinstance(plan, ResolvedTrainingPlan):
        raise TypeError(
            f"plan must be a ResolvedTrainingPlan, got {type(plan)!r}"
        )

    if config.project_name != plan.project_name:
        raise SchedulerConfigError(
            "training config and resolved plan belong to different projects"
        )
    if config.resolve() != plan:
        raise SchedulerConfigError(
            "training config and resolved plan do not describe the same "
            "execution"
        )

    return WarmupCosineScheduler(
        optimizer,
        total_updates=plan.total_updates,
        warmup_updates=plan.warmup_updates,
        max_learning_rate=config.learning_rate,
        min_learning_rate=config.min_learning_rate,
    )
