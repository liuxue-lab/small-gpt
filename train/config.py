from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


class TrainingConfigError(ValueError):
    """Raised when a training configuration violates the frozen contract."""


def _is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _require_positive_int(value: object, field: str) -> int:
    if not _is_plain_int(value) or int(value) <= 0:
        raise TrainingConfigError(f"{field} must be a positive integer, got {value!r}")
    return int(value)


def _require_non_negative_int(value: object, field: str) -> int:
    if not _is_plain_int(value) or int(value) < 0:
        raise TrainingConfigError(
            f"{field} must be a non-negative integer, got {value!r}"
        )
    return int(value)


def _require_optional_positive_int(value: object, field: str) -> int | None:
    if value is None:
        return None
    return _require_positive_int(value, field)


def _require_optional_non_negative_int(value: object, field: str) -> int | None:
    if value is None:
        return None
    return _require_non_negative_int(value, field)


def _require_finite(value: object, field: str) -> float:
    if not _is_finite_number(value):
        raise TrainingConfigError(f"{field} must be finite, got {value!r}")
    return float(value)


def _require_non_empty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrainingConfigError(f"{field} must be a non-empty string, got {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class ResolvedTrainingPlan:
    project_name: str
    seed: int
    context_length: int
    vocab_size: int
    requested_device: str
    precision: str
    micro_batch_size: int
    gradient_accumulation_steps: int
    num_workers: int
    pin_memory: bool
    tokens_per_micro_step: int
    tokens_per_update: int
    total_updates: int
    warmup_updates: int
    planned_tokens: int
    target_tokens: int | None
    token_overshoot: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    project_name: str
    seed: int
    context_length: int
    vocab_size: int
    device: str
    precision: str
    micro_batch_size: int | None
    gradient_accumulation_steps: int | None
    max_steps: int | None
    target_tokens: int | None
    learning_rate: float
    min_learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    adam_eps: float
    warmup_steps: int | None
    warmup_ratio: float | None
    gradient_clip: float
    log_interval: int
    eval_interval: int
    eval_batches: int | None
    save_interval: int
    num_workers: int | None
    pin_memory: bool | None
    deterministic: bool
    allow_tf32: bool
    run_dir: str
    checkpoint_dir: str

    TRAINING_FIELDS = frozenset(
        {
            "device",
            "precision",
            "micro_batch_size",
            "gradient_accumulation_steps",
            "max_steps",
            "target_tokens",
            "learning_rate",
            "min_learning_rate",
            "weight_decay",
            "beta1",
            "beta2",
            "adam_eps",
            "warmup_steps",
            "warmup_ratio",
            "gradient_clip",
            "log_interval",
            "eval_interval",
            "eval_batches",
            "save_interval",
            "num_workers",
            "pin_memory",
            "deterministic",
            "allow_tf32",
            "run_dir",
            "checkpoint_dir",
        }
    )

    def __post_init__(self) -> None:
        _require_non_empty_string(self.project_name, "project.name")
        _require_non_negative_int(self.seed, "project.seed")
        _require_positive_int(self.context_length, "model.context_length")
        _require_positive_int(self.vocab_size, "model.vocab_size")

        if self.device not in {"auto", "cpu", "cuda"}:
            raise TrainingConfigError(
                "training.device must be one of ['auto', 'cpu', 'cuda'], "
                f"got {self.device!r}"
            )
        if self.precision not in {"fp32", "bf16"}:
            raise TrainingConfigError(
                "training.precision must be one of ['fp32', 'bf16'], "
                f"got {self.precision!r}"
            )
        if self.device == "cpu" and self.precision == "bf16":
            raise TrainingConfigError(
                "training.precision='bf16' requires CUDA in the Day 7 contract"
            )

        _require_optional_positive_int(
            self.micro_batch_size,
            "training.micro_batch_size",
        )
        _require_optional_positive_int(
            self.gradient_accumulation_steps,
            "training.gradient_accumulation_steps",
        )
        _require_optional_positive_int(self.max_steps, "training.max_steps")
        _require_optional_positive_int(self.target_tokens, "training.target_tokens")
        if (self.max_steps is None) == (self.target_tokens is None):
            raise TrainingConfigError(
                "exactly one of training.max_steps and training.target_tokens "
                "must be non-null"
            )

        learning_rate = _require_finite(
            self.learning_rate,
            "training.learning_rate",
        )
        min_learning_rate = _require_finite(
            self.min_learning_rate,
            "training.min_learning_rate",
        )
        weight_decay = _require_finite(self.weight_decay, "training.weight_decay")
        beta1 = _require_finite(self.beta1, "training.beta1")
        beta2 = _require_finite(self.beta2, "training.beta2")
        adam_eps = _require_finite(self.adam_eps, "training.adam_eps")
        gradient_clip = _require_finite(
            self.gradient_clip,
            "training.gradient_clip",
        )

        if learning_rate <= 0.0:
            raise TrainingConfigError("training.learning_rate must be positive")
        if min_learning_rate < 0.0 or min_learning_rate > learning_rate:
            raise TrainingConfigError(
                "training.min_learning_rate must be non-negative and no greater "
                "than training.learning_rate"
            )
        if weight_decay < 0.0:
            raise TrainingConfigError("training.weight_decay must be non-negative")
        for value, field_name in ((beta1, "beta1"), (beta2, "beta2")):
            if not 0.0 <= value < 1.0:
                raise TrainingConfigError(
                    f"training.{field_name} must be in [0, 1), got {value!r}"
                )
        if adam_eps <= 0.0:
            raise TrainingConfigError("training.adam_eps must be positive")
        if gradient_clip <= 0.0:
            raise TrainingConfigError("training.gradient_clip must be positive")

        _require_optional_non_negative_int(
            self.warmup_steps,
            "training.warmup_steps",
        )
        if self.warmup_ratio is not None:
            warmup_ratio = _require_finite(
                self.warmup_ratio,
                "training.warmup_ratio",
            )
            if not 0.0 < warmup_ratio < 1.0:
                raise TrainingConfigError(
                    "training.warmup_ratio must be in (0, 1) when provided"
                )
        if (self.warmup_steps is None) == (self.warmup_ratio is None):
            raise TrainingConfigError(
                "exactly one of training.warmup_steps and training.warmup_ratio "
                "must be non-null"
            )

        for value, field_name in (
            (self.log_interval, "log_interval"),
            (self.eval_interval, "eval_interval"),
            (self.save_interval, "save_interval"),
        ):
            _require_positive_int(value, f"training.{field_name}")
        _require_optional_positive_int(self.eval_batches, "training.eval_batches")
        _require_optional_non_negative_int(self.num_workers, "training.num_workers")

        if self.pin_memory is not None and not isinstance(self.pin_memory, bool):
            raise TrainingConfigError(
                "training.pin_memory must be a boolean or null"
            )
        for value, field_name in (
            (self.deterministic, "deterministic"),
            (self.allow_tf32, "allow_tf32"),
        ):
            if not isinstance(value, bool):
                raise TrainingConfigError(
                    f"training.{field_name} must be a boolean, got {value!r}"
                )

        _require_non_empty_string(self.run_dir, "training.run_dir")
        _require_non_empty_string(self.checkpoint_dir, "training.checkpoint_dir")

    @property
    def unresolved_fields(self) -> tuple[str, ...]:
        nullable_resources = (
            "micro_batch_size",
            "gradient_accumulation_steps",
            "num_workers",
            "pin_memory",
        )
        return tuple(
            field_name
            for field_name in nullable_resources
            if getattr(self, field_name) is None
        )

    @property
    def is_execution_ready(self) -> bool:
        return not self.unresolved_fields

    def resolve(self) -> ResolvedTrainingPlan:
        unresolved = self.unresolved_fields
        if unresolved:
            raise TrainingConfigError(
                "training configuration has unresolved execution fields: "
                f"{list(unresolved)}"
            )

        assert self.micro_batch_size is not None
        assert self.gradient_accumulation_steps is not None
        assert self.num_workers is not None
        assert self.pin_memory is not None

        tokens_per_micro_step = self.micro_batch_size * self.context_length
        tokens_per_update = (
            tokens_per_micro_step * self.gradient_accumulation_steps
        )

        if self.max_steps is not None:
            total_updates = self.max_steps
            target_tokens = None
        else:
            assert self.target_tokens is not None
            total_updates = math.ceil(self.target_tokens / tokens_per_update)
            target_tokens = self.target_tokens

        if self.warmup_steps is not None:
            warmup_updates = self.warmup_steps
        else:
            assert self.warmup_ratio is not None
            warmup_updates = math.ceil(total_updates * self.warmup_ratio)

        if warmup_updates >= total_updates:
            raise TrainingConfigError(
                "resolved warmup updates must be smaller than total updates, "
                f"got warmup={warmup_updates}, total={total_updates}"
            )

        planned_tokens = total_updates * tokens_per_update
        token_overshoot = (
            0 if target_tokens is None else planned_tokens - target_tokens
        )

        return ResolvedTrainingPlan(
            project_name=self.project_name,
            seed=self.seed,
            context_length=self.context_length,
            vocab_size=self.vocab_size,
            requested_device=self.device,
            precision=self.precision,
            micro_batch_size=self.micro_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            tokens_per_micro_step=tokens_per_micro_step,
            tokens_per_update=tokens_per_update,
            total_updates=total_updates,
            warmup_updates=warmup_updates,
            planned_tokens=planned_tokens,
            target_tokens=target_tokens,
            token_overshoot=token_overshoot,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(
        cls,
        *,
        project: Mapping[str, Any],
        model: Mapping[str, Any],
        training: Mapping[str, Any],
    ) -> TrainingConfig:
        if not isinstance(project, Mapping):
            raise TrainingConfigError("project section must be a mapping")
        if not isinstance(model, Mapping):
            raise TrainingConfigError("model section must be a mapping")
        if not isinstance(training, Mapping):
            raise TrainingConfigError("training section must be a mapping")

        provided_fields = set(training)
        missing_fields = cls.TRAINING_FIELDS - provided_fields
        unknown_fields = provided_fields - cls.TRAINING_FIELDS
        if missing_fields:
            raise TrainingConfigError(
                f"training configuration is missing fields: {sorted(missing_fields)}"
            )
        if unknown_fields:
            raise TrainingConfigError(
                f"training configuration has unknown fields: {sorted(unknown_fields)}"
            )

        project_name = _require_non_empty_string(project.get("name"), "project.name")
        seed = _require_non_negative_int(project.get("seed"), "project.seed")
        context_length = _require_positive_int(
            model.get("context_length"),
            "model.context_length",
        )
        vocab_size = _require_positive_int(
            model.get("vocab_size"),
            "model.vocab_size",
        )

        try:
            return cls(
                project_name=project_name,
                seed=seed,
                context_length=context_length,
                vocab_size=vocab_size,
                **dict(training),
            )
        except TypeError as error:
            raise TrainingConfigError(
                f"could not construct TrainingConfig: {error}"
            ) from error

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingConfig:
        config_path = Path(path)
        try:
            with config_path.open("r", encoding="utf-8") as file:
                document = yaml.safe_load(file)
        except OSError as error:
            raise TrainingConfigError(
                f"could not read training configuration {config_path}: {error}"
            ) from error
        except yaml.YAMLError as error:
            raise TrainingConfigError(
                f"could not parse training configuration {config_path}: {error}"
            ) from error

        if not isinstance(document, dict):
            raise TrainingConfigError(
                f"{config_path}: top-level configuration must be a mapping"
            )

        for section_name in ("project", "model", "training"):
            if not isinstance(document.get(section_name), dict):
                raise TrainingConfigError(
                    f"{config_path}: {section_name} section must be a mapping"
                )

        try:
            return cls.from_mapping(
                project=document["project"],
                model=document["model"],
                training=document["training"],
            )
        except TrainingConfigError as error:
            raise TrainingConfigError(f"{config_path}: {error}") from error


def training_field_names() -> frozenset[str]:
    """Return the frozen YAML training-field set for validation and tests."""

    dataclass_only_fields = {
        "project_name",
        "seed",
        "context_length",
        "vocab_size",
    }
    return frozenset(
        field.name for field in fields(TrainingConfig)
    ) - dataclass_only_fields
