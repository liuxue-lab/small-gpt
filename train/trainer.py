from __future__ import annotations

import math
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from .config import ResolvedTrainingPlan, TrainingConfig
from .precision import PrecisionPolicy
from .scheduler import WarmupCosineScheduler
from .state import TrainerState


class TrainingStepError(RuntimeError):
    """Raised when one optimizer update cannot satisfy its contract."""


class BatchContractError(TrainingStepError):
    """Raised when a micro-batch does not match the resolved training plan."""


class NonFiniteTrainingError(TrainingStepError):
    """Raised before state commit when loss or gradient norm is non-finite."""


@dataclass(frozen=True, slots=True)
class UpdateMetrics:
    update_index: int
    completed_global_step: int
    raw_token_weighted_loss: float
    learning_rate: float
    grad_norm_before_clip: float
    micro_steps: int
    samples: int
    tokens: int
    elapsed_seconds: float
    tokens_per_second: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Trainer:
    """Execute transaction-like, full-boundary optimizer updates."""

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: WarmupCosineScheduler,
        state: TrainerState,
        config: TrainingConfig,
        plan: ResolvedTrainingPlan,
        precision: PrecisionPolicy,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError(f"model must be an nn.Module, got {type(model)!r}")
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(
                "optimizer must be a torch.optim.Optimizer, "
                f"got {type(optimizer)!r}"
            )
        if not isinstance(scheduler, WarmupCosineScheduler):
            raise TypeError(
                "scheduler must be a WarmupCosineScheduler, "
                f"got {type(scheduler)!r}"
            )
        if not isinstance(state, TrainerState):
            raise TypeError(
                f"state must be a TrainerState, got {type(state)!r}"
            )
        if not isinstance(config, TrainingConfig):
            raise TypeError(
                f"config must be a TrainingConfig, got {type(config)!r}"
            )
        if not isinstance(plan, ResolvedTrainingPlan):
            raise TypeError(
                "plan must be a ResolvedTrainingPlan, "
                f"got {type(plan)!r}"
            )
        if not isinstance(precision, PrecisionPolicy):
            raise TypeError(
                "precision must be a PrecisionPolicy, "
                f"got {type(precision)!r}"
            )
        if config.resolve() != plan:
            raise TrainingStepError(
                "training config and resolved plan do not describe the same "
                "execution"
            )

        expected_precision = PrecisionPolicy.from_config(config)
        if precision != expected_precision:
            raise TrainingStepError(
                "precision policy does not match the active training config: "
                f"expected {expected_precision.to_dict()}, "
                f"got {precision.to_dict()}"
            )
        if scheduler.optimizer is not optimizer:
            raise TrainingStepError(
                "scheduler and trainer must reference the same optimizer"
            )
        if scheduler.total_updates != plan.total_updates:
            raise TrainingStepError(
                "scheduler total_updates does not match the resolved plan"
            )
        if scheduler.warmup_updates != plan.warmup_updates:
            raise TrainingStepError(
                "scheduler warmup_updates does not match the resolved plan"
            )

        model_parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        optimizer_parameters = [
            parameter
            for parameter_group in optimizer.param_groups
            for parameter in parameter_group["params"]
        ]
        model_ids = {id(parameter) for parameter in model_parameters}
        optimizer_ids = [id(parameter) for parameter in optimizer_parameters]
        if len(optimizer_ids) != len(set(optimizer_ids)):
            raise TrainingStepError(
                "optimizer contains duplicate Parameter identities"
            )
        if set(optimizer_ids) != model_ids:
            raise TrainingStepError(
                "optimizer parameters do not exactly cover trainable model "
                "parameters"
            )
        wrong_device_names = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.device != precision.device
        ]
        if wrong_device_names:
            raise TrainingStepError(
                "trainable model parameters are not on the resolved device: "
                f"{wrong_device_names[:5]}"
            )

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.state = state
        self.config = config
        self.plan = plan
        self.precision = precision
        self._trainable_parameters = tuple(model_parameters)
        self._validate_update_boundary()

    def _validate_update_boundary(self) -> None:
        self.state.validate_for_plan(self.plan)
        if self.state.global_step >= self.plan.total_updates:
            raise TrainingStepError(
                "training has already reached the resolved update horizon"
            )
        if self.scheduler.next_update_index != self.state.global_step:
            raise TrainingStepError(
                "scheduler and trainer state disagree about the next update: "
                f"scheduler={self.scheduler.next_update_index}, "
                f"state={self.state.global_step}"
            )

    def _prepare_batch(
        self,
        batch: object,
        *,
        micro_step_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(batch, (tuple, list)) or len(batch) != 2:
            raise BatchContractError(
                f"micro-batch {micro_step_index} must be an (input, target) pair"
            )
        input_ids, targets = batch
        if not isinstance(input_ids, torch.Tensor) or not isinstance(
            targets,
            torch.Tensor,
        ):
            raise BatchContractError(
                f"micro-batch {micro_step_index} values must be torch.Tensor"
            )
        expected_shape = (
            self.plan.micro_batch_size,
            self.plan.context_length,
        )
        if tuple(input_ids.shape) != expected_shape:
            raise BatchContractError(
                f"micro-batch {micro_step_index} input shape must be "
                f"{expected_shape}, got {tuple(input_ids.shape)}"
            )
        if tuple(targets.shape) != expected_shape:
            raise BatchContractError(
                f"micro-batch {micro_step_index} target shape must be "
                f"{expected_shape}, got {tuple(targets.shape)}"
            )
        if input_ids.dtype != torch.long or targets.dtype != torch.long:
            raise BatchContractError(
                f"micro-batch {micro_step_index} input and target dtype must "
                "be torch.long"
            )

        non_blocking = self.plan.pin_memory and self.precision.device.type == "cuda"
        input_ids = input_ids.to(
            self.precision.device,
            non_blocking=non_blocking,
        )
        targets = targets.to(
            self.precision.device,
            non_blocking=non_blocking,
        )
        return input_ids, targets

    def _restore_pre_step_scheduler(
        self,
        scheduler_state: dict[str, Any],
        group_learning_rates: tuple[float, ...],
    ) -> None:
        self.scheduler.load_state_dict(scheduler_state)
        for parameter_group, learning_rate in zip(
            self.optimizer.param_groups,
            group_learning_rates,
            strict=True,
        ):
            parameter_group["lr"] = learning_rate

    def run_update(
        self,
        micro_batches: Iterable[object],
    ) -> UpdateMetrics:
        """Consume exactly N micro-batches and commit one optimizer update."""

        self._validate_update_boundary()
        try:
            batch_iterator = iter(micro_batches)
        except TypeError as error:
            raise BatchContractError("micro_batches must be iterable") from error

        update_index = self.state.global_step
        scheduler_state = self.scheduler.state_dict()
        group_learning_rates = tuple(
            float(parameter_group["lr"])
            for parameter_group in self.optimizer.param_groups
        )
        optimizer_step_completed = False
        start_time = time.perf_counter()

        try:
            learning_rate = self.scheduler.apply_for_update(update_index)
            if not math.isfinite(learning_rate) or learning_rate < 0.0:
                raise NonFiniteTrainingError(
                    f"scheduler returned invalid learning rate {learning_rate!r}"
                )
            self.optimizer.zero_grad(set_to_none=True)
            self.model.train()

            total_negative_log_likelihood = 0.0
            total_tokens = 0
            total_samples = 0

            for micro_step_index in range(
                self.plan.gradient_accumulation_steps
            ):
                try:
                    raw_batch = next(batch_iterator)
                except StopIteration as error:
                    raise BatchContractError(
                        "micro-batch source ended before one complete optimizer "
                        f"update: expected "
                        f"{self.plan.gradient_accumulation_steps} batches, "
                        f"received {micro_step_index}"
                    ) from error

                input_ids, targets = self._prepare_batch(
                    raw_batch,
                    micro_step_index=micro_step_index,
                )
                token_count = targets.numel()
                sample_count = targets.shape[0]

                with self.precision.autocast_context():
                    output = self.model(input_ids, targets)
                    loss = getattr(output, "loss", None)
                    if not isinstance(loss, torch.Tensor):
                        raise TrainingStepError(
                            "model output must expose a Tensor loss when targets "
                            "are provided"
                        )
                    if loss.ndim != 0 or not loss.is_floating_point():
                        raise TrainingStepError(
                            "model loss must be a scalar floating-point Tensor"
                        )

                detached_loss = loss.detach()
                if not bool(torch.isfinite(detached_loss).item()):
                    raise NonFiniteTrainingError(
                        "non-finite loss before backward at "
                        f"update={update_index}, micro_step={micro_step_index}"
                    )
                raw_loss = float(detached_loss.float().item())
                total_negative_log_likelihood += raw_loss * token_count
                total_tokens += token_count
                total_samples += sample_count

                scaled_loss = loss / self.plan.gradient_accumulation_steps
                scaled_loss.backward()

            missing_gradient_names = [
                name
                for name, parameter in self.model.named_parameters()
                if parameter.requires_grad and parameter.grad is None
            ]
            if missing_gradient_names:
                raise TrainingStepError(
                    "trainable parameters are missing gradients: "
                    f"{missing_gradient_names[:5]}"
                )

            try:
                grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                    self._trainable_parameters,
                    max_norm=self.config.gradient_clip,
                    error_if_nonfinite=True,
                )
            except RuntimeError as error:
                if "non-finite" in str(error).lower():
                    raise NonFiniteTrainingError(
                        "gradient norm is non-finite before optimizer step"
                    ) from error
                raise

            grad_norm = float(grad_norm_tensor.detach().float().item())
            if not math.isfinite(grad_norm):
                raise NonFiniteTrainingError(
                    "gradient norm is non-finite before optimizer step"
                )

            if total_tokens != self.plan.tokens_per_update:
                raise BatchContractError(
                    "micro-batches produced an unexpected token count: "
                    f"expected {self.plan.tokens_per_update}, got {total_tokens}"
                )
            expected_samples = (
                self.plan.micro_batch_size
                * self.plan.gradient_accumulation_steps
            )
            if total_samples != expected_samples:
                raise BatchContractError(
                    "micro-batches produced an unexpected sample count: "
                    f"expected {expected_samples}, got {total_samples}"
                )

            self.optimizer.step()
            optimizer_step_completed = True
            self.state.record_update(
                micro_steps=self.plan.gradient_accumulation_steps,
                tokens=total_tokens,
                samples=total_samples,
            )
            self.state.validate_for_plan(self.plan)
        except Exception:
            self.optimizer.zero_grad(set_to_none=True)
            if not optimizer_step_completed:
                self._restore_pre_step_scheduler(
                    scheduler_state,
                    group_learning_rates,
                )
            raise

        elapsed_seconds = max(time.perf_counter() - start_time, 1.0e-12)
        raw_token_weighted_loss = (
            total_negative_log_likelihood / total_tokens
        )
        return UpdateMetrics(
            update_index=update_index,
            completed_global_step=self.state.global_step,
            raw_token_weighted_loss=raw_token_weighted_loss,
            learning_rate=learning_rate,
            grad_norm_before_clip=grad_norm,
            micro_steps=self.plan.gradient_accumulation_steps,
            samples=total_samples,
            tokens=total_tokens,
            elapsed_seconds=elapsed_seconds,
            tokens_per_second=total_tokens / elapsed_seconds,
        )
