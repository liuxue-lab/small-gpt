from __future__ import annotations

import math
import sys
import time
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from itertools import islice
from typing import Any

import torch
from torch import nn

from .precision import PrecisionPolicy


class EvaluationError(RuntimeError):
    """Raised when deterministic validation cannot produce a valid metric."""


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    global_step: int
    validation_loss: float
    perplexity: float
    evaluated_batches: int
    evaluated_tokens: int
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_max_batches(max_batches: int | None) -> None:
    if max_batches is None:
        return
    if (
        isinstance(max_batches, bool)
        or not isinstance(max_batches, int)
        or max_batches <= 0
    ):
        raise EvaluationError(
            f"max_batches must be a positive integer or null, got {max_batches!r}"
        )


def _selected_batches(
    batches: Iterable[object],
    max_batches: int | None,
) -> Iterator[object]:
    try:
        iterator = iter(batches)
    except TypeError as error:
        raise EvaluationError("batches must be iterable") from error
    if max_batches is None:
        return iterator
    return islice(iterator, max_batches)


def _prepare_evaluation_batch(
    batch: object,
    *,
    batch_index: int,
    precision: PrecisionPolicy,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(batch, (tuple, list)) or len(batch) != 2:
        raise EvaluationError(
            f"evaluation batch {batch_index} must be an (input, target) pair"
        )
    input_ids, targets = batch
    if not isinstance(input_ids, torch.Tensor) or not isinstance(
        targets,
        torch.Tensor,
    ):
        raise EvaluationError(
            f"evaluation batch {batch_index} values must be torch.Tensor"
        )
    if input_ids.ndim != 2 or targets.shape != input_ids.shape:
        raise EvaluationError(
            f"evaluation batch {batch_index} input and target shapes must match "
            "and have rank 2"
        )
    if input_ids.numel() == 0:
        raise EvaluationError(f"evaluation batch {batch_index} is empty")
    if input_ids.dtype != torch.long or targets.dtype != torch.long:
        raise EvaluationError(
            f"evaluation batch {batch_index} input and target dtype must be "
            "torch.long"
        )

    non_blocking = precision.device.type == "cuda"
    return (
        input_ids.to(precision.device, non_blocking=non_blocking),
        targets.to(precision.device, non_blocking=non_blocking),
    )


def evaluate_model(
    model: nn.Module,
    batches: Iterable[object],
    *,
    precision: PrecisionPolicy,
    global_step: int,
    max_batches: int | None,
) -> EvaluationMetrics:
    """Evaluate fixed batches with a token-weighted mean and mode restoration."""

    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model)!r}")
    if not isinstance(precision, PrecisionPolicy):
        raise TypeError(
            "precision must be a PrecisionPolicy, "
            f"got {type(precision)!r}"
        )
    if (
        isinstance(global_step, bool)
        or not isinstance(global_step, int)
        or global_step < 0
    ):
        raise EvaluationError(
            f"global_step must be a non-negative integer, got {global_step!r}"
        )
    _validate_max_batches(max_batches)
    wrong_device_names = [
        name
        for name, parameter in model.named_parameters()
        if parameter.device != precision.device
    ]
    if wrong_device_names:
        raise EvaluationError(
            "model parameters are not on the evaluation device: "
            f"{wrong_device_names[:5]}"
        )

    selected_batches = _selected_batches(batches, max_batches)
    was_training = model.training
    start_time = time.perf_counter()
    total_negative_log_likelihood = 0.0
    evaluated_tokens = 0
    evaluated_batches = 0

    model.eval()
    try:
        with torch.no_grad():
            for batch_index, raw_batch in enumerate(selected_batches):
                input_ids, targets = _prepare_evaluation_batch(
                    raw_batch,
                    batch_index=batch_index,
                    precision=precision,
                )
                with precision.autocast_context():
                    output = model(input_ids, targets)
                    loss = getattr(output, "loss", None)
                if not isinstance(loss, torch.Tensor):
                    raise EvaluationError(
                        "model output must expose a Tensor loss during evaluation"
                    )
                if loss.ndim != 0 or not loss.is_floating_point():
                    raise EvaluationError(
                        "evaluation loss must be a scalar floating-point Tensor"
                    )
                detached_loss = loss.detach()
                if not bool(torch.isfinite(detached_loss).item()):
                    raise EvaluationError(
                        f"evaluation loss is non-finite at batch {batch_index}"
                    )
                batch_loss = float(detached_loss.float().item())
                if batch_loss < 0.0:
                    raise EvaluationError(
                        f"evaluation loss is negative at batch {batch_index}"
                    )
                batch_tokens = targets.numel()
                total_negative_log_likelihood += batch_loss * batch_tokens
                evaluated_tokens += batch_tokens
                evaluated_batches += 1
    finally:
        model.train(was_training)

    if evaluated_batches == 0 or evaluated_tokens == 0:
        raise EvaluationError("evaluation produced no complete batches")

    validation_loss = total_negative_log_likelihood / evaluated_tokens
    maximum_log = math.log(sys.float_info.max)
    perplexity = (
        math.exp(validation_loss)
        if validation_loss <= maximum_log
        else math.inf
    )
    elapsed_seconds = max(time.perf_counter() - start_time, 1.0e-12)
    return EvaluationMetrics(
        global_step=global_step,
        validation_loss=validation_loss,
        perplexity=perplexity,
        evaluated_batches=evaluated_batches,
        evaluated_tokens=evaluated_tokens,
        elapsed_seconds=elapsed_seconds,
    )
