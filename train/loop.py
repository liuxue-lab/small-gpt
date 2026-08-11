from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from .checkpoint import CheckpointRecord
from .evaluation import evaluate_model
from .run_logging import JsonlMetricLogger
from .trainer import Trainer


class TrainingLoopError(RuntimeError):
    """Raised when interval scheduling cannot preserve the training contract."""


@dataclass(frozen=True, slots=True)
class TrainingLoopResult:
    start_step: int
    stop_step: int
    updates_completed: int
    evaluation_steps: tuple[int, ...]
    checkpoint_records: tuple[CheckpointRecord, ...]

    @property
    def final_checkpoint(self) -> CheckpointRecord:
        if not self.checkpoint_records:
            raise TrainingLoopError("training loop produced no final checkpoint")
        return self.checkpoint_records[-1]


def _checkpoint_once(
    *,
    trainer: Trainer,
    logger: JsonlMetricLogger,
    checkpoint_writer: Callable[[int], CheckpointRecord],
    checkpoint_path_formatter: Callable[[Path], str],
    emit: Callable[[str], None],
) -> CheckpointRecord:
    step = trainer.state.global_step
    start_time = time.perf_counter()
    record = checkpoint_writer(step)
    elapsed_seconds = max(time.perf_counter() - start_time, 0.0)
    if not isinstance(record, CheckpointRecord):
        raise TrainingLoopError(
            "checkpoint_writer must return a CheckpointRecord"
        )
    if record.global_step != step:
        raise TrainingLoopError(
            "checkpoint writer returned a record for the wrong global step"
        )
    checkpoint_path = checkpoint_path_formatter(record.path)
    logger.log_checkpoint(
        record,
        state=trainer.state,
        checkpoint_path=checkpoint_path,
        elapsed_seconds=elapsed_seconds,
    )
    emit(
        f"checkpoint step={step} path={checkpoint_path} "
        f"bytes={record.file_size}"
    )
    return record


def run_training_loop(
    trainer: Trainer,
    training_batches: Iterable[object],
    validation_batches: Iterable[object],
    *,
    logger: JsonlMetricLogger,
    stop_at_step: int,
    checkpoint_writer: Callable[[int], CheckpointRecord],
    checkpoint_path_formatter: Callable[[Path], str] = str,
    emit: Callable[[str], None] = print,
) -> TrainingLoopResult:
    """Run updates, interval evaluation, and atomic update-boundary saves."""

    if not isinstance(trainer, Trainer):
        raise TypeError(f"trainer must be Trainer, got {type(trainer)!r}")
    if not isinstance(logger, JsonlMetricLogger):
        raise TypeError(
            f"logger must be JsonlMetricLogger, got {type(logger)!r}"
        )
    for value, name in (
        (checkpoint_writer, "checkpoint_writer"),
        (checkpoint_path_formatter, "checkpoint_path_formatter"),
        (emit, "emit"),
    ):
        if not callable(value):
            raise TypeError(f"{name} must be callable")
    if (
        isinstance(stop_at_step, bool)
        or not isinstance(stop_at_step, int)
        or stop_at_step <= 0
    ):
        raise TrainingLoopError(
            f"stop_at_step must be a positive integer, got {stop_at_step!r}"
        )

    state = trainer.state
    config = trainer.config
    plan = trainer.plan
    state.validate_for_plan(plan)
    if stop_at_step > plan.total_updates:
        raise TrainingLoopError(
            "stop_at_step exceeds the resolved training horizon: "
            f"stop={stop_at_step}, total={plan.total_updates}"
        )
    if stop_at_step <= state.global_step:
        raise TrainingLoopError(
            "stop_at_step must be greater than the restored global step: "
            f"stop={stop_at_step}, restored={state.global_step}"
        )

    try:
        training_iterator = iter(training_batches)
    except TypeError as error:
        raise TrainingLoopError("training_batches must be iterable") from error
    start_step = state.global_step
    evaluation_steps: list[int] = []
    checkpoint_records: list[CheckpointRecord] = []

    while state.global_step < stop_at_step:
        metrics = trainer.run_update(training_iterator)
        logger.log_train_update(
            metrics,
            state=state,
            precision=trainer.precision,
        )
        if (
            state.global_step == 1
            or state.global_step % config.log_interval == 0
            or state.global_step == stop_at_step
        ):
            emit(
                f"train step={state.global_step} "
                f"tokens_seen={state.tokens_seen} "
                f"loss={metrics.raw_token_weighted_loss:.6f} "
                f"lr={metrics.learning_rate:.12g} "
                f"grad_norm={metrics.grad_norm_before_clip:.6f} "
                f"tokens_per_second={metrics.tokens_per_second:.2f}"
            )

        if state.global_step % config.eval_interval == 0:
            evaluation = evaluate_model(
                trainer.model,
                validation_batches,
                precision=trainer.precision,
                global_step=state.global_step,
                max_batches=config.eval_batches,
            )
            state.record_evaluation(evaluation.validation_loss)
            logger.log_evaluation(evaluation, state=state)
            evaluation_steps.append(state.global_step)
            perplexity = (
                f"{evaluation.perplexity:.6f}"
                if math.isfinite(evaluation.perplexity)
                else "inf"
            )
            emit(
                f"eval step={state.global_step} "
                f"tokens={evaluation.evaluated_tokens} "
                f"loss={evaluation.validation_loss:.6f} "
                f"perplexity={perplexity}"
            )

        if state.global_step % config.save_interval == 0:
            checkpoint_records.append(
                _checkpoint_once(
                    trainer=trainer,
                    logger=logger,
                    checkpoint_writer=checkpoint_writer,
                    checkpoint_path_formatter=checkpoint_path_formatter,
                    emit=emit,
                )
            )

    if state.last_save_step != state.global_step:
        checkpoint_records.append(
            _checkpoint_once(
                trainer=trainer,
                logger=logger,
                checkpoint_writer=checkpoint_writer,
                checkpoint_path_formatter=checkpoint_path_formatter,
                emit=emit,
            )
        )

    state.validate_for_plan(plan)
    if state.last_save_step != state.global_step:
        raise TrainingLoopError(
            "training stopped without a checkpoint at the final update boundary"
        )
    return TrainingLoopResult(
        start_step=start_step,
        stop_step=state.global_step,
        updates_completed=state.global_step - start_step,
        evaluation_steps=tuple(evaluation_steps),
        checkpoint_records=tuple(checkpoint_records),
    )
