from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import yaml

from .evaluation import EvaluationMetrics
from .precision import PrecisionPolicy
from .state import TrainerState
from .trainer import UpdateMetrics


RUN_METADATA_SCHEMA_VERSION = 1
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class RunDirectoryError(RuntimeError):
    """Raised when a run directory cannot be created without overwriting data."""


class MetricLoggingError(RuntimeError):
    """Raised when a JSONL event violates the logging contract."""


@dataclass(frozen=True, slots=True)
class RunPaths:
    run_id: str
    run_dir: Path
    resolved_config_path: Path
    metadata_path: Path
    metrics_path: Path


def validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise RunDirectoryError(
            "run_id must match [A-Za-z0-9][A-Za-z0-9._-]{0,127}, "
            f"got {run_id!r}"
        )
    if run_id in {".", ".."}:
        raise RunDirectoryError(f"run_id is not safe: {run_id!r}")
    return run_id


def initialize_run_directory(
    runs_root: str | Path,
    *,
    run_id: str,
    resolved_config: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> RunPaths:
    """Create a new run directory and refuse every overwrite attempt."""

    run_id = validate_run_id(run_id)
    if not isinstance(resolved_config, Mapping):
        raise TypeError("resolved_config must be a mapping")
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")

    resolved_payload = dict(resolved_config)
    metadata_payload = dict(metadata)
    for field, expected in (
        ("schema_version", RUN_METADATA_SCHEMA_VERSION),
        ("run_id", run_id),
    ):
        if field in metadata_payload and metadata_payload[field] != expected:
            raise RunDirectoryError(
                f"metadata {field} conflicts with the run directory identity"
            )
        metadata_payload[field] = expected

    try:
        resolved_text = yaml.safe_dump(
            resolved_payload,
            sort_keys=True,
            allow_unicode=True,
        )
        metadata_text = json.dumps(
            metadata_payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ) + "\n"
    except (TypeError, ValueError, yaml.YAMLError) as error:
        raise RunDirectoryError(
            f"run metadata is not safely serializable: {error}"
        ) from error

    root = Path(runs_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / run_id
    if run_dir.exists():
        raise RunDirectoryError(
            f"run directory already exists and will not be overwritten: {run_dir}"
        )

    try:
        run_dir.mkdir(parents=False, exist_ok=False)
        resolved_config_path = run_dir / "resolved-config.yaml"
        metadata_path = run_dir / "run-metadata.json"
        metrics_path = run_dir / "metrics.jsonl"
        resolved_config_path.write_text(
            resolved_text,
            encoding="utf-8",
            newline="\n",
        )
        metadata_path.write_text(
            metadata_text,
            encoding="utf-8",
            newline="\n",
        )
        metrics_path.write_text("", encoding="utf-8", newline="\n")
    except OSError as error:
        raise RunDirectoryError(
            f"could not initialize run directory {run_dir}: {error}"
        ) from error

    return RunPaths(
        run_id=run_id,
        run_dir=run_dir,
        resolved_config_path=resolved_config_path,
        metadata_path=metadata_path,
        metrics_path=metrics_path,
    )


class JsonlMetricLogger:
    """Flush one strict JSON object per metrics line."""

    def __init__(self, path: str | Path, *, append: bool = True) -> None:
        if not isinstance(append, bool):
            raise TypeError("append must be a boolean")
        self.path = Path(path).resolve()
        if append and not self.path.is_file():
            raise MetricLoggingError(
                f"metrics file does not exist for append: {self.path}"
            )
        mode = "a" if append else "x"
        try:
            self._file: TextIO | None = self.path.open(
                mode,
                encoding="utf-8",
                newline="\n",
            )
        except OSError as error:
            raise MetricLoggingError(
                f"could not open metrics file {self.path}: {error}"
            ) from error

    @property
    def is_closed(self) -> bool:
        return self._file is None

    def write_event(self, event: Mapping[str, Any]) -> None:
        if self._file is None:
            raise MetricLoggingError("metrics logger is closed")
        if not isinstance(event, Mapping):
            raise MetricLoggingError("metrics event must be a mapping")
        event_name = event.get("event")
        if not isinstance(event_name, str) or not event_name:
            raise MetricLoggingError(
                "metrics event must contain a non-empty event name"
            )
        try:
            line = json.dumps(
                dict(event),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as error:
            raise MetricLoggingError(
                f"metrics event is not strict JSON: {error}"
            ) from error
        try:
            self._file.write(line + "\n")
            self._file.flush()
        except OSError as error:
            raise MetricLoggingError(
                f"could not write metrics event to {self.path}: {error}"
            ) from error

    def log_train_update(
        self,
        metrics: UpdateMetrics,
        *,
        state: TrainerState,
        precision: PrecisionPolicy,
    ) -> None:
        if not isinstance(metrics, UpdateMetrics):
            raise TypeError(
                f"metrics must be UpdateMetrics, got {type(metrics)!r}"
            )
        if not isinstance(state, TrainerState):
            raise TypeError(
                f"state must be TrainerState, got {type(state)!r}"
            )
        if not isinstance(precision, PrecisionPolicy):
            raise TypeError(
                "precision must be PrecisionPolicy, "
                f"got {type(precision)!r}"
            )
        state.validate()
        if metrics.completed_global_step != state.global_step:
            raise MetricLoggingError(
                "train metrics step does not match TrainerState"
            )

        self.write_event(
            {
                "event": "train_update",
                "step": state.global_step,
                "tokens_seen": state.tokens_seen,
                "train_loss": metrics.raw_token_weighted_loss,
                "learning_rate": metrics.learning_rate,
                "grad_norm": metrics.grad_norm_before_clip,
                "micro_steps": metrics.micro_steps,
                "samples": metrics.samples,
                "tokens": metrics.tokens,
                "elapsed_seconds": metrics.elapsed_seconds,
                "tokens_per_second": metrics.tokens_per_second,
                "device": str(precision.device),
                "precision": precision.precision,
            }
        )

    def log_evaluation(
        self,
        metrics: EvaluationMetrics,
        *,
        state: TrainerState,
    ) -> None:
        if not isinstance(metrics, EvaluationMetrics):
            raise TypeError(
                f"metrics must be EvaluationMetrics, got {type(metrics)!r}"
            )
        if not isinstance(state, TrainerState):
            raise TypeError(
                f"state must be TrainerState, got {type(state)!r}"
            )
        state.validate()
        if metrics.global_step != state.global_step:
            raise MetricLoggingError(
                "evaluation metrics step does not match TrainerState"
            )
        if state.last_eval_step != state.global_step:
            raise MetricLoggingError(
                "TrainerState must record evaluation before logging it"
            )
        if state.last_validation_loss != metrics.validation_loss:
            raise MetricLoggingError(
                "evaluation loss does not match TrainerState"
            )

        self.write_event(
            {
                "event": "evaluation",
                "step": state.global_step,
                "tokens_seen": state.tokens_seen,
                "validation_loss": metrics.validation_loss,
                "perplexity": (
                    metrics.perplexity
                    if math.isfinite(metrics.perplexity)
                    else None
                ),
                "evaluated_tokens": metrics.evaluated_tokens,
                "evaluated_batches": metrics.evaluated_batches,
                "elapsed_seconds": metrics.elapsed_seconds,
            }
        )

    def close(self) -> None:
        if self._file is None:
            return
        self._file.close()
        self._file = None

    def __enter__(self) -> JsonlMetricLogger:
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()
