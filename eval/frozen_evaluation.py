"""Strict single-split evaluation for a completed small-gpt checkpoint."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import torch

from model import GPT, GPTConfig
from train import (
    CheckpointIdentity,
    EvaluationDataStream,
    PrecisionPolicy,
    TrainingConfig,
    build_checkpoint_identity,
    evaluate_model,
    load_model_checkpoint,
)


FROZEN_EVALUATION_FORMAT_NAME = "small_gpt_frozen_split_evaluation"
FROZEN_EVALUATION_SCHEMA_VERSION = 1
_FROZEN_SPLITS = ("validation", "test")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class FrozenEvaluationError(RuntimeError):
    """Raised when a frozen evaluation would violate its evidence contract."""


def _validate_max_batches(max_batches: int | None) -> None:
    if max_batches is None:
        return
    if (
        isinstance(max_batches, bool)
        or not isinstance(max_batches, int)
        or max_batches <= 0
    ):
        raise FrozenEvaluationError(
            "max_batches must be a positive integer or null, "
            f"got {max_batches!r}"
        )


def _normalized_sha256(value: str, *, field: str) -> str:
    if not isinstance(value, str):
        raise FrozenEvaluationError(f"{field} must be a SHA-256 string")
    normalized = value.lower()
    if _SHA256_PATTERN.fullmatch(normalized) is None:
        raise FrozenEvaluationError(f"{field} must be a SHA-256 string")
    return normalized


def _checkpoint_training_config(
    resolved_config: Mapping[str, Any],
) -> TrainingConfig:
    training = resolved_config.get("training")
    if not isinstance(training, Mapping):
        raise FrozenEvaluationError(
            "checkpoint resolved_config.training must be a mapping"
        )
    try:
        return TrainingConfig(**dict(training))
    except Exception as error:
        raise FrozenEvaluationError(
            f"checkpoint training configuration is invalid: {error}"
        ) from error


def _identity_mismatches(
    expected: CheckpointIdentity,
    actual: CheckpointIdentity,
) -> list[str]:
    return [
        field.name
        for field in fields(CheckpointIdentity)
        if getattr(expected, field.name) != getattr(actual, field.name)
    ]


def _validate_evaluation_identity(
    *,
    model_config: GPTConfig,
    manifest_path: Path,
    checkpoint_identity: CheckpointIdentity,
) -> CheckpointIdentity:
    active_identity = build_checkpoint_identity(
        model_config.to_dict(),
        manifest_path,
        source_commit=checkpoint_identity.source_commit,
        source_dirty=checkpoint_identity.source_dirty,
    )
    mismatches = _identity_mismatches(checkpoint_identity, active_identity)
    if mismatches:
        raise FrozenEvaluationError(
            "checkpoint identity does not match the evaluation inputs: "
            f"{mismatches}"
        )
    return active_identity


def _runtime_payload(
    precision: PrecisionPolicy,
    training_config: TrainingConfig,
) -> dict[str, Any]:
    runtime: dict[str, Any] = {
        **precision.to_dict(),
        "torch_version": str(torch.__version__),
        "deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "configured_allow_tf32": training_config.allow_tf32,
    }
    if precision.device.type == "cuda":
        runtime["cuda_device_name"] = torch.cuda.get_device_name(
            precision.device
        )
        runtime["cuda_matmul_allow_tf32"] = (
            torch.backends.cuda.matmul.allow_tf32
        )
        runtime["cudnn_allow_tf32"] = torch.backends.cudnn.allow_tf32
        runtime["cudnn_deterministic"] = torch.backends.cudnn.deterministic
        runtime["cudnn_benchmark"] = torch.backends.cudnn.benchmark
    else:
        runtime["cuda_device_name"] = None
        runtime["cuda_matmul_allow_tf32"] = None
        runtime["cudnn_allow_tf32"] = None
        runtime["cudnn_deterministic"] = None
        runtime["cudnn_benchmark"] = None
    return runtime


@contextmanager
def _configured_evaluation_runtime(
    training_config: TrainingConfig,
    precision: PrecisionPolicy,
) -> Iterator[dict[str, Any]]:
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    warn_only_before = torch.is_deterministic_algorithms_warn_only_enabled()
    cuda_flags_before: tuple[bool, bool, bool, bool] | None = None
    if precision.device.type == "cuda":
        cuda_flags_before = (
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
            torch.backends.cudnn.deterministic,
            torch.backends.cudnn.benchmark,
        )

    try:
        torch.use_deterministic_algorithms(training_config.deterministic)
        if precision.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = training_config.allow_tf32
            torch.backends.cudnn.allow_tf32 = training_config.allow_tf32
            torch.backends.cudnn.deterministic = training_config.deterministic
            torch.backends.cudnn.benchmark = False
        yield _runtime_payload(precision, training_config)
    finally:
        torch.use_deterministic_algorithms(
            deterministic_before,
            warn_only=warn_only_before,
        )
        if cuda_flags_before is not None:
            (
                torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32,
                torch.backends.cudnn.deterministic,
                torch.backends.cudnn.benchmark,
            ) = cuda_flags_before


def _resolved_path(path: str | Path) -> str:
    return Path(path).resolve().as_posix()


def evaluate_frozen_split(
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    *,
    model_config: GPTConfig,
    expected_run_id: str,
    expected_checkpoint_sha256: str,
    split: str,
    precision: PrecisionPolicy,
    max_batches: int | None = None,
    evaluator_source_commit: str | None = None,
    evaluator_source_dirty: bool = True,
) -> dict[str, Any]:
    """Load model weights and evaluate one explicit immutable corpus split."""

    if not isinstance(model_config, GPTConfig):
        raise TypeError(
            f"model_config must be a GPTConfig, got {type(model_config)!r}"
        )
    if not isinstance(precision, PrecisionPolicy):
        raise TypeError(
            f"precision must be a PrecisionPolicy, got {type(precision)!r}"
        )
    if split not in _FROZEN_SPLITS:
        raise FrozenEvaluationError(
            f"split must be one of {_FROZEN_SPLITS}, got {split!r}"
        )
    if not isinstance(expected_run_id, str) or not expected_run_id.strip():
        raise FrozenEvaluationError("expected_run_id must be a non-empty string")
    if evaluator_source_commit is not None and (
        not isinstance(evaluator_source_commit, str)
        or re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", evaluator_source_commit)
        is None
    ):
        raise FrozenEvaluationError(
            "evaluator_source_commit must be a full lowercase Git object ID or null"
        )
    if not isinstance(evaluator_source_dirty, bool):
        raise TypeError("evaluator_source_dirty must be a boolean")
    _validate_max_batches(max_batches)
    expected_checkpoint_hash = _normalized_sha256(
        expected_checkpoint_sha256,
        field="expected_checkpoint_sha256",
    )

    resolved_checkpoint = Path(checkpoint_path).resolve()
    resolved_manifest = Path(manifest_path).resolve()
    try:
        checkpoint_sha256 = sha256_file(resolved_checkpoint)
    except OSError as error:
        raise FrozenEvaluationError(
            f"could not hash checkpoint {resolved_checkpoint}: {error}"
        ) from error
    if checkpoint_sha256 != expected_checkpoint_hash:
        raise FrozenEvaluationError(
            "checkpoint SHA-256 mismatch: "
            f"expected {expected_checkpoint_hash}, found {checkpoint_sha256}"
        )
    cuda_rng_devices = (
        []
        if precision.device.type == "cpu"
        else [precision.device.index]
    )
    with torch.random.fork_rng(devices=cuda_rng_devices):
        model = GPT(model_config).to(precision.device)
    loaded = load_model_checkpoint(
        resolved_checkpoint,
        model=model,
        expected_model_config=model_config.to_dict(),
        expected_run_id=expected_run_id,
    )
    if model.lm_head.weight is not model.token_embedding.weight:
        raise FrozenEvaluationError(
            "model input/output embeddings are not tied after checkpoint load"
        )
    model.eval()

    active_identity = _validate_evaluation_identity(
        model_config=model_config,
        manifest_path=resolved_manifest,
        checkpoint_identity=loaded.identity,
    )
    checkpoint_training = _checkpoint_training_config(loaded.resolved_config)
    try:
        checkpoint_plan = checkpoint_training.resolve()
    except Exception as error:
        raise FrozenEvaluationError(
            f"checkpoint training plan is invalid: {error}"
        ) from error

    with _configured_evaluation_runtime(
        checkpoint_training,
        precision,
    ) as runtime, EvaluationDataStream(
        resolved_manifest,
        split=split,
        plan=checkpoint_plan,
        verify_hashes=True,
    ) as stream:
        metrics = evaluate_model(
            model,
            stream,
            precision=precision,
            global_step=loaded.state.global_step,
            max_batches=max_batches,
        )
        available_batches = len(stream)
        selected_batches = (
            available_batches
            if max_batches is None
            else min(max_batches, available_batches)
        )
        selected_windows = min(
            stream.total_windows,
            selected_batches * checkpoint_plan.micro_batch_size,
        )
        expected_tokens = selected_windows * checkpoint_plan.context_length
        if metrics.evaluated_batches != selected_batches:
            raise FrozenEvaluationError(
                "evaluation batch count does not match the frozen stream"
            )
        if metrics.evaluated_tokens != expected_tokens:
            raise FrozenEvaluationError(
                "evaluation token count does not match the frozen stream"
            )

        is_full_split = (
            metrics.evaluated_batches == available_batches
            and metrics.evaluated_tokens == stream.total_evaluation_tokens
        )
        result = {
            "format_name": FROZEN_EVALUATION_FORMAT_NAME,
            "schema_version": FROZEN_EVALUATION_SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": loaded.state.run_id,
            "split": split,
            "checkpoint": {
                "path": _resolved_path(resolved_checkpoint),
                "bytes": loaded.record.file_size,
                "sha256": checkpoint_sha256,
                "global_step": loaded.record.global_step,
                "tokens_seen": loaded.record.tokens_seen,
                "identity": loaded.identity.to_dict(),
            },
            "evaluator": {
                "source_commit": evaluator_source_commit,
                "source_dirty": evaluator_source_dirty,
            },
            "model": {
                "config": model_config.to_dict(),
                "parameters": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "weight_tying_verified": True,
            },
            "data": {
                "manifest_path": _resolved_path(resolved_manifest),
                "manifest_sha256": active_identity.dataset_manifest_sha256,
                "config_fingerprint": (
                    active_identity.dataset_config_fingerprint
                ),
                "tokenizer_sha256": active_identity.tokenizer_sha256,
                "split_model_tokens": len(stream.store),
                "context_length": checkpoint_plan.context_length,
                "batch_size": checkpoint_plan.micro_batch_size,
                "num_workers": checkpoint_plan.num_workers,
                "pin_memory": checkpoint_plan.pin_memory,
                "window_mode": "sequential_non_overlapping",
                "verify_hashes": True,
            },
            "coverage": {
                "requested_max_batches": max_batches,
                "available_batches": available_batches,
                "total_windows": stream.total_windows,
                "full_evaluation_tokens": stream.total_evaluation_tokens,
                "trailing_tokens_discarded": stream.discarded_tokens,
                "is_full_split": is_full_split,
            },
            "metrics": {
                "loss": metrics.validation_loss,
                "perplexity": (
                    metrics.perplexity
                    if math.isfinite(metrics.perplexity)
                    else None
                ),
                "perplexity_is_finite": math.isfinite(metrics.perplexity),
                "evaluated_batches": metrics.evaluated_batches,
                "evaluated_tokens": metrics.evaluated_tokens,
                "elapsed_seconds": metrics.elapsed_seconds,
            },
            "runtime": runtime,
        }

    return result


def publish_evaluation_result(
    path: str | Path,
    result: Mapping[str, Any],
) -> Path:
    """Atomically publish strict JSON without replacing an existing result."""

    if not isinstance(result, Mapping):
        raise TypeError("result must be a mapping")
    try:
        encoded = json.dumps(
            dict(result),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n"
    except (TypeError, ValueError) as error:
        raise FrozenEvaluationError(
            f"evaluation result is not strict JSON: {error}"
        ) from error

    output_path = Path(path).resolve()
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise FrozenEvaluationError(
            f"could not create evaluation output directory: {error}"
        ) from error
    if output_path.exists():
        raise FrozenEvaluationError(
            f"evaluation output already exists and will not be overwritten: "
            f"{output_path}"
        )

    temporary_path = output_path.parent / (
        f".{output_path.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, output_path)
    except FileExistsError as error:
        raise FrozenEvaluationError(
            f"evaluation output already exists and will not be overwritten: "
            f"{output_path}"
        ) from error
    except OSError as error:
        raise FrozenEvaluationError(
            f"could not atomically publish evaluation result: {error}"
        ) from error
    finally:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
    return output_path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
