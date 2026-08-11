from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import torch
from torch import nn

from .config import ResolvedTrainingPlan
from .scheduler import WarmupCosineScheduler
from .state import TrainerState


CHECKPOINT_FORMAT_NAME = "small_gpt_training_checkpoint"
CHECKPOINT_SCHEMA_VERSION = 1
RNG_STATE_SCHEMA_VERSION = 1
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT_PATTERN = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_CHECKPOINT_FIELDS = frozenset(
    {
        "format_name",
        "schema_version",
        "created_at_utc",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
        "trainer_state",
        "rng_state",
        "resolved_config",
        "identity",
    }
)
_RNG_FIELDS = frozenset(
    {
        "schema_version",
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
    }
)
_NUMPY_RNG_FIELDS = frozenset(
    {
        "algorithm",
        "keys",
        "position",
        "has_gauss",
        "cached_gaussian",
    }
)


class CheckpointError(RuntimeError):
    """Base class for strict checkpoint failures."""


class CheckpointIdentityError(ValueError):
    """Raised when model, tokenizer, dataset, or source identity is invalid."""


class CheckpointSaveError(CheckpointError):
    """Raised when a complete checkpoint cannot be atomically published."""


class CheckpointLoadError(CheckpointError):
    """Raised when a checkpoint cannot be safely decoded or restored."""


class CheckpointCompatibilityError(CheckpointLoadError):
    """Raised before restoration when active and saved identities differ."""


class RngStateError(ValueError):
    """Raised when Python, NumPy, or Torch RNG state is malformed."""


def _is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise CheckpointIdentityError(
            f"{field} must be a lowercase SHA-256 string, got {value!r}"
        )
    return value


def _normalized_json_mapping(
    value: Mapping[str, Any],
    *,
    field: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    try:
        encoded = json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be strict JSON: {error}") from error
    if not isinstance(decoded, dict):
        raise ValueError(f"{field} must encode a JSON object")
    return decoded


def _mapping_sha256(value: Mapping[str, Any], *, field: str) -> str:
    normalized = _normalized_json_mapping(value, field=field)
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class CheckpointIdentity:
    model_config_sha256: str
    tokenizer_sha256: str
    dataset_manifest_sha256: str
    dataset_config_fingerprint: str
    source_commit: str | None
    source_dirty: bool

    def __post_init__(self) -> None:
        for field_name in (
            "model_config_sha256",
            "tokenizer_sha256",
            "dataset_manifest_sha256",
            "dataset_config_fingerprint",
        ):
            _require_sha256(getattr(self, field_name), field_name)
        if self.source_commit is not None and (
            not isinstance(self.source_commit, str)
            or _GIT_COMMIT_PATTERN.fullmatch(self.source_commit) is None
        ):
            raise CheckpointIdentityError(
                "source_commit must be a full lowercase Git object ID or null, "
                f"got {self.source_commit!r}"
            )
        if not isinstance(self.source_dirty, bool):
            raise CheckpointIdentityError("source_dirty must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> CheckpointIdentity:
        if not isinstance(value, Mapping):
            raise CheckpointIdentityError("checkpoint identity must be a mapping")
        expected_fields = {field.name for field in fields(cls)}
        provided_fields = set(value)
        missing_fields = expected_fields - provided_fields
        unknown_fields = provided_fields - expected_fields
        if missing_fields:
            raise CheckpointIdentityError(
                f"checkpoint identity is missing fields: {sorted(missing_fields)}"
            )
        if unknown_fields:
            raise CheckpointIdentityError(
                f"checkpoint identity has unknown fields: {sorted(unknown_fields)}"
            )
        try:
            return cls(**dict(value))
        except TypeError as error:
            raise CheckpointIdentityError(
                f"could not construct checkpoint identity: {error}"
            ) from error


def build_checkpoint_identity(
    model_config: Mapping[str, Any],
    manifest_path: str | Path,
    *,
    source_commit: str | None,
    source_dirty: bool,
) -> CheckpointIdentity:
    """Fingerprint immutable model, tokenizer, dataset, and source inputs."""

    model_config_sha256 = _mapping_sha256(
        model_config,
        field="model_config",
    )
    path = Path(manifest_path).resolve()
    try:
        manifest_bytes = path.read_bytes()
    except OSError as error:
        raise CheckpointIdentityError(
            f"could not read dataset manifest {path}: {error}"
        ) from error
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CheckpointIdentityError(
            f"dataset manifest is not valid UTF-8 JSON: {path}"
        ) from error
    if not isinstance(manifest, dict):
        raise CheckpointIdentityError("dataset manifest must be a JSON object")
    if manifest.get("status") != "complete":
        raise CheckpointIdentityError(
            "dataset manifest status must be 'complete'"
        )
    tokenizer = manifest.get("tokenizer")
    if not isinstance(tokenizer, Mapping):
        raise CheckpointIdentityError(
            "dataset manifest tokenizer must be a mapping"
        )

    return CheckpointIdentity(
        model_config_sha256=model_config_sha256,
        tokenizer_sha256=_require_sha256(
            tokenizer.get("sha256"),
            "manifest.tokenizer.sha256",
        ),
        dataset_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        dataset_config_fingerprint=_require_sha256(
            manifest.get("config_fingerprint"),
            "manifest.config_fingerprint",
        ),
        source_commit=source_commit,
        source_dirty=source_dirty,
    )


@dataclass(frozen=True, slots=True)
class CheckpointRecord:
    path: Path
    file_size: int
    global_step: int
    tokens_seen: int


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    record: CheckpointRecord
    state: TrainerState
    identity: CheckpointIdentity
    resolved_config: dict[str, Any]


def capture_rng_state(*, include_cuda: bool) -> dict[str, Any]:
    """Capture all RNG sources that can affect a Day 7 training continuation."""

    if not isinstance(include_cuda, bool):
        raise TypeError("include_cuda must be a boolean")
    numpy_state = np.random.get_state()
    torch_cuda: list[torch.Tensor] | None = None
    if include_cuda:
        if not torch.cuda.is_available():
            raise RngStateError(
                "CUDA RNG capture was requested, but CUDA is unavailable"
            )
        torch_cuda = [state.cpu().clone() for state in torch.cuda.get_rng_state_all()]
        if not torch_cuda:
            raise RngStateError("CUDA RNG capture produced no device states")

    return {
        "schema_version": RNG_STATE_SCHEMA_VERSION,
        "python": random.getstate(),
        "numpy": {
            "algorithm": str(numpy_state[0]),
            "keys": numpy_state[1].astype(np.uint32, copy=False).tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state().cpu().clone(),
        "torch_cuda": torch_cuda,
    }


def _validated_rng_state(
    state: Mapping[str, Any],
    *,
    require_cuda: bool,
) -> tuple[
    tuple[Any, ...],
    tuple[str, np.ndarray, int, int, float],
    torch.Tensor,
    list[torch.Tensor] | None,
]:
    if not isinstance(state, Mapping):
        raise RngStateError("RNG state must be a mapping")
    provided_fields = set(state)
    missing_fields = _RNG_FIELDS - provided_fields
    unknown_fields = provided_fields - _RNG_FIELDS
    if missing_fields:
        raise RngStateError(f"RNG state is missing fields: {sorted(missing_fields)}")
    if unknown_fields:
        raise RngStateError(
            f"RNG state has unknown fields: {sorted(unknown_fields)}"
        )
    if (
        not _is_plain_int(state["schema_version"])
        or state["schema_version"] != RNG_STATE_SCHEMA_VERSION
    ):
        raise RngStateError(
            "unsupported RNG state schema version: "
            f"{state['schema_version']!r}"
        )

    python_state = state["python"]
    try:
        validator = random.Random()
        validator.setstate(python_state)
    except (TypeError, ValueError) as error:
        raise RngStateError(f"invalid Python RNG state: {error}") from error

    numpy_payload = state["numpy"]
    if not isinstance(numpy_payload, Mapping):
        raise RngStateError("NumPy RNG state must be a mapping")
    numpy_fields = set(numpy_payload)
    if numpy_fields != _NUMPY_RNG_FIELDS:
        raise RngStateError(
            "NumPy RNG state fields do not match the checkpoint contract"
        )
    keys = numpy_payload["keys"]
    if not isinstance(keys, list) or not keys:
        raise RngStateError("NumPy RNG keys must be a non-empty list")
    if not all(
        _is_plain_int(value) and 0 <= value <= np.iinfo(np.uint32).max
        for value in keys
    ):
        raise RngStateError("NumPy RNG keys must contain uint32 integers")
    algorithm = numpy_payload["algorithm"]
    position = numpy_payload["position"]
    has_gauss = numpy_payload["has_gauss"]
    cached_gaussian = numpy_payload["cached_gaussian"]
    if not isinstance(algorithm, str) or not algorithm:
        raise RngStateError("NumPy RNG algorithm must be a non-empty string")
    if not _is_plain_int(position) or position < 0:
        raise RngStateError("NumPy RNG position must be non-negative")
    if not _is_plain_int(has_gauss) or has_gauss not in {0, 1}:
        raise RngStateError("NumPy RNG has_gauss must be 0 or 1")
    if (
        not isinstance(cached_gaussian, (int, float))
        or isinstance(cached_gaussian, bool)
        or not math.isfinite(float(cached_gaussian))
    ):
        raise RngStateError("NumPy cached Gaussian must be finite")
    numpy_state = (
        algorithm,
        np.asarray(keys, dtype=np.uint32),
        int(position),
        int(has_gauss),
        float(cached_gaussian),
    )
    try:
        numpy_validator = np.random.RandomState()
        numpy_validator.set_state(numpy_state)
    except (TypeError, ValueError) as error:
        raise RngStateError(f"invalid NumPy RNG state: {error}") from error

    torch_cpu = state["torch_cpu"]
    if (
        not isinstance(torch_cpu, torch.Tensor)
        or torch_cpu.dtype != torch.uint8
        or torch_cpu.ndim != 1
        or torch_cpu.device.type != "cpu"
        or torch_cpu.numel() == 0
    ):
        raise RngStateError("Torch CPU RNG state must be a non-empty CPU byte tensor")

    raw_cuda = state["torch_cuda"]
    torch_cuda: list[torch.Tensor] | None
    if raw_cuda is None:
        torch_cuda = None
    elif isinstance(raw_cuda, list) and raw_cuda:
        if not all(
            isinstance(value, torch.Tensor)
            and value.dtype == torch.uint8
            and value.ndim == 1
            and value.device.type == "cpu"
            and value.numel() > 0
            for value in raw_cuda
        ):
            raise RngStateError(
                "Torch CUDA RNG states must be non-empty CPU byte tensors"
            )
        torch_cuda = [value.clone() for value in raw_cuda]
    else:
        raise RngStateError("Torch CUDA RNG state must be a non-empty list or null")

    if require_cuda:
        if torch_cuda is None:
            raise RngStateError("CUDA training checkpoint is missing CUDA RNG state")
        if not torch.cuda.is_available():
            raise RngStateError(
                "CUDA training checkpoint cannot restore because CUDA is unavailable"
            )
        visible_devices = torch.cuda.device_count()
        if len(torch_cuda) != visible_devices:
            raise RngStateError(
                "CUDA RNG device count does not match the active runtime: "
                f"checkpoint={len(torch_cuda)}, runtime={visible_devices}"
            )
    elif torch_cuda is not None:
        raise RngStateError(
            "CPU training checkpoint unexpectedly contains CUDA RNG state"
        )

    return (
        python_state,
        numpy_state,
        torch_cpu.clone(),
        torch_cuda,
    )


def _apply_validated_rng_state(
    state: tuple[
        tuple[Any, ...],
        tuple[str, np.ndarray, int, int, float],
        torch.Tensor,
        list[torch.Tensor] | None,
    ],
) -> None:
    python_state, numpy_state, torch_cpu, torch_cuda = state
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_cpu)
    if torch_cuda is not None:
        torch.cuda.set_rng_state_all(torch_cuda)


def restore_rng_state(
    state: Mapping[str, Any],
    *,
    require_cuda: bool,
) -> None:
    """Validate every RNG source before mutating any global generator."""

    validated = _validated_rng_state(state, require_cuda=require_cuda)
    _apply_validated_rng_state(validated)


def _model_device(model: nn.Module) -> torch.device:
    devices = {parameter.device for parameter in model.parameters()}
    if not devices:
        raise CheckpointError("model has no parameters")
    if len(devices) != 1:
        raise CheckpointError(
            f"model parameters span multiple devices: {sorted(map(str, devices))}"
        )
    return next(iter(devices))


def _validate_state_boundary(
    *,
    state: TrainerState,
    plan: ResolvedTrainingPlan,
) -> None:
    state.validate_for_plan(plan)
    if state.data_epoch != 0:
        raise CheckpointError("Day 7 checkpoint requires data_epoch=0")
    if state.batches_consumed_in_epoch != state.micro_steps_seen:
        raise CheckpointError(
            "checkpoint data cursor does not end on a committed update boundary"
        )


def _validate_boundary(
    *,
    state: TrainerState,
    plan: ResolvedTrainingPlan,
    scheduler: WarmupCosineScheduler,
) -> None:
    _validate_state_boundary(state=state, plan=plan)
    if scheduler.next_update_index != state.global_step:
        raise CheckpointError(
            "scheduler and TrainerState disagree at checkpoint boundary: "
            f"scheduler={scheduler.next_update_index}, state={state.global_step}"
        )


def _resolved_config_snapshot(
    value: Mapping[str, Any],
    *,
    plan: ResolvedTrainingPlan,
) -> dict[str, Any]:
    normalized = _normalized_json_mapping(value, field="resolved_config")
    if normalized.get("plan") != plan.to_dict():
        raise ValueError(
            "resolved_config.plan does not match the active training plan"
        )
    return normalized


def _write_atomic_checkpoint(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise CheckpointSaveError(
            f"could not create checkpoint directory {path.parent}: {error}"
        ) from error
    if path.exists() and not path.is_file():
        raise CheckpointSaveError(
            f"checkpoint target exists and is not a file: {path}"
        )

    temporary_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary_path.open("xb") as handle:
            typed_handle: BinaryIO = handle
            torch.save(dict(payload), typed_handle)
            typed_handle.flush()
            os.fsync(typed_handle.fileno())
        os.replace(temporary_path, path)
    except Exception as error:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        if isinstance(error, CheckpointSaveError):
            raise
        raise CheckpointSaveError(
            f"could not atomically publish checkpoint {path}: {error}"
        ) from error


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    state: TrainerState,
    plan: ResolvedTrainingPlan,
    resolved_config: Mapping[str, Any],
    identity: CheckpointIdentity,
) -> CheckpointRecord:
    """Atomically save one complete optimizer-boundary training checkpoint."""

    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model)!r}")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch.optim.Optimizer")
    if not isinstance(scheduler, WarmupCosineScheduler):
        raise TypeError("scheduler must be a WarmupCosineScheduler")
    if not isinstance(state, TrainerState):
        raise TypeError("state must be a TrainerState")
    if not isinstance(plan, ResolvedTrainingPlan):
        raise TypeError("plan must be a ResolvedTrainingPlan")
    if not isinstance(identity, CheckpointIdentity):
        raise TypeError("identity must be a CheckpointIdentity")
    if scheduler.optimizer is not optimizer:
        raise CheckpointSaveError(
            "scheduler and checkpoint optimizer must be the same object"
        )

    try:
        _validate_boundary(state=state, plan=plan, scheduler=scheduler)
        config_snapshot = _resolved_config_snapshot(
            resolved_config,
            plan=plan,
        )
        device = _model_device(model)
    except (CheckpointError, TypeError, ValueError) as error:
        if isinstance(error, CheckpointSaveError):
            raise
        raise CheckpointSaveError(
            f"checkpoint preflight failed: {error}"
        ) from error

    checkpoint_state = TrainerState.from_state_dict(state.state_dict())
    checkpoint_state.record_checkpoint()
    try:
        rng_state = capture_rng_state(include_cuda=device.type == "cuda")
    except RngStateError as error:
        raise CheckpointSaveError(f"could not capture RNG state: {error}") from error

    payload = {
        "format_name": CHECKPOINT_FORMAT_NAME,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": None,
        "trainer_state": checkpoint_state.state_dict(),
        "rng_state": rng_state,
        "resolved_config": config_snapshot,
        "identity": identity.to_dict(),
    }
    final_path = Path(path).resolve()
    _write_atomic_checkpoint(final_path, payload)
    state.record_checkpoint()
    return CheckpointRecord(
        path=final_path,
        file_size=final_path.stat().st_size,
        global_step=state.global_step,
        tokens_seen=state.tokens_seen,
    )


def _read_checkpoint_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CheckpointLoadError(f"checkpoint file does not exist: {path}")
    try:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as error:
        raise CheckpointLoadError(
            f"could not decode a valid checkpoint from {path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise CheckpointLoadError("checkpoint root must be a mapping")
    provided_fields = set(payload)
    missing_fields = _CHECKPOINT_FIELDS - provided_fields
    unknown_fields = provided_fields - _CHECKPOINT_FIELDS
    if missing_fields:
        raise CheckpointLoadError(
            f"checkpoint is missing fields: {sorted(missing_fields)}"
        )
    if unknown_fields:
        raise CheckpointLoadError(
            f"checkpoint has unknown fields: {sorted(unknown_fields)}"
        )
    if payload["format_name"] != CHECKPOINT_FORMAT_NAME:
        raise CheckpointLoadError(
            f"unsupported checkpoint format: {payload['format_name']!r}"
        )
    if (
        not _is_plain_int(payload["schema_version"])
        or payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION
    ):
        raise CheckpointLoadError(
            "unsupported checkpoint schema version: "
            f"{payload['schema_version']!r}"
        )
    created_at = payload["created_at_utc"]
    if not isinstance(created_at, str):
        raise CheckpointLoadError("checkpoint created_at_utc must be a string")
    try:
        datetime.fromisoformat(created_at)
    except ValueError as error:
        raise CheckpointLoadError(
            "checkpoint created_at_utc is not a valid ISO timestamp"
        ) from error
    return payload


def _identity_mismatches(
    expected: CheckpointIdentity,
    actual: CheckpointIdentity,
) -> list[str]:
    return [
        field.name
        for field in fields(CheckpointIdentity)
        if getattr(expected, field.name) != getattr(actual, field.name)
    ]


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    plan: ResolvedTrainingPlan,
    resolved_config: Mapping[str, Any],
    identity: CheckpointIdentity,
    expected_run_id: str,
) -> LoadedCheckpoint:
    """Strictly validate then restore all trainable and stochastic state."""

    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model)!r}")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch.optim.Optimizer")
    if not isinstance(scheduler, WarmupCosineScheduler):
        raise TypeError("scheduler must be a WarmupCosineScheduler")
    if not isinstance(plan, ResolvedTrainingPlan):
        raise TypeError("plan must be a ResolvedTrainingPlan")
    if not isinstance(identity, CheckpointIdentity):
        raise TypeError("identity must be a CheckpointIdentity")
    if not isinstance(expected_run_id, str) or not expected_run_id.strip():
        raise TypeError("expected_run_id must be a non-empty string")
    if scheduler.optimizer is not optimizer:
        raise CheckpointLoadError(
            "scheduler and checkpoint optimizer must be the same object"
        )

    try:
        active_config = _resolved_config_snapshot(
            resolved_config,
            plan=plan,
        )
        device = _model_device(model)
    except (CheckpointError, TypeError, ValueError) as error:
        raise CheckpointLoadError(
            f"checkpoint restore preflight failed: {error}"
        ) from error

    checkpoint_path = Path(path).resolve()
    payload = _read_checkpoint_payload(checkpoint_path)
    try:
        saved_identity = CheckpointIdentity.from_mapping(payload["identity"])
    except CheckpointIdentityError as error:
        raise CheckpointLoadError(
            f"checkpoint identity is invalid: {error}"
        ) from error
    mismatches = _identity_mismatches(identity, saved_identity)
    if mismatches:
        raise CheckpointCompatibilityError(
            f"checkpoint identity does not match active inputs: {mismatches}"
        )

    try:
        saved_config = _normalized_json_mapping(
            payload["resolved_config"],
            field="checkpoint resolved_config",
        )
    except (TypeError, ValueError) as error:
        raise CheckpointLoadError(
            f"checkpoint resolved config is invalid: {error}"
        ) from error
    if saved_config != active_config:
        raise CheckpointCompatibilityError(
            "checkpoint resolved config does not match the active execution"
        )

    try:
        restored_state = TrainerState.from_state_dict(payload["trainer_state"])
        _validate_state_boundary(
            state=restored_state,
            plan=plan,
        )
    except Exception as error:
        raise CheckpointLoadError(
            f"checkpoint TrainerState is invalid: {error}"
        ) from error
    if restored_state.run_id != expected_run_id:
        raise CheckpointCompatibilityError(
            "checkpoint run_id does not match the requested run: "
            f"checkpoint={restored_state.run_id!r}, "
            f"requested={expected_run_id!r}"
        )
    if restored_state.last_save_step != restored_state.global_step:
        raise CheckpointLoadError(
            "checkpoint TrainerState was not committed at its saved step"
        )

    scheduler_state = payload["scheduler_state_dict"]
    expected_last_update = (
        None if restored_state.global_step == 0 else restored_state.global_step - 1
    )
    if not isinstance(scheduler_state, Mapping):
        raise CheckpointLoadError("checkpoint scheduler state must be a mapping")
    if scheduler_state.get("last_applied_update") != expected_last_update:
        raise CheckpointLoadError(
            "checkpoint scheduler state does not match TrainerState global_step"
        )
    if payload["scaler_state_dict"] is not None:
        raise CheckpointLoadError(
            "Day 7 FP32/BF16 checkpoint must contain a null scaler state"
        )
    if not isinstance(payload["model_state_dict"], Mapping):
        raise CheckpointLoadError("checkpoint model state must be a mapping")
    if not isinstance(payload["optimizer_state_dict"], Mapping):
        raise CheckpointLoadError("checkpoint optimizer state must be a mapping")
    try:
        validated_rng = _validated_rng_state(
            payload["rng_state"],
            require_cuda=device.type == "cuda",
        )
    except RngStateError as error:
        raise CheckpointLoadError(
            f"checkpoint RNG state is invalid: {error}"
        ) from error

    try:
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(scheduler_state)
    except Exception as error:
        raise CheckpointLoadError(
            f"checkpoint trainable state could not be restored: {error}"
        ) from error
    if scheduler.next_update_index != restored_state.global_step:
        raise CheckpointLoadError(
            "restored scheduler does not point at TrainerState global_step"
        )
    _apply_validated_rng_state(validated_rng)

    record = CheckpointRecord(
        path=checkpoint_path,
        file_size=checkpoint_path.stat().st_size,
        global_step=restored_state.global_step,
        tokens_seen=restored_state.tokens_seen,
    )
    return LoadedCheckpoint(
        record=record,
        state=restored_state,
        identity=saved_identity,
        resolved_config=saved_config,
    )
