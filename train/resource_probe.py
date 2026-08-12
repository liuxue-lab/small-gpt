from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from .config import ResolvedTrainingPlan, TrainingConfig


RESOURCE_PROBE_SCHEMA_VERSION = 1
RESOURCE_FIELDS = (
    "micro_batch_size",
    "gradient_accumulation_steps",
    "num_workers",
    "pin_memory",
)
RESULT_STATUSES = frozenset({"ok", "oom", "error", "timeout"})


class ResourceProbeError(RuntimeError):
    """Raised when a resource probe would violate its strict contract."""


def _is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _positive_int(value: object, field: str) -> int:
    if not _is_plain_int(value) or int(value) <= 0:
        raise ResourceProbeError(
            f"{field} must be a positive integer, got {value!r}"
        )
    return int(value)


def _non_negative_int(value: object, field: str) -> int:
    if not _is_plain_int(value) or int(value) < 0:
        raise ResourceProbeError(
            f"{field} must be a non-negative integer, got {value!r}"
        )
    return int(value)


def _finite_float(value: object, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ResourceProbeError(f"{field} must be finite, got {value!r}")
    return float(value)


def _utc_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ResourceProbeError(f"{field} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True, order=True)
class ProbeCandidate:
    """One isolated execution-resource tuple."""

    micro_batch_size: int
    gradient_accumulation_steps: int
    num_workers: int
    pin_memory: bool

    def __post_init__(self) -> None:
        _positive_int(self.micro_batch_size, "micro_batch_size")
        _positive_int(
            self.gradient_accumulation_steps,
            "gradient_accumulation_steps",
        )
        _non_negative_int(self.num_workers, "num_workers")
        if not isinstance(self.pin_memory, bool):
            raise ResourceProbeError(
                f"pin_memory must be a boolean, got {self.pin_memory!r}"
            )

    @property
    def key(self) -> str:
        pin = 1 if self.pin_memory else 0
        return (
            f"b{self.micro_batch_size}"
            f"-a{self.gradient_accumulation_steps}"
            f"-w{self.num_workers}"
            f"-p{pin}"
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ProbeCandidate:
        if not isinstance(payload, Mapping):
            raise ResourceProbeError("candidate must be a mapping")
        expected = set(RESOURCE_FIELDS)
        provided = set(payload)
        if provided != expected:
            raise ResourceProbeError(
                "candidate fields must be exactly "
                f"{sorted(expected)}, got {sorted(provided)}"
            )
        return cls(
            micro_batch_size=payload["micro_batch_size"],
            gradient_accumulation_steps=payload[
                "gradient_accumulation_steps"
            ],
            num_workers=payload["num_workers"],
            pin_memory=payload["pin_memory"],
        )


@dataclass(frozen=True, slots=True)
class ProbeSettings:
    """Bounded work and safety limits shared by every candidate process."""

    warmup_updates: int
    measured_updates: int
    candidate_timeout_seconds: int
    max_reserved_fraction: float
    verify_hashes: bool = False

    def __post_init__(self) -> None:
        _positive_int(self.warmup_updates, "warmup_updates")
        _positive_int(self.measured_updates, "measured_updates")
        _positive_int(
            self.candidate_timeout_seconds,
            "candidate_timeout_seconds",
        )
        fraction = _finite_float(
            self.max_reserved_fraction,
            "max_reserved_fraction",
        )
        if not 0.0 < fraction < 1.0:
            raise ResourceProbeError(
                "max_reserved_fraction must be in (0, 1)"
            )
        if not isinstance(self.verify_hashes, bool):
            raise ResourceProbeError("verify_hashes must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AccumulationProposal:
    """Arithmetic proposal only; it is not an automatically frozen setting."""

    micro_batch_size: int
    context_length: int
    requested_tokens_per_update: int
    gradient_accumulation_steps: int
    tokens_per_micro_step: int
    tokens_per_update: int
    proposal_overshoot_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def unique_candidates(
    candidates: Iterable[ProbeCandidate],
) -> tuple[ProbeCandidate, ...]:
    result: list[ProbeCandidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, ProbeCandidate):
            raise TypeError(
                "candidates must contain ProbeCandidate instances, "
                f"got {type(candidate)!r}"
            )
        if candidate.key in seen:
            raise ResourceProbeError(
                f"duplicate probe candidate {candidate.key!r}"
            )
        seen.add(candidate.key)
        result.append(candidate)
    if not result:
        raise ResourceProbeError("at least one probe candidate is required")
    return tuple(result)


def build_micro_batch_candidates(
    micro_batch_sizes: Sequence[int],
    *,
    gradient_accumulation_steps: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[ProbeCandidate, ...]:
    if isinstance(micro_batch_sizes, (str, bytes)):
        raise ResourceProbeError("micro_batch_sizes must be an integer sequence")
    candidates = (
        ProbeCandidate(
            micro_batch_size=value,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        for value in micro_batch_sizes
    )
    return unique_candidates(candidates)


def build_loader_candidates(
    *,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    num_workers_values: Sequence[int],
    pin_memory_values: Sequence[bool],
) -> tuple[ProbeCandidate, ...]:
    if isinstance(num_workers_values, (str, bytes)):
        raise ResourceProbeError("num_workers_values must be an integer sequence")
    if isinstance(pin_memory_values, (str, bytes)):
        raise ResourceProbeError("pin_memory_values must be a boolean sequence")
    candidates = (
        ProbeCandidate(
            micro_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_workers=workers,
            pin_memory=pin_memory,
        )
        for workers in num_workers_values
        for pin_memory in pin_memory_values
    )
    return unique_candidates(candidates)


def resolve_candidate_config(
    base_config: TrainingConfig,
    candidate: ProbeCandidate,
) -> tuple[TrainingConfig, ResolvedTrainingPlan]:
    """Resolve only the four resource fields without mutating the baseline."""

    if not isinstance(base_config, TrainingConfig):
        raise TypeError(
            f"base_config must be TrainingConfig, got {type(base_config)!r}"
        )
    if not isinstance(candidate, ProbeCandidate):
        raise TypeError(
            f"candidate must be ProbeCandidate, got {type(candidate)!r}"
        )
    if base_config.device != "cuda":
        raise ResourceProbeError(
            "baseline resource probing requires training.device='cuda'"
        )
    if base_config.precision != "bf16":
        raise ResourceProbeError(
            "baseline resource probing requires training.precision='bf16'"
        )

    resolved_config = replace(base_config, **candidate.to_dict())
    before = base_config.to_dict()
    after = resolved_config.to_dict()
    changed = {
        field
        for field in before
        if before[field] != after[field]
    }
    if not changed.issubset(set(RESOURCE_FIELDS)):
        raise ResourceProbeError(
            "candidate resolution changed non-resource fields: "
            f"{sorted(changed - set(RESOURCE_FIELDS))}"
        )
    return resolved_config, resolved_config.resolve()


def propose_accumulation(
    *,
    micro_batch_size: int,
    context_length: int,
    requested_tokens_per_update: int,
) -> AccumulationProposal:
    micro_batch_size = _positive_int(
        micro_batch_size,
        "micro_batch_size",
    )
    context_length = _positive_int(context_length, "context_length")
    requested_tokens_per_update = _positive_int(
        requested_tokens_per_update,
        "requested_tokens_per_update",
    )
    tokens_per_micro_step = micro_batch_size * context_length
    accumulation = math.ceil(
        requested_tokens_per_update / tokens_per_micro_step
    )
    tokens_per_update = tokens_per_micro_step * accumulation
    return AccumulationProposal(
        micro_batch_size=micro_batch_size,
        context_length=context_length,
        requested_tokens_per_update=requested_tokens_per_update,
        gradient_accumulation_steps=accumulation,
        tokens_per_micro_step=tokens_per_micro_step,
        tokens_per_update=tokens_per_update,
        proposal_overshoot_tokens=(
            tokens_per_update - requested_tokens_per_update
        ),
    )


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    try:
        with source.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ResourceProbeError(f"could not hash {source}: {error}") from error
    return digest.hexdigest()


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    if not isinstance(payload, Mapping):
        raise TypeError(f"payload must be a mapping, got {type(payload)!r}")
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Durably replace one JSON report without exposing a partial file."""

    if not isinstance(payload, Mapping):
        raise TypeError(f"payload must be a mapping, got {type(payload)!r}")
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as file:
            json.dump(
                payload,
                file,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
    except (OSError, TypeError, ValueError) as error:
        raise ResourceProbeError(
            f"could not atomically write JSON report {destination}: {error}"
        ) from error
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
    return destination


def load_json_object(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ResourceProbeError(
            f"could not read valid JSON object {source}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise ResourceProbeError(f"JSON report must be an object: {source}")
    return payload


def validate_candidate_result(
    payload: Mapping[str, Any],
    *,
    expected_candidate: ProbeCandidate | None = None,
) -> dict[str, Any]:
    """Validate fields the parent process needs before trusting a worker file."""

    if not isinstance(payload, Mapping):
        raise ResourceProbeError("candidate result must be a mapping")
    if payload.get("schema_version") != RESOURCE_PROBE_SCHEMA_VERSION:
        raise ResourceProbeError(
            "candidate result has an unsupported schema_version"
        )
    candidate = ProbeCandidate.from_mapping(payload.get("candidate", {}))
    if expected_candidate is not None and candidate != expected_candidate:
        raise ResourceProbeError(
            "worker result candidate does not match the requested candidate"
        )
    if payload.get("candidate_key") != candidate.key:
        raise ResourceProbeError("candidate_key does not match candidate fields")
    status = payload.get("status")
    if status not in RESULT_STATUSES:
        raise ResourceProbeError(
            f"candidate result status must be one of {sorted(RESULT_STATUSES)}"
        )
    _utc_text(payload.get("started_at_utc"), "started_at_utc")
    _utc_text(payload.get("finished_at_utc"), "finished_at_utc")
    if status == "ok":
        measured = _positive_int(
            payload.get("measured_updates_completed"),
            "measured_updates_completed",
        )
        tokens = _positive_int(payload.get("measured_tokens"), "measured_tokens")
        elapsed = _finite_float(payload.get("elapsed_seconds"), "elapsed_seconds")
        throughput = _finite_float(
            payload.get("tokens_per_second"),
            "tokens_per_second",
        )
        if measured <= 0 or tokens <= 0 or elapsed <= 0.0 or throughput <= 0.0:
            raise ResourceProbeError("successful candidate metrics must be positive")
        total_memory = _positive_int(
            payload.get("total_device_memory_bytes"),
            "total_device_memory_bytes",
        )
        peak_reserved = _non_negative_int(
            payload.get("peak_reserved_bytes"),
            "peak_reserved_bytes",
        )
        fraction = _finite_float(
            payload.get("peak_reserved_fraction"),
            "peak_reserved_fraction",
        )
        if not 0.0 <= fraction <= 1.0:
            raise ResourceProbeError(
                "peak_reserved_fraction must be in [0, 1]"
            )
        expected_fraction = peak_reserved / total_memory
        if not math.isclose(fraction, expected_fraction, rel_tol=1.0e-9):
            raise ResourceProbeError(
                "peak_reserved_fraction does not match memory byte counts"
            )
    else:
        _utc_text(payload.get("error_type"), "error_type")
        _utc_text(payload.get("error_message"), "error_message")
    return dict(payload)


def _successful_results(
    results: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    successful: list[dict[str, Any]] = []
    for payload in results:
        validated = validate_candidate_result(payload)
        if validated["status"] == "ok":
            successful.append(validated)
    return successful


def recommend_micro_batch(
    results: Iterable[Mapping[str, Any]],
    *,
    max_reserved_fraction: float,
) -> dict[str, Any] | None:
    """Return the largest measured candidate that preserves memory headroom."""

    limit = _finite_float(
        max_reserved_fraction,
        "max_reserved_fraction",
    )
    if not 0.0 < limit < 1.0:
        raise ResourceProbeError(
            "max_reserved_fraction must be in (0, 1)"
        )
    eligible = [
        result
        for result in _successful_results(results)
        if float(result["peak_reserved_fraction"]) <= limit
    ]
    if not eligible:
        return None
    selected = max(
        eligible,
        key=lambda result: (
            ProbeCandidate.from_mapping(result["candidate"]).micro_batch_size,
            float(result["tokens_per_second"]),
        ),
    )
    candidate = ProbeCandidate.from_mapping(selected["candidate"])
    return {
        "status": "preliminary",
        "candidate_key": candidate.key,
        "candidate": candidate.to_dict(),
        "peak_reserved_fraction": selected["peak_reserved_fraction"],
        "tokens_per_second": selected["tokens_per_second"],
        "max_reserved_fraction": limit,
        "reason": (
            "largest successful measured micro batch within the configured "
            "reserved-memory safety limit; requires the later stability gate"
        ),
    }


def recommend_loader(
    results: Iterable[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Return the fastest successful measured workers/pin-memory tuple."""

    successful = _successful_results(results)
    if not successful:
        return None
    selected = max(
        successful,
        key=lambda result: float(result["tokens_per_second"]),
    )
    candidate = ProbeCandidate.from_mapping(selected["candidate"])
    return {
        "status": "preliminary",
        "candidate_key": candidate.key,
        "candidate": candidate.to_dict(),
        "tokens_per_second": selected["tokens_per_second"],
        "reason": (
            "highest measured end-to-end throughput among successful loader "
            "candidates; requires repeat measurements and resume validation"
        ),
    }


def is_cuda_oom(error: BaseException | str) -> bool:
    text = str(error).lower()
    markers = (
        "cuda out of memory",
        "cuda error: out of memory",
        "cudaerror_memoryallocation",
        "cublas_status_alloc_failed",
    )
    return any(marker in text for marker in markers)


def tail_text(value: object, *, limit: int = 4_000) -> str:
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ResourceProbeError("tail limit must be a positive integer")
    text = "" if value is None else str(value)
    return text[-limit:]
