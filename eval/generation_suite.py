"""Frozen multi-prompt generation protocol and JSONL evidence bundle."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from model import GPTConfig
from train import PrecisionPolicy

from .frozen_evaluation import sha256_file
from .generation import (
    GenerationError,
    GenerationSettings,
    generate_with_session,
    load_generation_session,
)


GENERATION_PROTOCOL_FORMAT_NAME = "small_gpt_generation_protocol"
GENERATION_PROTOCOL_SCHEMA_VERSION = 1
GENERATION_SUITE_FORMAT_NAME = "small_gpt_generation_suite"
GENERATION_SAMPLE_FORMAT_NAME = "small_gpt_generation_sample"
GENERATION_SUITE_SCHEMA_VERSION = 1
GENERATION_SAMPLES_FILENAME = "samples.jsonl"
GENERATION_MANIFEST_FILENAME = "manifest.json"

_ORDERING = "prompt_then_decoding"
_ID_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_GIT_OBJECT_PATTERN = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_REQUIRED_ROLES = frozenset(
    {
        "greedy",
        "sample_temperature_1",
        "lower_temperature",
        "top_k",
        "top_p",
    }
)
_MAX_SEED = (1 << 63) - 1


class GenerationSuiteError(GenerationError):
    """Raised when a generation suite violates its frozen contract."""


@dataclass(frozen=True, slots=True)
class PromptSpec:
    prompt_id: str
    text: str

    def to_dict(self) -> dict[str, str]:
        return {"id": self.prompt_id, "text": self.text}


@dataclass(frozen=True, slots=True)
class DecodingSpec:
    decoding_id: str
    role: str
    settings: GenerationSettings

    def to_dict(self) -> dict[str, Any]:
        settings = self.settings.to_dict()
        settings.pop("max_new_tokens")
        return {
            "id": self.decoding_id,
            "role": self.role,
            **settings,
        }


@dataclass(frozen=True, slots=True)
class FrozenGenerationProtocol:
    protocol_id: str
    ordering: str
    max_new_tokens: int
    stochastic_seed: int
    prompts: tuple[PromptSpec, ...]
    decodings: tuple[DecodingSpec, ...]

    @property
    def sample_count(self) -> int:
        return len(self.prompts) * len(self.decodings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_name": GENERATION_PROTOCOL_FORMAT_NAME,
            "schema_version": GENERATION_PROTOCOL_SCHEMA_VERSION,
            "protocol_id": self.protocol_id,
            "ordering": self.ordering,
            "max_new_tokens": self.max_new_tokens,
            "stochastic_seed": self.stochastic_seed,
            "prompts": [prompt.to_dict() for prompt in self.prompts],
            "decodings": [decoding.to_dict() for decoding in self.decodings],
        }


def _is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _as_mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GenerationSuiteError(f"{field} must be a mapping")
    return value


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    field: str,
) -> None:
    provided = set(value)
    missing = expected - provided
    unknown = provided - expected
    if missing:
        raise GenerationSuiteError(
            f"{field} is missing fields: {sorted(missing)}"
        )
    if unknown:
        raise GenerationSuiteError(
            f"{field} has unknown fields: {sorted(unknown)}"
        )


def _require_id(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _ID_PATTERN.fullmatch(value) is None:
        raise GenerationSuiteError(
            f"{field} must be a lowercase kebab-case identifier"
        )
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GenerationSuiteError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _parse_prompt(value: object, *, index: int) -> PromptSpec:
    field = f"prompts[{index}]"
    prompt = _as_mapping(value, field=field)
    _require_exact_fields(prompt, {"id", "text"}, field=field)
    prompt_id = _require_id(prompt["id"], field=f"{field}.id")
    text = prompt["text"]
    if not isinstance(text, str) or not text.strip():
        raise GenerationSuiteError(f"{field}.text must be non-empty")
    if len(text) > 512:
        raise GenerationSuiteError(
            f"{field}.text must contain at most 512 characters"
        )
    if any(token in text for token in ("<bos>", "<eos>", "<pad>", "<unk>")):
        raise GenerationSuiteError(
            f"{field}.text must not contain literal special tokens"
        )
    return PromptSpec(prompt_id=prompt_id, text=text)


def _parse_decoding(
    value: object,
    *,
    index: int,
    max_new_tokens: int,
) -> DecodingSpec:
    field = f"decodings[{index}]"
    decoding = _as_mapping(value, field=field)
    _require_exact_fields(
        decoding,
        {
            "id",
            "role",
            "strategy",
            "temperature",
            "top_k",
            "top_p",
            "seed",
        },
        field=field,
    )
    decoding_id = _require_id(decoding["id"], field=f"{field}.id")
    role = decoding["role"]
    if not isinstance(role, str) or role not in _REQUIRED_ROLES:
        raise GenerationSuiteError(
            f"{field}.role must be one of {sorted(_REQUIRED_ROLES)}"
        )
    try:
        settings = GenerationSettings(
            strategy=decoding["strategy"],
            max_new_tokens=max_new_tokens,
            temperature=decoding["temperature"],
            top_k=decoding["top_k"],
            top_p=decoding["top_p"],
            seed=decoding["seed"],
        )
    except (GenerationError, TypeError) as error:
        raise GenerationSuiteError(f"{field} is invalid: {error}") from error
    return DecodingSpec(
        decoding_id=decoding_id,
        role=role,
        settings=settings,
    )


def _validate_decoding_roles(
    decodings: tuple[DecodingSpec, ...],
    *,
    stochastic_seed: int,
) -> None:
    by_role = {decoding.role: decoding for decoding in decodings}
    if set(by_role) != _REQUIRED_ROLES or len(by_role) != len(decodings):
        raise GenerationSuiteError(
            "decodings must contain each required role exactly once"
        )

    greedy = by_role["greedy"].settings
    if greedy.strategy != "greedy":
        raise GenerationSuiteError("greedy role must use greedy strategy")

    pure_sample = by_role["sample_temperature_1"].settings
    if (
        pure_sample.strategy != "sample"
        or pure_sample.temperature != 1.0
        or pure_sample.top_k is not None
        or pure_sample.top_p is not None
    ):
        raise GenerationSuiteError(
            "sample_temperature_1 must be unfiltered temperature=1 sampling"
        )

    lower_temperature = by_role["lower_temperature"].settings
    if (
        lower_temperature.strategy != "sample"
        or not 0.0 < lower_temperature.temperature < 1.0
        or lower_temperature.top_k is not None
        or lower_temperature.top_p is not None
    ):
        raise GenerationSuiteError(
            "lower_temperature must be unfiltered sampling below 1.0"
        )

    top_k = by_role["top_k"].settings
    if (
        top_k.strategy != "sample"
        or top_k.temperature != 1.0
        or top_k.top_k is None
        or top_k.top_k <= 1
        or top_k.top_p is not None
    ):
        raise GenerationSuiteError(
            "top_k role must isolate top-k filtering at temperature=1.0"
        )

    top_p = by_role["top_p"].settings
    if (
        top_p.strategy != "sample"
        or top_p.temperature != 1.0
        or top_p.top_k is not None
        or top_p.top_p is None
        or not 0.0 < top_p.top_p < 1.0
    ):
        raise GenerationSuiteError(
            "top_p role must isolate top-p filtering at temperature=1.0"
        )

    for decoding in decodings:
        if decoding.settings.strategy == "sample" and (
            decoding.settings.seed != stochastic_seed
        ):
            raise GenerationSuiteError(
                "all stochastic decoding seeds must equal stochastic_seed"
            )


def load_generation_protocol(
    path: str | Path,
) -> FrozenGenerationProtocol:
    """Load and strictly validate one frozen generation protocol JSON file."""

    protocol_path = Path(path).resolve()
    try:
        raw_bytes = protocol_path.read_bytes()
    except OSError as error:
        raise GenerationSuiteError(
            f"could not read generation protocol {protocol_path}: {error}"
        ) from error
    try:
        document = json.loads(
            raw_bytes.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except GenerationSuiteError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GenerationSuiteError(
            f"generation protocol must be valid UTF-8 JSON: {error}"
        ) from error

    root = _as_mapping(document, field="protocol")
    _require_exact_fields(
        root,
        {
            "format_name",
            "schema_version",
            "protocol_id",
            "ordering",
            "max_new_tokens",
            "stochastic_seed",
            "prompts",
            "decodings",
        },
        field="protocol",
    )
    if root["format_name"] != GENERATION_PROTOCOL_FORMAT_NAME:
        raise GenerationSuiteError(
            f"format_name must be {GENERATION_PROTOCOL_FORMAT_NAME!r}"
        )
    if root["schema_version"] != GENERATION_PROTOCOL_SCHEMA_VERSION:
        raise GenerationSuiteError(
            f"schema_version must be {GENERATION_PROTOCOL_SCHEMA_VERSION}"
        )
    protocol_id = _require_id(root["protocol_id"], field="protocol_id")
    if root["ordering"] != _ORDERING:
        raise GenerationSuiteError(f"ordering must be {_ORDERING!r}")

    max_new_tokens = root["max_new_tokens"]
    if (
        not _is_plain_int(max_new_tokens)
        or max_new_tokens <= 0
        or max_new_tokens > 256
    ):
        raise GenerationSuiteError(
            "max_new_tokens must be an integer in [1, 256]"
        )
    stochastic_seed = root["stochastic_seed"]
    if (
        not _is_plain_int(stochastic_seed)
        or stochastic_seed < 0
        or stochastic_seed > _MAX_SEED
    ):
        raise GenerationSuiteError(
            f"stochastic_seed must be an integer in [0, {_MAX_SEED}]"
        )

    raw_prompts = root["prompts"]
    if not isinstance(raw_prompts, list) or not 5 <= len(raw_prompts) <= 12:
        raise GenerationSuiteError("prompts must contain between 5 and 12 items")
    prompts = tuple(
        _parse_prompt(value, index=index)
        for index, value in enumerate(raw_prompts)
    )
    prompt_ids = [prompt.prompt_id for prompt in prompts]
    prompt_texts = [prompt.text for prompt in prompts]
    if len(set(prompt_ids)) != len(prompt_ids):
        raise GenerationSuiteError("prompt IDs must be unique")
    if len(set(prompt_texts)) != len(prompt_texts):
        raise GenerationSuiteError("prompt texts must be unique")

    raw_decodings = root["decodings"]
    if not isinstance(raw_decodings, list) or len(raw_decodings) != 5:
        raise GenerationSuiteError("decodings must contain exactly five items")
    decodings = tuple(
        _parse_decoding(
            value,
            index=index,
            max_new_tokens=max_new_tokens,
        )
        for index, value in enumerate(raw_decodings)
    )
    decoding_ids = [decoding.decoding_id for decoding in decodings]
    if len(set(decoding_ids)) != len(decoding_ids):
        raise GenerationSuiteError("decoding IDs must be unique")
    _validate_decoding_roles(
        decodings,
        stochastic_seed=stochastic_seed,
    )

    return FrozenGenerationProtocol(
        protocol_id=protocol_id,
        ordering=_ORDERING,
        max_new_tokens=max_new_tokens,
        stochastic_seed=stochastic_seed,
        prompts=prompts,
        decodings=decodings,
    )


def generation_protocol_fingerprint(
    protocol: FrozenGenerationProtocol,
) -> str:
    if not isinstance(protocol, FrozenGenerationProtocol):
        raise TypeError("protocol must be a FrozenGenerationProtocol")
    encoded = json.dumps(
        protocol.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_source_identity(
    source_commit: str | None,
    source_dirty: bool,
) -> None:
    if source_commit is not None and (
        not isinstance(source_commit, str)
        or _GIT_OBJECT_PATTERN.fullmatch(source_commit) is None
    ):
        raise GenerationSuiteError(
            "generator_source_commit must be a full lowercase Git object ID or null"
        )
    if not isinstance(source_dirty, bool):
        raise TypeError("generator_source_dirty must be a boolean")


def _strict_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                dict(value),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise GenerationSuiteError(
            f"generation suite value is not strict JSON: {error}"
        ) from error


def _strict_jsonl_bytes(records: list[Mapping[str, Any]]) -> bytes:
    lines: list[str] = []
    try:
        for record in records:
            lines.append(
                json.dumps(
                    dict(record),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
    except (TypeError, ValueError) as error:
        raise GenerationSuiteError(
            f"generation sample is not strict JSON: {error}"
        ) from error
    return (("\n".join(lines)) + "\n").encode("utf-8")


def _write_fsynced(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _publish_suite_directory(
    output_dir: Path,
    *,
    samples_payload: bytes,
    manifest_payload: bytes,
) -> Path:
    try:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise GenerationSuiteError(
            f"could not create suite output parent: {error}"
        ) from error
    if output_dir.exists():
        raise GenerationSuiteError(
            f"generation suite output already exists: {output_dir}"
        )

    staging = output_dir.parent / (
        f".{output_dir.name}.{uuid.uuid4().hex}.tmp"
    )
    output_created = False
    try:
        staging.mkdir()
        staging_samples = staging / GENERATION_SAMPLES_FILENAME
        staging_manifest = staging / GENERATION_MANIFEST_FILENAME
        _write_fsynced(staging_samples, samples_payload)
        _write_fsynced(staging_manifest, manifest_payload)

        output_dir.mkdir()
        output_created = True
        os.link(staging_samples, output_dir / GENERATION_SAMPLES_FILENAME)
        os.link(staging_manifest, output_dir / GENERATION_MANIFEST_FILENAME)
    except FileExistsError as error:
        if output_created:
            shutil.rmtree(output_dir, ignore_errors=True)
        raise GenerationSuiteError(
            f"generation suite output already exists: {output_dir}"
        ) from error
    except OSError as error:
        if output_created:
            shutil.rmtree(output_dir, ignore_errors=True)
        raise GenerationSuiteError(
            f"could not atomically publish generation suite: {error}"
        ) from error
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return output_dir


def run_generation_suite(
    checkpoint_path: str | Path,
    tokenizer_path: str | Path,
    protocol_path: str | Path,
    output_dir: str | Path,
    *,
    model_config: GPTConfig,
    expected_run_id: str,
    expected_checkpoint_sha256: str,
    expected_tokenizer_sha256: str,
    precision: PrecisionPolicy,
    generator_source_commit: str | None = None,
    generator_source_dirty: bool = True,
) -> tuple[Path, dict[str, Any]]:
    """Run one complete frozen protocol with one artifact/model load."""

    resolved_output = Path(output_dir).resolve()
    if resolved_output.exists():
        raise GenerationSuiteError(
            f"generation suite output already exists: {resolved_output}"
        )
    _validate_source_identity(
        generator_source_commit,
        generator_source_dirty,
    )
    protocol = load_generation_protocol(protocol_path)
    resolved_protocol = Path(protocol_path).resolve()
    protocol_file_sha256 = sha256_file(resolved_protocol)
    protocol_fingerprint = generation_protocol_fingerprint(protocol)

    suite_created_at = datetime.now(timezone.utc).isoformat()
    wall_start = time.perf_counter()
    session = load_generation_session(
        checkpoint_path,
        tokenizer_path,
        model_config=model_config,
        expected_run_id=expected_run_id,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_tokenizer_sha256=expected_tokenizer_sha256,
        precision=precision,
    )

    records: list[dict[str, Any]] = []
    stop_reason_counts = {"eos": 0, "max_new_tokens": 0}
    total_prompt_tokens = 0
    total_generated_tokens = 0
    total_forward_passes = 0
    total_context_crop_events = 0
    summed_generation_seconds = 0.0
    common: dict[str, Any] | None = None

    for prompt in protocol.prompts:
        for decoding in protocol.decodings:
            result = generate_with_session(
                session,
                prompt.text,
                settings=decoding.settings,
                generator_source_commit=generator_source_commit,
                generator_source_dirty=generator_source_dirty,
            )
            current_common = {
                "run_id": result["run_id"],
                "checkpoint": result["checkpoint"],
                "generator": result["generator"],
                "model": result["model"],
                "tokenizer": result["tokenizer"],
            }
            if common is None:
                common = current_common
            elif current_common != common:
                raise GenerationSuiteError(
                    "generation sample artifact identity changed within the suite"
                )

            expected_settings = decoding.settings.to_dict()
            actual_settings = {
                key: result["protocol"][key]
                for key in expected_settings
            }
            if actual_settings != expected_settings:
                raise GenerationSuiteError(
                    "generation sample decoding settings do not match the protocol"
                )
            if result["prompt"]["text"] != prompt.text:
                raise GenerationSuiteError(
                    "generation sample prompt does not match the protocol"
                )

            generation = result["generation"]
            stop_reason = generation["stop_reason"]
            if stop_reason not in stop_reason_counts:
                raise GenerationSuiteError(
                    f"unexpected generation stop reason: {stop_reason!r}"
                )
            stop_reason_counts[stop_reason] += 1
            total_prompt_tokens += result["prompt"]["token_count"]
            total_generated_tokens += generation["token_count"]
            total_forward_passes += generation["forward_passes"]
            total_context_crop_events += generation["context_crop_events"]
            summed_generation_seconds += generation["elapsed_seconds"]

            sample_index = len(records)
            records.append(
                {
                    "format_name": GENERATION_SAMPLE_FORMAT_NAME,
                    "schema_version": GENERATION_SUITE_SCHEMA_VERSION,
                    "protocol_id": protocol.protocol_id,
                    "run_id": result["run_id"],
                    "sample_index": sample_index,
                    "sample_key": (
                        f"{prompt.prompt_id}/{decoding.decoding_id}"
                    ),
                    "prompt_id": prompt.prompt_id,
                    "decoding_id": decoding.decoding_id,
                    "decoding_role": decoding.role,
                    "created_at_utc": result["created_at_utc"],
                    "prompt": result["prompt"],
                    "protocol": result["protocol"],
                    "generation": generation,
                    "runtime": result["runtime"],
                }
            )

    if common is None or len(records) != protocol.sample_count:
        raise GenerationSuiteError(
            "generation suite did not produce the expected sample count"
        )

    samples_payload = _strict_jsonl_bytes(records)
    samples_sha256 = hashlib.sha256(samples_payload).hexdigest()
    wall_elapsed_seconds = max(time.perf_counter() - wall_start, 1.0e-12)
    manifest = {
        "format_name": GENERATION_SUITE_FORMAT_NAME,
        "schema_version": GENERATION_SUITE_SCHEMA_VERSION,
        "status": "complete",
        "created_at_utc": suite_created_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        **common,
        "protocol": {
            "path": resolved_protocol.as_posix(),
            "bytes": resolved_protocol.stat().st_size,
            "file_sha256": protocol_file_sha256,
            "config_fingerprint": protocol_fingerprint,
            "definition": protocol.to_dict(),
        },
        "execution": {
            "ordering": protocol.ordering,
            "prompt_count": len(protocol.prompts),
            "decoding_count": len(protocol.decodings),
            "expected_samples": protocol.sample_count,
            "completed_samples": len(records),
            "artifacts_loaded_once": True,
            "model_loads": 1,
            "wall_elapsed_seconds": wall_elapsed_seconds,
        },
        "samples": {
            "filename": GENERATION_SAMPLES_FILENAME,
            "records": len(records),
            "bytes": len(samples_payload),
            "sha256": samples_sha256,
        },
        "summary": {
            "stop_reason_counts": stop_reason_counts,
            "sample_prompt_tokens": total_prompt_tokens,
            "generated_tokens": total_generated_tokens,
            "forward_passes": total_forward_passes,
            "context_crop_events": total_context_crop_events,
            "summed_generation_seconds": summed_generation_seconds,
        },
    }
    manifest_payload = _strict_json_bytes(manifest)
    published = _publish_suite_directory(
        resolved_output,
        samples_payload=samples_payload,
        manifest_payload=manifest_payload,
    )
    return published, manifest
