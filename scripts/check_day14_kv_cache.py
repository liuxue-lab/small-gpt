"""Strict protocol and dry-run gate for the frozen Day 14 KV-cache work."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


sys.dont_write_bytecode = True


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


FORMAT_NAME = "small_gpt_day14_kv_cache_protocol"
SCHEMA_VERSION = 1
PROTOCOL_ID = "day14-kv-cache-v1"
PROTOCOL_STATUS = "frozen_after_user_approval"
EXPECTED_PROTOCOL_BYTES = 16_134
EXPECTED_PROTOCOL_SHA256 = (
    "fe676b127c7bd0e08d920816dce547037669065d8aad721cc17dad0569b08b1c"
)
EXPECTED_SOURCE_HEAD = "6ef391625a091fc652ea85478d1e72abdb1bb56e"
EXPECTED_PARAMETERS = 33_833_984
EXPECTED_STATE_DICT_KEYS = 69
EXPECTED_LAYER_COUNT = 8
EXPECTED_HEAD_COUNT = 8
EXPECTED_HEAD_DIMENSION = 64
EXPECTED_CONTEXT_LENGTH = 512
EXPECTED_VOCAB_SIZE = 16_384
EXPECTED_FUNCTIONAL_FILES = (
    "configs/day14_kv_cache_protocol.json",
    "model/__init__.py",
    "model/attention.py",
    "model/block.py",
    "model/gpt.py",
    "scripts/check_day14_kv_cache.py",
    "scripts/benchmark_day14_kv_cache.py",
    "tests/test_day14_kv_cache.py",
)
EXPECTED_SCENARIOS = {
    "bridge": (3, 64, 67, 66),
    "short": (16, 64, 80, 79),
    "medium": (128, 128, 256, 255),
    "long": (384, 128, 512, 511),
}
EXPECTED_CPU_TOLERANCE = {"rtol": 1.0e-5, "atol": 1.0e-6}
EXPECTED_JETSON_FP32_TOLERANCE = {"rtol": 1.0e-4, "atol": 1.0e-5}
EXPECTED_JETSON_FP16_TOLERANCE = {"rtol": 1.0e-2, "atol": 1.0e-2}
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class Day14KVCacheError(RuntimeError):
    """Raised when the Day 14 protocol or an inference-only gate fails."""


class Day14ProtocolError(Day14KVCacheError):
    """Raised when the frozen Day 14 protocol is missing or malformed."""


@dataclass(frozen=True, slots=True)
class FileIdentity:
    path: Path
    bytes: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path.as_posix(),
            "bytes": self.bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    branch: str
    head: str
    remote_url: str
    worktree_entries: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch": self.branch,
            "head": self.head,
            "remote_url": self.remote_url,
            "worktree_entries": self.worktree_entries,
        }


@dataclass(frozen=True, slots=True)
class RuntimeSession:
    protocol: dict[str, Any]
    protocol_fingerprint: str
    model: Any
    tokenizer: Any
    model_config: Any
    loaded_checkpoint: Any
    device: Any
    dtype: Any
    precision: str
    source: SourceIdentity
    config_identity: FileIdentity
    checkpoint_identity: FileIdentity
    tokenizer_identity: FileIdentity
    tokenizer_config_identity: FileIdentity
    model_load_seconds: float


def _require(condition: object, message: str) -> None:
    if not bool(condition):
        raise Day14ProtocolError(message)


def _require_mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Day14ProtocolError(f"{field} must be a JSON object")
    return value


def _require_exact(value: object, expected: object, *, field: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise Day14ProtocolError(
            f"{field} must equal {expected!r}, got {value!r}"
        )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Day14ProtocolError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_non_finite_constant(value: str) -> None:
    raise Day14ProtocolError(
        f"protocol contains non-finite numeric constant {value!r}"
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_sha256(value: object) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise Day14ProtocolError(
            f"value is not canonical strict JSON: {error}"
        ) from error
    return sha256_bytes(payload)


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    normalized = validate_token_ids(
        token_ids,
        vocab_size=EXPECTED_VOCAB_SIZE,
        label="token_ids",
    )
    payload = json.dumps(
        list(normalized),
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256_bytes(payload)


def validate_token_ids(
    token_ids: Sequence[int],
    *,
    vocab_size: int,
    label: str,
) -> tuple[int, ...]:
    if isinstance(token_ids, (str, bytes)) or not isinstance(token_ids, Sequence):
        raise Day14KVCacheError(f"{label} must be a sequence of integers")
    normalized = tuple(token_ids)
    if not normalized:
        raise Day14KVCacheError(f"{label} must not be empty")
    for index, token_id in enumerate(normalized):
        if not isinstance(token_id, int) or isinstance(token_id, bool):
            raise Day14KVCacheError(
                f"{label}[{index}] must be a plain integer"
            )
        if token_id < 0 or token_id >= vocab_size:
            raise Day14KVCacheError(
                f"{label}[{index}]={token_id} is outside [0, {vocab_size})"
            )
    return normalized


def repeat_truncate(token_ids: Sequence[int], target_length: int) -> tuple[int, ...]:
    source = validate_token_ids(
        token_ids,
        vocab_size=EXPECTED_VOCAB_SIZE,
        label="base_token_ids",
    )
    if not isinstance(target_length, int) or isinstance(target_length, bool):
        raise Day14KVCacheError("target_length must be a positive integer")
    if target_length <= 0:
        raise Day14KVCacheError("target_length must be a positive integer")
    repetitions = (target_length + len(source) - 1) // len(source)
    return (source * repetitions)[:target_length]


def validate_protocol_document(document: Mapping[str, Any]) -> None:
    _require_exact(document.get("format_name"), FORMAT_NAME, field="format_name")
    _require_exact(
        document.get("schema_version"),
        SCHEMA_VERSION,
        field="schema_version",
    )
    _require_exact(document.get("protocol_id"), PROTOCOL_ID, field="protocol_id")
    _require_exact(document.get("status"), PROTOCOL_STATUS, field="status")

    approval = _require_mapping(document.get("approval"), field="approval")
    _require_exact(approval.get("approved_by"), "user", field="approval.approved_by")
    _require_exact(
        approval.get("approval_text"),
        "批准 Day14 KV Cache v1 协议",
        field="approval.approval_text",
    )

    source = _require_mapping(document.get("source"), field="source")
    _require_exact(
        source.get("day13_final_head"),
        EXPECTED_SOURCE_HEAD,
        field="source.day13_final_head",
    )
    _require_exact(
        source.get("required_branch"),
        "main",
        field="source.required_branch",
    )

    architecture = _require_mapping(
        document.get("architecture"),
        field="architecture",
    )
    expected_architecture = {
        "parameters": EXPECTED_PARAMETERS,
        "state_dict_key_count": EXPECTED_STATE_DICT_KEYS,
        "layer_count": EXPECTED_LAYER_COUNT,
        "head_count": EXPECTED_HEAD_COUNT,
        "head_dimension": EXPECTED_HEAD_DIMENSION,
        "context_length": EXPECTED_CONTEXT_LENGTH,
        "vocab_size": EXPECTED_VOCAB_SIZE,
        "dropout": 0.0,
        "tied_embeddings": True,
    }
    for key, expected in expected_architecture.items():
        _require_exact(
            architecture.get(key),
            expected,
            field=f"architecture.{key}",
        )

    mutation_scope = _require_mapping(
        document.get("mutation_scope"),
        field="mutation_scope",
    )
    _require_exact(
        mutation_scope.get("functional_files"),
        list(EXPECTED_FUNCTIONAL_FILES),
        field="mutation_scope.functional_files",
    )
    _require(
        "eval/generation.py" in mutation_scope.get("forbidden_files", []),
        "eval/generation.py must remain outside the mutation scope",
    )
    _require(
        "scripts/benchmark_jetson_inference.py"
        in mutation_scope.get("forbidden_files", []),
        "Day 13 benchmark must remain outside the mutation scope",
    )

    api_route = _require_mapping(document.get("api_route"), field="api_route")
    expected_api = {
        "route_id": "separate_forward_cached_methods_v1",
        "original_forward_preserved": True,
        "training_call_sites_changed": False,
        "optional_use_cache_keyword_added": False,
        "attention_method": "CausalSelfAttention.forward_cached",
        "block_method": "TransformerBlock.forward_cached",
        "gpt_method": "GPT.forward_cached",
        "layer_cache_type": "LayerKVCache",
        "past_key_values_type": "PastKeyValues",
        "cached_output_type": "GPTCachedOutput",
        "cached_output_fields": ["logits", "past_key_values"],
        "cached_path_is_inference_only": True,
        "requires_eval_mode": True,
        "requires_torch_inference_mode": True,
    }
    for key, expected in expected_api.items():
        _require_exact(api_route.get(key), expected, field=f"api_route.{key}")

    cache_contract = _require_mapping(
        document.get("cache_contract"),
        field="cache_contract",
    )
    _require_exact(
        cache_contract.get("layer_cache_members"),
        ["key", "value"],
        field="cache_contract.layer_cache_members",
    )
    append = _require_mapping(
        cache_contract.get("append"),
        field="cache_contract.append",
    )
    _require_exact(
        append.get("sequence_dimension"),
        2,
        field="append.sequence_dimension",
    )
    _require_exact(append.get("operation"), "torch_concat", field="append.operation")
    _require_exact(
        append.get("input_past_modified_in_place"),
        False,
        field="append.input_past_modified_in_place",
    )
    final_length = _require_mapping(
        cache_contract.get("final_length_semantics"),
        field="cache_contract.final_length_semantics",
    )
    _require_exact(
        final_length.get("unnecessary_final_forward_allowed"),
        False,
        field="final_length_semantics.unnecessary_final_forward_allowed",
    )

    context = _require_mapping(
        document.get("context_policy"),
        field="context_policy",
    )
    for key, expected in {
        "policy_id": "strict_no_overflow_v1",
        "rolling_window": False,
        "cache_eviction": False,
        "position_remapping": False,
        "prompt_511_generate_1_allowed": True,
        "prompt_512_generate_0_allowed": True,
        "prompt_512_generate_1_allowed": False,
        "prompt_513_allowed": False,
        "past_512_then_input_1_allowed": False,
        "inconsistent_layer_lengths_allowed": False,
    }.items():
        _require_exact(context.get(key), expected, field=f"context_policy.{key}")

    correctness = _require_mapping(
        document.get("correctness"),
        field="correctness",
    )
    _require_exact(
        correctness.get("cpu_fp32"),
        EXPECTED_CPU_TOLERANCE,
        field="correctness.cpu_fp32",
    )
    _require_exact(
        correctness.get("jetson_fp32"),
        EXPECTED_JETSON_FP32_TOLERANCE,
        field="correctness.jetson_fp32",
    )
    _require_exact(
        correctness.get("jetson_fp16"),
        EXPECTED_JETSON_FP16_TOLERANCE,
        field="correctness.jetson_fp16",
    )
    _require_exact(
        correctness.get("tolerance_may_be_relaxed_after_failure"),
        False,
        field="correctness.tolerance_may_be_relaxed_after_failure",
    )

    prompt_builder = _require_mapping(
        document.get("prompt_builder"),
        field="prompt_builder",
    )
    scenarios = prompt_builder.get("scenarios")
    _require(isinstance(scenarios, list), "prompt_builder.scenarios must be a list")
    _require(len(scenarios) == 4, "prompt_builder.scenarios must contain four entries")
    observed_names: list[str] = []
    for index, raw_scenario in enumerate(scenarios):
        scenario = _require_mapping(
            raw_scenario,
            field=f"prompt_builder.scenarios[{index}]",
        )
        name = scenario.get("name")
        _require(isinstance(name, str), f"scenario {index} name must be a string")
        _require(name in EXPECTED_SCENARIOS, f"unknown scenario {name!r}")
        observed_names.append(name)
        expected = EXPECTED_SCENARIOS[name]
        for key, expected_value in zip(
            (
                "prompt_length",
                "max_new_tokens",
                "returned_sequence_length",
                "expected_final_cache_length",
            ),
            expected,
            strict=True,
        ):
            _require_exact(
                scenario.get(key),
                expected_value,
                field=f"scenario.{name}.{key}",
            )
        prompt_hash = scenario.get("prompt_token_ids_sha256")
        _require(
            isinstance(prompt_hash, str)
            and _SHA256_PATTERN.fullmatch(prompt_hash) is not None,
            f"scenario {name} token hash is invalid",
        )
    _require_exact(
        observed_names,
        ["bridge", "short", "medium", "long"],
        field="prompt_builder.scenario_order",
    )

    benchmark = _require_mapping(document.get("benchmark"), field="benchmark")
    for key, expected in {
        "precision": "fp16",
        "decoding": "greedy",
        "stop_on_eos": False,
        "reference_strategy": "full_prefix_recompute",
        "cached_strategy": "kv_cache",
        "single_model_load": True,
        "warmup_pairs_per_scenario": 3,
        "measured_pairs_per_scenario": 10,
        "even_pair_order": ["reference", "kv_cache"],
        "odd_pair_order": ["kv_cache", "reference"],
        "warmups_excluded_from_summary": True,
        "empty_cache_between_requests": False,
        "primary_scenarios": ["medium", "long"],
        "bridge_is_historical_only": True,
    }.items():
        _require_exact(benchmark.get(key), expected, field=f"benchmark.{key}")

    safety = _require_mapping(document.get("safety"), field="safety")
    for key, value in safety.items():
        _require_exact(value, False, field=f"safety.{key}")


def load_protocol(path: str | Path) -> dict[str, Any]:
    protocol_path = Path(path).resolve()
    try:
        payload = protocol_path.read_bytes()
    except OSError as error:
        raise Day14ProtocolError(
            f"could not read protocol {protocol_path}: {error}"
        ) from error
    if len(payload) != EXPECTED_PROTOCOL_BYTES:
        raise Day14ProtocolError(
            "protocol byte count mismatch: "
            f"{len(payload)} != {EXPECTED_PROTOCOL_BYTES}"
        )
    actual_sha256 = sha256_bytes(payload)
    if actual_sha256 != EXPECTED_PROTOCOL_SHA256:
        raise Day14ProtocolError(
            "protocol SHA-256 mismatch: "
            f"{actual_sha256} != {EXPECTED_PROTOCOL_SHA256}"
        )
    try:
        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite_constant,
        )
    except Day14ProtocolError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Day14ProtocolError(
            f"protocol must be valid strict UTF-8 JSON: {error}"
        ) from error
    if not isinstance(document, dict):
        raise Day14ProtocolError("protocol root must be a JSON object")
    validate_protocol_document(document)
    return document


def strict_json_bytes(value: Mapping[str, Any]) -> bytes:
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
        raise Day14KVCacheError(
            f"value is not strict finite JSON: {error}"
        ) from error


def strict_jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    try:
        lines = [
            json.dumps(
                dict(row),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            for row in rows
        ]
    except (TypeError, ValueError) as error:
        raise Day14KVCacheError(
            f"rows are not strict finite JSON: {error}"
        ) from error
    if not lines:
        raise Day14KVCacheError("JSONL publication requires at least one row")
    return ("\n".join(lines) + "\n").encode("utf-8")


def atomic_write_bytes_exclusive(path: str | Path, payload: bytes) -> Path:
    output_path = Path(path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise Day14KVCacheError(
            f"output already exists and will not be overwritten: {output_path}"
        )
    temporary = output_path.with_name(
        f".{output_path.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output_path)
    except FileExistsError as error:
        raise Day14KVCacheError(
            f"output already exists and will not be overwritten: {output_path}"
        ) from error
    except OSError as error:
        raise Day14KVCacheError(
            f"could not atomically publish {output_path}: {error}"
        ) from error
    finally:
        temporary.unlink(missing_ok=True)
    return output_path


def atomic_write_json_exclusive(
    path: str | Path,
    value: Mapping[str, Any],
) -> Path:
    return atomic_write_bytes_exclusive(path, strict_json_bytes(value))


def verify_file_identity(
    path: str | Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
    label: str,
) -> FileIdentity:
    resolved = Path(path).resolve()
    try:
        actual_bytes = resolved.stat().st_size
        digest = hashlib.sha256()
        with resolved.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as error:
        raise Day14KVCacheError(
            f"could not read {label} {resolved}: {error}"
        ) from error
    actual = FileIdentity(
        path=resolved,
        bytes=actual_bytes,
        sha256=digest.hexdigest(),
    )
    if actual.bytes != expected_bytes:
        raise Day14KVCacheError(
            f"{label} byte count mismatch: "
            f"{actual.bytes} != {expected_bytes}"
        )
    if actual.sha256 != expected_sha256:
        raise Day14KVCacheError(
            f"{label} SHA-256 mismatch: "
            f"{actual.sha256} != {expected_sha256}"
        )
    return actual


def _run_git(project_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise Day14KVCacheError(
            f"git {' '.join(arguments)} failed with "
            f"exit {completed.returncode}: {detail}"
        )
    return completed.stdout


def git_source_identity(
    project_root: str | Path = PROJECT_ROOT,
) -> SourceIdentity:
    root = Path(project_root).resolve()
    branch = _run_git(root, "branch", "--show-current").strip()
    head = _run_git(root, "rev-parse", "HEAD").strip()
    remote_url = _run_git(root, "remote", "get-url", "origin").strip()
    status = _run_git(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ).splitlines()
    return SourceIdentity(
        branch=branch,
        head=head,
        remote_url=remote_url,
        worktree_entries=len(status),
    )


def assert_ntp_synchronized() -> None:
    completed = subprocess.run(
        ["timedatectl", "show", "-p", "NTPSynchronized", "--value"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    value = completed.stdout.strip().lower()
    if completed.returncode != 0 or value not in {"yes", "true"}:
        detail = completed.stderr.strip() or value or "unknown"
        raise Day14KVCacheError(
            f"NTP synchronization is required before a formal run: {detail}"
        )


def validate_run_id(value: str, *, mode: str) -> str:
    prefixes = {
        "correctness": "day14-jetson-kv-cache-correctness-",
        "smoke": "day14-jetson-kv-cache-smoke-",
        "benchmark": "day14-jetson-kv-cache-paired-benchmark-",
        "stability": "day14-jetson-kv-cache-stability-",
    }
    prefix = prefixes.get(mode)
    if prefix is None:
        raise Day14KVCacheError(f"unknown run-ID mode: {mode!r}")
    pattern = re.compile(
        rf"{re.escape(prefix)}[0-9]{{8}}T[0-9]{{6}}Z"
    )
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise Day14KVCacheError(
            f"run ID must match {prefix}YYYYMMDDTHHMMSSZ"
        )
    timestamp = value[len(prefix):]
    try:
        datetime.strptime(timestamp, "%Y%m%dT%H%M%SZ")
    except ValueError as error:
        raise Day14KVCacheError(
            f"run ID contains an invalid UTC timestamp: {timestamp}"
        ) from error
    return value


def reserve_output_directory(path: str | Path, *, run_id: str) -> Path:
    output_dir = preflight_output_directory(path, run_id=run_id)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_dir.mkdir()
    except FileExistsError as error:
        raise Day14KVCacheError(
            f"output directory already exists: {output_dir}"
        ) from error
    except OSError as error:
        raise Day14KVCacheError(
            f"could not reserve output directory {output_dir}: {error}"
        ) from error
    return output_dir


def preflight_output_directory(path: str | Path, *, run_id: str) -> Path:
    output_dir = Path(path).resolve()
    if output_dir.name != run_id:
        raise Day14KVCacheError(
            "output directory basename must equal the frozen run ID"
        )
    if output_dir.exists():
        raise Day14KVCacheError(
            f"output directory already exists: {output_dir}"
        )
    return output_dir


def require_external_output_path(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    if resolved == PROJECT_ROOT or PROJECT_ROOT in resolved.parents:
        raise Day14KVCacheError(
            "runtime outputs must be outside the immutable Git checkout"
        )
    return resolved


def _precision_dtype(torch: Any, precision: str) -> Any:
    if precision == "fp32":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    raise Day14KVCacheError(
        f"precision must be 'fp32' or 'fp16', got {precision!r}"
    )


def _resolve_device(torch: Any, requested: str) -> Any:
    try:
        device = torch.device(requested)
    except Exception as error:
        raise Day14KVCacheError(
            f"invalid requested device {requested!r}: {error}"
        ) from error
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise Day14KVCacheError("CUDA was requested but is not available")
        if device.index is None:
            device = torch.device("cuda:0")
        if device.index >= torch.cuda.device_count():
            raise Day14KVCacheError(
                f"requested CUDA index {device.index} is unavailable"
            )
    elif device.type != "cpu":
        raise Day14KVCacheError(
            f"device type must be cpu or cuda, got {device.type!r}"
        )
    return device


def load_runtime_session(
    *,
    protocol_path: str | Path,
    config_path: str | Path,
    checkpoint_path: str | Path,
    tokenizer_path: str | Path,
    tokenizer_config_path: str | Path,
    requested_device: str,
    precision: str,
    expected_functional_head: str,
    project_root: str | Path = PROJECT_ROOT,
) -> RuntimeSession:
    try:
        torch = importlib.import_module("torch")
        model_module = importlib.import_module("model")
        deployment = importlib.import_module(
            "scripts.check_jetson_deployment"
        )
        tokenizer_module = importlib.import_module("tokenizer")
    except Exception as error:
        raise Day14KVCacheError(
            f"could not import the audited inference runtime: {error}"
        ) from error

    protocol = load_protocol(protocol_path)
    frozen = protocol["frozen_artifacts"]
    config_spec = frozen["baseline_config"]
    checkpoint_spec = frozen["control_checkpoint"]
    tokenizer_spec = frozen["tokenizer_json"]
    tokenizer_config_spec = frozen["tokenizer_config"]
    config_identity = verify_file_identity(
        config_path,
        expected_bytes=int(config_spec["bytes"]),
        expected_sha256=str(config_spec["sha256"]),
        label="baseline config",
    )
    checkpoint_identity = verify_file_identity(
        checkpoint_path,
        expected_bytes=int(checkpoint_spec["bytes"]),
        expected_sha256=str(checkpoint_spec["sha256"]),
        label="control checkpoint",
    )
    tokenizer_identity = verify_file_identity(
        tokenizer_path,
        expected_bytes=int(tokenizer_spec["bytes"]),
        expected_sha256=str(tokenizer_spec["sha256"]),
        label="tokenizer",
    )
    tokenizer_config_identity = verify_file_identity(
        tokenizer_config_path,
        expected_bytes=int(tokenizer_config_spec["bytes"]),
        expected_sha256=str(tokenizer_config_spec["sha256"]),
        label="tokenizer config",
    )

    source = git_source_identity(project_root)
    if re.fullmatch(r"[0-9a-f]{40}", expected_functional_head) is None:
        raise Day14KVCacheError(
            "expected functional HEAD must be a 40-character lowercase SHA"
        )
    if source.branch != protocol["source"]["required_branch"]:
        raise Day14KVCacheError("runtime branch is not the required main branch")
    if source.head != expected_functional_head:
        raise Day14KVCacheError(
            f"runtime HEAD mismatch: {source.head} != {expected_functional_head}"
        )
    if source.remote_url != protocol["source"]["remote_url"]:
        raise Day14KVCacheError("runtime origin URL changed")
    if source.worktree_entries != 0:
        raise Day14KVCacheError("runtime worktree must be clean")

    try:
        model_config = model_module.GPTConfig.from_yaml(config_identity.path)
    except Exception as error:
        raise Day14KVCacheError(
            f"could not load baseline model config: {error}"
        ) from error
    architecture = protocol["architecture"]
    expected_model_values = {
        "n_layer": architecture["layer_count"],
        "n_head": architecture["head_count"],
        "n_embd": architecture["embedding_dimension"],
        "ffn_hidden": architecture["ffn_hidden_dimension"],
        "context_length": architecture["context_length"],
        "vocab_size": architecture["vocab_size"],
        "dropout": architecture["dropout"],
    }
    for field, expected in expected_model_values.items():
        if getattr(model_config, field) != expected:
            raise Day14KVCacheError(
                f"model config {field} mismatch: "
                f"{getattr(model_config, field)!r} != {expected!r}"
            )
    try:
        tokenizer = tokenizer_module.load_tokenizer(tokenizer_identity.path)
    except Exception as error:
        raise Day14KVCacheError(f"could not load tokenizer: {error}") from error
    if int(tokenizer.get_vocab_size()) != int(architecture["vocab_size"]):
        raise Day14KVCacheError("tokenizer vocabulary size changed")

    device = _resolve_device(torch, requested_device)
    if device.type != "cuda":
        raise Day14KVCacheError(
            "Day 14 runtime modes require an audited CUDA device"
        )
    dtype = _precision_dtype(torch, precision)
    load_started = time.perf_counter()
    try:
        with torch.random.fork_rng(devices=[]):
            model = model_module.GPT(model_config)
        loaded = deployment.model_only_load(
            checkpoint_identity.path,
            model=model,
            model_config=model_config,
            expected_run_id=str(checkpoint_spec["run_id"]),
        )
        loaded_tokenizer_sha = getattr(
            getattr(loaded, "identity", None),
            "tokenizer_sha256",
            None,
        )
        if loaded_tokenizer_sha != tokenizer_identity.sha256:
            raise Day14KVCacheError(
                "checkpoint tokenizer identity does not match frozen tokenizer"
            )
        model.to(device=device, dtype=dtype)
        model.eval()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    except Day14KVCacheError:
        raise
    except Exception as error:
        raise Day14KVCacheError(f"model-only load failed: {error}") from error
    model_load_seconds = max(time.perf_counter() - load_started, 1.0e-12)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    state_key_count = len(model.state_dict())
    if parameter_count != int(architecture["parameters"]):
        raise Day14KVCacheError("runtime model parameter count changed")
    if state_key_count != int(architecture["state_dict_key_count"]):
        raise Day14KVCacheError("runtime model state_dict key count changed")
    if model.training:
        raise Day14KVCacheError("runtime model must be in eval mode")
    if model.lm_head.weight is not model.token_embedding.weight:
        raise Day14KVCacheError("runtime model embedding weights are not tied")
    wrong_devices = [
        name
        for name, parameter in model.named_parameters()
        if parameter.device != device
    ]
    wrong_dtypes = [
        name
        for name, parameter in model.named_parameters()
        if parameter.is_floating_point() and parameter.dtype != dtype
    ]
    if wrong_devices:
        raise Day14KVCacheError(
            f"runtime parameters use wrong devices: {wrong_devices[:5]}"
        )
    if wrong_dtypes:
        raise Day14KVCacheError(
            f"runtime parameters use wrong dtypes: {wrong_dtypes[:5]}"
        )
    return RuntimeSession(
        protocol=dict(protocol),
        protocol_fingerprint=canonical_sha256(protocol),
        model=model,
        tokenizer=tokenizer,
        model_config=model_config,
        loaded_checkpoint=loaded,
        device=device,
        dtype=dtype,
        precision=precision,
        source=source,
        config_identity=config_identity,
        checkpoint_identity=checkpoint_identity,
        tokenizer_identity=tokenizer_identity,
        tokenizer_config_identity=tokenizer_config_identity,
        model_load_seconds=model_load_seconds,
    )


def build_load_only_summary(session: RuntimeSession) -> dict[str, Any]:
    torch = importlib.import_module("torch")

    probe_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long, device=session.device)
    with torch.inference_mode():
        output = session.model(probe_ids)
        logits = output.logits
        finite = bool(torch.isfinite(logits).all().item())
    if tuple(logits.shape) != (1, 4, EXPECTED_VOCAB_SIZE):
        raise Day14KVCacheError("load-only forward logits shape changed")
    if logits.dtype != session.dtype or logits.device != session.device:
        raise Day14KVCacheError("load-only forward logits runtime changed")
    if not finite:
        raise Day14KVCacheError("load-only forward logits are non-finite")
    loaded = session.loaded_checkpoint
    return {
        "format_name": "small_gpt_day14_kv_cache_load_only",
        "schema_version": 1,
        "status": "complete",
        "gate": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": session.protocol["protocol_id"],
        "protocol_fingerprint": session.protocol_fingerprint,
        "source": session.source.to_dict(),
        "artifacts": {
            "config": session.config_identity.to_dict(),
            "checkpoint": session.checkpoint_identity.to_dict(),
            "tokenizer": session.tokenizer_identity.to_dict(),
            "tokenizer_config": session.tokenizer_config_identity.to_dict(),
        },
        "checkpoint": {
            "run_id": getattr(getattr(loaded, "state", None), "run_id", None),
            "global_step": getattr(getattr(loaded, "record", None), "global_step", None),
            "tokens_seen": getattr(getattr(loaded, "record", None), "tokens_seen", None),
            "load_mode": "model_only",
            "strict_state_dict_load": True,
            "missing_keys": 0,
            "unexpected_keys": 0,
            "optimizer_state_restored": False,
            "scheduler_state_restored": False,
            "training_resume": False,
        },
        "model": {
            "parameters": sum(
                parameter.numel() for parameter in session.model.parameters()
            ),
            "state_dict_key_count": len(session.model.state_dict()),
            "training": False,
            "eval_mode": True,
            "weight_tied": True,
        },
        "runtime": {
            "device": str(session.device),
            "precision": session.precision,
            "dtype": str(session.dtype),
            "model_load_seconds": session.model_load_seconds,
            "inference_mode": True,
        },
        "forward_probe": {
            "input_shape": [1, 4],
            "logits_shape": [1, 4, EXPECTED_VOCAB_SIZE],
            "logits_finite": True,
        },
        "safety": {
            "training_attempted": False,
            "backward_called": False,
            "optimizer_created": False,
            "checkpoint_written": False,
            "text_generated": False,
        },
    }


def _failure_document(error: BaseException, *, mode: str, run_id: str) -> dict[str, Any]:
    return {
        "format_name": "small_gpt_day14_kv_cache_failure",
        "schema_version": 1,
        "status": "failed",
        "mode": mode,
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "manifest_published": False,
        "training_attempted": False,
        "backward_called": False,
        "optimizer_created": False,
        "checkpoint_written": False,
    }


def validate_correctness_output(
    *,
    output_dir: str | Path,
    protocol: Mapping[str, Any],
    run_id: str,
) -> dict[str, Any]:
    validate_run_id(run_id, mode="correctness")
    root = Path(output_dir).resolve()
    if root.name != run_id:
        raise Day14KVCacheError(
            "correctness output directory basename changed"
        )
    if not root.is_dir():
        raise Day14KVCacheError(
            "correctness output path is not a directory"
        )
    expected_files = {"correctness-summary.json", "comparisons.jsonl"}
    try:
        actual_entries = tuple(root.iterdir())
    except OSError as error:
        raise Day14KVCacheError(
            f"could not enumerate correctness output: {error}"
        ) from error
    actual_names = {path.name for path in actual_entries}
    if (
        actual_names != expected_files
        or any(not path.is_file() for path in actual_entries)
    ):
        raise Day14KVCacheError(
            f"correctness output file set mismatch: {sorted(actual_names)}"
        )
    try:
        summary = json.loads(
            (root / "correctness-summary.json").read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite_constant,
        )
        lines = (root / "comparisons.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Day14KVCacheError(
            f"could not parse correctness output: {error}"
        ) from error
    if not isinstance(summary, dict):
        raise Day14KVCacheError("correctness summary must be a JSON object")
    if not lines or any(not line.strip() for line in lines):
        raise Day14KVCacheError(
            "comparisons JSONL must contain only non-empty rows"
        )
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        try:
            row = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_non_finite_constant,
            )
        except json.JSONDecodeError as error:
            raise Day14KVCacheError(
                f"comparison row {index} is invalid JSON: {error}"
            ) from error
        if not isinstance(row, dict):
            raise Day14KVCacheError(
                f"comparison row {index} must be a JSON object"
            )
        rows.append(row)
    if summary.get("run_id") != run_id:
        raise Day14KVCacheError("correctness summary run ID mismatch")
    if summary.get("protocol_fingerprint") != canonical_sha256(protocol):
        raise Day14KVCacheError(
            "correctness summary protocol fingerprint mismatch"
        )
    if summary.get("protocol_id") != protocol["protocol_id"]:
        raise Day14KVCacheError("correctness summary protocol ID mismatch")
    if summary.get("gate") != "PASS":
        raise Day14KVCacheError("correctness summary gate is not PASS")
    if summary.get("status") != "complete":
        raise Day14KVCacheError("correctness summary status is not complete")
    if summary.get("format_name") != (
        "small_gpt_day14_kv_cache_correctness_summary"
    ):
        raise Day14KVCacheError("correctness summary format changed")
    comparisons_payload = (root / "comparisons.jsonl").read_bytes()
    published_files = summary.get("published_files")
    if not isinstance(published_files, dict):
        raise Day14KVCacheError(
            "correctness published-files mapping is missing"
        )
    if set(published_files) != {"comparisons.jsonl"}:
        raise Day14KVCacheError(
            "correctness published-files set changed"
        )
    expected_identity = published_files.get("comparisons.jsonl")
    actual_identity = {
        "bytes": len(comparisons_payload),
        "sha256": sha256_bytes(comparisons_payload),
    }
    if expected_identity != actual_identity:
        raise Day14KVCacheError(
            "correctness comparisons identity mismatch"
        )
    if [row.get("sequence_index") for row in rows] != list(range(len(rows))):
        raise Day14KVCacheError(
            "correctness sequence_index must be contiguous and 0-based"
        )
    comparison = summary.get("comparison")
    if not isinstance(comparison, dict):
        raise Day14KVCacheError("correctness comparison summary is missing")
    if comparison.get("comparison_position_count") != len(rows):
        raise Day14KVCacheError("correctness comparison row count mismatch")
    if comparison.get("pass") is not True:
        raise Day14KVCacheError("correctness comparison did not pass")
    if any(row.get("within_tolerance") is not True for row in rows):
        raise Day14KVCacheError("one or more positions exceed tolerance")
    if any(row.get("all_finite") is not True for row in rows):
        raise Day14KVCacheError("one or more positions are non-finite")
    context_boundaries = summary.get("context_boundaries")
    if not isinstance(context_boundaries, dict):
        raise Day14KVCacheError("correctness boundary summary is missing")
    if context_boundaries.get("pass") is not True:
        raise Day14KVCacheError("correctness boundary summary did not pass")
    scenario_name = summary.get("scenario")
    scenario = next(
        (
            item
            for item in protocol["prompt_builder"]["scenarios"]
            if item["name"] == scenario_name
        ),
        None,
    )
    if scenario is None:
        raise Day14KVCacheError("correctness summary scenario is unknown")
    expected_row_count = int(scenario["max_new_tokens"])
    if len(rows) != expected_row_count:
        raise Day14KVCacheError(
            "correctness row count does not match frozen scenario"
        )
    runtime = summary.get("runtime")
    if not isinstance(runtime, dict):
        raise Day14KVCacheError("correctness runtime identity is missing")
    tolerance_name = (
        "jetson_fp16"
        if runtime.get("precision") == "fp16"
        else "jetson_fp32"
    )
    expected_tolerance = protocol["correctness"][tolerance_name]
    summary_tolerance = summary.get("tolerance", {})
    if summary_tolerance != {
        "rtol": float(expected_tolerance["rtol"]),
        "atol": float(expected_tolerance["atol"]),
        "relaxed_after_failure": False,
    }:
        raise Day14KVCacheError("correctness tolerance contract changed")
    prompt_length = int(scenario["prompt_length"])
    architecture = protocol["architecture"]
    precision = runtime.get("precision")
    if precision not in {"fp32", "fp16"}:
        raise Day14KVCacheError("correctness precision is invalid")
    expected_dtype = "torch.float16" if precision == "fp16" else "torch.float32"
    if runtime.get("dtype") != expected_dtype:
        raise Day14KVCacheError("correctness runtime dtype changed")
    if not str(runtime.get("device", "")).startswith("cuda"):
        raise Day14KVCacheError("correctness runtime is not CUDA")
    model_load_seconds = runtime.get("model_load_seconds")
    if (
        not isinstance(model_load_seconds, (int, float))
        or isinstance(model_load_seconds, bool)
        or not math.isfinite(float(model_load_seconds))
        or float(model_load_seconds) <= 0.0
    ):
        raise Day14KVCacheError("correctness model load time is invalid")
    bytes_per_element = 2 if precision == "fp16" else 4
    for index, row in enumerate(rows):
        expected_cache_length = prompt_length + index
        expected_shape = [
            1,
            int(architecture["head_count"]),
            expected_cache_length,
            int(architecture["head_dimension"]),
        ]
        if row.get("prefix_length") != expected_cache_length:
            raise Day14KVCacheError("correctness prefix length changed")
        if row.get("expected_cache_length") != expected_cache_length:
            raise Day14KVCacheError("correctness expected cache length changed")
        if row.get("actual_cache_length") != expected_cache_length:
            raise Day14KVCacheError("correctness actual cache length changed")
        if row.get("cache_key_shape") != expected_shape:
            raise Day14KVCacheError("correctness cache key shape changed")
        if row.get("cache_value_shape") != expected_shape:
            raise Day14KVCacheError("correctness cache value shape changed")
        if row.get("cache_layer_count") != int(architecture["layer_count"]):
            raise Day14KVCacheError("correctness cache layer count changed")
        expected_payload_bytes = (
            2
            * int(architecture["layer_count"])
            * int(architecture["head_count"])
            * expected_cache_length
            * int(architecture["head_dimension"])
            * bytes_per_element
        )
        if row.get("cache_payload_bytes") != expected_payload_bytes:
            raise Day14KVCacheError("correctness cache payload bytes changed")
        for field in ("maximum_absolute_error", "mean_absolute_error"):
            value = row.get(field)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise Day14KVCacheError(
                    f"correctness row {field} is invalid"
                )
        if row.get("rtol") != float(expected_tolerance["rtol"]):
            raise Day14KVCacheError("correctness row rtol changed")
        if row.get("atol") != float(expected_tolerance["atol"]):
            raise Day14KVCacheError("correctness row atol changed")
        if row.get("argmax_exact_match") is not True:
            raise Day14KVCacheError("correctness row argmax mismatch")
        for field in (
            "maximum_error_token_id",
            "reference_argmax",
            "cached_argmax",
        ):
            value = row.get(field)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                or value >= int(architecture["vocab_size"])
            ):
                raise Day14KVCacheError(
                    f"correctness row {field} is outside vocabulary"
                )
        overlap = row.get("top5_token_set_overlap")
        if (
            not isinstance(overlap, int)
            or isinstance(overlap, bool)
            or not 0 <= overlap <= 5
        ):
            raise Day14KVCacheError("correctness top-5 overlap is invalid")
        if row.get("finite_count") != 2 * int(architecture["vocab_size"]):
            raise Day14KVCacheError("correctness finite count changed")
    if rows[-1].get("actual_cache_length") != int(
        scenario["expected_final_cache_length"]
    ):
        raise Day14KVCacheError("correctness final cache length mismatch")
    required_comparison_flags = (
        "generated_token_ids_exact_match",
        "all_logits_finite",
        "all_positions_within_tolerance",
        "all_argmax_exact_match",
        "parameter_count_stable",
        "state_dict_key_set_stable",
    )
    if any(
        comparison.get(field) is not True
        for field in required_comparison_flags
    ):
        raise Day14KVCacheError(
            "correctness comparison invariant is not true"
        )
    if comparison.get("generated_token_count") != expected_row_count:
        raise Day14KVCacheError("correctness generated token count changed")
    reference_ids = comparison.get("reference_generated_token_ids")
    cached_ids = comparison.get("cached_generated_token_ids")
    if (
        not isinstance(reference_ids, list)
        or not isinstance(cached_ids, list)
        or len(reference_ids) != expected_row_count
        or reference_ids != cached_ids
    ):
        raise Day14KVCacheError("correctness generated sequences changed")
    if comparison.get("finite_count") != (
        expected_row_count * 2 * int(architecture["vocab_size"])
    ):
        raise Day14KVCacheError("correctness aggregate finite count changed")
    for field in ("maximum_absolute_error", "mean_absolute_error"):
        value = comparison.get(field)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise Day14KVCacheError(
                f"correctness aggregate {field} is invalid"
            )
    maximum_error_index = comparison.get("maximum_error_index")
    if not isinstance(maximum_error_index, dict):
        raise Day14KVCacheError("correctness maximum-error index is missing")
    if maximum_error_index.get("sequence_index") not in range(
        expected_row_count
    ):
        raise Day14KVCacheError("correctness maximum-error sequence is invalid")
    maximum_error_token = maximum_error_index.get("token_id")
    if (
        not isinstance(maximum_error_token, int)
        or isinstance(maximum_error_token, bool)
        or maximum_error_token < 0
        or maximum_error_token >= int(architecture["vocab_size"])
    ):
        raise Day14KVCacheError("correctness maximum-error token is invalid")
    minimum_top5 = comparison.get("minimum_top5_token_set_overlap")
    if (
        not isinstance(minimum_top5, int)
        or isinstance(minimum_top5, bool)
        or not 0 <= minimum_top5 <= 5
    ):
        raise Day14KVCacheError("correctness minimum top-5 overlap is invalid")
    expected_boundaries = {
        "prompt_context_minus_one_generate_one_allowed": True,
        "prompt_context_generate_one_rejected": True,
        "cache_context_then_append_one_rejected": True,
        "inconsistent_layer_lengths_rejected": True,
        "context_length": int(architecture["context_length"]),
        "pass": True,
    }
    if context_boundaries != expected_boundaries:
        raise Day14KVCacheError("correctness boundary contract changed")
    model_summary = summary.get("model")
    if not isinstance(model_summary, dict):
        raise Day14KVCacheError("correctness model summary is missing")
    if model_summary.get("parameters") != int(architecture["parameters"]):
        raise Day14KVCacheError("correctness parameter count changed")
    if model_summary.get("state_dict_key_count") != int(
        architecture["state_dict_key_count"]
    ):
        raise Day14KVCacheError("correctness state key count changed")
    checkpoint = summary.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise Day14KVCacheError("correctness checkpoint summary is missing")
    if checkpoint.get("load_mode") != "model_only":
        raise Day14KVCacheError("correctness checkpoint load mode changed")
    if checkpoint.get("strict_state_dict_load") is not True:
        raise Day14KVCacheError("correctness strict checkpoint load is false")
    if checkpoint.get("run_id") != protocol["frozen_artifacts"][
        "control_checkpoint"
    ]["run_id"]:
        raise Day14KVCacheError("correctness checkpoint run ID changed")
    for field in ("missing_keys", "unexpected_keys"):
        if checkpoint.get(field) != 0:
            raise Day14KVCacheError(
                f"correctness checkpoint {field} is nonzero"
            )
    for field in (
        "optimizer_state_restored",
        "scheduler_state_restored",
        "training_resume",
    ):
        if checkpoint.get(field) is not False:
            raise Day14KVCacheError(
                f"correctness checkpoint safety field {field} changed"
            )
    if model_summary.get("training") is not False:
        raise Day14KVCacheError("correctness model reports training mode")
    if model_summary.get("weight_tied") is not True:
        raise Day14KVCacheError("correctness model weight tying changed")
    strategies = summary.get("strategies")
    if strategies != {
        "reference": protocol["benchmark"]["reference_strategy"],
        "cached": protocol["benchmark"]["cached_strategy"],
        "decoding": "greedy",
        "stop_on_eos": False,
    }:
        raise Day14KVCacheError("correctness strategy contract changed")
    cache_contract = summary.get("cache_contract")
    if not isinstance(cache_contract, dict):
        raise Day14KVCacheError("correctness cache contract is missing")
    if cache_contract.get("layer_count") != int(architecture["layer_count"]):
        raise Day14KVCacheError("correctness cache contract layer count changed")
    if cache_contract.get("head_count") != int(architecture["head_count"]):
        raise Day14KVCacheError("correctness cache contract head count changed")
    if cache_contract.get("head_dimension") != int(
        architecture["head_dimension"]
    ):
        raise Day14KVCacheError("correctness cache contract head dimension changed")
    if cache_contract.get("dtype") != expected_dtype:
        raise Day14KVCacheError("correctness cache contract dtype changed")
    if cache_contract.get("device") != runtime.get("device"):
        raise Day14KVCacheError("correctness cache contract device changed")
    source = summary.get("source")
    if not isinstance(source, dict):
        raise Day14KVCacheError("correctness source identity is missing")
    if source.get("branch") != protocol["source"]["required_branch"]:
        raise Day14KVCacheError("correctness source branch changed")
    if source.get("remote_url") != protocol["source"]["remote_url"]:
        raise Day14KVCacheError("correctness source remote changed")
    if source.get("worktree_entries") != 0:
        raise Day14KVCacheError("correctness source worktree was not clean")
    if re.fullmatch(r"[0-9a-f]{40}", str(source.get("head"))) is None:
        raise Day14KVCacheError("correctness source HEAD is invalid")
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, dict):
        raise Day14KVCacheError("correctness artifact identities are missing")
    artifact_specs = {
        "config": protocol["frozen_artifacts"]["baseline_config"],
        "checkpoint": protocol["frozen_artifacts"]["control_checkpoint"],
        "tokenizer": protocol["frozen_artifacts"]["tokenizer_json"],
        "tokenizer_config": protocol["frozen_artifacts"]["tokenizer_config"],
    }
    for label, spec in artifact_specs.items():
        identity = artifacts.get(label)
        if not isinstance(identity, dict):
            raise Day14KVCacheError(
                f"correctness artifact {label} is missing"
            )
        if identity.get("bytes") != int(spec["bytes"]):
            raise Day14KVCacheError(
                f"correctness artifact {label} byte count changed"
            )
        if identity.get("sha256") != spec["sha256"]:
            raise Day14KVCacheError(
                f"correctness artifact {label} hash changed"
            )
    safety = summary.get("safety")
    if not isinstance(safety, dict):
        raise Day14KVCacheError("correctness safety summary is missing")
    for field in (
        "formal_test_access",
        "training_attempted",
        "backward_called",
        "optimizer_created",
        "checkpoint_written",
    ):
        if safety.get(field) is not False:
            raise Day14KVCacheError(
                f"correctness safety field {field} is not false"
            )
    return {
        "format_name": "small_gpt_day14_kv_cache_correctness_validation",
        "schema_version": 1,
        "status": "complete",
        "gate": "PASS",
        "run_id": run_id,
        "file_count": len(actual_names),
        "comparison_row_count": len(rows),
        "sequence_index_base": 0,
        "comparisons_identity_valid": True,
        "output_exact_set": True,
        "context_boundary_pass": True,
        "training_attempted": False,
        "checkpoint_written": False,
    }


def run_correctness(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    from scripts import benchmark_day14_kv_cache as benchmark

    run_id = validate_run_id(args.run_id, mode="correctness")
    assert_ntp_synchronized()
    output_candidate = preflight_output_directory(
        require_external_output_path(args.output_dir),
        run_id=run_id,
    )
    session = load_runtime_session(
        protocol_path=args.protocol,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        tokenizer_config_path=args.tokenizer_config,
        requested_device=args.device,
        precision=args.precision,
        expected_functional_head=args.expected_functional_head,
        project_root=PROJECT_ROOT,
    )
    prompts = benchmark.materialize_protocol_prompts(
        session.protocol,
        session.tokenizer,
    )
    scenario = next(
        item
        for item in session.protocol["prompt_builder"]["scenarios"]
        if item["name"] == args.scenario
    )
    tolerance = session.protocol["correctness"][
        "jetson_fp16" if args.precision == "fp16" else "jetson_fp32"
    ]
    output_dir = reserve_output_directory(output_candidate, run_id=run_id)
    try:
        comparison = benchmark.run_stepwise_correctness(
            session.model,
            prompts[args.scenario],
            max_new_tokens=int(scenario["max_new_tokens"]),
            rtol=float(tolerance["rtol"]),
            atol=float(tolerance["atol"]),
        )
        boundaries = benchmark.run_context_boundary_checks(session.model)
        if comparison["pass"] is not True:
            raise Day14KVCacheError("reference/cached correctness comparison failed")
        if boundaries["pass"] is not True:
            raise Day14KVCacheError("context boundary checks failed")
        comparison_path = atomic_write_bytes_exclusive(
            output_dir / "comparisons.jsonl",
            strict_jsonl_bytes(comparison["rows"]),
        )
        summary = {
            "format_name": "small_gpt_day14_kv_cache_correctness_summary",
            "schema_version": 1,
            "status": "complete",
            "gate": "PASS",
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_id": session.protocol["protocol_id"],
            "protocol_fingerprint": session.protocol_fingerprint,
            "source": session.source.to_dict(),
            "artifacts": {
                "config": session.config_identity.to_dict(),
                "checkpoint": session.checkpoint_identity.to_dict(),
                "tokenizer": session.tokenizer_identity.to_dict(),
                "tokenizer_config": session.tokenizer_config_identity.to_dict(),
            },
            "runtime": {
                "device": str(session.device),
                "precision": session.precision,
                "dtype": str(session.dtype),
                "model_load_seconds": session.model_load_seconds,
            },
            "scenario": args.scenario,
            "strategies": {
                "reference": benchmark.REFERENCE_STRATEGY,
                "cached": benchmark.CACHED_STRATEGY,
                "decoding": "greedy",
                "stop_on_eos": False,
            },
            "checkpoint": {
                "run_id": getattr(
                    getattr(session.loaded_checkpoint, "state", None),
                    "run_id",
                    None,
                ),
                "global_step": getattr(
                    getattr(session.loaded_checkpoint, "record", None),
                    "global_step",
                    None,
                ),
                "tokens_seen": getattr(
                    getattr(session.loaded_checkpoint, "record", None),
                    "tokens_seen",
                    None,
                ),
                "load_mode": "model_only",
                "strict_state_dict_load": True,
                "missing_keys": 0,
                "unexpected_keys": 0,
                "optimizer_state_restored": False,
                "scheduler_state_restored": False,
                "training_resume": False,
            },
            "model": {
                "parameters": sum(
                    parameter.numel()
                    for parameter in session.model.parameters()
                ),
                "state_dict_key_count": len(session.model.state_dict()),
                "training": False,
                "weight_tied": True,
            },
            "cache_contract": {
                "layer_count": int(session.model_config.n_layer),
                "head_count": int(session.model_config.n_head),
                "head_dimension": int(session.model_config.head_dim),
                "shape_order": [
                    "batch_size",
                    "head_count",
                    "cache_sequence_length",
                    "head_dimension",
                ],
                "dtype": str(session.dtype),
                "device": str(session.device),
            },
            "tolerance": {
                "rtol": float(tolerance["rtol"]),
                "atol": float(tolerance["atol"]),
                "relaxed_after_failure": False,
            },
            "comparison": {
                key: value
                for key, value in comparison.items()
                if key != "rows"
            },
            "context_boundaries": boundaries,
            "published_files": {
                "comparisons.jsonl": {
                    "bytes": comparison_path.stat().st_size,
                    "sha256": sha256_bytes(comparison_path.read_bytes()),
                }
            },
            "safety": {
                "formal_test_access": False,
                "training_attempted": False,
                "backward_called": False,
                "optimizer_created": False,
                "checkpoint_written": False,
            },
        }
        summary_path = atomic_write_json_exclusive(
            output_dir / "correctness-summary.json",
            summary,
        )
        validation = validate_correctness_output(
            output_dir=output_dir,
            protocol=session.protocol,
            run_id=run_id,
        )
        return {**summary, "post_publication_validation": validation}, summary_path
    except BaseException as error:
        failure_path = output_dir / "failure.json"
        if not failure_path.exists():
            try:
                atomic_write_json_exclusive(
                    failure_path,
                    _failure_document(error, mode="correctness", run_id=run_id),
                )
            except BaseException:
                pass
        raise


def build_dry_run_summary(protocol: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "format_name": "small_gpt_day14_kv_cache_dry_run",
        "schema_version": 1,
        "status": "complete",
        "gate": "PASS",
        "dry_run": True,
        "protocol_loaded": True,
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": canonical_sha256(protocol),
        "source_head_expected": protocol["source"]["day13_final_head"],
        "functional_file_count": len(
            protocol["mutation_scope"]["functional_files"]
        ),
        "parameters_expected": protocol["architecture"]["parameters"],
        "state_dict_keys_expected": protocol["architecture"][
            "state_dict_key_count"
        ],
        "context_length": protocol["architecture"]["context_length"],
        "reference_strategy": protocol["benchmark"]["reference_strategy"],
        "cached_strategy": protocol["benchmark"]["cached_strategy"],
        "checkpoint_read": False,
        "config_read": False,
        "tokenizer_read": False,
        "model_constructed": False,
        "cuda_queried": False,
        "cuda_allocated": False,
        "generation_executed": False,
        "formal_test_access": False,
        "training_attempted": False,
        "backward_called": False,
        "optimizer_created": False,
        "checkpoint_written": False,
        "output_written": False,
        "jetson_execution_claim": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the frozen Day 14 KV-cache protocol in dry-run, "
            "model-only load, or reference/cached correctness mode."
        )
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "configs" / "day14_kv_cache_protocol.json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "baseline.yaml",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=(
            PROJECT_ROOT
            / "tokenizer"
            / "artifacts"
            / "tokenizer.json"
        ),
    )
    parser.add_argument("--tokenizer-config", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument(
        "--mode",
        choices=("dry-run", "load-only", "correctness"),
        required=True,
    )
    parser.add_argument("--expected-functional-head")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--scenario",
        choices=("bridge", "short", "medium", "long"),
        default="short",
    )
    parser.add_argument("--validate-only", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        protocol = load_protocol(args.protocol)
        if args.mode == "dry-run":
            if args.validate_only:
                raise Day14KVCacheError(
                    "dry-run does not use --validate-only"
                )
            if args.output is not None or args.output_dir is not None:
                raise Day14KVCacheError(
                    "dry-run forbids --output/--output-dir and performs no writes"
                )
            summary = build_dry_run_summary(protocol)
        elif args.mode == "load-only":
            if args.validate_only:
                raise Day14KVCacheError(
                    "load-only does not use --validate-only"
                )
            if args.checkpoint is None:
                raise Day14KVCacheError("load-only requires --checkpoint")
            if args.tokenizer_config is None:
                raise Day14KVCacheError("load-only requires --tokenizer-config")
            if args.expected_functional_head is None:
                raise Day14KVCacheError(
                    "load-only requires --expected-functional-head"
                )
            if args.output is None:
                raise Day14KVCacheError("load-only requires --output")
            if args.output_dir is not None:
                raise Day14KVCacheError("load-only forbids --output-dir")
            if args.output.exists():
                raise Day14KVCacheError(
                    f"output already exists: {args.output.resolve()}"
                )
            require_external_output_path(args.output)
            session = load_runtime_session(
                protocol_path=args.protocol,
                config_path=args.config,
                checkpoint_path=args.checkpoint,
                tokenizer_path=args.tokenizer,
                tokenizer_config_path=args.tokenizer_config,
                requested_device=args.device,
                precision=args.precision,
                expected_functional_head=args.expected_functional_head,
                project_root=PROJECT_ROOT,
            )
            summary = build_load_only_summary(session)
            atomic_write_json_exclusive(args.output, summary)
        else:
            if args.validate_only:
                if args.run_id is None:
                    raise Day14KVCacheError(
                        "correctness validation requires --run-id"
                    )
                if args.output_dir is None:
                    raise Day14KVCacheError(
                        "correctness validation requires --output-dir"
                    )
                if args.output is not None:
                    raise Day14KVCacheError(
                        "correctness validation forbids --output"
                    )
                summary = validate_correctness_output(
                    output_dir=args.output_dir,
                    protocol=protocol,
                    run_id=args.run_id,
                )
                print(
                    json.dumps(
                        summary,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                )
                return 0
            if args.checkpoint is None:
                raise Day14KVCacheError("correctness requires --checkpoint")
            if args.tokenizer_config is None:
                raise Day14KVCacheError(
                    "correctness requires --tokenizer-config"
                )
            if args.expected_functional_head is None:
                raise Day14KVCacheError(
                    "correctness requires --expected-functional-head"
                )
            if args.run_id is None:
                raise Day14KVCacheError("correctness requires --run-id")
            if args.output_dir is None:
                raise Day14KVCacheError("correctness requires --output-dir")
            if args.output is not None:
                raise Day14KVCacheError("correctness forbids --output")
            summary, _ = run_correctness(args)
        print(
            json.dumps(
                summary,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
        return 0
    except Day14KVCacheError as error:
        print(f"Day 14 KV Cache error: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(
            "Day 14 KV Cache unexpected error: "
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
