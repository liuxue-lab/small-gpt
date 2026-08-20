"""Strict, inference-only deployment gate for the frozen Day 13 Jetson run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
import tokenizers as tokenizers_library  # noqa: E402
from torch import nn  # noqa: E402

from model import GPT, GPTConfig  # noqa: E402
from tokenizer import encode_text, load_tokenizer  # noqa: E402
from train import LoadedCheckpoint, load_model_checkpoint  # noqa: E402


FORMAT_NAME = "small_gpt_jetson_deployment_protocol"
SCHEMA_VERSION = 1
PROTOCOL_ID = "day13-jetson-pytorch-inference-v1"
SMOKE_PROTOCOL_ID = "day13-jetson-greedy-smoke-v1"
BENCHMARK_PROTOCOL_ID = "day13-jetson-benchmark-v1"
STABILITY_PROTOCOL_ID = "day13-jetson-stability-v1"
STARTING_REPOSITORY_HEAD = "0319f80766991eead65556df564497036605d1a3"
CONTROL_RUN_ID = "baseline-full-300m-20260813-232952"
EXPECTED_CONFIG_BYTES = 1_258
EXPECTED_CONFIG_SHA256 = (
    "ca8524c425e1e5e3a600de5773f9a526ef3674741040635bee91fe31f4b24c0e"
)
EXPECTED_CHECKPOINT_BYTES = 406_108_827
EXPECTED_CHECKPOINT_SHA256 = (
    "a39f8378ebe4012afb992be451d355e814b856ffb5e690ac011758f9db614b51"
)
EXPECTED_TOKENIZER_BYTES = 1_137_073
EXPECTED_TOKENIZER_SHA256 = (
    "b26835e02eebf777a257c4732abdd6f9732a115967d2ad839f3a1a00e45ee8c5"
)
EXPECTED_TOKENIZER_CONFIG_BYTES = 2_988
EXPECTED_TOKENIZER_CONFIG_SHA256 = (
    "8622711407aab3f299996b7d3009d4f4447ae35879ca8e50451b5f0adbdf5141"
)
EXPECTED_PARAMETERS = 33_833_984
EXPECTED_VOCAB_SIZE = 16_384
EXPECTED_CONTEXT_LENGTH = 512
EXPECTED_DROPOUT = 0.0
EXPECTED_SPECIAL_TOKEN_IDS = {"bos": 0, "eos": 1, "pad": 2, "unk": 3}
EXPECTED_SPECIAL_TOKEN_TEXT = {
    "bos": "<bos>",
    "eos": "<eos>",
    "pad": "<pad>",
    "unk": "<unk>",
}
EXPECTED_PROMPTS = (
    ("prompt_01", "Once upon a time"),
    ("prompt_02", "The solar system"),
    ("prompt_03", "To make a cup of tea"),
)
STRICT_STATE_DICT_LOAD = True

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_GIT_OBJECT_PATTERN = re.compile(r"[0-9a-f]{40}")
_CUDA_DEVICE_PATTERN = re.compile(r"cuda(?::([0-9]+))?")
_TOP_LEVEL_FIELDS = frozenset(
    {
        "format_name",
        "schema_version",
        "protocol_id",
        "status",
        "starting_repository_head",
        "functional_source_mode",
        "model_role",
        "control_run_id",
        "config",
        "checkpoint",
        "tokenizer",
        "model",
        "smoke",
        "benchmark",
        "stability",
        "runtime",
        "implementation",
        "safety",
    }
)
_CONFIG_FIELDS = frozenset({"path", "bytes", "sha256"})
_CHECKPOINT_FIELDS = frozenset({"bytes", "sha256", "load_mode"})
_TOKENIZER_FIELDS = frozenset(
    {"bytes", "sha256", "vocab_size", "special_token_ids", "config"}
)
_MODEL_FIELDS = frozenset({"parameters", "dropout", "context_length"})
_SMOKE_FIELDS = frozenset(
    {
        "protocol_id",
        "batch_size",
        "max_new_tokens",
        "stop_on_eos",
        "decoding",
        "seed",
        "precisions",
        "expected_generated_tokens_per_precision",
        "prompts",
    }
)
_BENCHMARK_FIELDS = frozenset(
    {
        "protocol_id",
        "prompt_id",
        "batch_size",
        "warmup_runs",
        "measured_runs",
        "max_new_tokens",
        "stop_on_eos",
        "decoding",
        "synchronize_cuda",
        "reset_peak_memory_before_run",
        "summary_statistics",
    }
)
_STABILITY_FIELDS = frozenset(
    {
        "protocol_id",
        "prompt_id",
        "batch_size",
        "sequential_requests",
        "max_new_tokens",
        "stop_on_eos",
        "decoding",
        "concurrent_requests",
        "expected_total_generated_tokens",
    }
)
_RUNTIME_FIELDS = frozenset(
    {
        "route",
        "jetson_linux_version",
        "python_version",
        "torch_version",
        "torch_cuda_version",
        "torch_cudnn_version",
        "runtime_identity_sha256",
        "tokenizers_version",
        "tokenizers_wheel",
        "tensor_rt_enabled",
    }
)
_WHEEL_FIELDS = frozenset({"filename", "bytes", "sha256"})
_IMPLEMENTATION_FIELDS = frozenset(
    {"kv_cache_enabled", "decode_implementation", "fp16_mode"}
)
_SAFETY_FIELDS = frozenset(
    {
        "formal_test_access",
        "overwrite_outputs",
        "training_allowed",
        "cross_machine_bitwise_claim",
        "power_mode_change_allowed",
        "jetson_clocks_allowed",
    }
)
_CREDENTIAL_KEY_FRAGMENTS = (
    "password",
    "passphrase",
    "private_key",
    "secret",
    "ssh_key",
    "username",
    "ip_address",
)
_FORMAL_TEST_PATH_PREFIXES = (
    "data/test",
    "data/formal-test",
    "data/formal_test",
    "tests/",
)


class DeploymentError(RuntimeError):
    """Raised when a frozen deployment identity or runtime gate fails."""


class ProtocolError(DeploymentError):
    """Raised when the checked-in Day 13 protocol is malformed or changed."""


class ArtifactIdentityError(DeploymentError):
    """Raised when an immutable deployment artifact does not match its lock."""


class CudaDeviceError(DeploymentError):
    """Raised when a requested CUDA device cannot be used exactly as requested."""


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
    head: str
    dirty: bool
    branch: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DeploymentSession:
    protocol: dict[str, Any]
    protocol_fingerprint: str
    model: GPT
    tokenizer: Any
    model_config: GPTConfig
    loaded_checkpoint: LoadedCheckpoint
    device: torch.device
    precision: str
    weight_dtype: torch.dtype
    compute_dtype: torch.dtype
    source: SourceIdentity
    config_identity: FileIdentity
    checkpoint_identity: FileIdentity
    tokenizer_identity: FileIdentity
    tokenizer_config_identity: FileIdentity | None
    tokenizer_contract: dict[str, Any]
    runtime: dict[str, Any]
    model_load_seconds: float
    forward_probe: dict[str, Any] | None


def _is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _strict_equal(left: object, right: object) -> bool:
    return type(left) is type(right) and left == right


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolError(message)


def _require_mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{field} must be a JSON object")
    return value


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    field: str,
) -> None:
    provided = set(value)
    missing = expected - provided
    unknown = provided - expected
    if missing:
        raise ProtocolError(f"{field} is missing fields: {sorted(missing)}")
    if unknown:
        raise ProtocolError(f"{field} has unknown fields: {sorted(unknown)}")


def _require_frozen(
    value: Mapping[str, Any],
    key: str,
    expected: object,
    *,
    prefix: str,
) -> None:
    actual = value[key]
    if not _strict_equal(actual, expected):
        raise ProtocolError(
            f"{prefix}.{key} must equal {expected!r}, got {actual!r}"
        )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_non_finite_constant(value: str) -> None:
    raise ProtocolError(f"protocol contains non-finite numeric constant {value!r}")


def canonical_sha256(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ProtocolError(f"value is not canonical strict JSON: {error}") from error
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_key_values(value: object, *, key_path: str = ""):
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{key_path}.{key}" if key_path else str(key)
            yield path, str(key), item
            yield from _iter_key_values(item, key_path=path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            path = f"{key_path}[{index}]"
            yield from _iter_key_values(item, key_path=path)


def assert_no_credentials(document: Mapping[str, Any]) -> None:
    for key_path, key, _ in _iter_key_values(document):
        lowered = key.lower()
        if any(fragment in lowered for fragment in _CREDENTIAL_KEY_FRAGMENTS):
            raise ProtocolError(
                f"protocol must not contain credential or host field {key_path!r}"
            )


def assert_no_formal_test_paths(document: Mapping[str, Any]) -> None:
    for key_path, key, value in _iter_key_values(document):
        lowered_key = key.lower()
        if not (lowered_key.endswith("path") or lowered_key.endswith("paths")):
            continue
        candidates: Sequence[object]
        if isinstance(value, list):
            candidates = value
        else:
            candidates = (value,)
        for candidate in candidates:
            if not isinstance(candidate, str):
                continue
            normalized = candidate.replace("\\", "/").lower().lstrip("./")
            if any(
                normalized == prefix.rstrip("/")
                or normalized.startswith(prefix)
                for prefix in _FORMAL_TEST_PATH_PREFIXES
            ):
                raise ProtocolError(
                    f"formal test path is forbidden at {key_path}: {candidate!r}"
                )


def load_protocol(path: str | Path) -> dict[str, Any]:
    protocol_path = Path(path).resolve()
    try:
        raw = protocol_path.read_bytes()
    except OSError as error:
        raise ProtocolError(f"could not read protocol {protocol_path}: {error}") from error
    try:
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite_constant,
        )
    except ProtocolError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProtocolError(f"protocol must be valid UTF-8 JSON: {error}") from error
    if not isinstance(document, dict):
        raise ProtocolError("protocol root must be a JSON object")
    validate_protocol_document(document)
    return document


def validate_protocol_document(document: Mapping[str, Any]) -> None:
    """Validate the complete frozen protocol and reject any hidden expansion."""

    assert_no_credentials(document)
    assert_no_formal_test_paths(document)
    _require_exact_fields(document, _TOP_LEVEL_FIELDS, field="protocol")

    frozen_top = {
        "format_name": FORMAT_NAME,
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "frozen_after_user_approval",
        "starting_repository_head": STARTING_REPOSITORY_HEAD,
        "functional_source_mode": "runtime_clean_git_head",
        "model_role": "control",
        "control_run_id": CONTROL_RUN_ID,
    }
    for key, expected in frozen_top.items():
        _require_frozen(document, key, expected, prefix="protocol")

    config = _require_mapping(document["config"], field="config")
    _require_exact_fields(config, _CONFIG_FIELDS, field="config")
    for key, expected in {
        "path": "configs/baseline.yaml",
        "bytes": EXPECTED_CONFIG_BYTES,
        "sha256": EXPECTED_CONFIG_SHA256,
    }.items():
        _require_frozen(config, key, expected, prefix="config")

    checkpoint = _require_mapping(document["checkpoint"], field="checkpoint")
    _require_exact_fields(checkpoint, _CHECKPOINT_FIELDS, field="checkpoint")
    for key, expected in {
        "bytes": EXPECTED_CHECKPOINT_BYTES,
        "sha256": EXPECTED_CHECKPOINT_SHA256,
        "load_mode": "model_only",
    }.items():
        _require_frozen(checkpoint, key, expected, prefix="checkpoint")

    tokenizer = _require_mapping(document["tokenizer"], field="tokenizer")
    _require_exact_fields(tokenizer, _TOKENIZER_FIELDS, field="tokenizer")
    for key, expected in {
        "bytes": EXPECTED_TOKENIZER_BYTES,
        "sha256": EXPECTED_TOKENIZER_SHA256,
        "vocab_size": EXPECTED_VOCAB_SIZE,
    }.items():
        _require_frozen(tokenizer, key, expected, prefix="tokenizer")
    _require(
        _strict_equal(tokenizer["special_token_ids"], EXPECTED_SPECIAL_TOKEN_IDS),
        "tokenizer.special_token_ids do not match the frozen IDs",
    )
    tokenizer_config = _require_mapping(
        tokenizer["config"], field="tokenizer.config"
    )
    _require_exact_fields(tokenizer_config, frozenset({"bytes", "sha256"}), field="tokenizer.config")
    _require_frozen(
        tokenizer_config,
        "bytes",
        EXPECTED_TOKENIZER_CONFIG_BYTES,
        prefix="tokenizer.config",
    )
    _require_frozen(
        tokenizer_config,
        "sha256",
        EXPECTED_TOKENIZER_CONFIG_SHA256,
        prefix="tokenizer.config",
    )

    model = _require_mapping(document["model"], field="model")
    _require_exact_fields(model, _MODEL_FIELDS, field="model")
    for key, expected in {
        "parameters": EXPECTED_PARAMETERS,
        "dropout": EXPECTED_DROPOUT,
        "context_length": EXPECTED_CONTEXT_LENGTH,
    }.items():
        _require_frozen(model, key, expected, prefix="model")

    smoke = _require_mapping(document["smoke"], field="smoke")
    _require_exact_fields(smoke, _SMOKE_FIELDS, field="smoke")
    for key, expected in {
        "protocol_id": SMOKE_PROTOCOL_ID,
        "batch_size": 1,
        "max_new_tokens": 64,
        "stop_on_eos": False,
        "decoding": "greedy",
        "seed": 1337,
        "precisions": ["fp32", "fp16"],
        "expected_generated_tokens_per_precision": 192,
    }.items():
        _require_frozen(smoke, key, expected, prefix="smoke")
    prompts = smoke["prompts"]
    _require(isinstance(prompts, list), "smoke.prompts must be a JSON list")
    normalized_prompts: list[tuple[str, str]] = []
    for index, prompt in enumerate(prompts):
        prompt_mapping = _require_mapping(prompt, field=f"smoke.prompts[{index}]")
        _require_exact_fields(
            prompt_mapping,
            frozenset({"prompt_id", "text"}),
            field=f"smoke.prompts[{index}]",
        )
        prompt_id = prompt_mapping["prompt_id"]
        text = prompt_mapping["text"]
        _require(
            isinstance(prompt_id, str) and bool(prompt_id),
            f"smoke.prompts[{index}].prompt_id must be non-empty",
        )
        _require(
            isinstance(text, str) and bool(text.strip()),
            f"smoke.prompts[{index}].text must be non-empty",
        )
        normalized_prompts.append((prompt_id, text))
    _require(
        tuple(normalized_prompts) == EXPECTED_PROMPTS,
        "smoke.prompts do not match the frozen prompt order and text",
    )

    benchmark = _require_mapping(document["benchmark"], field="benchmark")
    _require_exact_fields(benchmark, _BENCHMARK_FIELDS, field="benchmark")
    expected_benchmark = {
        "protocol_id": BENCHMARK_PROTOCOL_ID,
        "prompt_id": "prompt_02",
        "batch_size": 1,
        "warmup_runs": 3,
        "measured_runs": 10,
        "max_new_tokens": 64,
        "stop_on_eos": False,
        "decoding": "greedy",
        "synchronize_cuda": True,
        "reset_peak_memory_before_run": True,
        "summary_statistics": ["mean", "median", "min", "max", "p95"],
    }
    for key, expected in expected_benchmark.items():
        _require_frozen(benchmark, key, expected, prefix="benchmark")

    stability = _require_mapping(document["stability"], field="stability")
    _require_exact_fields(stability, _STABILITY_FIELDS, field="stability")
    expected_stability = {
        "protocol_id": STABILITY_PROTOCOL_ID,
        "prompt_id": "prompt_02",
        "batch_size": 1,
        "sequential_requests": 10,
        "max_new_tokens": 64,
        "stop_on_eos": False,
        "decoding": "greedy",
        "concurrent_requests": 1,
        "expected_total_generated_tokens": 640,
    }
    for key, expected in expected_stability.items():
        _require_frozen(stability, key, expected, prefix="stability")

    runtime = _require_mapping(document["runtime"], field="runtime")
    _require_exact_fields(runtime, _RUNTIME_FIELDS, field="runtime")
    expected_runtime = {
        "route": "native_existing",
        "jetson_linux_version": "36.4.3",
        "python_version": "3.10.12",
        "torch_version": "2.5.0a0+872d972e41.nv24.08",
        "torch_cuda_version": "12.6",
        "torch_cudnn_version": 90600,
        "runtime_identity_sha256": "44b65f0ac7be423affd46a4a0f80299c49a69ccd7f49385809ba3d5762d3ad40",
        "tokenizers_version": "0.23.1",
        "tensor_rt_enabled": False,
    }
    for key, expected in expected_runtime.items():
        _require_frozen(runtime, key, expected, prefix="runtime")
    wheel = _require_mapping(runtime["tokenizers_wheel"], field="runtime.tokenizers_wheel")
    _require_exact_fields(wheel, _WHEEL_FIELDS, field="runtime.tokenizers_wheel")
    expected_wheel = {
        "filename": "tokenizers-0.23.1-cp310-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl",
        "bytes": 3_374_081,
        "sha256": "1bf13402aff9bc533c89cb849ec3b412dc3fbeacc9744840e423d7bf3f7dc0e3",
    }
    for key, expected in expected_wheel.items():
        _require_frozen(wheel, key, expected, prefix="runtime.tokenizers_wheel")

    implementation = _require_mapping(
        document["implementation"], field="implementation"
    )
    _require_exact_fields(
        implementation, _IMPLEMENTATION_FIELDS, field="implementation"
    )
    expected_implementation = {
        "kv_cache_enabled": False,
        "decode_implementation": "full_prefix_recompute",
        "fp16_mode": "float16_weights_and_ops",
    }
    for key, expected in expected_implementation.items():
        _require_frozen(implementation, key, expected, prefix="implementation")

    safety = _require_mapping(document["safety"], field="safety")
    _require_exact_fields(safety, _SAFETY_FIELDS, field="safety")
    for key in _SAFETY_FIELDS:
        _require_frozen(safety, key, False, prefix="safety")


def verify_file_identity(
    path: str | Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
    label: str,
) -> FileIdentity:
    if not isinstance(path, (str, Path)) or not str(path).strip():
        raise ArtifactIdentityError(f"{label} path must be non-empty")
    if not _is_plain_int(expected_bytes) or expected_bytes <= 0:
        raise ArtifactIdentityError(f"{label} expected bytes must be positive")
    if (
        not isinstance(expected_sha256, str)
        or _SHA256_PATTERN.fullmatch(expected_sha256) is None
    ):
        raise ArtifactIdentityError(f"{label} expected SHA-256 is invalid")
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ArtifactIdentityError(f"{label} does not exist: {resolved}")
    actual_bytes = resolved.stat().st_size
    if actual_bytes != expected_bytes:
        raise ArtifactIdentityError(
            f"{label} byte count mismatch: expected {expected_bytes}, "
            f"found {actual_bytes}"
        )
    try:
        actual_sha256 = sha256_file(resolved)
    except OSError as error:
        raise ArtifactIdentityError(f"could not hash {label}: {error}") from error
    if actual_sha256 != expected_sha256:
        raise ArtifactIdentityError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, "
            f"found {actual_sha256}"
        )
    return FileIdentity(resolved, actual_bytes, actual_sha256)


def resolve_cuda_device(requested: str) -> torch.device:
    if not isinstance(requested, str) or _CUDA_DEVICE_PATTERN.fullmatch(requested) is None:
        raise CudaDeviceError(
            f"device must be 'cuda' or 'cuda:N'; CPU fallback is forbidden, got {requested!r}"
        )
    if not torch.cuda.is_available():
        raise CudaDeviceError(
            "CUDA was requested but torch.cuda.is_available() is False; "
            "CPU fallback is forbidden"
        )
    match = _CUDA_DEVICE_PATTERN.fullmatch(requested)
    if match is None:  # defensive
        raise CudaDeviceError(f"invalid CUDA device {requested!r}")
    if match.group(1) is None:
        try:
            index = int(torch.cuda.current_device())
        except Exception as error:
            raise CudaDeviceError(f"could not resolve current CUDA device: {error}") from error
    else:
        index = int(match.group(1))
    count = int(torch.cuda.device_count())
    if index < 0 or index >= count:
        raise CudaDeviceError(
            f"CUDA device index {index} is outside the visible range [0, {count})"
        )
    return torch.device("cuda", index)


def _run_git(project_root: Path, arguments: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
    except OSError as error:
        raise DeploymentError(f"could not execute git: {error}") from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise DeploymentError(
            f"git {' '.join(arguments)} failed with exit {completed.returncode}: {detail}"
        )
    return completed.stdout.strip()


def git_source_identity(project_root: str | Path = PROJECT_ROOT) -> SourceIdentity:
    root = Path(project_root).resolve()
    head = _run_git(root, ("rev-parse", "HEAD"))
    if _GIT_OBJECT_PATTERN.fullmatch(head) is None:
        raise DeploymentError(f"source HEAD is not a full lowercase Git commit: {head!r}")
    branch_text = _run_git(root, ("branch", "--show-current"))
    status = _run_git(root, ("status", "--porcelain=v1", "--untracked-files=all"))
    return SourceIdentity(
        head=head,
        dirty=bool(status),
        branch=branch_text or None,
    )


def validate_runtime_source(
    source: SourceIdentity,
    *,
    project_root: str | Path = PROJECT_ROOT,
    starting_head: str = STARTING_REPOSITORY_HEAD,
) -> None:
    if source.dirty:
        raise DeploymentError("runtime source worktree is dirty")
    if _GIT_OBJECT_PATTERN.fullmatch(source.head) is None:
        raise DeploymentError("runtime source HEAD is invalid")
    root = Path(project_root).resolve()
    try:
        completed = subprocess.run(
            ["git", "merge-base", "--is-ancestor", starting_head, source.head],
            cwd=root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
    except OSError as error:
        raise DeploymentError(f"could not verify source ancestry: {error}") from error
    if completed.returncode != 0:
        raise DeploymentError(
            "runtime source HEAD does not descend from the frozen starting commit"
        )


def _l4t_version() -> str | None:
    path = Path("/etc/nv_tegra_release")
    try:
        first_line = path.read_text(encoding="utf-8").splitlines()[0]
    except (OSError, IndexError, UnicodeDecodeError):
        return None
    match = re.search(r"R([0-9]+).*REVISION:\s*([0-9]+)\.([0-9]+)", first_line)
    if match is None:
        return None
    return f"{match.group(1)}.{match.group(2)}.{match.group(3)}"


def validate_runtime_lock(protocol: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    expected = protocol["runtime"]
    actual_python = platform.python_version()
    actual_torch = str(torch.__version__)
    actual_cuda = str(torch.version.cuda)
    actual_cudnn_raw = torch.backends.cudnn.version()
    actual_cudnn = None if actual_cudnn_raw is None else int(actual_cudnn_raw)
    actual_tokenizers = str(tokenizers_library.__version__)
    actual_l4t = _l4t_version()
    comparisons = {
        "python_version": actual_python,
        "torch_version": actual_torch,
        "torch_cuda_version": actual_cuda,
        "torch_cudnn_version": actual_cudnn,
        "tokenizers_version": actual_tokenizers,
        "jetson_linux_version": actual_l4t,
    }
    for field, actual in comparisons.items():
        expected_value = expected[field]
        if not _strict_equal(actual, expected_value):
            raise DeploymentError(
                f"runtime {field} mismatch: expected {expected_value!r}, found {actual!r}"
            )
    try:
        device_name = torch.cuda.get_device_name(device)
        capability = list(torch.cuda.get_device_capability(device))
    except Exception as error:
        raise DeploymentError(f"could not inspect CUDA device {device}: {error}") from error
    return {
        **comparisons,
        "device": str(device),
        "cuda_device_name": device_name,
        "cuda_compute_capability": capability,
        "runtime_route": expected["route"],
        "runtime_identity_sha256_expected": expected["runtime_identity_sha256"],
        "tensor_rt_enabled": False,
    }


def validate_tokenizer_contract(
    tokenizer: Any,
    *,
    expected_vocab_size: int = EXPECTED_VOCAB_SIZE,
    expected_special_ids: Mapping[str, int] = EXPECTED_SPECIAL_TOKEN_IDS,
) -> dict[str, Any]:
    try:
        vocab_size = int(tokenizer.get_vocab_size(with_added_tokens=True))
    except Exception as error:
        raise DeploymentError(f"could not inspect tokenizer vocabulary: {error}") from error
    if vocab_size != expected_vocab_size:
        raise DeploymentError(
            f"tokenizer vocabulary mismatch: expected {expected_vocab_size}, found {vocab_size}"
        )
    actual_special_ids: dict[str, int] = {}
    for name, expected_id in expected_special_ids.items():
        token_text = EXPECTED_SPECIAL_TOKEN_TEXT[name]
        try:
            actual_id = tokenizer.token_to_id(token_text)
        except Exception as error:
            raise DeploymentError(
                f"could not inspect tokenizer token {token_text}: {error}"
            ) from error
        if actual_id != expected_id:
            raise DeploymentError(
                f"tokenizer token {token_text} has ID {actual_id}; expected {expected_id}"
            )
        actual_special_ids[name] = int(actual_id)

    probe_text = "Once upon a time"
    try:
        token_ids = list(encode_text(tokenizer, probe_text))
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
    except Exception as error:
        raise DeploymentError(f"tokenizer runtime probe failed: {error}") from error
    if not token_ids:
        raise DeploymentError("tokenizer runtime probe produced no token IDs")
    if any(not _is_plain_int(value) or not 0 <= value < vocab_size for value in token_ids):
        raise DeploymentError("tokenizer runtime probe produced an out-of-range token ID")
    if not isinstance(decoded, str) or not decoded.strip():
        raise DeploymentError("tokenizer runtime probe decoded to empty text")
    unknown_count = sum(value == expected_special_ids["unk"] for value in token_ids)
    if unknown_count != 0:
        raise DeploymentError("tokenizer runtime probe unexpectedly produced <unk>")
    return {
        "library": "tokenizers",
        "library_version": str(tokenizers_library.__version__),
        "vocab_size": vocab_size,
        "special_token_ids": actual_special_ids,
        "probe_text": probe_text,
        "probe_token_ids": token_ids,
        "probe_token_count": len(token_ids),
        "probe_decoded_text": decoded,
        "probe_unknown_token_count": unknown_count,
        "probe_token_ids_in_range": True,
    }


def precision_dtype(precision: str) -> torch.dtype:
    if precision == "fp32":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    raise DeploymentError(f"precision must be 'fp32' or 'fp16', got {precision!r}")


def model_only_load(
    checkpoint_path: str | Path,
    *,
    model: nn.Module,
    model_config: GPTConfig,
    expected_run_id: str,
) -> LoadedCheckpoint:
    """Use the project's audited model-only loader; it enforces strict=True."""

    if not STRICT_STATE_DICT_LOAD:
        raise DeploymentError("strict model-state loading is not enabled")
    return load_model_checkpoint(
        checkpoint_path,
        model=model,
        expected_model_config=model_config.to_dict(),
        expected_run_id=expected_run_id,
    )


def validate_model_runtime_contract(
    model: nn.Module,
    *,
    model_config: Any,
    protocol: Mapping[str, Any],
    device: torch.device,
    expected_dtype: torch.dtype,
) -> dict[str, Any]:
    expected_model = protocol["model"]
    expected_tokenizer = protocol["tokenizer"]
    if float(model_config.dropout) != float(expected_model["dropout"]):
        raise DeploymentError(
            f"model dropout mismatch: {model_config.dropout} != {expected_model['dropout']}"
        )
    if float(model_config.dropout) != 0.0:
        raise DeploymentError("deployment model dropout must equal 0.0")
    if int(model_config.vocab_size) != int(expected_tokenizer["vocab_size"]):
        raise DeploymentError("model and tokenizer vocabulary sizes do not match")
    if int(model_config.context_length) != int(expected_model["context_length"]):
        raise DeploymentError("model context length does not match protocol")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != int(expected_model["parameters"]):
        raise DeploymentError(
            f"model parameter count mismatch: {parameter_count} != {expected_model['parameters']}"
        )
    if model.training:
        raise DeploymentError("model.training is True; eval mode is required")
    wrong_devices = [
        name for name, parameter in model.named_parameters() if parameter.device != device
    ]
    wrong_dtypes = [
        name
        for name, parameter in model.named_parameters()
        if parameter.is_floating_point() and parameter.dtype != expected_dtype
    ]
    if wrong_devices:
        raise DeploymentError(
            f"model parameters are not on {device}: {wrong_devices[:5]}"
        )
    if wrong_dtypes:
        raise DeploymentError(
            f"model parameters have the wrong dtype: {wrong_dtypes[:5]}"
        )
    dropout_modules = [module for module in model.modules() if isinstance(module, nn.Dropout)]
    if any(float(module.p) != 0.0 for module in dropout_modules):
        raise DeploymentError("one or more model dropout modules have p != 0.0")
    token_embedding = getattr(model, "token_embedding", None)
    lm_head = getattr(model, "lm_head", None)
    weight_tied = (
        token_embedding is not None
        and lm_head is not None
        and getattr(lm_head, "weight", None) is getattr(token_embedding, "weight", None)
    )
    if not weight_tied:
        raise DeploymentError("model input/output embedding weights are not tied")
    return {
        "parameters": parameter_count,
        "dropout": float(model_config.dropout),
        "context_length": int(model_config.context_length),
        "vocab_size": int(model_config.vocab_size),
        "training": False,
        "eval_mode": True,
        "weight_tying_verified": True,
        "dropout_module_count": len(dropout_modules),
        "weight_dtype": str(expected_dtype),
        "device": str(device),
    }


def validate_token_id(token_id: object, *, vocab_size: int) -> int:
    if not _is_plain_int(token_id):
        raise DeploymentError(f"generated token ID must be an integer, got {token_id!r}")
    if token_id < 0 or token_id >= vocab_size:
        raise DeploymentError(
            f"generated token ID {token_id} is outside [0, {vocab_size})"
        )
    return int(token_id)


def checked_forward(
    model: nn.Module,
    token_ids: Sequence[int],
    *,
    device: torch.device,
    precision: str,
    vocab_size: int,
) -> dict[str, Any]:
    if model.training:
        raise DeploymentError("model must be in eval mode before forward")
    normalized = [validate_token_id(value, vocab_size=vocab_size) for value in token_ids]
    if not normalized:
        raise DeploymentError("forward probe token IDs must not be empty")
    expected_dtype = precision_dtype(precision)
    with torch.inference_mode():
        if not torch.is_inference_mode_enabled():
            raise DeploymentError("torch inference mode did not activate")
        input_ids = torch.tensor([normalized], dtype=torch.long, device=device)
        output = model(input_ids)
        logits = getattr(output, "logits", None)
        if not isinstance(logits, torch.Tensor):
            raise DeploymentError("model output does not expose Tensor logits")
        expected_shape = (1, len(normalized), vocab_size)
        if tuple(logits.shape) != expected_shape:
            raise DeploymentError(
                f"forward logits shape mismatch: expected {expected_shape}, found {tuple(logits.shape)}"
            )
        if logits.device != device:
            raise DeploymentError(
                f"forward logits are on {logits.device}; expected {device}"
            )
        if logits.dtype != expected_dtype:
            raise DeploymentError(
                f"forward logits dtype is {logits.dtype}; expected {expected_dtype}"
            )
        if not bool(torch.isfinite(logits).all().item()):
            raise DeploymentError("forward logits contain NaN or infinity")
    return {
        "input_shape": [1, len(normalized)],
        "logits_shape": [1, len(normalized), vocab_size],
        "logits_finite": True,
        "logits_device": str(device),
        "logits_dtype": str(expected_dtype),
        "token_ids_in_range": True,
        "inference_mode": True,
    }


def load_deployment_session(
    *,
    protocol_path: str | Path,
    config_path: str | Path,
    checkpoint_path: str | Path,
    tokenizer_path: str | Path,
    tokenizer_config_path: str | Path | None,
    requested_device: str,
    precision: str,
    project_root: str | Path = PROJECT_ROOT,
    enforce_source: bool = True,
    perform_forward_probe: bool = True,
) -> DeploymentSession:
    protocol = load_protocol(protocol_path)
    protocol_fingerprint = canonical_sha256(protocol)
    config_identity = verify_file_identity(
        config_path,
        expected_bytes=protocol["config"]["bytes"],
        expected_sha256=protocol["config"]["sha256"],
        label="baseline config",
    )
    checkpoint_identity = verify_file_identity(
        checkpoint_path,
        expected_bytes=protocol["checkpoint"]["bytes"],
        expected_sha256=protocol["checkpoint"]["sha256"],
        label="control checkpoint",
    )
    tokenizer_identity = verify_file_identity(
        tokenizer_path,
        expected_bytes=protocol["tokenizer"]["bytes"],
        expected_sha256=protocol["tokenizer"]["sha256"],
        label="tokenizer",
    )
    tokenizer_config_identity: FileIdentity | None = None
    if tokenizer_config_path is not None:
        tokenizer_config_identity = verify_file_identity(
            tokenizer_config_path,
            expected_bytes=protocol["tokenizer"]["config"]["bytes"],
            expected_sha256=protocol["tokenizer"]["config"]["sha256"],
            label="tokenizer config",
        )

    device = resolve_cuda_device(requested_device)
    dtype = precision_dtype(precision)
    runtime = validate_runtime_lock(protocol, device)
    source = git_source_identity(project_root)
    if enforce_source:
        validate_runtime_source(
            source,
            project_root=project_root,
            starting_head=protocol["starting_repository_head"],
        )
    try:
        model_config = GPTConfig.from_yaml(config_identity.path)
    except Exception as error:
        raise DeploymentError(f"could not load baseline model config: {error}") from error
    if float(model_config.dropout) != 0.0:
        raise DeploymentError("baseline config dropout must equal 0.0")
    if int(model_config.vocab_size) != int(protocol["tokenizer"]["vocab_size"]):
        raise DeploymentError("baseline config vocabulary does not match protocol")
    if int(model_config.context_length) != int(protocol["model"]["context_length"]):
        raise DeploymentError("baseline config context length does not match protocol")
    try:
        tokenizer = load_tokenizer(tokenizer_identity.path)
    except Exception as error:
        raise DeploymentError(f"could not load tokenizer: {error}") from error
    tokenizer_contract = validate_tokenizer_contract(
        tokenizer,
        expected_vocab_size=protocol["tokenizer"]["vocab_size"],
        expected_special_ids=protocol["tokenizer"]["special_token_ids"],
    )

    load_started = time.perf_counter()
    try:
        with torch.random.fork_rng(devices=[]):
            model = GPT(model_config)
        loaded = model_only_load(
            checkpoint_identity.path,
            model=model,
            model_config=model_config,
            expected_run_id=protocol["control_run_id"],
        )
        if loaded.identity.tokenizer_sha256 != tokenizer_identity.sha256:
            raise DeploymentError(
                "checkpoint tokenizer identity does not match the frozen tokenizer"
            )
        model.to(device=device, dtype=dtype)
        model.eval()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    except DeploymentError:
        raise
    except Exception as error:
        raise DeploymentError(f"model-only load failed: {error}") from error
    model_load_seconds = max(time.perf_counter() - load_started, 1.0e-12)

    validate_model_runtime_contract(
        model,
        model_config=model_config,
        protocol=protocol,
        device=device,
        expected_dtype=dtype,
    )
    forward_probe: dict[str, Any] | None = None
    if perform_forward_probe:
        forward_probe = checked_forward(
            model,
            tokenizer_contract["probe_token_ids"],
            device=device,
            precision=precision,
            vocab_size=model_config.vocab_size,
        )
    return DeploymentSession(
        protocol=dict(protocol),
        protocol_fingerprint=protocol_fingerprint,
        model=model,
        tokenizer=tokenizer,
        model_config=model_config,
        loaded_checkpoint=loaded,
        device=device,
        precision=precision,
        weight_dtype=dtype,
        compute_dtype=dtype,
        source=source,
        config_identity=config_identity,
        checkpoint_identity=checkpoint_identity,
        tokenizer_identity=tokenizer_identity,
        tokenizer_config_identity=tokenizer_config_identity,
        tokenizer_contract=tokenizer_contract,
        runtime=runtime,
        model_load_seconds=model_load_seconds,
        forward_probe=forward_probe,
    )


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
        raise DeploymentError(f"result is not strict finite JSON: {error}") from error


def atomic_write_bytes_exclusive(path: str | Path, payload: bytes) -> Path:
    output_path = Path(path).resolve()
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise DeploymentError(f"could not create output parent: {error}") from error
    if output_path.exists():
        raise DeploymentError(f"output already exists and will not be overwritten: {output_path}")
    temporary = output_path.parent / f".{output_path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output_path)
    except FileExistsError as error:
        raise DeploymentError(
            f"output already exists and will not be overwritten: {output_path}"
        ) from error
    except OSError as error:
        raise DeploymentError(f"could not atomically publish {output_path}: {error}") from error
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
    return output_path


def atomic_write_json_exclusive(path: str | Path, value: Mapping[str, Any]) -> Path:
    return atomic_write_bytes_exclusive(path, strict_json_bytes(value))


def build_load_only_summary(session: DeploymentSession) -> dict[str, Any]:
    loaded = session.loaded_checkpoint
    return {
        "format_name": "small_gpt_jetson_load_gate",
        "schema_version": 1,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gate": "PASS",
        "protocol_id": session.protocol["protocol_id"],
        "protocol_fingerprint": session.protocol_fingerprint,
        "source": session.source.to_dict(),
        "model_role": "control",
        "control_run_id": loaded.state.run_id,
        "artifacts": {
            "config": session.config_identity.to_dict(),
            "checkpoint": {
                **session.checkpoint_identity.to_dict(),
                "load_mode": "model_only",
                "global_step": loaded.record.global_step,
                "tokens_seen": loaded.record.tokens_seen,
            },
            "tokenizer": session.tokenizer_identity.to_dict(),
            "tokenizer_config": (
                None
                if session.tokenizer_config_identity is None
                else session.tokenizer_config_identity.to_dict()
            ),
        },
        "model": {
            "parameters": sum(parameter.numel() for parameter in session.model.parameters()),
            "dropout": float(session.model_config.dropout),
            "context_length": int(session.model_config.context_length),
            "vocab_size": int(session.model_config.vocab_size),
            "strict_state_dict_load": True,
            "missing_keys": 0,
            "unexpected_keys": 0,
            "training": False,
            "eval_mode": True,
            "optimizer_state_restored": False,
            "scheduler_state_restored": False,
            "training_resume": False,
        },
        "tokenizer": session.tokenizer_contract,
        "runtime": {
            **session.runtime,
            "precision": session.precision,
            "weight_dtype": str(session.weight_dtype),
            "compute_dtype": str(session.compute_dtype),
            "model_load_seconds": session.model_load_seconds,
            "inference_mode": True,
        },
        "forward_probe": session.forward_probe,
        "safety": {
            **session.protocol["safety"],
            "text_generated": False,
            "checkpoint_written": False,
        },
    }


def build_dry_run_summary(protocol: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "format_name": "small_gpt_jetson_deployment_dry_run",
        "schema_version": 1,
        "status": "complete",
        "gate": "PASS",
        "dry_run": True,
        "protocol_loaded": True,
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": canonical_sha256(protocol),
        "checkpoint_read": False,
        "config_read": False,
        "tokenizer_read": False,
        "device": "cpu",
        "precision": "fp32",
        "parameters_expected": protocol["model"]["parameters"],
        "formal_test_access": protocol["safety"]["formal_test_access"],
        "training_allowed": protocol["safety"]["training_allowed"],
        "output_written": False,
        "jetson_deployment_claim": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the frozen Day 13 Jetson deployment in dry-run or "
            "CUDA model-only load mode."
        )
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "configs" / "day13_jetson_deployment_protocol.json",
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
        default=PROJECT_ROOT / "tokenizer" / "artifacts" / "tokenizer.json",
    )
    parser.add_argument("--tokenizer-config", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--mode", choices=("dry-run", "load-only"), required=True)
    parser.add_argument("--output", type=Path)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        protocol = load_protocol(args.protocol)
        if args.mode == "dry-run":
            if args.output is not None:
                raise DeploymentError("dry-run forbids --output and performs no writes")
            summary = build_dry_run_summary(protocol)
        else:
            if args.checkpoint is None:
                raise DeploymentError("load-only requires --checkpoint")
            if args.tokenizer_config is None:
                raise DeploymentError("load-only requires --tokenizer-config")
            if args.output is None:
                raise DeploymentError("load-only requires --output")
            if args.output.exists():
                raise DeploymentError(
                    f"output already exists and will not be overwritten: {args.output.resolve()}"
                )
            session = load_deployment_session(
                protocol_path=args.protocol,
                config_path=args.config,
                checkpoint_path=args.checkpoint,
                tokenizer_path=args.tokenizer,
                tokenizer_config_path=args.tokenizer_config,
                requested_device=args.device,
                precision=args.precision,
                project_root=PROJECT_ROOT,
                enforce_source=True,
                perform_forward_probe=True,
            )
            summary = build_load_only_summary(session)
            atomic_write_json_exclusive(args.output, summary)
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
    except (DeploymentError, OSError, ValueError, TypeError) as error:
        payload = {
            "format_name": "small_gpt_jetson_deployment_error",
            "schema_version": 1,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        print(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
