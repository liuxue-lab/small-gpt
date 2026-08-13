"""Strict autoregressive generation for a completed small-gpt checkpoint."""

from __future__ import annotations

import json
import math
import os
import re
import time
import uuid
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import torch
import tokenizers as tokenizers_library
from torch import nn

from model import GPT, GPTConfig
from tokenizer import encode_text, load_tokenizer
from train import PrecisionPolicy, load_model_checkpoint

from .frozen_evaluation import sha256_file


GENERATION_FORMAT_NAME = "small_gpt_text_generation"
GENERATION_SCHEMA_VERSION = 1
_GENERATION_STRATEGIES = ("greedy", "sample")
_SPECIAL_TOKEN_IDS = {
    "bos": 0,
    "eos": 1,
    "pad": 2,
    "unk": 3,
}
_SPECIAL_TOKEN_TEXT = {
    "bos": "<bos>",
    "eos": "<eos>",
    "pad": "<pad>",
    "unk": "<unk>",
}
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_GIT_OBJECT_PATTERN = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_MAX_SEED = (1 << 63) - 1


class GenerationError(RuntimeError):
    """Raised when generation would violate the frozen protocol."""


def _is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _normalized_sha256(value: str, *, field: str) -> str:
    if not isinstance(value, str):
        raise GenerationError(f"{field} must be a SHA-256 string")
    normalized = value.lower()
    if _SHA256_PATTERN.fullmatch(normalized) is None:
        raise GenerationError(f"{field} must be a SHA-256 string")
    return normalized


@dataclass(frozen=True, slots=True)
class GenerationSettings:
    """One explicit decoding strategy and its reproducibility controls."""

    strategy: str
    max_new_tokens: int
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.strategy not in _GENERATION_STRATEGIES:
            raise GenerationError(
                f"strategy must be one of {_GENERATION_STRATEGIES}, "
                f"got {self.strategy!r}"
            )
        if not _is_plain_int(self.max_new_tokens) or self.max_new_tokens <= 0:
            raise GenerationError("max_new_tokens must be a positive integer")
        if not _is_finite_number(self.temperature) or self.temperature <= 0.0:
            raise GenerationError("temperature must be finite and positive")
        if self.top_k is not None and (
            not _is_plain_int(self.top_k) or self.top_k <= 0
        ):
            raise GenerationError("top_k must be a positive integer or null")
        if self.top_p is not None and (
            not _is_finite_number(self.top_p)
            or not 0.0 < float(self.top_p) <= 1.0
        ):
            raise GenerationError("top_p must be in (0, 1] or null")
        if self.seed is not None and (
            not _is_plain_int(self.seed)
            or self.seed < 0
            or self.seed > _MAX_SEED
        ):
            raise GenerationError(
                f"seed must be an integer in [0, {_MAX_SEED}] or null"
            )

        if self.strategy == "greedy":
            if float(self.temperature) != 1.0:
                raise GenerationError(
                    "greedy generation requires temperature=1.0"
                )
            if self.top_k is not None or self.top_p is not None:
                raise GenerationError(
                    "greedy generation does not accept top_k or top_p"
                )
            if self.seed is not None:
                raise GenerationError("greedy generation does not accept a seed")
        elif self.seed is None:
            raise GenerationError(
                "sample generation requires an explicit non-negative seed"
            )

    def validate_for_vocab(self, vocab_size: int) -> None:
        if not _is_plain_int(vocab_size) or vocab_size <= 0:
            raise GenerationError("vocab_size must be a positive integer")
        if self.top_k is not None and self.top_k > vocab_size:
            raise GenerationError(
                f"top_k exceeds model vocabulary: {self.top_k} > {vocab_size}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "temperature": float(self.temperature),
            "top_p": None if self.top_p is None else float(self.top_p),
        }


@dataclass(frozen=True, slots=True)
class GenerationTrace:
    """Raw token-level output from one batch-size-one decode."""

    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    stop_reason: str
    initial_prompt_tokens_discarded: int
    context_crop_events: int
    forward_passes: int
    elapsed_seconds: float

    @property
    def full_token_ids(self) -> tuple[int, ...]:
        return self.prompt_token_ids + self.generated_token_ids


def _model_contract(model: nn.Module) -> tuple[int, int]:
    config = getattr(model, "config", None)
    context_length = getattr(config, "context_length", None)
    vocab_size = getattr(config, "vocab_size", None)
    if not _is_plain_int(context_length) or context_length <= 0:
        raise GenerationError(
            "model.config.context_length must be a positive integer"
        )
    if not _is_plain_int(vocab_size) or vocab_size <= 0:
        raise GenerationError("model.config.vocab_size must be a positive integer")
    return context_length, vocab_size


def _validated_prompt_tokens(
    prompt_token_ids: Sequence[int],
    *,
    vocab_size: int,
) -> tuple[int, ...]:
    if isinstance(prompt_token_ids, (str, bytes)) or not isinstance(
        prompt_token_ids,
        Sequence,
    ):
        raise GenerationError("prompt_token_ids must be a sequence of integers")
    normalized = tuple(prompt_token_ids)
    if not normalized:
        raise GenerationError("prompt_token_ids must not be empty")
    for index, token_id in enumerate(normalized):
        if not _is_plain_int(token_id):
            raise GenerationError(
                f"prompt_token_ids[{index}] must be an integer"
            )
        if token_id < 0 or token_id >= vocab_size:
            raise GenerationError(
                f"prompt_token_ids[{index}] is outside [0, {vocab_size})"
            )
    return normalized


def _validate_model_device(
    model: nn.Module,
    precision: PrecisionPolicy,
) -> None:
    wrong_parameters = [
        name
        for name, parameter in model.named_parameters()
        if parameter.device != precision.device
    ]
    wrong_buffers = [
        name
        for name, buffer in model.named_buffers()
        if buffer.device != precision.device
    ]
    if wrong_parameters or wrong_buffers:
        raise GenerationError(
            "model tensors are not on the generation device: "
            f"parameters={wrong_parameters[:5]}, buffers={wrong_buffers[:5]}"
        )


def _sample_next_token(
    logits: torch.Tensor,
    *,
    settings: GenerationSettings,
    generator: torch.Generator,
) -> int:
    filtered = logits / float(settings.temperature)
    if not bool(torch.isfinite(filtered).all().item()):
        raise GenerationError("temperature-scaled logits are non-finite")
    if settings.top_k is not None:
        top_values, top_indices = torch.topk(filtered, settings.top_k)
        top_k_filtered = torch.full_like(filtered, -torch.inf)
        top_k_filtered.scatter_(0, top_indices, top_values)
        filtered = top_k_filtered

    if settings.top_p is not None:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
        sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
        cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)
        remove = cumulative_probabilities > float(settings.top_p)
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
        probabilities = torch.softmax(sorted_logits, dim=-1)
        if not bool(torch.isfinite(probabilities).all().item()):
            raise GenerationError("top-p sampling probabilities are non-finite")
        sampled_sorted_index = torch.multinomial(
            probabilities,
            num_samples=1,
            generator=generator,
        )
        return int(sorted_indices[sampled_sorted_index].item())

    probabilities = torch.softmax(filtered, dim=-1)
    if not bool(torch.isfinite(probabilities).all().item()):
        raise GenerationError("sampling probabilities are non-finite")
    return int(
        torch.multinomial(
            probabilities,
            num_samples=1,
            generator=generator,
        ).item()
    )


def generate_token_ids(
    model: nn.Module,
    prompt_token_ids: Sequence[int],
    *,
    settings: GenerationSettings,
    precision: PrecisionPolicy,
    eos_token_id: int = _SPECIAL_TOKEN_IDS["eos"],
) -> GenerationTrace:
    """Generate one continuation without mutating global Torch RNG state."""

    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model)!r}")
    if not isinstance(settings, GenerationSettings):
        raise TypeError(
            f"settings must be GenerationSettings, got {type(settings)!r}"
        )
    if not isinstance(precision, PrecisionPolicy):
        raise TypeError(
            f"precision must be PrecisionPolicy, got {type(precision)!r}"
        )

    context_length, vocab_size = _model_contract(model)
    settings.validate_for_vocab(vocab_size)
    prompt = _validated_prompt_tokens(
        prompt_token_ids,
        vocab_size=vocab_size,
    )
    if not _is_plain_int(eos_token_id) or not 0 <= eos_token_id < vocab_size:
        raise GenerationError(
            f"eos_token_id must be inside [0, {vocab_size})"
        )
    _validate_model_device(model, precision)

    generator: torch.Generator | None = None
    if settings.strategy == "sample":
        try:
            generator = torch.Generator(device=precision.device)
            generator.manual_seed(settings.seed)
        except Exception as error:
            raise GenerationError(
                f"could not initialize the sampling generator: {error}"
            ) from error

    full_sequence = list(prompt)
    generated: list[int] = []
    stop_reason = "max_new_tokens"
    context_crop_events = 0
    forward_passes = 0
    was_training = model.training
    start_time = time.perf_counter()

    model.eval()
    try:
        with torch.inference_mode():
            for _ in range(settings.max_new_tokens):
                if len(full_sequence) > context_length:
                    context_crop_events += 1
                conditioning = full_sequence[-context_length:]
                input_ids = torch.tensor(
                    [conditioning],
                    dtype=torch.long,
                    device=precision.device,
                )
                with precision.autocast_context():
                    output = model(input_ids)
                logits = getattr(output, "logits", None)
                if not isinstance(logits, torch.Tensor):
                    raise GenerationError(
                        "model output must expose a Tensor logits field"
                    )
                expected_shape = (1, len(conditioning), vocab_size)
                if tuple(logits.shape) != expected_shape:
                    raise GenerationError(
                        "model logits have the wrong shape: "
                        f"expected {expected_shape}, got {tuple(logits.shape)}"
                    )
                next_logits = logits[0, -1].float()
                if not bool(torch.isfinite(next_logits).all().item()):
                    raise GenerationError("next-token logits are non-finite")

                if settings.strategy == "greedy":
                    next_token = int(torch.argmax(next_logits).item())
                else:
                    if generator is None:  # defensive; settings enforces this
                        raise GenerationError("sampling generator is unavailable")
                    next_token = _sample_next_token(
                        next_logits,
                        settings=settings,
                        generator=generator,
                    )

                generated.append(next_token)
                full_sequence.append(next_token)
                forward_passes += 1
                if next_token == eos_token_id:
                    stop_reason = "eos"
                    break
    finally:
        model.train(was_training)

    elapsed_seconds = max(time.perf_counter() - start_time, 1.0e-12)
    return GenerationTrace(
        prompt_token_ids=prompt,
        generated_token_ids=tuple(generated),
        stop_reason=stop_reason,
        initial_prompt_tokens_discarded=max(0, len(prompt) - context_length),
        context_crop_events=context_crop_events,
        forward_passes=forward_passes,
        elapsed_seconds=elapsed_seconds,
    )


def _validate_tokenizer_contract(
    tokenizer: Any,
    *,
    expected_vocab_size: int,
) -> dict[str, Any]:
    try:
        actual_vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    except Exception as error:
        raise GenerationError(
            f"could not inspect tokenizer vocabulary: {error}"
        ) from error
    if actual_vocab_size != expected_vocab_size:
        raise GenerationError(
            "tokenizer vocabulary does not match the model: "
            f"{actual_vocab_size} != {expected_vocab_size}"
        )

    actual_special_ids: dict[str, int] = {}
    for name, expected_id in _SPECIAL_TOKEN_IDS.items():
        token_text = _SPECIAL_TOKEN_TEXT[name]
        try:
            actual_id = tokenizer.token_to_id(token_text)
        except Exception as error:
            raise GenerationError(
                f"could not inspect tokenizer special token {token_text}: {error}"
            ) from error
        if actual_id != expected_id:
            raise GenerationError(
                f"tokenizer special token {token_text} has ID {actual_id}; "
                f"expected {expected_id}"
            )
        actual_special_ids[name] = actual_id
    return {
        "library": "tokenizers",
        "library_version": tokenizers_library.__version__,
        "vocab_size": actual_vocab_size,
        "special_token_ids": actual_special_ids,
    }


def _generation_runtime_payload(
    precision: PrecisionPolicy,
) -> dict[str, Any]:
    runtime: dict[str, Any] = {
        **precision.to_dict(),
        "torch_version": str(torch.__version__),
        "deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "generation_allow_tf32": False,
        "reproducibility_scope": "same_hardware_and_runtime",
    }
    if precision.device.type == "cuda":
        runtime.update(
            {
                "cuda_device_name": torch.cuda.get_device_name(precision.device),
                "cuda_matmul_allow_tf32": (
                    torch.backends.cuda.matmul.allow_tf32
                ),
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
            }
        )
    else:
        runtime.update(
            {
                "cuda_device_name": None,
                "cuda_matmul_allow_tf32": None,
                "cudnn_allow_tf32": None,
                "cudnn_deterministic": None,
                "cudnn_benchmark": None,
            }
        )
    return runtime


@contextmanager
def _configured_generation_runtime(
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
        torch.use_deterministic_algorithms(True)
        if precision.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        yield _generation_runtime_payload(precision)
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


def generate_from_checkpoint(
    checkpoint_path: str | Path,
    tokenizer_path: str | Path,
    prompt: str,
    *,
    model_config: GPTConfig,
    expected_run_id: str,
    expected_checkpoint_sha256: str,
    expected_tokenizer_sha256: str,
    settings: GenerationSettings,
    precision: PrecisionPolicy,
    generator_source_commit: str | None = None,
    generator_source_dirty: bool = True,
) -> dict[str, Any]:
    """Strictly load immutable artifacts and generate one auditable sample."""

    if not isinstance(model_config, GPTConfig):
        raise TypeError(
            f"model_config must be GPTConfig, got {type(model_config)!r}"
        )
    if not isinstance(settings, GenerationSettings):
        raise TypeError(
            f"settings must be GenerationSettings, got {type(settings)!r}"
        )
    if not isinstance(precision, PrecisionPolicy):
        raise TypeError(
            f"precision must be PrecisionPolicy, got {type(precision)!r}"
        )
    if not isinstance(expected_run_id, str) or not expected_run_id.strip():
        raise GenerationError("expected_run_id must be a non-empty string")
    if not isinstance(prompt, str) or not prompt.strip():
        raise GenerationError("prompt must be a non-empty, non-whitespace string")
    if generator_source_commit is not None and (
        not isinstance(generator_source_commit, str)
        or _GIT_OBJECT_PATTERN.fullmatch(generator_source_commit) is None
    ):
        raise GenerationError(
            "generator_source_commit must be a full lowercase Git object ID or null"
        )
    if not isinstance(generator_source_dirty, bool):
        raise TypeError("generator_source_dirty must be a boolean")

    settings.validate_for_vocab(model_config.vocab_size)
    expected_checkpoint_hash = _normalized_sha256(
        expected_checkpoint_sha256,
        field="expected_checkpoint_sha256",
    )
    expected_tokenizer_hash = _normalized_sha256(
        expected_tokenizer_sha256,
        field="expected_tokenizer_sha256",
    )

    resolved_checkpoint = Path(checkpoint_path).resolve()
    resolved_tokenizer = Path(tokenizer_path).resolve()
    try:
        checkpoint_sha256 = sha256_file(resolved_checkpoint)
    except OSError as error:
        raise GenerationError(
            f"could not hash checkpoint {resolved_checkpoint}: {error}"
        ) from error
    if checkpoint_sha256 != expected_checkpoint_hash:
        raise GenerationError(
            "checkpoint SHA-256 mismatch: "
            f"expected {expected_checkpoint_hash}, found {checkpoint_sha256}"
        )
    try:
        tokenizer_sha256 = sha256_file(resolved_tokenizer)
    except OSError as error:
        raise GenerationError(
            f"could not hash tokenizer {resolved_tokenizer}: {error}"
        ) from error
    if tokenizer_sha256 != expected_tokenizer_hash:
        raise GenerationError(
            "tokenizer SHA-256 mismatch: "
            f"expected {expected_tokenizer_hash}, found {tokenizer_sha256}"
        )

    try:
        tokenizer = load_tokenizer(resolved_tokenizer)
    except Exception as error:
        raise GenerationError(f"could not load tokenizer: {error}") from error
    tokenizer_contract = _validate_tokenizer_contract(
        tokenizer,
        expected_vocab_size=model_config.vocab_size,
    )

    cuda_rng_devices = (
        [] if precision.device.type == "cpu" else [precision.device.index]
    )
    with torch.random.fork_rng(devices=cuda_rng_devices):
        model = GPT(model_config).to(precision.device)
    loaded = load_model_checkpoint(
        resolved_checkpoint,
        model=model,
        expected_model_config=model_config.to_dict(),
        expected_run_id=expected_run_id,
    )
    if loaded.identity.tokenizer_sha256 != tokenizer_sha256:
        raise GenerationError(
            "tokenizer SHA-256 does not match checkpoint identity: "
            f"checkpoint={loaded.identity.tokenizer_sha256}, "
            f"tokenizer={tokenizer_sha256}"
        )
    if model.lm_head.weight is not model.token_embedding.weight:
        raise GenerationError(
            "model input/output embeddings are not tied after checkpoint load"
        )

    try:
        prompt_token_ids = encode_text(tokenizer, prompt)
    except Exception as error:
        raise GenerationError(f"could not tokenize prompt: {error}") from error
    if not prompt_token_ids:
        raise GenerationError("prompt produced no tokenizer IDs")

    with _configured_generation_runtime(precision) as runtime:
        trace = generate_token_ids(
            model,
            prompt_token_ids,
            settings=settings,
            precision=precision,
            eos_token_id=_SPECIAL_TOKEN_IDS["eos"],
        )

    try:
        decoded_prompt = tokenizer.decode(
            list(trace.prompt_token_ids),
            skip_special_tokens=True,
        )
        continuation_text = tokenizer.decode(
            list(trace.generated_token_ids),
            skip_special_tokens=True,
        )
        full_text = tokenizer.decode(
            list(trace.full_token_ids),
            skip_special_tokens=True,
        )
    except Exception as error:
        raise GenerationError(f"could not decode generated IDs: {error}") from error

    initial_conditioning = trace.prompt_token_ids[-model_config.context_length :]
    return {
        "format_name": GENERATION_FORMAT_NAME,
        "schema_version": GENERATION_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": loaded.state.run_id,
        "checkpoint": {
            "path": _resolved_path(resolved_checkpoint),
            "bytes": loaded.record.file_size,
            "sha256": checkpoint_sha256,
            "global_step": loaded.record.global_step,
            "tokens_seen": loaded.record.tokens_seen,
            "identity": loaded.identity.to_dict(),
        },
        "generator": {
            "source_commit": generator_source_commit,
            "source_dirty": generator_source_dirty,
        },
        "model": {
            "config": model_config.to_dict(),
            "parameters": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "weight_tying_verified": True,
        },
        "tokenizer": {
            "path": _resolved_path(resolved_tokenizer),
            "bytes": resolved_tokenizer.stat().st_size,
            "sha256": tokenizer_sha256,
            **tokenizer_contract,
        },
        "protocol": {
            **settings.to_dict(),
            "batch_size": 1,
            "add_bos": False,
            "append_eos_to_prompt": False,
            "eos_token_id": _SPECIAL_TOKEN_IDS["eos"],
            "context_policy": "left_crop_conditioning_window",
            "kv_cache": False,
            "filter_order": ["temperature", "top_k", "top_p"],
        },
        "prompt": {
            "text": prompt,
            "decoded_text": decoded_prompt,
            "token_ids": list(trace.prompt_token_ids),
            "token_count": len(trace.prompt_token_ids),
            "initial_conditioning_token_ids": list(initial_conditioning),
            "initial_conditioning_token_count": len(initial_conditioning),
            "initial_tokens_discarded": (
                trace.initial_prompt_tokens_discarded
            ),
        },
        "generation": {
            "token_ids": list(trace.generated_token_ids),
            "token_count": len(trace.generated_token_ids),
            "full_token_ids": list(trace.full_token_ids),
            "continuation_text": continuation_text,
            "full_text": full_text,
            "stop_reason": trace.stop_reason,
            "eos_generated": trace.stop_reason == "eos",
            "context_crop_events": trace.context_crop_events,
            "forward_passes": trace.forward_passes,
            "elapsed_seconds": trace.elapsed_seconds,
        },
        "runtime": runtime,
    }


def publish_generation_result(
    path: str | Path,
    result: Mapping[str, Any],
) -> Path:
    """Atomically publish strict JSON without replacing existing evidence."""

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
        raise GenerationError(
            f"generation result is not strict JSON: {error}"
        ) from error

    output_path = Path(path).resolve()
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise GenerationError(
            f"could not create generation output directory: {error}"
        ) from error
    if output_path.exists():
        raise GenerationError(
            "generation output already exists and will not be overwritten: "
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
        raise GenerationError(
            "generation output already exists and will not be overwritten: "
            f"{output_path}"
        ) from error
    except OSError as error:
        raise GenerationError(
            f"could not atomically publish generation result: {error}"
        ) from error
    finally:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
    return output_path
