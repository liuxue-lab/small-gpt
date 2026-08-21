"""Reference and KV-cache greedy generation primitives for Day 14."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


sys.dont_write_bytecode = True


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from scripts.check_day14_kv_cache import (  # noqa: E402
    Day14KVCacheError,
    assert_ntp_synchronized,
    atomic_write_bytes_exclusive,
    atomic_write_json_exclusive,
    canonical_sha256,
    load_protocol,
    load_runtime_session,
    preflight_output_directory,
    repeat_truncate,
    require_external_output_path,
    reserve_output_directory,
    sha256_bytes,
    strict_jsonl_bytes,
    token_ids_sha256,
    validate_run_id,
    validate_token_ids,
)
from tokenizer import encode_text  # noqa: E402


REFERENCE_STRATEGY = "full_prefix_recompute"
CACHED_STRATEGY = "kv_cache"
PAIRED_STRATEGY = "paired"
MANIFEST_FILENAME = "manifest.json"
SAMPLES_FILENAME = "samples.jsonl"
SUMMARY_FILENAME = "benchmark-summary.json"
TEGRASTATS_FILENAME = "tegrastats.log"
FAILURE_FILENAME = "failure.json"


class GenerationError(Day14KVCacheError):
    """Raised when an inference-only generation contract is violated."""


@dataclass(frozen=True, slots=True)
class GenerationTrace:
    """Strategy-neutral result schema for one greedy request."""

    decode_strategy: str
    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    returned_token_ids: tuple[int, ...]
    prefill_input_tokens: int
    final_cache_length: int
    cache_layer_count: int
    cache_payload_bytes: int
    cache_theoretical_bytes: int
    prompt_preparation_seconds: float
    prefill_seconds: float
    ttft_seconds: float
    decode_seconds: float
    decode_token_count: int
    request_wall_seconds: float
    per_token_latency_seconds: tuple[float, ...]
    device: str
    dtype: str
    inference_mode: bool
    all_logits_finite: bool

    def to_dict(self) -> dict[str, Any]:
        decode_tokens_per_second: float | None = None
        if self.decode_token_count > 0 and self.decode_seconds > 0.0:
            decode_tokens_per_second = (
                self.decode_token_count / self.decode_seconds
            )
        end_to_end_tokens_per_second: float | None = None
        if self.generated_token_ids and self.request_wall_seconds > 0.0:
            end_to_end_tokens_per_second = (
                len(self.generated_token_ids) / self.request_wall_seconds
            )
        return {
            "format_name": "small_gpt_day14_generation_trace",
            "schema_version": 1,
            "decode_strategy": self.decode_strategy,
            "prompt": {
                "token_ids": list(self.prompt_token_ids),
                "token_count": len(self.prompt_token_ids),
            },
            "generation": {
                "decoding": "greedy",
                "stop_on_eos": False,
                "token_ids": list(self.generated_token_ids),
                "token_count": len(self.generated_token_ids),
                "returned_token_ids": list(self.returned_token_ids),
                "returned_sequence_length": len(self.returned_token_ids),
                "all_token_ids_in_range": True,
                "all_logits_finite": self.all_logits_finite,
                "context_crop_events": 0,
                "stop_reason": "fixed_max_new_tokens",
            },
            "cache": {
                "prefill_input_tokens": self.prefill_input_tokens,
                "final_cache_length": self.final_cache_length,
                "cache_layer_count": self.cache_layer_count,
                "cache_payload_bytes": self.cache_payload_bytes,
                "cache_theoretical_bytes": self.cache_theoretical_bytes,
                "query_cached": False,
                "input_past_modified_in_place": False,
                "global_model_cache_used": False,
            },
            "timing": {
                "prompt_preparation_seconds": self.prompt_preparation_seconds,
                "prefill_seconds": self.prefill_seconds,
                "ttft_seconds": self.ttft_seconds,
                "decode_seconds": self.decode_seconds,
                "decode_token_count": self.decode_token_count,
                "decode_tokens_per_second": decode_tokens_per_second,
                "request_wall_seconds": self.request_wall_seconds,
                "end_to_end_tokens_per_second": (
                    end_to_end_tokens_per_second
                ),
                "per_token_latency_seconds": list(
                    self.per_token_latency_seconds
                ),
            },
            "runtime": {
                "device": self.device,
                "dtype": self.dtype,
                "model_training": False,
                "inference_mode": self.inference_mode,
                "kv_cache_enabled": self.decode_strategy == CACHED_STRATEGY,
            },
        }


@dataclass(frozen=True, slots=True)
class RunSpec:
    sequence_index: int
    scenario: str
    phase: str
    strategy: str
    pair_index: int | None
    phase_pair_index: int | None
    order_index: int
    max_new_tokens: int


def _is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _model_runtime(model: nn.Module) -> tuple[Any, torch.device, torch.dtype]:
    if not isinstance(model, nn.Module):
        raise GenerationError(
            f"model must be a torch.nn.Module, got {type(model)!r}"
        )
    if model.training:
        raise GenerationError("model.training is True; eval mode is required")
    config = getattr(model, "config", None)
    if config is None:
        raise GenerationError("model must expose its frozen config")
    floating_parameters = tuple(
        parameter
        for parameter in model.parameters()
        if parameter.is_floating_point()
    )
    if not floating_parameters:
        raise GenerationError("model has no floating-point parameters")
    device = floating_parameters[0].device
    dtype = floating_parameters[0].dtype
    if any(parameter.device != device for parameter in floating_parameters):
        raise GenerationError("model parameters span multiple devices")
    if any(parameter.dtype != dtype for parameter in floating_parameters):
        raise GenerationError("model parameters span multiple floating dtypes")
    return config, device, dtype


def validate_generation_request(
    prompt_token_ids: Any,
    *,
    max_new_tokens: int,
    vocab_size: int,
    context_length: int,
) -> tuple[int, ...]:
    prompt = validate_token_ids(
        prompt_token_ids,
        vocab_size=vocab_size,
        label="prompt_token_ids",
    )
    if not _is_plain_int(max_new_tokens) or max_new_tokens < 0:
        raise GenerationError("max_new_tokens must be a non-negative integer")
    if len(prompt) > context_length:
        raise GenerationError(
            "prompt length exceeds configured context length: "
            f"{len(prompt)} > {context_length}"
        )
    returned_length = len(prompt) + max_new_tokens
    if returned_length > context_length:
        raise GenerationError(
            "prompt length plus max_new_tokens exceeds configured context: "
            f"{len(prompt)} + {max_new_tokens} > {context_length}"
        )
    return prompt


def _validate_logits(
    logits: object,
    *,
    expected_shape: tuple[int, int, int],
    expected_device: torch.device,
    expected_dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(logits, torch.Tensor):
        raise GenerationError("model output logits must be a torch.Tensor")
    if tuple(logits.shape) != expected_shape:
        raise GenerationError(
            f"logits shape mismatch: {tuple(logits.shape)} != {expected_shape}"
        )
    if logits.device != expected_device:
        raise GenerationError(
            f"logits device mismatch: {logits.device} != {expected_device}"
        )
    if logits.dtype != expected_dtype:
        raise GenerationError(
            f"logits dtype mismatch: {logits.dtype} != {expected_dtype}"
        )
    if not bool(torch.isfinite(logits).all().item()):
        raise GenerationError("logits contain NaN or infinity")
    return logits


def _next_token_id(logits: torch.Tensor, *, vocab_size: int) -> int:
    token_id = int(torch.argmax(logits[0, -1].float()).item())
    if token_id < 0 or token_id >= vocab_size:
        raise GenerationError(
            f"generated token ID {token_id} is outside [0, {vocab_size})"
        )
    return token_id


def validate_past_key_values(
    past_key_values: object,
    *,
    batch_size: int,
    layer_count: int,
    head_count: int,
    head_dimension: int,
    expected_length: int,
    expected_device: torch.device,
    expected_dtype: torch.dtype,
) -> tuple[int, int]:
    if not isinstance(past_key_values, tuple):
        raise GenerationError("past_key_values must be a tuple")
    if len(past_key_values) != layer_count:
        raise GenerationError(
            "cache layer count mismatch: "
            f"{len(past_key_values)} != {layer_count}"
        )
    payload_bytes = 0
    observed_length: int | None = None
    for layer_index, layer_cache in enumerate(past_key_values):
        if not isinstance(layer_cache, tuple) or len(layer_cache) != 2:
            raise GenerationError(
                f"cache layer {layer_index} must be a two-item tuple"
            )
        key, value = layer_cache
        for cache_name, tensor in (("key", key), ("value", value)):
            if not isinstance(tensor, torch.Tensor):
                raise GenerationError(
                    f"cache layer {layer_index} {cache_name} must be a Tensor"
                )
            if tensor.ndim != 4:
                raise GenerationError(
                    f"cache layer {layer_index} {cache_name} must have rank 4"
                )
            if tensor.device != expected_device:
                raise GenerationError(
                    f"cache layer {layer_index} {cache_name} device mismatch"
                )
            if tensor.dtype != expected_dtype:
                raise GenerationError(
                    f"cache layer {layer_index} {cache_name} dtype mismatch"
                )
            payload_bytes += tensor.numel() * tensor.element_size()
        if key.shape != value.shape:
            raise GenerationError(
                f"cache layer {layer_index} key/value shapes do not match"
            )
        expected_prefix = (batch_size, head_count)
        if tuple(key.shape[:2]) != expected_prefix:
            raise GenerationError(
                f"cache layer {layer_index} batch/head dimensions mismatch"
            )
        if key.shape[3] != head_dimension:
            raise GenerationError(
                f"cache layer {layer_index} head dimension mismatch"
            )
        layer_length = int(key.shape[2])
        if observed_length is None:
            observed_length = layer_length
        elif layer_length != observed_length:
            raise GenerationError("cache layers have inconsistent lengths")
    if observed_length is None:
        raise GenerationError("cache must contain at least one layer")
    if observed_length != expected_length:
        raise GenerationError(
            f"cache length mismatch: {observed_length} != {expected_length}"
        )
    return observed_length, payload_bytes


def _empty_trace(
    *,
    strategy: str,
    prompt: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> GenerationTrace:
    return GenerationTrace(
        decode_strategy=strategy,
        prompt_token_ids=prompt,
        generated_token_ids=(),
        returned_token_ids=prompt,
        prefill_input_tokens=0,
        final_cache_length=0,
        cache_layer_count=0,
        cache_payload_bytes=0,
        cache_theoretical_bytes=0,
        prompt_preparation_seconds=0.0,
        prefill_seconds=0.0,
        ttft_seconds=0.0,
        decode_seconds=0.0,
        decode_token_count=0,
        request_wall_seconds=0.0,
        per_token_latency_seconds=(),
        device=str(device),
        dtype=str(dtype),
        inference_mode=True,
        all_logits_finite=True,
    )


def run_reference_generation(
    model: nn.Module,
    prompt_token_ids: Any,
    *,
    max_new_tokens: int,
) -> GenerationTrace:
    """Generate greedily by recomputing the complete prefix at every step."""

    config, device, dtype = _model_runtime(model)
    context_length = int(config.context_length)
    vocab_size = int(config.vocab_size)
    prompt = validate_generation_request(
        prompt_token_ids,
        max_new_tokens=max_new_tokens,
        vocab_size=vocab_size,
        context_length=context_length,
    )
    if max_new_tokens == 0:
        return _empty_trace(
            strategy=REFERENCE_STRATEGY,
            prompt=prompt,
            device=device,
            dtype=dtype,
        )

    full_sequence = list(prompt)
    generated: list[int] = []
    per_token_seconds: list[float] = []
    request_started = time.perf_counter()
    first_token_ready: float | None = None
    prompt_preparation_seconds = 0.0
    prefill_seconds = 0.0

    with torch.inference_mode():
        if not torch.is_inference_mode_enabled():
            raise GenerationError("torch inference mode did not activate")
        for token_index in range(max_new_tokens):
            preparation_started = time.perf_counter()
            input_ids = torch.tensor(
                [full_sequence],
                dtype=torch.long,
                device=device,
            )
            if token_index == 0:
                prompt_preparation_seconds = max(
                    time.perf_counter() - preparation_started,
                    1.0e-12,
                )
            _synchronize(device)
            step_started = time.perf_counter()
            output = model(input_ids)
            _synchronize(device)
            step_seconds = max(time.perf_counter() - step_started, 1.0e-12)
            per_token_seconds.append(step_seconds)
            logits = _validate_logits(
                getattr(output, "logits", None),
                expected_shape=(1, len(full_sequence), vocab_size),
                expected_device=device,
                expected_dtype=dtype,
            )
            next_token = _next_token_id(logits, vocab_size=vocab_size)
            generated.append(next_token)
            full_sequence.append(next_token)
            if token_index == 0:
                prefill_seconds = step_seconds
                first_token_ready = time.perf_counter()

    _synchronize(device)
    request_finished = time.perf_counter()
    if first_token_ready is None:
        raise GenerationError("reference generation produced no first token")
    request_seconds = max(request_finished - request_started, 1.0e-12)
    ttft_seconds = max(first_token_ready - request_started, 1.0e-12)
    decode_seconds = max(request_finished - first_token_ready, 0.0)
    decode_token_count = max(max_new_tokens - 1, 0)
    return GenerationTrace(
        decode_strategy=REFERENCE_STRATEGY,
        prompt_token_ids=prompt,
        generated_token_ids=tuple(generated),
        returned_token_ids=tuple(full_sequence),
        prefill_input_tokens=len(prompt),
        final_cache_length=0,
        cache_layer_count=0,
        cache_payload_bytes=0,
        cache_theoretical_bytes=0,
        prompt_preparation_seconds=prompt_preparation_seconds,
        prefill_seconds=prefill_seconds,
        ttft_seconds=ttft_seconds,
        decode_seconds=decode_seconds,
        decode_token_count=decode_token_count,
        request_wall_seconds=request_seconds,
        per_token_latency_seconds=tuple(per_token_seconds),
        device=str(device),
        dtype=str(dtype),
        inference_mode=True,
        all_logits_finite=True,
    )


def run_cached_generation(
    model: nn.Module,
    prompt_token_ids: Any,
    *,
    max_new_tokens: int,
) -> GenerationTrace:
    """Generate with one prompt prefill followed by single-token cached decode."""

    config, device, dtype = _model_runtime(model)
    context_length = int(config.context_length)
    vocab_size = int(config.vocab_size)
    layer_count = int(config.n_layer)
    head_count = int(config.n_head)
    head_dimension = int(config.head_dim)
    prompt = validate_generation_request(
        prompt_token_ids,
        max_new_tokens=max_new_tokens,
        vocab_size=vocab_size,
        context_length=context_length,
    )
    if max_new_tokens == 0:
        return _empty_trace(
            strategy=CACHED_STRATEGY,
            prompt=prompt,
            device=device,
            dtype=dtype,
        )
    cached_forward = getattr(model, "forward_cached", None)
    if not callable(cached_forward):
        raise GenerationError("model does not expose callable forward_cached")

    full_sequence = list(prompt)
    generated: list[int] = []
    per_token_seconds: list[float] = []
    request_started = time.perf_counter()
    first_token_ready: float | None = None
    prompt_preparation_seconds = 0.0
    prefill_seconds = 0.0
    past_key_values: Any = None
    final_cache_length = 0
    final_payload_bytes = 0

    with torch.inference_mode():
        if not torch.is_inference_mode_enabled():
            raise GenerationError("torch inference mode did not activate")
        preparation_started = time.perf_counter()
        prompt_tensor = torch.tensor(
            [prompt],
            dtype=torch.long,
            device=device,
        )
        prompt_preparation_seconds = max(
            time.perf_counter() - preparation_started,
            1.0e-12,
        )
        _synchronize(device)
        prefill_started = time.perf_counter()
        prefill_output = cached_forward(prompt_tensor)
        _synchronize(device)
        prefill_seconds = max(
            time.perf_counter() - prefill_started,
            1.0e-12,
        )
        per_token_seconds.append(prefill_seconds)
        prefill_logits = _validate_logits(
            getattr(prefill_output, "logits", None),
            expected_shape=(1, len(prompt), vocab_size),
            expected_device=device,
            expected_dtype=dtype,
        )
        past_key_values = getattr(prefill_output, "past_key_values", None)
        final_cache_length, final_payload_bytes = validate_past_key_values(
            past_key_values,
            batch_size=1,
            layer_count=layer_count,
            head_count=head_count,
            head_dimension=head_dimension,
            expected_length=len(prompt),
            expected_device=device,
            expected_dtype=dtype,
        )
        first_token = _next_token_id(prefill_logits, vocab_size=vocab_size)
        generated.append(first_token)
        full_sequence.append(first_token)
        first_token_ready = time.perf_counter()

        for decode_index in range(1, max_new_tokens):
            decode_input = torch.tensor(
                [[generated[-1]]],
                dtype=torch.long,
                device=device,
            )
            _synchronize(device)
            decode_started = time.perf_counter()
            decode_output = cached_forward(decode_input, past_key_values)
            _synchronize(device)
            decode_step_seconds = max(
                time.perf_counter() - decode_started,
                1.0e-12,
            )
            per_token_seconds.append(decode_step_seconds)
            decode_logits = _validate_logits(
                getattr(decode_output, "logits", None),
                expected_shape=(1, 1, vocab_size),
                expected_device=device,
                expected_dtype=dtype,
            )
            past_key_values = getattr(
                decode_output,
                "past_key_values",
                None,
            )
            expected_cache_length = len(prompt) + decode_index
            final_cache_length, final_payload_bytes = (
                validate_past_key_values(
                    past_key_values,
                    batch_size=1,
                    layer_count=layer_count,
                    head_count=head_count,
                    head_dimension=head_dimension,
                    expected_length=expected_cache_length,
                    expected_device=device,
                    expected_dtype=dtype,
                )
            )
            next_token = _next_token_id(
                decode_logits,
                vocab_size=vocab_size,
            )
            generated.append(next_token)
            full_sequence.append(next_token)

    _synchronize(device)
    request_finished = time.perf_counter()
    if first_token_ready is None:
        raise GenerationError("cached generation produced no first token")
    expected_final_cache_length = len(prompt) + max_new_tokens - 1
    if final_cache_length != expected_final_cache_length:
        raise GenerationError(
            "final cache length mismatch: "
            f"{final_cache_length} != {expected_final_cache_length}"
        )
    theoretical_bytes = (
        2
        * layer_count
        * head_count
        * head_dimension
        * final_cache_length
        * torch.empty((), dtype=dtype).element_size()
    )
    if final_payload_bytes != theoretical_bytes:
        raise GenerationError(
            "cache payload byte count mismatch: "
            f"{final_payload_bytes} != {theoretical_bytes}"
        )
    request_seconds = max(request_finished - request_started, 1.0e-12)
    ttft_seconds = max(first_token_ready - request_started, 1.0e-12)
    decode_seconds = max(request_finished - first_token_ready, 0.0)
    decode_token_count = max(max_new_tokens - 1, 0)
    return GenerationTrace(
        decode_strategy=CACHED_STRATEGY,
        prompt_token_ids=prompt,
        generated_token_ids=tuple(generated),
        returned_token_ids=tuple(full_sequence),
        prefill_input_tokens=len(prompt),
        final_cache_length=final_cache_length,
        cache_layer_count=layer_count,
        cache_payload_bytes=final_payload_bytes,
        cache_theoretical_bytes=theoretical_bytes,
        prompt_preparation_seconds=prompt_preparation_seconds,
        prefill_seconds=prefill_seconds,
        ttft_seconds=ttft_seconds,
        decode_seconds=decode_seconds,
        decode_token_count=decode_token_count,
        request_wall_seconds=request_seconds,
        per_token_latency_seconds=tuple(per_token_seconds),
        device=str(device),
        dtype=str(dtype),
        inference_mode=True,
        all_logits_finite=True,
    )


def materialize_protocol_prompts(
    protocol: Any,
    tokenizer: Any,
) -> dict[str, tuple[int, ...]]:
    prompt_builder = protocol["prompt_builder"]
    try:
        bridge_ids = tuple(
            encode_text(tokenizer, prompt_builder["bridge_text"])
        )
        base_ids = tuple(
            encode_text(tokenizer, prompt_builder["exact_length_base_text"])
        )
    except Exception as error:
        raise GenerationError(f"could not tokenize frozen prompts: {error}") from error
    bridge_ids = validate_token_ids(
        bridge_ids,
        vocab_size=int(protocol["architecture"]["vocab_size"]),
        label="bridge_prompt",
    )
    base_ids = validate_token_ids(
        base_ids,
        vocab_size=int(protocol["architecture"]["vocab_size"]),
        label="exact_length_base",
    )
    if len(bridge_ids) != int(prompt_builder["bridge_base_token_count"]):
        raise GenerationError("bridge prompt token count changed")
    if token_ids_sha256(bridge_ids) != prompt_builder[
        "bridge_base_token_ids_sha256"
    ]:
        raise GenerationError("bridge prompt token hash changed")
    if len(base_ids) != int(prompt_builder["exact_length_base_token_count"]):
        raise GenerationError("exact-length base token count changed")
    if token_ids_sha256(base_ids) != prompt_builder[
        "exact_length_base_token_ids_sha256"
    ]:
        raise GenerationError("exact-length base token hash changed")

    prompts: dict[str, tuple[int, ...]] = {}
    for scenario in prompt_builder["scenarios"]:
        name = str(scenario["name"])
        target_length = int(scenario["prompt_length"])
        if name == "bridge":
            token_ids = bridge_ids
        else:
            token_ids = repeat_truncate(base_ids, target_length)
        if len(token_ids) != target_length:
            raise GenerationError(f"{name} prompt length changed")
        if token_ids_sha256(token_ids) != scenario["prompt_token_ids_sha256"]:
            raise GenerationError(f"{name} prompt token hash changed")
        prompts[name] = token_ids
    if tuple(prompts) != ("bridge", "short", "medium", "long"):
        raise GenerationError("scenario materialization order changed")
    return prompts


def compare_generation_traces(
    reference: GenerationTrace,
    cached: GenerationTrace,
) -> dict[str, Any]:
    if reference.decode_strategy != REFERENCE_STRATEGY:
        raise GenerationError("reference trace uses the wrong strategy")
    if cached.decode_strategy != CACHED_STRATEGY:
        raise GenerationError("cached trace uses the wrong strategy")
    if reference.prompt_token_ids != cached.prompt_token_ids:
        raise GenerationError("reference and cached prompts differ")

    first_divergence: int | None = None
    comparison_count = min(
        len(reference.generated_token_ids),
        len(cached.generated_token_ids),
    )
    for index in range(comparison_count):
        if (
            reference.generated_token_ids[index]
            != cached.generated_token_ids[index]
        ):
            first_divergence = index
            break
    if first_divergence is None and (
        len(reference.generated_token_ids)
        != len(cached.generated_token_ids)
    ):
        first_divergence = comparison_count

    reference_payload = reference.to_dict()
    cached_payload = cached.to_dict()
    schema_keys_match = set(reference_payload) == set(cached_payload)
    nested_schema_keys_match = all(
        set(reference_payload[key]) == set(cached_payload[key])
        for key in ("prompt", "generation", "cache", "timing", "runtime")
    )
    generated_exact = (
        reference.generated_token_ids == cached.generated_token_ids
    )
    returned_exact = reference.returned_token_ids == cached.returned_token_ids
    return {
        "generated_token_ids_exact_match": generated_exact,
        "generated_length_exact_match": (
            len(reference.generated_token_ids)
            == len(cached.generated_token_ids)
        ),
        "returned_token_ids_exact_match": returned_exact,
        "returned_length_exact_match": (
            len(reference.returned_token_ids)
            == len(cached.returned_token_ids)
        ),
        "first_divergence_step": first_divergence,
        "comparison_count": comparison_count,
        "top_level_schema_match": schema_keys_match,
        "nested_schema_match": nested_schema_keys_match,
        "pass": (
            generated_exact
            and returned_exact
            and first_divergence is None
            and schema_keys_match
            and nested_schema_keys_match
        ),
    }


def _evaluate_decision_contract(
    *,
    contract: Mapping[str, Any],
    maximum_absolute_error: float,
    mean_absolute_error: float,
    generated_token_ids_exact_match: bool,
    generated_length_exact_match: bool,
    all_argmax_exact_match: bool,
    minimum_top5_token_set_overlap: int,
    all_logits_finite: bool,
    nonfinite_count: int,
    parameter_count_stable: bool,
    state_dict_key_set_stable: bool,
    context_boundaries_pass: bool | None,
    oom_count: int,
) -> tuple[str, dict[str, bool]]:
    """Evaluate the frozen v2 FP16 contract without changing its thresholds."""

    contract_id = contract.get("contract_id")
    if not isinstance(contract_id, str) or not contract_id:
        raise GenerationError("decision contract ID is missing")
    hard_gates = contract.get("hard_gates")
    if not isinstance(hard_gates, Mapping):
        raise GenerationError("decision contract hard_gates must be a mapping")
    expected_keys = {
        "all_argmax_exact_match_required",
        "all_logits_finite_required",
        "context_boundaries_pass_required",
        "generated_length_exact_match_required",
        "generated_token_ids_exact_match_required",
        "maximum_absolute_error_lte",
        "mean_absolute_error_lte",
        "minimum_top5_token_set_overlap",
        "nonfinite_count_lte",
        "oom_count_lte",
        "parameter_count_stable_required",
        "state_dict_key_set_stable_required",
    }
    if set(hard_gates) != expected_keys:
        raise GenerationError("decision contract hard-gate key set changed")
    for key in (
        "all_argmax_exact_match_required",
        "all_logits_finite_required",
        "context_boundaries_pass_required",
        "generated_length_exact_match_required",
        "generated_token_ids_exact_match_required",
        "parameter_count_stable_required",
        "state_dict_key_set_stable_required",
    ):
        if hard_gates[key] is not True:
            raise GenerationError(f"decision contract {key} must be true")
    for key in (
        "maximum_absolute_error_lte",
        "mean_absolute_error_lte",
    ):
        value = hard_gates[key]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise GenerationError(f"decision contract {key} is invalid")
    for key in (
        "minimum_top5_token_set_overlap",
        "nonfinite_count_lte",
        "oom_count_lte",
    ):
        value = hard_gates[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise GenerationError(f"decision contract {key} is invalid")
    if context_boundaries_pass is not True:
        raise GenerationError(
            "v2 decision contract requires a passing context-boundary result"
        )
    results = {
        "all_argmax_exact_match_required": all_argmax_exact_match,
        "all_logits_finite_required": all_logits_finite,
        "context_boundaries_pass_required": context_boundaries_pass,
        "generated_length_exact_match_required": generated_length_exact_match,
        "generated_token_ids_exact_match_required": (
            generated_token_ids_exact_match
        ),
        "maximum_absolute_error_lte": maximum_absolute_error
        <= float(hard_gates["maximum_absolute_error_lte"]),
        "mean_absolute_error_lte": mean_absolute_error
        <= float(hard_gates["mean_absolute_error_lte"]),
        "minimum_top5_token_set_overlap": minimum_top5_token_set_overlap
        >= int(hard_gates["minimum_top5_token_set_overlap"]),
        "nonfinite_count_lte": nonfinite_count
        <= int(hard_gates["nonfinite_count_lte"]),
        "oom_count_lte": oom_count <= int(hard_gates["oom_count_lte"]),
        "parameter_count_stable_required": parameter_count_stable,
        "state_dict_key_set_stable_required": state_dict_key_set_stable,
    }
    return contract_id, results


def run_stepwise_correctness(
    model: nn.Module,
    prompt_token_ids: Any,
    *,
    max_new_tokens: int,
    rtol: float,
    atol: float,
    decision_contract: Mapping[str, Any] | None = None,
    context_boundaries_pass: bool | None = None,
) -> dict[str, Any]:
    """Compare reference and cached logits on the same absolute prefixes."""

    config, device, dtype = _model_runtime(model)
    prompt = validate_generation_request(
        prompt_token_ids,
        max_new_tokens=max_new_tokens,
        vocab_size=int(config.vocab_size),
        context_length=int(config.context_length),
    )
    if max_new_tokens <= 0:
        raise GenerationError(
            "stepwise correctness requires at least one generated token"
        )
    if not isinstance(rtol, (int, float)) or not math.isfinite(float(rtol)):
        raise GenerationError("rtol must be finite")
    if not isinstance(atol, (int, float)) or not math.isfinite(float(atol)):
        raise GenerationError("atol must be finite")
    if float(rtol) < 0.0 or float(atol) < 0.0:
        raise GenerationError("correctness tolerances must be non-negative")
    if decision_contract is not None:
        expected_positions = decision_contract.get("comparison_position_count")
        if (
            not isinstance(expected_positions, int)
            or isinstance(expected_positions, bool)
            or expected_positions <= 0
            or expected_positions != max_new_tokens
        ):
            raise GenerationError(
                "decision contract comparison position count changed"
            )

    parameters_before = sum(parameter.numel() for parameter in model.parameters())
    state_keys_before = tuple(model.state_dict())
    full_sequence = list(prompt)
    reference_tokens: list[int] = []
    cached_tokens: list[int] = []
    rows: list[dict[str, Any]] = []
    past_key_values: Any = None
    all_within_tolerance = True
    all_finite = True
    maximum_absolute_error = -1.0
    maximum_error_step = -1
    maximum_error_token_id = -1
    mean_error_sum = 0.0
    finite_count = 0
    nonfinite_count = 0

    with torch.inference_mode():
        if not torch.is_inference_mode_enabled():
            raise GenerationError("torch inference mode did not activate")
        for sequence_index in range(max_new_tokens):
            reference_input = torch.tensor(
                [full_sequence],
                dtype=torch.long,
                device=device,
            )
            reference_output = model(reference_input)
            reference_logits = _validate_logits(
                getattr(reference_output, "logits", None),
                expected_shape=(1, len(full_sequence), int(config.vocab_size)),
                expected_device=device,
                expected_dtype=dtype,
            )[0, -1]

            if sequence_index == 0:
                cached_input = torch.tensor(
                    [prompt],
                    dtype=torch.long,
                    device=device,
                )
                cached_output = model.forward_cached(cached_input)
                expected_logits_shape = (
                    1,
                    len(prompt),
                    int(config.vocab_size),
                )
            else:
                cached_input = torch.tensor(
                    [[full_sequence[-1]]],
                    dtype=torch.long,
                    device=device,
                )
                cached_output = model.forward_cached(
                    cached_input,
                    past_key_values,
                )
                expected_logits_shape = (1, 1, int(config.vocab_size))
            cached_all_logits = _validate_logits(
                getattr(cached_output, "logits", None),
                expected_shape=expected_logits_shape,
                expected_device=device,
                expected_dtype=dtype,
            )
            cached_logits = cached_all_logits[0, -1]
            past_key_values = getattr(cached_output, "past_key_values", None)
            expected_cache_length = len(prompt) + sequence_index
            actual_cache_length, payload_bytes = validate_past_key_values(
                past_key_values,
                batch_size=1,
                layer_count=int(config.n_layer),
                head_count=int(config.n_head),
                head_dimension=int(config.head_dim),
                expected_length=expected_cache_length,
                expected_device=device,
                expected_dtype=dtype,
            )

            difference = (reference_logits.float() - cached_logits.float()).abs()
            row_max_error, row_max_index = torch.max(difference, dim=0)
            row_mean_error = float(difference.mean().item())
            row_max_error_value = float(row_max_error.item())
            row_max_token_id = int(row_max_index.item())
            reference_finite_count = int(
                torch.isfinite(reference_logits).sum().item()
            )
            cached_finite_count = int(
                torch.isfinite(cached_logits).sum().item()
            )
            row_finite_count = reference_finite_count + cached_finite_count
            row_nonfinite_count = (
                2 * int(config.vocab_size) - row_finite_count
            )
            row_finite = row_nonfinite_count == 0
            row_within_tolerance = bool(
                torch.allclose(
                    reference_logits.float(),
                    cached_logits.float(),
                    rtol=float(rtol),
                    atol=float(atol),
                )
            )
            reference_argmax = int(
                torch.argmax(reference_logits.float()).item()
            )
            cached_argmax = int(torch.argmax(cached_logits.float()).item())
            reference_top5 = set(
                int(value)
                for value in torch.topk(reference_logits.float(), k=5).indices.tolist()
            )
            cached_top5 = set(
                int(value)
                for value in torch.topk(cached_logits.float(), k=5).indices.tolist()
            )
            top5_overlap = len(reference_top5 & cached_top5)
            row = {
                "format_name": "small_gpt_day14_kv_cache_comparison",
                "schema_version": 2,
                "sequence_index": sequence_index,
                "prefix_length": len(full_sequence),
                "expected_cache_length": expected_cache_length,
                "actual_cache_length": actual_cache_length,
                "cache_layer_count": int(config.n_layer),
                "cache_key_shape": [
                    1,
                    int(config.n_head),
                    actual_cache_length,
                    int(config.head_dim),
                ],
                "cache_value_shape": [
                    1,
                    int(config.n_head),
                    actual_cache_length,
                    int(config.head_dim),
                ],
                "cache_payload_bytes": payload_bytes,
                "maximum_absolute_error": row_max_error_value,
                "mean_absolute_error": row_mean_error,
                "maximum_error_token_id": row_max_token_id,
                "reference_argmax": reference_argmax,
                "cached_argmax": cached_argmax,
                "argmax_exact_match": reference_argmax == cached_argmax,
                "top5_token_set_overlap": top5_overlap,
                "finite_count": row_finite_count,
                "nonfinite_count": row_nonfinite_count,
                "all_finite": row_finite,
                "within_tolerance": row_within_tolerance,
                "rtol": float(rtol),
                "atol": float(atol),
            }
            rows.append(row)
            reference_tokens.append(reference_argmax)
            cached_tokens.append(cached_argmax)
            full_sequence.append(reference_argmax)
            all_within_tolerance = all_within_tolerance and row_within_tolerance
            all_finite = all_finite and row_finite
            finite_count += row["finite_count"]
            nonfinite_count += row["nonfinite_count"]
            mean_error_sum += row_mean_error
            if row_max_error_value > maximum_absolute_error:
                maximum_absolute_error = row_max_error_value
                maximum_error_step = sequence_index
                maximum_error_token_id = row_max_token_id

    parameters_after = sum(parameter.numel() for parameter in model.parameters())
    state_keys_after = tuple(model.state_dict())
    sequences_match = reference_tokens == cached_tokens
    generated_length_exact_match = (
        len(reference_tokens) == len(cached_tokens) == max_new_tokens
    )
    parameter_count_stable = parameters_before == parameters_after
    state_keys_stable = state_keys_before == state_keys_after
    mean_absolute_error = mean_error_sum / len(rows)
    all_argmax_exact_match = all(
        row["argmax_exact_match"] for row in rows
    )
    minimum_top5_overlap = min(
        row["top5_token_set_overlap"] for row in rows
    )
    legacy_passing_positions = sum(
        row["within_tolerance"] is True for row in rows
    )
    legacy_failing_positions = len(rows) - legacy_passing_positions
    decision_contract_id: str | None = None
    hard_gate_results: dict[str, bool] | None = None
    if decision_contract is not None:
        decision_contract_id, hard_gate_results = _evaluate_decision_contract(
            contract=decision_contract,
            maximum_absolute_error=maximum_absolute_error,
            mean_absolute_error=mean_absolute_error,
            generated_token_ids_exact_match=sequences_match,
            generated_length_exact_match=generated_length_exact_match,
            all_argmax_exact_match=all_argmax_exact_match,
            minimum_top5_token_set_overlap=minimum_top5_overlap,
            all_logits_finite=all_finite,
            nonfinite_count=nonfinite_count,
            parameter_count_stable=parameter_count_stable,
            state_dict_key_set_stable=state_keys_stable,
            context_boundaries_pass=context_boundaries_pass,
            oom_count=0,
        )
        overall_pass = all(hard_gate_results.values())
    else:
        overall_pass = (
            sequences_match
            and generated_length_exact_match
            and all_finite
            and all_within_tolerance
            and parameter_count_stable
            and state_keys_stable
        )
    return {
        "comparison_position_count": len(rows),
        "generated_token_count": max_new_tokens,
        "generated_token_ids_exact_match": sequences_match,
        "generated_length_exact_match": generated_length_exact_match,
        "reference_generated_token_ids": reference_tokens,
        "cached_generated_token_ids": cached_tokens,
        "maximum_absolute_error": maximum_absolute_error,
        "mean_absolute_error": mean_absolute_error,
        "maximum_error_index": {
            "sequence_index": maximum_error_step,
            "token_id": maximum_error_token_id,
        },
        "finite_count": finite_count,
        "nonfinite_count": nonfinite_count,
        "oom_count": 0,
        "all_logits_finite": all_finite,
        "all_positions_within_tolerance": all_within_tolerance,
        "legacy_elementwise_allclose_pass": all_within_tolerance,
        "legacy_elementwise_allclose_position_count": len(rows),
        "legacy_elementwise_allclose_passing_position_count": (
            legacy_passing_positions
        ),
        "legacy_elementwise_allclose_failing_position_count": (
            legacy_failing_positions
        ),
        "all_argmax_exact_match": all_argmax_exact_match,
        "minimum_top5_token_set_overlap": minimum_top5_overlap,
        "parameter_count_stable": parameter_count_stable,
        "state_dict_key_set_stable": state_keys_stable,
        "decision_contract_id": decision_contract_id,
        "decision_contract_applied": decision_contract is not None,
        "decision_contract_pass": (
            all(hard_gate_results.values())
            if hard_gate_results is not None
            else None
        ),
        "hard_gate_results": hard_gate_results,
        "rows": rows,
        "pass": overall_pass,
    }


def run_context_boundary_checks(model: nn.Module) -> dict[str, Any]:
    config, device, _ = _model_runtime(model)
    context_length = int(config.context_length)
    vocab_size = int(config.vocab_size)
    if context_length < 2:
        raise GenerationError("context boundary checks require context >= 2")
    prompt_511_equivalent = tuple(
        index % vocab_size for index in range(context_length - 1)
    )
    prompt_512_equivalent = tuple(
        index % vocab_size for index in range(context_length)
    )

    allowed = run_cached_generation(
        model,
        prompt_511_equivalent,
        max_new_tokens=1,
    )
    allowed_pass = (
        len(allowed.returned_token_ids) == context_length
        and allowed.final_cache_length == context_length - 1
    )

    overflow_rejected = False
    try:
        run_cached_generation(
            model,
            prompt_512_equivalent,
            max_new_tokens=1,
        )
    except GenerationError:
        overflow_rejected = True

    cache_512_append_rejected = False
    inconsistent_layers_rejected = False
    with torch.inference_mode():
        full_prompt_tensor = torch.tensor(
            [prompt_512_equivalent],
            dtype=torch.long,
            device=device,
        )
        full_cache = model.forward_cached(full_prompt_tensor).past_key_values
        try:
            model.forward_cached(
                torch.tensor([[0]], dtype=torch.long, device=device),
                full_cache,
            )
        except ValueError:
            cache_512_append_rejected = True

        short_length = min(4, context_length - 1)
        short_prompt = torch.tensor(
            [[index % vocab_size for index in range(short_length)]],
            dtype=torch.long,
            device=device,
        )
        short_cache = list(model.forward_cached(short_prompt).past_key_values)
        if len(short_cache) >= 2:
            key, value = short_cache[1]
            short_cache[1] = (key[:, :, :-1, :], value[:, :, :-1, :])
            try:
                model.forward_cached(
                    torch.tensor([[0]], dtype=torch.long, device=device),
                    tuple(short_cache),
                )
            except ValueError:
                inconsistent_layers_rejected = True
        else:
            raise GenerationError(
                "boundary checks require at least two transformer layers"
            )

    return {
        "prompt_context_minus_one_generate_one_allowed": allowed_pass,
        "prompt_context_generate_one_rejected": overflow_rejected,
        "cache_context_then_append_one_rejected": cache_512_append_rejected,
        "inconsistent_layer_lengths_rejected": inconsistent_layers_rejected,
        "context_length": context_length,
        "pass": (
            allowed_pass
            and overflow_rejected
            and cache_512_append_rejected
            and inconsistent_layers_rejected
        ),
    }


def _scenario_map(protocol: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(item["name"]): item
        for item in protocol["prompt_builder"]["scenarios"]
    }


def build_run_plan(
    protocol: Mapping[str, Any],
    mode: str,
    strategy: str,
) -> tuple[RunSpec, ...]:
    scenarios = _scenario_map(protocol)
    plan: list[RunSpec] = []
    sequence_index = 0
    if mode == "smoke":
        if strategy != PAIRED_STRATEGY:
            raise GenerationError("smoke mode requires paired strategy")
        scenario_names = ("bridge",)
        phase_counts = (("measured", 1),)
    elif mode == "benchmark":
        if strategy != PAIRED_STRATEGY:
            raise GenerationError("benchmark mode requires paired strategy")
        scenario_names = ("short", "medium", "long")
        phase_counts = (
            ("warmup", int(protocol["benchmark"]["warmup_pairs_per_scenario"])),
            (
                "measured",
                int(protocol["benchmark"]["measured_pairs_per_scenario"]),
            ),
        )
    elif mode == "stability":
        if strategy != CACHED_STRATEGY:
            raise GenerationError("stability mode requires kv_cache strategy")
        rotation = tuple(protocol["stability"]["scenario_rotation"])
        request_count = int(protocol["stability"]["sequential_requests"])
        for request_index in range(request_count):
            scenario_name = str(rotation[request_index % len(rotation)])
            scenario = scenarios[scenario_name]
            plan.append(
                RunSpec(
                    sequence_index=sequence_index,
                    scenario=scenario_name,
                    phase="measured",
                    strategy=CACHED_STRATEGY,
                    pair_index=None,
                    phase_pair_index=None,
                    order_index=0,
                    max_new_tokens=int(scenario["max_new_tokens"]),
                )
            )
            sequence_index += 1
        return tuple(plan)
    else:
        raise GenerationError(
            f"mode must be smoke, benchmark, or stability, got {mode!r}"
        )

    for scenario_name in scenario_names:
        scenario = scenarios[scenario_name]
        pair_index = 0
        for phase, pair_count in phase_counts:
            for phase_pair_index in range(pair_count):
                order_names = (
                    protocol["benchmark"]["even_pair_order"]
                    if pair_index % 2 == 0
                    else protocol["benchmark"]["odd_pair_order"]
                )
                for order_index, strategy_name in enumerate(order_names):
                    normalized_strategy = {
                        "reference": REFERENCE_STRATEGY,
                        "kv_cache": CACHED_STRATEGY,
                    }.get(str(strategy_name))
                    if normalized_strategy is None:
                        raise GenerationError(
                            f"unknown frozen pair strategy {strategy_name!r}"
                        )
                    plan.append(
                        RunSpec(
                            sequence_index=sequence_index,
                            scenario=scenario_name,
                            phase=phase,
                            strategy=normalized_strategy,
                            pair_index=pair_index,
                            phase_pair_index=phase_pair_index,
                            order_index=order_index,
                            max_new_tokens=int(scenario["max_new_tokens"]),
                        )
                    )
                    sequence_index += 1
                pair_index += 1
    return tuple(plan)


def summary_statistics(values: Sequence[float]) -> dict[str, float]:
    normalized = [float(value) for value in values]
    if not normalized:
        raise GenerationError("summary statistics require at least one value")
    if any(not math.isfinite(value) for value in normalized):
        raise GenerationError("summary statistics values must be finite")
    ordered = sorted(normalized)
    p95_index = max(math.ceil(0.95 * len(ordered)) - 1, 0)
    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": ordered[p95_index],
        "minimum": ordered[0],
        "maximum": ordered[-1],
    }


def build_stability_summary(
    protocol: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected_requests = int(protocol["stability"]["sequential_requests"])
    if len(rows) != expected_requests:
        raise GenerationError(
            f"stability row count mismatch: {len(rows)} != {expected_requests}"
        )
    expected_rotation = tuple(protocol["stability"]["scenario_rotation"])
    observed_rotation = tuple(str(row["scenario"]) for row in rows)
    expected_observed = tuple(
        str(expected_rotation[index % len(expected_rotation)])
        for index in range(expected_requests)
    )
    if observed_rotation != expected_observed:
        raise GenerationError("stability scenario rotation changed")

    def series(field: str) -> list[int]:
        values = [row["memory"].get(field) for row in rows]
        if any(not _is_plain_int(value) or int(value) < 0 for value in values):
            raise GenerationError(
                f"stability memory series {field} contains invalid values"
            )
        return [int(value) for value in values]

    def delta_report(values: Sequence[int]) -> dict[str, Any]:
        first = int(values[0])
        last = int(values[-1])
        monotonic_growth = (
            any(current > previous for previous, current in zip(values, values[1:]))
            and all(current >= previous for previous, current in zip(values, values[1:]))
        )
        return {
            "first": first,
            "last": last,
            "delta": last - first,
            "minimum": min(values),
            "maximum": max(values),
            "monotonic_growth_pattern_observed": monotonic_growth,
        }

    scenario_counts = {
        str(name): observed_rotation.count(str(name))
        for name in expected_rotation
    }
    all_completed = all(
        int(row["trace"]["generation"]["token_count"])
        == int(_scenario_map(protocol)[str(row["scenario"])]["max_new_tokens"])
        for row in rows
    )
    all_tokens_valid = all(
        row["trace"]["generation"]["all_token_ids_in_range"] is True
        for row in rows
    )
    all_logits_finite = all(
        row["trace"]["generation"]["all_logits_finite"] is True
        for row in rows
    )
    all_cache_lengths_valid = all(
        int(row["trace"]["cache"]["final_cache_length"])
        == int(
            _scenario_map(protocol)[str(row["scenario"])][
                "expected_final_cache_length"
            ]
        )
        for row in rows
    )
    if not (
        all_completed
        and all_tokens_valid
        and all_logits_finite
        and all_cache_lengths_valid
    ):
        raise GenerationError("stability completion or safety invariant failed")
    return {
        "performed": True,
        "precision": protocol["stability"]["precision"],
        "decode_strategy": protocol["stability"]["decode_strategy"],
        "sequential_request_count": len(rows),
        "concurrent_request_count": 1,
        "scenario_rotation": list(expected_rotation),
        "requests_per_scenario": scenario_counts,
        "new_empty_cache_per_request": True,
        "cache_reused_across_requests": False,
        "all_requests_completed": all_completed,
        "all_token_ids_in_range": all_tokens_valid,
        "all_logits_finite": all_logits_finite,
        "all_final_cache_lengths_valid": all_cache_lengths_valid,
        "failed_request_count": 0,
        "oom_count": 0,
        "nonfinite_count": 0,
        "context_overflow_count": 0,
        "cuda_allocated_after": delta_report(
            series("cuda_allocated_after")
        ),
        "cuda_reserved_after": delta_report(series("cuda_reserved_after")),
        "mem_available_after": delta_report(series("mem_available_after")),
        "swap_after": delta_report(series("swap_after")),
        "finite_window_request_count": len(rows),
        "absolute_memory_leak_freedom_claimed": False,
        "continuous_7x24_stability_claimed": False,
        "concurrent_stability_claimed": False,
    }


def system_memory_snapshot() -> dict[str, int | None]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, raw = line.split(":", 1)
            fields = raw.strip().split()
            if fields:
                values[key] = int(fields[0]) * 1024
    except (OSError, ValueError):
        return {
            "mem_available_bytes": None,
            "swap_free_bytes": None,
        }
    return {
        "mem_available_bytes": values.get("MemAvailable"),
        "swap_free_bytes": values.get("SwapFree"),
    }


def _require_system_memory_snapshot(
    snapshot: Mapping[str, int | None],
    *,
    label: str,
) -> None:
    for field in ("mem_available_bytes", "swap_free_bytes"):
        value = snapshot.get(field)
        if not _is_plain_int(value) or int(value) < 0:
            raise GenerationError(
                f"{label} system memory field {field} is unavailable or invalid"
            )


def cuda_memory_snapshot(device: torch.device) -> dict[str, int]:
    if device.type != "cuda":
        return {
            "cuda_allocated_bytes": 0,
            "cuda_reserved_bytes": 0,
            "cuda_peak_allocated_bytes": 0,
            "cuda_peak_reserved_bytes": 0,
        }
    return {
        "cuda_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "cuda_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "cuda_peak_allocated_bytes": int(
            torch.cuda.max_memory_allocated(device)
        ),
        "cuda_peak_reserved_bytes": int(
            torch.cuda.max_memory_reserved(device)
        ),
    }


def parse_tegrastats(payload: str) -> dict[str, Any]:
    lines = [line.strip() for line in payload.splitlines() if line.strip()]
    if not lines:
        raise GenerationError("tegrastats log contains no samples")
    cpu_temperatures: list[float] = []
    gpu_temperatures: list[float] = []
    all_temperatures: list[float] = []
    gr3d_values: list[int] = []
    input_power_mw: list[int] = []
    for line in lines:
        temperatures = re.findall(
            r"([A-Za-z0-9_]+)@([0-9]+(?:\.[0-9]+)?)C",
            line,
        )
        for label, value in temperatures:
            temperature = float(value)
            all_temperatures.append(temperature)
            lowered = label.lower()
            if lowered.startswith("cpu"):
                cpu_temperatures.append(temperature)
            if lowered.startswith("gpu"):
                gpu_temperatures.append(temperature)
        gr3d_values.extend(
            int(value)
            for value in re.findall(r"GR3D_FREQ\s+([0-9]+)%", line)
        )
        input_power_mw.extend(
            int(value)
            for value in re.findall(r"VDD_IN\s+([0-9]+)mW/", line)
        )
    if not all_temperatures:
        raise GenerationError("tegrastats log contains no temperature values")
    if not cpu_temperatures:
        raise GenerationError("tegrastats log contains no CPU temperature values")
    if not gpu_temperatures:
        raise GenerationError("tegrastats log contains no GPU temperature values")
    if not gr3d_values:
        raise GenerationError("tegrastats log contains no GR3D utilization values")
    if not input_power_mw:
        raise GenerationError("tegrastats log contains no input-power values")
    return {
        "sample_count": len(lines),
        "maximum_cpu_temperature_c": max(cpu_temperatures),
        "maximum_gpu_temperature_c": max(gpu_temperatures),
        "maximum_any_temperature_c": max(all_temperatures),
        "maximum_gr3d_percent": max(gr3d_values),
        "maximum_input_power_mw": max(input_power_mw),
    }


def _run_strategy(
    model: nn.Module,
    strategy: str,
    prompt: tuple[int, ...],
    *,
    max_new_tokens: int,
) -> GenerationTrace:
    if strategy == REFERENCE_STRATEGY:
        return run_reference_generation(
            model,
            prompt,
            max_new_tokens=max_new_tokens,
        )
    if strategy == CACHED_STRATEGY:
        return run_cached_generation(
            model,
            prompt,
            max_new_tokens=max_new_tokens,
        )
    raise GenerationError(f"unknown generation strategy: {strategy!r}")


def run_request_with_memory(
    model: nn.Module,
    strategy: str,
    prompt: tuple[int, ...],
    *,
    max_new_tokens: int,
) -> tuple[GenerationTrace, dict[str, Any]]:
    _, device, _ = _model_runtime(model)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    system_before = system_memory_snapshot()
    _require_system_memory_snapshot(system_before, label="before")
    cuda_before = cuda_memory_snapshot(device)
    trace = _run_strategy(
        model,
        strategy,
        prompt,
        max_new_tokens=max_new_tokens,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    cuda_after = cuda_memory_snapshot(device)
    system_after = system_memory_snapshot()
    _require_system_memory_snapshot(system_after, label="after")
    memory = {
        "cuda_allocated_before": cuda_before["cuda_allocated_bytes"],
        "cuda_reserved_before": cuda_before["cuda_reserved_bytes"],
        "cuda_peak_allocated": cuda_after["cuda_peak_allocated_bytes"],
        "cuda_peak_reserved": cuda_after["cuda_peak_reserved_bytes"],
        "cuda_allocated_after": cuda_after["cuda_allocated_bytes"],
        "cuda_reserved_after": cuda_after["cuda_reserved_bytes"],
        "mem_available_before": system_before["mem_available_bytes"],
        "mem_available_after": system_after["mem_available_bytes"],
        "swap_before": system_before["swap_free_bytes"],
        "swap_after": system_after["swap_free_bytes"],
        "cache_theoretical_bytes": trace.cache_theoretical_bytes,
        "cache_payload_bytes": trace.cache_payload_bytes,
        "final_cache_length": trace.final_cache_length,
    }
    return trace, memory


def execute_run_plan(
    model: nn.Module,
    prompts: Mapping[str, tuple[int, ...]],
    plan: Sequence[RunSpec],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    traces_by_pair: dict[
        tuple[str, str, int],
        dict[str, GenerationTrace],
    ] = {}
    for spec in plan:
        trace, memory = run_request_with_memory(
            model,
            spec.strategy,
            prompts[spec.scenario],
            max_new_tokens=spec.max_new_tokens,
        )
        trace_payload = trace.to_dict()
        row = {
            "format_name": "small_gpt_day14_benchmark_sample",
            "schema_version": 1,
            "sequence_index": spec.sequence_index,
            "scenario": spec.scenario,
            "phase": spec.phase,
            "strategy": spec.strategy,
            "pair_index": spec.pair_index,
            "phase_pair_index": spec.phase_pair_index,
            "order_index": spec.order_index,
            "max_new_tokens": spec.max_new_tokens,
            "trace": trace_payload,
            "memory": memory,
        }
        rows.append(row)
        if spec.pair_index is not None:
            key = (spec.scenario, spec.phase, spec.pair_index)
            traces_by_pair.setdefault(key, {})[spec.strategy] = trace

    pair_rows: list[dict[str, Any]] = []
    for pair_sequence_index, (key, traces) in enumerate(
        sorted(traces_by_pair.items())
    ):
        if set(traces) != {REFERENCE_STRATEGY, CACHED_STRATEGY}:
            raise GenerationError(f"paired trace set is incomplete for {key}")
        comparison = compare_generation_traces(
            traces[REFERENCE_STRATEGY],
            traces[CACHED_STRATEGY],
        )
        if comparison["pass"] is not True:
            raise GenerationError(f"paired token alignment failed for {key}")
        reference_payload = traces[REFERENCE_STRATEGY].to_dict()
        cached_payload = traces[CACHED_STRATEGY].to_dict()
        reference_decode = reference_payload["timing"][
            "decode_tokens_per_second"
        ]
        cached_decode = cached_payload["timing"]["decode_tokens_per_second"]
        decode_speedup = None
        if reference_decode and cached_decode:
            decode_speedup = float(cached_decode) / float(reference_decode)
        pair_rows.append(
            {
                "pair_sequence_index": pair_sequence_index,
                "scenario": key[0],
                "phase": key[1],
                "pair_index": key[2],
                "comparison": comparison,
                "decode_speedup": decode_speedup,
            }
        )
    return rows, pair_rows


def _trace_metric(row: Mapping[str, Any], name: str) -> float:
    value = row["trace"]["timing"][name]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise GenerationError(
            f"sample timing metric {name} is not numeric: {value!r}"
        )
    normalized = float(value)
    if not math.isfinite(normalized):
        raise GenerationError(f"sample timing metric {name} is non-finite")
    return normalized


def build_scenario_statistics(
    rows: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    measured_rows = [row for row in rows if row["phase"] == "measured"]
    groups: dict[str, dict[str, Any]] = {}
    scenario_names = sorted({str(row["scenario"]) for row in measured_rows})
    for scenario in scenario_names:
        scenario_summary: dict[str, Any] = {}
        for strategy_name in (REFERENCE_STRATEGY, CACHED_STRATEGY):
            strategy_rows = [
                row
                for row in measured_rows
                if row["scenario"] == scenario
                and row["strategy"] == strategy_name
            ]
            if not strategy_rows:
                continue
            scenario_summary[strategy_name] = {
                "request_count": len(strategy_rows),
                "ttft_seconds": summary_statistics(
                    [_trace_metric(row, "ttft_seconds") for row in strategy_rows]
                ),
                "decode_tokens_per_second": summary_statistics(
                    [
                        _trace_metric(row, "decode_tokens_per_second")
                        for row in strategy_rows
                    ]
                ),
                "end_to_end_tokens_per_second": summary_statistics(
                    [
                        _trace_metric(row, "end_to_end_tokens_per_second")
                        for row in strategy_rows
                    ]
                ),
                "cuda_peak_allocated": summary_statistics(
                    [
                        float(row["memory"]["cuda_peak_allocated"])
                        for row in strategy_rows
                    ]
                ),
                "cuda_peak_reserved": summary_statistics(
                    [
                        float(row["memory"]["cuda_peak_reserved"])
                        for row in strategy_rows
                    ]
                ),
            }
        measured_pairs = [
            row
            for row in pair_rows
            if row["scenario"] == scenario and row["phase"] == "measured"
        ]
        speedups = [
            float(row["decode_speedup"])
            for row in measured_pairs
            if row["decode_speedup"] is not None
        ]
        if speedups:
            scenario_summary["paired_decode_speedup"] = summary_statistics(
                speedups
            )
        scenario_summary["measured_pair_count"] = len(measured_pairs)
        groups[scenario] = scenario_summary
    return groups


def build_adoption_summary(
    protocol: Mapping[str, Any],
    mode: str,
    groups: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    adoption_gates = protocol["adoption_gates"]
    primary_scenarios = tuple(protocol["benchmark"]["primary_scenarios"])
    primary_speedups: dict[str, float | None] = {}
    for scenario in primary_scenarios:
        speedup_summary = groups.get(scenario, {}).get(
            "paired_decode_speedup"
        )
        primary_speedups[scenario] = (
            None if speedup_summary is None else float(speedup_summary["mean"])
        )
    adoption_supported = mode == "benchmark" and all(
        primary_speedups.get(name) is not None
        and float(primary_speedups[name])
        >= float(adoption_gates[f"{name}_decode_speedup_minimum"])
        for name in primary_scenarios
    )
    return {
        "primary_speedups": primary_speedups,
        "medium_minimum": adoption_gates[
            "medium_decode_speedup_minimum"
        ],
        "long_minimum": adoption_gates["long_decode_speedup_minimum"],
        "optimization_adoption": (
            "SUPPORTED" if adoption_supported else "NOT_SUPPORTED"
        ),
        "performance_gate_controls_day14_completion": False,
        "performance_claim_scope": "same_device_descriptive",
        "reliability_regression": False,
        "selective_rerun_for_better_numbers_allowed": False,
    }


def build_benchmark_summary(
    *,
    protocol: Mapping[str, Any],
    run_id: str,
    mode: str,
    strategy: str,
    rows: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    tegrastats: Mapping[str, Any],
    model_load_seconds: float,
    source: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    measured_rows = [row for row in rows if row["phase"] == "measured"]
    warmup_rows = [row for row in rows if row["phase"] == "warmup"]
    groups = build_scenario_statistics(rows, pair_rows)

    expected_generated = {
        name: int(item["max_new_tokens"])
        for name, item in _scenario_map(protocol).items()
    }
    all_requests_completed = all(
        int(row["trace"]["generation"]["token_count"])
        == expected_generated[str(row["scenario"])]
        for row in rows
    )
    pairwise_alignment = all(
        row["comparison"]["pass"] is True for row in pair_rows
    )
    return {
        "format_name": "small_gpt_day14_kv_cache_benchmark_summary",
        "schema_version": 1,
        "status": "complete",
        "gate": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "mode": mode,
        "strategy": strategy,
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": canonical_sha256(protocol),
        "source": dict(source),
        "artifacts": dict(artifacts),
        "runtime": {
            **dict(runtime),
            "model_load_seconds": model_load_seconds,
        },
        "execution": {
            "sample_row_count": len(rows),
            "warmup_request_count": len(warmup_rows),
            "measured_request_count": len(measured_rows),
            "pair_count": len(pair_rows),
            "warmups_excluded_from_summary": True,
            "sequence_index_base": 0,
            "single_model_load": True,
            "all_requests_completed": all_requests_completed,
            "pairwise_token_alignment": pairwise_alignment,
            "failed_request_count": 0,
            "oom_count": 0,
            "nonfinite_count": 0,
            "context_overflow_count": 0,
        },
        "scenarios": groups,
        "stability": (
            build_stability_summary(protocol, rows)
            if mode == "stability"
            else {
                "performed": False,
                "absolute_memory_leak_freedom_claimed": False,
                "continuous_7x24_stability_claimed": False,
                "concurrent_stability_claimed": False,
            }
        ),
        "adoption": build_adoption_summary(protocol, mode, groups),
        "tegrastats": dict(tegrastats),
        "safety": {
            "formal_test_access": False,
            "training_attempted": False,
            "backward_called": False,
            "optimizer_created": False,
            "checkpoint_written": False,
            "power_mode_mutation_attempted": False,
            "jetson_clocks_attempted": False,
        },
    }


def _published_identity(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def publish_success_outputs(
    *,
    output_dir: Path,
    run_id: str,
    mode: str,
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    tegrastats_payload: bytes,
) -> Path:
    if not tegrastats_payload.strip():
        raise GenerationError("tegrastats payload must not be empty")
    samples_path = atomic_write_bytes_exclusive(
        output_dir / SAMPLES_FILENAME,
        strict_jsonl_bytes(rows),
    )
    summary_path = atomic_write_json_exclusive(
        output_dir / SUMMARY_FILENAME,
        summary,
    )
    tegrastats_path = atomic_write_bytes_exclusive(
        output_dir / TEGRASTATS_FILENAME,
        tegrastats_payload,
    )
    manifest = {
        "format_name": "small_gpt_day14_kv_cache_manifest",
        "schema_version": 1,
        "status": "complete",
        "gate": "PASS",
        "run_id": run_id,
        "mode": mode,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": summary["protocol_id"],
        "protocol_fingerprint": summary["protocol_fingerprint"],
        "files": {
            SAMPLES_FILENAME: _published_identity(samples_path),
            SUMMARY_FILENAME: _published_identity(summary_path),
            TEGRASTATS_FILENAME: _published_identity(tegrastats_path),
        },
        "manifest_published_last": True,
        "failure_record_present": False,
        "checkpoint_included": False,
        "tokenizer_binary_included": False,
        "credentials_included": False,
    }
    return atomic_write_json_exclusive(
        output_dir / MANIFEST_FILENAME,
        manifest,
    )


def _load_strict_json(path: Path) -> Any:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite constant {value}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise GenerationError(f"could not parse strict JSON {path}: {error}") from error


def _load_strict_jsonl(path: Path) -> list[dict[str, Any]]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise GenerationError(f"could not read JSONL {path}: {error}") from error
    if not lines or any(not line.strip() for line in lines):
        raise GenerationError("samples JSONL must contain non-empty lines")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        try:
            value = json.loads(
                line,
                object_pairs_hook=reject_duplicate_keys,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"non-finite constant {token}")
                ),
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise GenerationError(
                f"samples JSONL line {index} is invalid: {error}"
            ) from error
        if not isinstance(value, dict):
            raise GenerationError(f"samples JSONL line {index} is not an object")
        rows.append(value)
    return rows


def validate_published_run(
    *,
    output_dir: str | Path,
    protocol: Mapping[str, Any],
    mode: str,
    strategy: str,
    run_id: str,
) -> dict[str, Any]:
    root = Path(output_dir).resolve()
    validate_run_id(run_id, mode=mode)
    if root.name != run_id:
        raise GenerationError("validated output directory basename changed")
    if not root.is_dir():
        raise GenerationError("validated output path is not a directory")
    expected_files = {
        MANIFEST_FILENAME,
        SAMPLES_FILENAME,
        SUMMARY_FILENAME,
        TEGRASTATS_FILENAME,
    }
    try:
        actual_entries = tuple(root.iterdir())
    except OSError as error:
        raise GenerationError(
            f"could not enumerate published output directory: {error}"
        ) from error
    actual_names = {path.name for path in actual_entries}
    if (
        actual_names != expected_files
        or any(not path.is_file() for path in actual_entries)
    ):
        raise GenerationError(
            f"published file set mismatch: {sorted(actual_names)} "
            f"!= {sorted(expected_files)}"
        )
    manifest = _load_strict_json(root / MANIFEST_FILENAME)
    if not isinstance(manifest, dict):
        raise GenerationError("manifest must be a JSON object")
    if manifest.get("format_name") != "small_gpt_day14_kv_cache_manifest":
        raise GenerationError("published manifest format changed")
    if manifest.get("status") != "complete" or manifest.get("gate") != "PASS":
        raise GenerationError("published manifest is not a complete PASS")
    if manifest.get("run_id") != run_id:
        raise GenerationError("published run ID mismatch")
    if manifest.get("mode") != mode:
        raise GenerationError("published mode mismatch")
    if manifest.get("manifest_published_last") is not True:
        raise GenerationError("manifest-last contract is not recorded")
    manifest_files = manifest.get("files")
    if not isinstance(manifest_files, dict):
        raise GenerationError("manifest files mapping is missing")
    if set(manifest_files) != {
        SAMPLES_FILENAME,
        SUMMARY_FILENAME,
        TEGRASTATS_FILENAME,
    }:
        raise GenerationError("manifest files mapping changed")
    if manifest.get("protocol_fingerprint") != canonical_sha256(protocol):
        raise GenerationError("manifest protocol fingerprint mismatch")
    if manifest.get("protocol_id") != protocol["protocol_id"]:
        raise GenerationError("manifest protocol ID mismatch")
    if manifest.get("failure_record_present") is not False:
        raise GenerationError("success manifest reports a failure record")
    for field in (
        "checkpoint_included",
        "tokenizer_binary_included",
        "credentials_included",
    ):
        if manifest.get(field) is not False:
            raise GenerationError(f"success manifest safety field {field} changed")
    for filename in (SAMPLES_FILENAME, SUMMARY_FILENAME, TEGRASTATS_FILENAME):
        expected_identity = manifest_files.get(filename)
        actual_identity = _published_identity(root / filename)
        if expected_identity != actual_identity:
            raise GenerationError(f"published identity mismatch for {filename}")

    # The manifest is the publication commit marker.  Verify every payload byte
    # identity before parsing any payload so post-publication mutation always
    # fails at the immutable publication boundary, independent of how the
    # mutation affects JSON/JSONL/text syntax.
    summary = _load_strict_json(root / SUMMARY_FILENAME)
    rows = _load_strict_jsonl(root / SAMPLES_FILENAME)
    try:
        tegrastats_payload = (root / TEGRASTATS_FILENAME).read_text(
            encoding="utf-8"
        )
    except (OSError, UnicodeDecodeError) as error:
        raise GenerationError(
            f"could not read published tegrastats log: {error}"
        ) from error
    tegrastats = parse_tegrastats(tegrastats_payload)
    if not isinstance(summary, dict):
        raise GenerationError("summary must be a JSON object")
    if summary.get("status") != "complete" or summary.get("gate") != "PASS":
        raise GenerationError("published summary is not a complete PASS")
    if summary.get("run_id") != run_id:
        raise GenerationError("published run ID mismatch")
    if summary.get("mode") != mode:
        raise GenerationError("published mode mismatch")
    if summary.get("strategy") != strategy:
        raise GenerationError("published strategy mismatch")
    if summary.get("protocol_fingerprint") != canonical_sha256(protocol):
        raise GenerationError("published protocol fingerprint mismatch")
    if summary.get("protocol_id") != protocol["protocol_id"]:
        raise GenerationError("summary protocol ID mismatch")
    if [row.get("sequence_index") for row in rows] != list(range(len(rows))):
        raise GenerationError("sample sequence_index must be contiguous and 0-based")

    expected_plan = build_run_plan(protocol, mode, strategy)
    if len(rows) != len(expected_plan):
        raise GenerationError(
            f"sample row count mismatch: {len(rows)} != {len(expected_plan)}"
        )
    scenarios = _scenario_map(protocol)
    paired_groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row, spec in zip(rows, expected_plan, strict=True):
        expected_fields = {
            "sequence_index": spec.sequence_index,
            "scenario": spec.scenario,
            "phase": spec.phase,
            "strategy": spec.strategy,
            "pair_index": spec.pair_index,
            "phase_pair_index": spec.phase_pair_index,
            "order_index": spec.order_index,
            "max_new_tokens": spec.max_new_tokens,
        }
        for field, expected in expected_fields.items():
            if row.get(field) != expected:
                raise GenerationError(
                    f"sample {spec.sequence_index} {field} mismatch"
                )
        trace = row.get("trace")
        memory = row.get("memory")
        if not isinstance(trace, dict) or not isinstance(memory, dict):
            raise GenerationError("sample trace/memory schema is invalid")
        expected_scenario = scenarios[spec.scenario]
        generation = trace.get("generation", {})
        cache = trace.get("cache", {})
        prompt = trace.get("prompt", {})
        timing = trace.get("timing", {})
        trace_runtime = trace.get("runtime", {})
        if not all(
            isinstance(value, dict)
            for value in (generation, cache, prompt, timing, trace_runtime)
        ):
            raise GenerationError("sample nested trace schema is invalid")
        if trace.get("decode_strategy") != spec.strategy:
            raise GenerationError("sample trace decode strategy mismatch")
        if trace.get("format_name") != "small_gpt_day14_generation_trace":
            raise GenerationError("sample trace format changed")
        if row.get("format_name") != "small_gpt_day14_benchmark_sample":
            raise GenerationError("sample row format changed")
        expected_prompt_length = int(expected_scenario["prompt_length"])
        if prompt.get("token_count") != expected_prompt_length:
            raise GenerationError("sample prompt token count mismatch")
        prompt_ids = validate_token_ids(
            prompt.get("token_ids"),
            vocab_size=int(protocol["architecture"]["vocab_size"]),
            label="published_prompt_token_ids",
        )
        if token_ids_sha256(prompt_ids) != expected_scenario[
            "prompt_token_ids_sha256"
        ]:
            raise GenerationError("sample prompt token identity mismatch")
        required_memory_fields = protocol["memory_and_thermal"][
            "required_memory_fields"
        ]
        if set(required_memory_fields) - set(memory):
            raise GenerationError("sample required memory fields are missing")
        for field in required_memory_fields:
            value = memory.get(field)
            if not _is_plain_int(value) or int(value) < 0:
                raise GenerationError(
                    f"sample memory field {field} is invalid"
                )
        if generation.get("token_count") != int(
            expected_scenario["max_new_tokens"]
        ):
            raise GenerationError("sample generated token count mismatch")
        if generation.get("returned_sequence_length") != int(
            expected_scenario["returned_sequence_length"]
        ):
            raise GenerationError("sample returned sequence length mismatch")
        generated_ids = validate_token_ids(
            generation.get("token_ids"),
            vocab_size=int(protocol["architecture"]["vocab_size"]),
            label="published_generated_token_ids",
        )
        returned_ids = validate_token_ids(
            generation.get("returned_token_ids"),
            vocab_size=int(protocol["architecture"]["vocab_size"]),
            label="published_returned_token_ids",
        )
        if returned_ids != prompt_ids + generated_ids:
            raise GenerationError("sample returned sequence is not prompt+generated")
        if generation.get("context_crop_events") != 0:
            raise GenerationError("sample reports a context crop")
        if generation.get("decoding") != "greedy":
            raise GenerationError("sample decoding mode changed")
        if generation.get("stop_reason") != "fixed_max_new_tokens":
            raise GenerationError("sample stop reason changed")
        if generation.get("all_logits_finite") is not True:
            raise GenerationError("sample reports non-finite logits")
        if generation.get("all_token_ids_in_range") is not True:
            raise GenerationError("sample reports out-of-range tokens")
        required_timing_fields = set(
            protocol["timing_and_metrics"]["required_fields"]
        ) - {"model_load_seconds"}
        if required_timing_fields - set(timing):
            raise GenerationError("sample required timing fields are missing")
        numeric_timing_fields = (
            "prompt_preparation_seconds",
            "prefill_seconds",
            "ttft_seconds",
            "decode_seconds",
            "request_wall_seconds",
            "end_to_end_tokens_per_second",
        )
        for field in numeric_timing_fields:
            value = timing.get(field)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise GenerationError(f"sample timing field {field} is invalid")
        if timing.get("decode_token_count") != spec.max_new_tokens - 1:
            raise GenerationError("sample decode token count mismatch")
        decode_rate = timing.get("decode_tokens_per_second")
        if (
            not isinstance(decode_rate, (int, float))
            or isinstance(decode_rate, bool)
            or not math.isfinite(float(decode_rate))
            or float(decode_rate) <= 0.0
        ):
            raise GenerationError("sample decode token rate is invalid")
        per_token = timing.get("per_token_latency_seconds")
        if (
            not isinstance(per_token, list)
            or len(per_token) != spec.max_new_tokens
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) <= 0.0
                for value in per_token
            )
        ):
            raise GenerationError("sample per-token latency series is invalid")
        if float(timing["prompt_preparation_seconds"]) <= 0.0:
            raise GenerationError("sample prompt preparation time is not positive")
        if float(timing["prefill_seconds"]) <= 0.0:
            raise GenerationError("sample prefill time is not positive")
        if float(timing["ttft_seconds"]) <= 0.0:
            raise GenerationError("sample TTFT is not positive")
        if float(timing["request_wall_seconds"]) <= 0.0:
            raise GenerationError("sample request wall time is not positive")
        if float(timing["end_to_end_tokens_per_second"]) <= 0.0:
            raise GenerationError("sample end-to-end token rate is not positive")
        if float(timing["decode_seconds"]) <= 0.0:
            raise GenerationError("sample decode time is not positive")
        expected_decode_rate = (
            (spec.max_new_tokens - 1) / float(timing["decode_seconds"])
        )
        if not math.isclose(
            float(decode_rate),
            expected_decode_rate,
            rel_tol=1.0e-12,
            abs_tol=0.0,
        ):
            raise GenerationError("sample decode token rate is inconsistent")
        expected_end_to_end_rate = spec.max_new_tokens / float(
            timing["request_wall_seconds"]
        )
        if not math.isclose(
            float(timing["end_to_end_tokens_per_second"]),
            expected_end_to_end_rate,
            rel_tol=1.0e-12,
            abs_tol=0.0,
        ):
            raise GenerationError(
                "sample end-to-end token rate is inconsistent"
            )
        if float(timing["prefill_seconds"]) > float(timing["ttft_seconds"]):
            raise GenerationError("sample prefill time exceeds TTFT")
        if float(timing["ttft_seconds"]) > float(
            timing["request_wall_seconds"]
        ):
            raise GenerationError("sample TTFT exceeds request wall time")
        if trace_runtime.get("model_training") is not False:
            raise GenerationError("sample runtime reports training mode")
        if trace_runtime.get("inference_mode") is not True:
            raise GenerationError("sample runtime lacks inference mode")
        if trace_runtime.get("kv_cache_enabled") is not (
            spec.strategy == CACHED_STRATEGY
        ):
            raise GenerationError("sample runtime KV-cache flag mismatch")
        if not str(trace_runtime.get("device", "")).startswith("cuda"):
            raise GenerationError("sample runtime is not CUDA")
        if trace_runtime.get("dtype") != "torch.float16":
            raise GenerationError("sample runtime dtype changed")
        if cache.get("prefill_input_tokens") != expected_prompt_length:
            raise GenerationError("sample prefill input token count mismatch")
        for field in (
            "query_cached",
            "input_past_modified_in_place",
            "global_model_cache_used",
        ):
            if cache.get(field) is not False:
                raise GenerationError(f"sample cache safety field {field} changed")
        if spec.strategy == CACHED_STRATEGY:
            if cache.get("final_cache_length") != int(
                expected_scenario["expected_final_cache_length"]
            ):
                raise GenerationError("cached final cache length mismatch")
            if memory.get("cache_payload_bytes") != cache.get(
                "cache_payload_bytes"
            ):
                raise GenerationError("sample cache payload memory mismatch")
            if cache.get("cache_payload_bytes") != cache.get(
                "cache_theoretical_bytes"
            ):
                raise GenerationError("cached payload/theoretical bytes mismatch")
            if memory.get("final_cache_length") != cache.get(
                "final_cache_length"
            ):
                raise GenerationError("sample cache length memory mismatch")
            if cache.get("cache_layer_count") != int(
                protocol["architecture"]["layer_count"]
            ):
                raise GenerationError("cached sample layer count mismatch")
            expected_cache_bytes = (
                2
                * int(protocol["architecture"]["layer_count"])
                * int(protocol["architecture"]["head_count"])
                * int(expected_scenario["expected_final_cache_length"])
                * int(protocol["architecture"]["head_dimension"])
                * 2
            )
            if cache.get("cache_payload_bytes") != expected_cache_bytes:
                raise GenerationError("cached sample payload byte count changed")
            if memory.get("cache_theoretical_bytes") != expected_cache_bytes:
                raise GenerationError("cached memory theoretical bytes changed")
        else:
            if cache.get("final_cache_length") != 0:
                raise GenerationError("reference sample unexpectedly reports cache")
            if cache.get("cache_payload_bytes") != 0:
                raise GenerationError("reference sample reports cache payload")
            if cache.get("cache_theoretical_bytes") != 0:
                raise GenerationError("reference sample reports theoretical cache")
            if cache.get("cache_layer_count") != 0:
                raise GenerationError("reference sample reports cache layers")
            if memory.get("cache_theoretical_bytes") != 0:
                raise GenerationError("reference memory reports cache bytes")
            if memory.get("final_cache_length") != 0:
                raise GenerationError("reference memory reports cache length")
        if spec.pair_index is not None:
            key = (spec.scenario, spec.phase, spec.pair_index)
            paired_groups.setdefault(key, []).append(row)
    validated_pair_rows: list[dict[str, Any]] = []
    for key, pair in paired_groups.items():
        if len(pair) != 2:
            raise GenerationError(f"published pair cardinality mismatch for {key}")
        by_strategy = {row["strategy"]: row for row in pair}
        if set(by_strategy) != {REFERENCE_STRATEGY, CACHED_STRATEGY}:
            raise GenerationError(f"published pair strategy mismatch for {key}")
        reference_ids = by_strategy[REFERENCE_STRATEGY]["trace"]["generation"][
            "token_ids"
        ]
        cached_ids = by_strategy[CACHED_STRATEGY]["trace"]["generation"][
            "token_ids"
        ]
        if reference_ids != cached_ids:
            raise GenerationError(f"published pair token mismatch for {key}")
        reference_returned = by_strategy[REFERENCE_STRATEGY]["trace"][
            "generation"
        ]["returned_token_ids"]
        cached_returned = by_strategy[CACHED_STRATEGY]["trace"][
            "generation"
        ]["returned_token_ids"]
        if reference_returned != cached_returned:
            raise GenerationError(
                f"published pair returned-sequence mismatch for {key}"
            )
        reference_decode_rate = by_strategy[REFERENCE_STRATEGY]["trace"][
            "timing"
        ]["decode_tokens_per_second"]
        cached_decode_rate = by_strategy[CACHED_STRATEGY]["trace"]["timing"][
            "decode_tokens_per_second"
        ]
        validated_pair_rows.append(
            {
                "scenario": key[0],
                "phase": key[1],
                "pair_index": key[2],
                "comparison": {"pass": True},
                "decode_speedup": (
                    float(cached_decode_rate) / float(reference_decode_rate)
                ),
            }
        )

    execution = summary.get("execution")
    if not isinstance(execution, dict):
        raise GenerationError("summary execution record is missing")
    measured_count = sum(row["phase"] == "measured" for row in rows)
    warmup_count = sum(row["phase"] == "warmup" for row in rows)
    if execution.get("sample_row_count") != len(rows):
        raise GenerationError("summary sample row count mismatch")
    if execution.get("measured_request_count") != measured_count:
        raise GenerationError("summary measured request count mismatch")
    if execution.get("warmup_request_count") != warmup_count:
        raise GenerationError("summary warmup request count mismatch")
    if execution.get("warmups_excluded_from_summary") is not True:
        raise GenerationError("summary does not exclude warmups")
    if execution.get("sequence_index_base") != 0:
        raise GenerationError("summary sequence index base changed")
    if execution.get("all_requests_completed") is not True:
        raise GenerationError("summary reports incomplete requests")
    if execution.get("pairwise_token_alignment") is not True:
        raise GenerationError("summary reports pairwise token divergence")
    if execution.get("pair_count") != len(validated_pair_rows):
        raise GenerationError("summary pair count mismatch")
    expected_scenarios = build_scenario_statistics(rows, validated_pair_rows)
    if summary.get("scenarios") != expected_scenarios:
        raise GenerationError(
            "summary scenario statistics mismatch or include warmups"
        )
    expected_adoption = build_adoption_summary(
        protocol,
        mode,
        expected_scenarios,
    )
    if summary.get("adoption") != expected_adoption:
        raise GenerationError("summary adoption analysis mismatch")
    if summary.get("tegrastats") != tegrastats:
        raise GenerationError("summary tegrastats aggregation mismatch")
    if tegrastats.get("maximum_gr3d_percent") is None:
        raise GenerationError("tegrastats GR3D samples are missing")
    if tegrastats.get("maximum_input_power_mw") is None:
        raise GenerationError("tegrastats input-power samples are missing")
    if float(tegrastats["maximum_any_temperature_c"]) >= float(
        protocol["memory_and_thermal"]["maximum_allowed_temperature_c"]
    ):
        raise GenerationError("published temperature is not below frozen gate")
    source = summary.get("source")
    if not isinstance(source, dict):
        raise GenerationError("summary source identity is missing")
    if source.get("branch") != protocol["source"]["required_branch"]:
        raise GenerationError("summary source branch changed")
    if source.get("remote_url") != protocol["source"]["remote_url"]:
        raise GenerationError("summary source remote changed")
    if source.get("worktree_entries") != 0:
        raise GenerationError("summary source worktree was not clean")
    if re.fullmatch(r"[0-9a-f]{40}", str(source.get("head"))) is None:
        raise GenerationError("summary source HEAD is invalid")
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, dict):
        raise GenerationError("summary artifact identities are missing")
    artifact_specs = {
        "config": protocol["frozen_artifacts"]["baseline_config"],
        "checkpoint": protocol["frozen_artifacts"]["control_checkpoint"],
        "tokenizer": protocol["frozen_artifacts"]["tokenizer_json"],
        "tokenizer_config": protocol["frozen_artifacts"]["tokenizer_config"],
    }
    for label, spec in artifact_specs.items():
        identity = artifacts.get(label)
        if not isinstance(identity, dict):
            raise GenerationError(f"summary artifact {label} is missing")
        if identity.get("bytes") != int(spec["bytes"]):
            raise GenerationError(f"summary artifact {label} byte count changed")
        if identity.get("sha256") != spec["sha256"]:
            raise GenerationError(f"summary artifact {label} hash changed")
    runtime = summary.get("runtime")
    if not isinstance(runtime, dict):
        raise GenerationError("summary runtime identity is missing")
    if not str(runtime.get("device", "")).startswith("cuda"):
        raise GenerationError("summary runtime is not CUDA")
    if runtime.get("precision") != protocol["benchmark"]["precision"]:
        raise GenerationError("summary runtime precision changed")
    if runtime.get("dtype") != "torch.float16":
        raise GenerationError("summary runtime dtype changed")
    if runtime.get("model_training") is not False:
        raise GenerationError("summary runtime reports training mode")
    if runtime.get("inference_mode") is not True:
        raise GenerationError("summary runtime lacks inference mode")
    architecture = protocol["architecture"]
    for field in ("parameter_count_before", "parameter_count_after"):
        if runtime.get(field) != int(architecture["parameters"]):
            raise GenerationError(f"summary runtime {field} changed")
    for field in ("state_dict_key_count_before", "state_dict_key_count_after"):
        if runtime.get(field) != int(architecture["state_dict_key_count"]):
            raise GenerationError(f"summary runtime {field} changed")
    if runtime.get("parameter_count_stable") is not True:
        raise GenerationError("summary parameter count was not stable")
    if runtime.get("state_dict_key_set_stable") is not True:
        raise GenerationError("summary state key set was not stable")
    model_load_seconds = runtime.get("model_load_seconds")
    if (
        not isinstance(model_load_seconds, (int, float))
        or isinstance(model_load_seconds, bool)
        or not math.isfinite(float(model_load_seconds))
        or float(model_load_seconds) <= 0.0
    ):
        raise GenerationError("summary model load time is invalid")
    safety = summary.get("safety")
    if not isinstance(safety, dict):
        raise GenerationError("summary safety record is missing")
    for field in (
        "formal_test_access",
        "training_attempted",
        "backward_called",
        "optimizer_created",
        "checkpoint_written",
        "power_mode_mutation_attempted",
        "jetson_clocks_attempted",
    ):
        if safety.get(field) is not False:
            raise GenerationError(f"summary safety field {field} changed")
    expected_stability = (
        build_stability_summary(protocol, rows)
        if mode == "stability"
        else {
            "performed": False,
            "absolute_memory_leak_freedom_claimed": False,
            "continuous_7x24_stability_claimed": False,
            "concurrent_stability_claimed": False,
        }
    )
    if summary.get("stability") != expected_stability:
        raise GenerationError("summary stability analysis mismatch")
    return {
        "format_name": "small_gpt_day14_kv_cache_validation",
        "schema_version": 1,
        "status": "complete",
        "gate": "PASS",
        "run_id": run_id,
        "mode": mode,
        "strategy": strategy,
        "file_count": len(actual_names),
        "sample_row_count": len(rows),
        "measured_request_count": measured_count,
        "warmup_request_count": warmup_count,
        "sequence_index_base": 0,
        "pairwise_token_alignment": True,
        "manifest_hashes_valid": True,
        "output_exact_set": True,
        "training_attempted": False,
        "checkpoint_written": False,
    }


def _failure_document(
    error: BaseException,
    *,
    mode: str,
    strategy: str,
    run_id: str,
) -> dict[str, Any]:
    return {
        "format_name": "small_gpt_day14_kv_cache_failure",
        "schema_version": 1,
        "status": "failed",
        "mode": mode,
        "strategy": strategy,
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


def run_benchmark(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    protocol = load_protocol(args.protocol)
    run_id = validate_run_id(args.run_id, mode=args.mode)
    plan = build_run_plan(protocol, args.mode, args.strategy)
    assert_ntp_synchronized()
    output_candidate = preflight_output_directory(
        require_external_output_path(args.output_dir),
        run_id=run_id,
    )
    tegrastats_source = require_external_output_path(args.tegrastats_log)
    if (
        tegrastats_source == output_candidate
        or output_candidate in tegrastats_source.parents
    ):
        raise GenerationError(
            "tegrastats source must be outside the immutable output directory"
        )
    if not tegrastats_source.is_file():
        raise GenerationError(
            f"tegrastats source does not exist: {tegrastats_source}"
        )
    if args.precision != str(protocol["benchmark"]["precision"]):
        raise GenerationError("formal benchmark precision changed")
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
    prompts = materialize_protocol_prompts(
        session.protocol,
        session.tokenizer,
    )
    output_dir = reserve_output_directory(output_candidate, run_id=run_id)
    try:
        parameter_count_before = sum(
            parameter.numel() for parameter in session.model.parameters()
        )
        state_keys_before = tuple(session.model.state_dict())
        rows, pair_rows = execute_run_plan(session.model, prompts, plan)
        parameter_count_after = sum(
            parameter.numel() for parameter in session.model.parameters()
        )
        state_keys_after = tuple(session.model.state_dict())
        if parameter_count_after != parameter_count_before:
            raise GenerationError("model parameter count changed during benchmark")
        if state_keys_after != state_keys_before:
            raise GenerationError("model state_dict keys changed during benchmark")
        try:
            tegrastats_payload = tegrastats_source.read_bytes()
        except OSError as error:
            raise GenerationError(
                f"could not read tegrastats log {tegrastats_source}: {error}"
            ) from error
        try:
            tegrastats_text = tegrastats_payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise GenerationError("tegrastats log is not UTF-8") from error
        tegrastats = parse_tegrastats(tegrastats_text)
        temperature_gate = float(
            protocol["memory_and_thermal"]["maximum_allowed_temperature_c"]
        )
        if float(tegrastats["maximum_any_temperature_c"]) >= temperature_gate:
            raise GenerationError(
                "tegrastats temperature is not strictly below frozen gate"
            )
        artifacts = {
            "config": session.config_identity.to_dict(),
            "checkpoint": session.checkpoint_identity.to_dict(),
            "tokenizer": session.tokenizer_identity.to_dict(),
            "tokenizer_config": session.tokenizer_config_identity.to_dict(),
        }
        runtime = {
            "device": str(session.device),
            "precision": session.precision,
            "dtype": str(session.dtype),
            "model_training": False,
            "inference_mode": True,
            "parameter_count_before": parameter_count_before,
            "parameter_count_after": parameter_count_after,
            "parameter_count_stable": True,
            "state_dict_key_count_before": len(state_keys_before),
            "state_dict_key_count_after": len(state_keys_after),
            "state_dict_key_set_stable": True,
        }
        summary = build_benchmark_summary(
            protocol=session.protocol,
            run_id=run_id,
            mode=args.mode,
            strategy=args.strategy,
            rows=rows,
            pair_rows=pair_rows,
            tegrastats=tegrastats,
            model_load_seconds=session.model_load_seconds,
            source=session.source.to_dict(),
            artifacts=artifacts,
            runtime=runtime,
        )
        manifest_path = publish_success_outputs(
            output_dir=output_dir,
            run_id=run_id,
            mode=args.mode,
            summary=summary,
            rows=rows,
            tegrastats_payload=tegrastats_payload,
        )
        validation = validate_published_run(
            output_dir=output_dir,
            protocol=session.protocol,
            mode=args.mode,
            strategy=args.strategy,
            run_id=run_id,
        )
        result = {**summary, "post_publication_validation": validation}
        return result, manifest_path
    except BaseException as error:
        failure_path = output_dir / FAILURE_FILENAME
        if not failure_path.exists():
            try:
                atomic_write_json_exclusive(
                    failure_path,
                    _failure_document(
                        error,
                        mode=args.mode,
                        strategy=args.strategy,
                        run_id=run_id,
                    ),
                )
            except BaseException:
                pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run or validate frozen Day 14 reference/KV-cache smoke, "
            "paired benchmark, and sequential stability protocols."
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
            PROJECT_ROOT / "tokenizer" / "artifacts" / "tokenizer.json"
        ),
    )
    parser.add_argument("--tokenizer-config", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp16")
    parser.add_argument(
        "--mode",
        choices=("smoke", "benchmark", "stability"),
        required=True,
    )
    parser.add_argument(
        "--strategy",
        choices=(REFERENCE_STRATEGY, CACHED_STRATEGY, PAIRED_STRATEGY),
        required=True,
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-functional-head")
    parser.add_argument("--tegrastats-log", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        protocol = load_protocol(args.protocol)
        if args.validate_only:
            validation = validate_published_run(
                output_dir=args.output_dir,
                protocol=protocol,
                mode=args.mode,
                strategy=args.strategy,
                run_id=args.run_id,
            )
            result = validation
        else:
            if args.checkpoint is None:
                raise GenerationError("benchmark execution requires --checkpoint")
            if args.tokenizer_config is None:
                raise GenerationError(
                    "benchmark execution requires --tokenizer-config"
                )
            if args.expected_functional_head is None:
                raise GenerationError(
                    "benchmark execution requires --expected-functional-head"
                )
            if args.tegrastats_log is None:
                raise GenerationError(
                    "benchmark execution requires --tegrastats-log"
                )
            result, _ = run_benchmark(args)
        print(
            json.dumps(
                result,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
        return 0
    except Day14KVCacheError as error:
        print(f"Day 14 KV Cache benchmark error: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(
            "Day 14 KV Cache benchmark unexpected error: "
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
