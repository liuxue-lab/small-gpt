from __future__ import annotations

import ast
import inspect
import json
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest
import torch

from model import (
    CausalSelfAttention,
    GPT,
    GPTCachedOutput,
    GPTConfig,
    GPTOutput,
    LayerKVCache,
    PastKeyValues,
    TransformerBlock,
)
from scripts import benchmark_day14_kv_cache as benchmark
from scripts import check_day14_kv_cache as check


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT_ROOT / "configs" / "day14_kv_cache_protocol.json"
CPU_RTOL = 1.0e-5
CPU_ATOL = 1.0e-6
EXPECTED_RUNTIME_BRANCH = "day14-kv-cache-v2"
EXPECTED_F0_APPROVAL_TEXT = (
    "批准 Day14 v2 Stage F0 分支契约修正：将 required_branch 从 main 修正为 "
    "day14-kv-cache-v2，更新 protocol、checker 与测试，执行定向及完整回归，创建"
    "新提交并普通推送功能分支，重建冻结 package，并部署到全新 v2-r1 隔离目录；"
    "禁止修改 v1、覆盖现有 v2、amend、force push、复制或链接 checkpoint、安装或"
    "升级 Torch、训练及自动重试。"
)


def tiny_config(**overrides) -> GPTConfig:
    values = {
        "architecture": "decoder_only_gpt",
        "n_layer": 2,
        "n_head": 2,
        "n_embd": 8,
        "ffn_hidden": 32,
        "context_length": 16,
        "vocab_size": 64,
        "dropout": 0.0,
        "tie_embeddings": True,
        "normalization": "layernorm",
        "norm_position": "pre",
        "layer_norm_eps": 1.0e-5,
        "activation": "gelu",
        "gelu_approximate": "tanh",
        "position_encoding": "learned_absolute",
        "linear_bias": False,
        "lm_head_bias": False,
        "layer_norm_affine": True,
        "init_std": 0.02,
        "scale_residual_projections": True,
    }
    values.update(overrides)
    return GPTConfig(**values)


def make_cache(
    model: GPT,
    *,
    batch_size: int,
    length: int,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> PastKeyValues:
    cache_dtype = model.token_embedding.weight.dtype if dtype is None else dtype
    cache_device = model.token_embedding.weight.device if device is None else device
    shape = (
        batch_size,
        model.config.n_head,
        length,
        model.config.head_dim,
    )
    return tuple(
        (
            torch.zeros(shape, dtype=cache_dtype, device=cache_device),
            torch.zeros(shape, dtype=cache_dtype, device=cache_device),
        )
        for _ in range(model.config.n_layer)
    )


def assert_logits_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(
        actual,
        expected,
        rtol=CPU_RTOL,
        atol=CPU_ATOL,
    )


def test_cached_output_contract_is_frozen_and_slotted():
    assert [field.name for field in fields(GPTCachedOutput)] == [
        "logits",
        "past_key_values",
    ]
    assert GPTCachedOutput.__slots__ == ("logits", "past_key_values")
    output = GPTCachedOutput(logits=torch.zeros(1), past_key_values=())

    with pytest.raises(FrozenInstanceError):
        output.logits = torch.ones(1)


def test_cache_type_aliases_are_exported():
    assert LayerKVCache is not None
    assert PastKeyValues is not None


def test_prefill_returns_one_key_value_pair_per_layer():
    model = GPT(tiny_config()).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    with torch.inference_mode():
        output = model.forward_cached(input_ids)

    assert isinstance(output, GPTCachedOutput)
    assert len(output.past_key_values) == model.config.n_layer
    assert all(len(layer_cache) == 2 for layer_cache in output.past_key_values)


def test_prefill_cache_shapes_match_bhsd_contract():
    model = GPT(tiny_config()).eval()
    input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])

    with torch.inference_mode():
        output = model.forward_cached(input_ids)

    expected = (2, model.config.n_head, 4, model.config.head_dim)
    for key, value in output.past_key_values:
        assert tuple(key.shape) == expected
        assert tuple(value.shape) == expected


def test_single_decode_appends_exactly_one_position_to_every_layer():
    model = GPT(tiny_config()).eval()

    with torch.inference_mode():
        prefill = model.forward_cached(torch.tensor([[1, 2, 3, 4]]))
        decoded = model.forward_cached(
            torch.tensor([[5]]),
            prefill.past_key_values,
        )

    for old_layer, new_layer in zip(
        prefill.past_key_values,
        decoded.past_key_values,
        strict=True,
    ):
        assert old_layer[0].shape[2] == 4
        assert old_layer[1].shape[2] == 4
        assert new_layer[0].shape[2] == 5
        assert new_layer[1].shape[2] == 5


def test_cache_dtype_and_device_match_model_computation():
    model = GPT(tiny_config()).eval()

    with torch.inference_mode():
        output = model.forward_cached(torch.tensor([[1, 2, 3, 4]]))

    expected_dtype = model.token_embedding.weight.dtype
    expected_device = model.token_embedding.weight.device
    for key, value in output.past_key_values:
        assert key.dtype == expected_dtype
        assert value.dtype == expected_dtype
        assert key.device == expected_device
        assert value.device == expected_device


def test_decode_does_not_modify_input_cache_in_place():
    model = GPT(tiny_config()).eval()

    with torch.inference_mode():
        prefill = model.forward_cached(torch.tensor([[1, 2, 3, 4]]))
        snapshots = tuple(
            (key.clone(), value.clone())
            for key, value in prefill.past_key_values
        )
        pointers = tuple(
            (key.data_ptr(), value.data_ptr())
            for key, value in prefill.past_key_values
        )
        decoded = model.forward_cached(
            torch.tensor([[5]]),
            prefill.past_key_values,
        )

    for layer_index, ((key, value), (key_copy, value_copy)) in enumerate(
        zip(prefill.past_key_values, snapshots, strict=True)
    ):
        torch.testing.assert_close(key, key_copy, rtol=0, atol=0)
        torch.testing.assert_close(value, value_copy, rtol=0, atol=0)
        assert key.data_ptr() == pointers[layer_index][0]
        assert value.data_ptr() == pointers[layer_index][1]
        assert decoded.past_key_values[layer_index][0].data_ptr() != key.data_ptr()
        assert decoded.past_key_values[layer_index][1].data_ptr() != value.data_ptr()


def test_cache_payload_bytes_match_tensor_storage_payload():
    model = GPT(tiny_config()).eval()

    with torch.inference_mode():
        output = model.forward_cached(torch.tensor([[1, 2, 3, 4]]))

    length, payload_bytes = benchmark.validate_past_key_values(
        output.past_key_values,
        batch_size=1,
        layer_count=model.config.n_layer,
        head_count=model.config.n_head,
        head_dimension=model.config.head_dim,
        expected_length=4,
        expected_device=model.token_embedding.weight.device,
        expected_dtype=model.token_embedding.weight.dtype,
    )
    expected = 2 * 2 * 2 * 4 * 4 * 4
    assert length == 4
    assert payload_bytes == expected


def test_cache_never_enters_state_dict_or_named_parameters():
    model = GPT(tiny_config()).eval()
    state_before = tuple(model.state_dict())
    parameters_before = tuple(dict(model.named_parameters()))

    with torch.inference_mode():
        model.forward_cached(torch.tensor([[1, 2, 3, 4]]))

    assert tuple(model.state_dict()) == state_before
    assert tuple(dict(model.named_parameters())) == parameters_before
    assert not any("cache" in key.lower() for key in model.state_dict())


def test_prefill_position_ids_begin_at_zero():
    model = GPT(tiny_config()).eval()
    captured = []
    handle = model.position_embedding.register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].clone())
    )
    try:
        with torch.inference_mode():
            model.forward_cached(torch.tensor([[1, 2, 3, 4]]))
    finally:
        handle.remove()

    torch.testing.assert_close(captured[0], torch.arange(4), rtol=0, atol=0)


@pytest.mark.parametrize("past_length", (4, 127, 511))
def test_decode_position_id_equals_past_length(past_length):
    config = tiny_config(context_length=512)
    model = GPT(config).eval()
    cache = make_cache(model, batch_size=1, length=past_length)
    captured = []
    handle = model.position_embedding.register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].clone())
    )
    try:
        with torch.inference_mode():
            model.forward_cached(torch.tensor([[1]]), cache)
    finally:
        handle.remove()

    torch.testing.assert_close(
        captured[0],
        torch.tensor([past_length]),
        rtol=0,
        atol=0,
    )


def test_past_length_512_plus_one_token_is_rejected():
    model = GPT(tiny_config(context_length=512)).eval()
    cache = make_cache(model, batch_size=1, length=512)

    with torch.inference_mode(), pytest.raises(
        ValueError,
        match="exceeds configured context",
    ):
        model.forward_cached(torch.tensor([[1]]), cache)


def test_cached_attention_single_query_can_attend_all_past_and_current():
    config = tiny_config(n_embd=4, n_head=2, ffn_hidden=16, context_length=8)
    attention = CausalSelfAttention(config).eval()
    past_hidden = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]
    )
    current_hidden = torch.tensor([[[9.0, 10.0, 11.0, 12.0]]])
    with torch.no_grad():
        attention.qkv_proj.weight.zero_()
        attention.qkv_proj.weight[8:12].copy_(torch.eye(4))
        attention.out_proj.weight.copy_(torch.eye(4))

    with torch.inference_mode():
        _, past = attention.forward_cached(past_hidden)
        output, present = attention.forward_cached(current_hidden, past)

    expected = torch.cat((past_hidden, current_hidden), dim=1).mean(
        dim=1,
        keepdim=True,
    )
    assert_logits_close(output, expected)
    assert present[0].shape[2] == 3
    assert present[1].shape[2] == 3


def test_cached_prefill_preserves_causal_isolation():
    torch.manual_seed(1401)
    model = GPT(tiny_config()).eval()
    original = torch.randint(0, model.config.vocab_size, (1, 8))
    modified = original.clone()
    modified[:, 5:] = (modified[:, 5:] + 1) % model.config.vocab_size

    with torch.inference_mode():
        original_logits = model.forward_cached(original).logits
        modified_logits = model.forward_cached(modified).logits

    torch.testing.assert_close(
        original_logits[:, :5],
        modified_logits[:, :5],
        rtol=0,
        atol=1.0e-7,
    )


def test_attention_prefill_cached_output_matches_original_attention():
    torch.manual_seed(1402)
    attention = CausalSelfAttention(tiny_config()).eval()
    hidden_states = torch.randn(2, 7, attention.config.n_embd)

    with torch.inference_mode():
        reference = attention(hidden_states)
        cached, present = attention.forward_cached(hidden_states)

    assert_logits_close(cached, reference)
    assert present[0].shape[2] == 7


@pytest.mark.parametrize("sequence_length", (1, 4, 8, 16))
def test_gpt_prefill_logits_match_full_forward(sequence_length):
    torch.manual_seed(1403 + sequence_length)
    model = GPT(tiny_config()).eval()
    input_ids = torch.randint(
        0,
        model.config.vocab_size,
        (1, sequence_length),
    )

    with torch.inference_mode():
        reference = model(input_ids).logits
        cached = model.forward_cached(input_ids).logits

    assert_logits_close(cached, reference)


@pytest.mark.parametrize("sequence_length", (2, 3, 5, 8, 12, 16))
def test_single_cached_step_matches_full_prefix_logits(sequence_length):
    torch.manual_seed(1420 + sequence_length)
    model = GPT(tiny_config()).eval()
    full_sequence = torch.randint(
        0,
        model.config.vocab_size,
        (1, sequence_length),
    )

    with torch.inference_mode():
        prefill = model.forward_cached(full_sequence[:, :-1])
        cached = model.forward_cached(
            full_sequence[:, -1:],
            prefill.past_key_values,
        )
        reference = model(full_sequence)

    assert_logits_close(cached.logits, reference.logits[:, -1:])


def test_multiple_cached_decode_steps_each_match_full_prefix():
    torch.manual_seed(1437)
    model = GPT(tiny_config()).eval()
    full_sequence = torch.randint(0, model.config.vocab_size, (1, 12))

    with torch.inference_mode():
        cached = model.forward_cached(full_sequence[:, :3])
        for end in range(4, 13):
            cached = model.forward_cached(
                full_sequence[:, end - 1 : end],
                cached.past_key_values,
            )
            reference = model(full_sequence[:, :end])
            assert_logits_close(cached.logits, reference.logits[:, -1:])


def test_batch_two_cached_step_matches_full_prefix():
    torch.manual_seed(1438)
    model = GPT(tiny_config()).eval()
    full_sequence = torch.randint(0, model.config.vocab_size, (2, 9))

    with torch.inference_mode():
        prefill = model.forward_cached(full_sequence[:, :-1])
        cached = model.forward_cached(
            full_sequence[:, -1:],
            prefill.past_key_values,
        )
        reference = model(full_sequence)

    assert_logits_close(cached.logits, reference.logits[:, -1:])


def test_cached_logits_are_finite_across_sequential_decode():
    torch.manual_seed(1439)
    model = GPT(tiny_config()).eval()

    with torch.inference_mode():
        output = model.forward_cached(torch.tensor([[1, 2, 3, 4]]))
        assert torch.isfinite(output.logits).all()
        for token_id in (5, 6, 7, 8):
            output = model.forward_cached(
                torch.tensor([[token_id]]),
                output.past_key_values,
            )
            assert torch.isfinite(output.logits).all()


@pytest.mark.parametrize(
    ("prompt", "max_new_tokens"),
    (
        ((1, 2, 3, 4), 8),
        ((1, 2, 3, 4, 5, 6, 7, 8), 4),
        (tuple(range(1, 16)), 1),
    ),
)
def test_reference_and_cached_greedy_sequences_match(prompt, max_new_tokens):
    torch.manual_seed(1440)
    model = GPT(tiny_config()).eval()

    reference = benchmark.run_reference_generation(
        model,
        prompt,
        max_new_tokens=max_new_tokens,
    )
    cached = benchmark.run_cached_generation(
        model,
        prompt,
        max_new_tokens=max_new_tokens,
    )
    comparison = benchmark.compare_generation_traces(reference, cached)

    assert comparison["pass"] is True
    assert comparison["first_divergence_step"] is None


def test_zero_new_tokens_is_allowed_at_full_context_boundary():
    model = GPT(tiny_config()).eval()
    prompt = tuple(range(16))

    reference = benchmark.run_reference_generation(
        model,
        prompt,
        max_new_tokens=0,
    )
    cached = benchmark.run_cached_generation(
        model,
        prompt,
        max_new_tokens=0,
    )

    assert reference.returned_token_ids == prompt
    assert cached.returned_token_ids == prompt
    assert reference.generated_token_ids == ()
    assert cached.generated_token_ids == ()
    assert cached.final_cache_length == 0


def test_cached_generation_avoids_unnecessary_final_forward(monkeypatch):
    torch.manual_seed(1441)
    model = GPT(tiny_config()).eval()
    original = model.forward_cached
    call_count = 0

    def counted(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "forward_cached", counted)
    trace = benchmark.run_cached_generation(
        model,
        (1, 2, 3, 4),
        max_new_tokens=8,
    )

    assert call_count == 8
    assert trace.final_cache_length == 11
    assert len(trace.returned_token_ids) == 12
    assert trace.prompt_preparation_seconds > 0.0
    assert trace.to_dict()["timing"]["prompt_preparation_seconds"] > 0.0


def test_reference_generation_never_calls_cached_path(monkeypatch):
    model = GPT(tiny_config()).eval()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("reference called forward_cached")

    monkeypatch.setattr(model, "forward_cached", forbidden)
    trace = benchmark.run_reference_generation(
        model,
        (1, 2, 3, 4),
        max_new_tokens=2,
    )

    assert trace.decode_strategy == benchmark.REFERENCE_STRATEGY
    assert trace.prompt_preparation_seconds > 0.0


def test_cached_generation_never_calls_original_forward(monkeypatch):
    model = GPT(tiny_config()).eval()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("cached path called original forward")

    monkeypatch.setattr(model, "forward", forbidden)
    trace = benchmark.run_cached_generation(
        model,
        (1, 2, 3, 4),
        max_new_tokens=2,
    )

    assert trace.decode_strategy == benchmark.CACHED_STRATEGY


def test_reference_and_cached_output_schemas_match():
    torch.manual_seed(1442)
    model = GPT(tiny_config()).eval()
    reference = benchmark.run_reference_generation(
        model,
        (1, 2, 3, 4),
        max_new_tokens=2,
    )
    cached = benchmark.run_cached_generation(
        model,
        (1, 2, 3, 4),
        max_new_tokens=2,
    )
    comparison = benchmark.compare_generation_traces(reference, cached)

    assert comparison["top_level_schema_match"] is True
    assert comparison["nested_schema_match"] is True
    assert set(reference.to_dict()["cache"]) == set(cached.to_dict()["cache"])


def test_original_forward_signatures_are_unchanged():
    assert tuple(inspect.signature(CausalSelfAttention.forward).parameters) == (
        "self",
        "hidden_states",
    )
    assert tuple(inspect.signature(TransformerBlock.forward).parameters) == (
        "self",
        "hidden_states",
    )
    assert tuple(inspect.signature(GPT.forward).parameters) == (
        "self",
        "input_ids",
        "targets",
    )


def test_original_forward_still_returns_logits_and_optional_loss():
    torch.manual_seed(1443)
    model = GPT(tiny_config()).eval()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 7))
    targets = torch.randint(0, model.config.vocab_size, (2, 7))

    without_targets = model(input_ids)
    with_targets = model(input_ids, targets)

    assert isinstance(without_targets, GPTOutput)
    assert without_targets.loss is None
    assert with_targets.loss is not None
    assert with_targets.loss.ndim == 0
    assert torch.isfinite(with_targets.loss)


def test_parameter_count_state_keys_and_weight_tying_remain_stable():
    config = tiny_config()
    model = GPT(config).eval()
    parameters_before = sum(parameter.numel() for parameter in model.parameters())
    state_before = tuple(model.state_dict())

    with torch.inference_mode():
        model.forward_cached(torch.tensor([[1, 2, 3, 4]]))

    assert parameters_before == config.parameter_count
    assert sum(parameter.numel() for parameter in model.parameters()) == parameters_before
    assert tuple(model.state_dict()) == state_before
    assert model.lm_head.weight is model.token_embedding.weight
    assert model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr()


def test_attention_cached_path_requires_eval_mode():
    attention = CausalSelfAttention(tiny_config()).train()

    with torch.inference_mode(), pytest.raises(RuntimeError, match="eval mode"):
        attention.forward_cached(torch.randn(1, 2, attention.config.n_embd))


def test_gpt_cached_path_requires_eval_mode():
    model = GPT(tiny_config()).train()

    with torch.inference_mode(), pytest.raises(RuntimeError, match="eval mode"):
        model.forward_cached(torch.tensor([[1, 2]]))


def test_gpt_cached_path_requires_inference_mode():
    model = GPT(tiny_config()).eval()

    with pytest.raises(RuntimeError, match="inference_mode"):
        model.forward_cached(torch.tensor([[1, 2]]))


def test_wrong_cache_layer_count_is_rejected():
    model = GPT(tiny_config()).eval()
    cache = make_cache(model, batch_size=1, length=4)[:-1]

    with torch.inference_mode(), pytest.raises(ValueError, match="layer count"):
        model.forward_cached(torch.tensor([[1]]), cache)


def test_non_tuple_past_key_values_is_rejected():
    model = GPT(tiny_config()).eval()
    cache = list(make_cache(model, batch_size=1, length=4))

    with torch.inference_mode(), pytest.raises(TypeError, match="must be a tuple"):
        model.forward_cached(torch.tensor([[1]]), cache)


def test_cache_layer_missing_value_is_rejected():
    model = GPT(tiny_config()).eval()
    valid = make_cache(model, batch_size=1, length=4)
    invalid = ((valid[0][0],), valid[1])

    with torch.inference_mode(), pytest.raises(TypeError, match="two-item tuple"):
        model.forward_cached(torch.tensor([[1]]), invalid)


def test_non_tensor_cache_member_is_rejected():
    model = GPT(tiny_config()).eval()
    valid = make_cache(model, batch_size=1, length=4)
    invalid = (("not-a-tensor", valid[0][1]), valid[1])

    with torch.inference_mode(), pytest.raises(TypeError, match="torch.Tensor"):
        model.forward_cached(torch.tensor([[1]]), invalid)


def test_cache_rank_mismatch_is_rejected():
    model = GPT(tiny_config()).eval()
    valid = make_cache(model, batch_size=1, length=4)
    wrong = torch.zeros(1, model.config.n_head, 4)
    invalid = ((wrong, wrong.clone()), valid[1])

    with torch.inference_mode(), pytest.raises(ValueError, match="shape"):
        model.forward_cached(torch.tensor([[1]]), invalid)


def test_cache_key_value_shape_mismatch_is_rejected():
    model = GPT(tiny_config()).eval()
    valid = make_cache(model, batch_size=1, length=4)
    wrong_value = torch.zeros(1, model.config.n_head, 3, model.config.head_dim)
    invalid = ((valid[0][0], wrong_value), valid[1])

    with torch.inference_mode(), pytest.raises(ValueError, match="shapes must match"):
        model.forward_cached(torch.tensor([[1]]), invalid)


def test_cache_batch_mismatch_is_rejected():
    model = GPT(tiny_config()).eval()
    cache = make_cache(model, batch_size=2, length=4)

    with torch.inference_mode(), pytest.raises(ValueError, match="batch dimension"):
        model.forward_cached(torch.tensor([[1]]), cache)


def test_cache_head_count_mismatch_is_rejected():
    model = GPT(tiny_config()).eval()
    shape = (1, model.config.n_head + 1, 4, model.config.head_dim)
    layer = (torch.zeros(shape), torch.zeros(shape))
    cache = tuple(layer for _ in range(model.config.n_layer))

    with torch.inference_mode(), pytest.raises(ValueError, match="head dimension"):
        model.forward_cached(torch.tensor([[1]]), cache)


def test_cache_head_dimension_mismatch_is_rejected():
    model = GPT(tiny_config()).eval()
    shape = (1, model.config.n_head, 4, model.config.head_dim + 1)
    layer = (torch.zeros(shape), torch.zeros(shape))
    cache = tuple(layer for _ in range(model.config.n_layer))

    with torch.inference_mode(), pytest.raises(ValueError, match="head_dim"):
        model.forward_cached(torch.tensor([[1]]), cache)


def test_cache_dtype_mismatch_is_rejected():
    model = GPT(tiny_config()).eval()
    cache = make_cache(model, batch_size=1, length=4, dtype=torch.float64)

    with torch.inference_mode(), pytest.raises(TypeError, match="same dtype"):
        model.forward_cached(torch.tensor([[1]]), cache)


def test_cache_device_mismatch_is_rejected():
    model = GPT(tiny_config()).eval()
    cache = make_cache(model, batch_size=1, length=4, device="meta")

    with torch.inference_mode(), pytest.raises(ValueError, match="same device"):
        model.forward_cached(torch.tensor([[1]]), cache)


def test_inconsistent_cache_layer_lengths_are_rejected():
    model = GPT(tiny_config()).eval()
    cache = list(make_cache(model, batch_size=1, length=4))
    shape = (1, model.config.n_head, 3, model.config.head_dim)
    cache[1] = (torch.zeros(shape), torch.zeros(shape))

    with torch.inference_mode(), pytest.raises(
        ValueError,
        match="same sequence length",
    ):
        model.forward_cached(torch.tensor([[1]]), tuple(cache))


def test_zero_length_cache_is_rejected():
    model = GPT(tiny_config()).eval()
    cache = make_cache(model, batch_size=1, length=0)

    with torch.inference_mode(), pytest.raises(ValueError, match="must be positive"):
        model.forward_cached(torch.tensor([[1]]), cache)


def test_cached_decode_rejects_more_than_one_current_token():
    model = GPT(tiny_config()).eval()
    cache = make_cache(model, batch_size=1, length=4)

    with torch.inference_mode(), pytest.raises(ValueError, match="length 1"):
        model.forward_cached(torch.tensor([[1, 2]]), cache)


def test_generation_rejects_empty_prompt():
    model = GPT(tiny_config()).eval()

    with pytest.raises(check.Day14KVCacheError, match="must not be empty"):
        benchmark.run_cached_generation(model, (), max_new_tokens=1)


def test_generation_rejects_negative_max_new_tokens():
    model = GPT(tiny_config()).eval()

    with pytest.raises(benchmark.GenerationError, match="non-negative"):
        benchmark.run_cached_generation(
            model,
            (1, 2, 3),
            max_new_tokens=-1,
        )


def test_generation_rejects_context_overflow_without_cropping():
    model = GPT(tiny_config()).eval()

    with pytest.raises(benchmark.GenerationError, match="exceeds configured context"):
        benchmark.run_cached_generation(
            model,
            tuple(range(15)),
            max_new_tokens=2,
        )


def test_generation_rejects_training_mode_model():
    model = GPT(tiny_config()).train()

    with pytest.raises(benchmark.GenerationError, match="eval mode"):
        benchmark.run_cached_generation(
            model,
            (1, 2, 3),
            max_new_tokens=1,
        )


def test_frozen_protocol_loads_with_exact_identity():
    protocol = check.load_protocol(PROTOCOL_PATH)

    assert protocol["schema_version"] == 2
    assert protocol["protocol_id"] == "day14-kv-cache-v2"
    assert protocol["status"] == "frozen_after_user_approval"
    assert protocol["correctness"]["cpu_fp32"] == {
        "rtol": CPU_RTOL,
        "atol": CPU_ATOL,
    }


def test_f0_branch_contract_correction_and_isolated_namespace_are_frozen():
    protocol = check.load_protocol(PROTOCOL_PATH)
    correction = protocol["revision"]["branch_contract_correction"]

    assert protocol["source"]["required_branch"] == EXPECTED_RUNTIME_BRANCH
    assert correction["approval_text"] == EXPECTED_F0_APPROVAL_TEXT
    assert correction["approved_by"] == "user"
    assert correction["corrected_before_stage_f_runtime_execution"] is True
    assert correction["changed_fields"]["source.required_branch"] == {
        "from": "main",
        "to": EXPECTED_RUNTIME_BRANCH,
    }
    assert correction["unchanged_contracts"] == [
        "architecture",
        "api_route",
        "cache_contract",
        "context_policy",
        "correctness",
        "memory_and_thermal",
        "prompt_builder",
        "safety",
        "timing_and_metrics",
    ]
    namespace = protocol["deployment_namespace"]
    assert namespace["jetson_deploy_root"] == (
        "/home/jetson/small-gpt-day14-v2-r1"
    )
    assert namespace["jetson_incoming_pattern"] == (
        "/home/jetson/small-gpt-day14-v2-r1-incoming-<functional_short_sha>"
    )
    assert namespace["windows_package_build_pattern"] == (
        "D:/model-backups/small-gpt/day14-v2-r1/packages/"
        "build-<functional_short_sha>"
    )
    assert namespace["windows_package_root"] == (
        "D:/model-backups/small-gpt/day14-v2-r1/packages"
    )


def test_f0_runtime_branch_gate_uses_the_frozen_feature_branch():
    check.validate_runtime_branch(
        EXPECTED_RUNTIME_BRANCH,
        EXPECTED_RUNTIME_BRANCH,
    )

    with pytest.raises(
        check.Day14KVCacheError,
        match=(
            "runtime branch mismatch: 'main' != required branch "
            "'day14-kv-cache-v2'"
        ),
    ):
        check.validate_runtime_branch("main", EXPECTED_RUNTIME_BRANCH)


def test_mutated_protocol_is_rejected_by_frozen_identity(tmp_path):
    document = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    document["protocol_id"] = "mutated"
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(check.Day14ProtocolError, match="byte count|SHA-256"):
        check.load_protocol(path)


def test_dry_run_reads_only_protocol_and_reports_no_side_effects(capsys):
    exit_code = check.main(
        ["--protocol", str(PROTOCOL_PATH), "--mode", "dry-run"]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert captured.err == ""
    assert payload["gate"] == "PASS"
    for field in (
        "checkpoint_read",
        "config_read",
        "tokenizer_read",
        "model_constructed",
        "cuda_queried",
        "cuda_allocated",
        "generation_executed",
        "output_written",
        "training_attempted",
        "backward_called",
        "optimizer_created",
        "checkpoint_written",
    ):
        assert payload[field] is False


def test_dry_run_rejects_output_path_without_writing(tmp_path, capsys):
    output_path = tmp_path / "forbidden.json"
    exit_code = check.main(
        [
            "--protocol",
            str(PROTOCOL_PATH),
            "--mode",
            "dry-run",
            "--output",
            str(output_path),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "forbids --output" in captured.err
    assert captured.out == ""
    assert not output_path.exists()


def test_dry_run_checker_has_no_torch_model_or_training_imports():
    source = (PROJECT_ROOT / "scripts" / "check_day14_kv_cache.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])

    assert {"torch", "model", "train", "tokenizer"}.isdisjoint(imported_roots)


def test_day14_cli_exposes_no_training_resume_or_optimizer_controls():
    options = {
        option
        for action in check.build_parser()._actions
        for option in action.option_strings
    }

    for forbidden in ("--train", "--training", "--resume", "--optimizer"):
        assert forbidden not in options


def test_stage_f_check_cli_exposes_frozen_modes_and_runtime_inputs():
    parser = check.build_parser()
    actions = {
        option: action
        for action in parser._actions
        for option in action.option_strings
    }

    assert actions["--mode"].choices == (
        "dry-run",
        "load-only",
        "correctness",
    )
    for option in (
        "--protocol",
        "--config",
        "--checkpoint",
        "--tokenizer",
        "--tokenizer-config",
        "--device",
        "--precision",
        "--expected-functional-head",
        "--output",
        "--run-id",
        "--output-dir",
        "--scenario",
        "--validate-only",
    ):
        assert option in actions


def test_stage_f_benchmark_cli_exposes_frozen_modes_and_strategies():
    parser = benchmark.build_parser()
    actions = {
        option: action
        for action in parser._actions
        for option in action.option_strings
    }

    assert actions["--mode"].choices == (
        "smoke",
        "benchmark",
        "stability",
    )
    assert actions["--strategy"].choices == (
        benchmark.REFERENCE_STRATEGY,
        benchmark.CACHED_STRATEGY,
        benchmark.PAIRED_STRATEGY,
    )
    assert "--validate-only" in actions
    assert "--tegrastats-log" in actions


def test_stage_f_runtime_modes_fail_closed_before_artifact_loading(capsys):
    load_exit = check.main(
        ["--protocol", str(PROTOCOL_PATH), "--mode", "load-only"]
    )
    load_capture = capsys.readouterr()
    correctness_exit = check.main(
        ["--protocol", str(PROTOCOL_PATH), "--mode", "correctness"]
    )
    correctness_capture = capsys.readouterr()

    assert load_exit == 1
    assert "requires --checkpoint" in load_capture.err
    assert load_capture.out == ""
    assert correctness_exit == 1
    assert "requires --checkpoint" in correctness_capture.err
    assert correctness_capture.out == ""


@pytest.mark.parametrize(
    ("mode", "run_id"),
    (
        (
            "correctness",
            "day14-v2-jetson-kv-cache-correctness-20260821T010203Z",
        ),
        ("smoke", "day14-v2-jetson-kv-cache-smoke-20260821T010203Z"),
        (
            "benchmark",
            "day14-v2-jetson-kv-cache-paired-benchmark-20260821T010203Z",
        ),
        (
            "stability",
            "day14-v2-jetson-kv-cache-stability-20260821T010203Z",
        ),
    ),
)
def test_stage_f_run_id_contract_accepts_exact_utc_shape(mode, run_id):
    assert check.validate_run_id(run_id, mode=mode) == run_id


@pytest.mark.parametrize(
    ("mode", "run_id"),
    (
        ("smoke", "day13-jetson-kv-cache-smoke-20260821T010203Z"),
        ("smoke", "day14-jetson-kv-cache-smoke-20260821T010203Z"),
        ("smoke", "day14-v2-jetson-kv-cache-smoke-local"),
        ("smoke", "day14-v2-jetson-kv-cache-smoke-20261340T256199Z"),
        ("benchmark", "day14-v2-jetson-kv-cache-smoke-20260821T010203Z"),
        ("unknown", "day14-v2-jetson-kv-cache-smoke-20260821T010203Z"),
    ),
)
def test_stage_f_run_id_contract_rejects_wrong_identity(mode, run_id):
    with pytest.raises(check.Day14KVCacheError):
        check.validate_run_id(run_id, mode=mode)


def test_stage_f_output_directory_reservation_is_unique_and_exact(tmp_path):
    run_id = "day14-v2-jetson-kv-cache-smoke-20260821T010203Z"
    output_dir = check.reserve_output_directory(
        tmp_path / run_id,
        run_id=run_id,
    )

    assert output_dir.is_dir()
    with pytest.raises(check.Day14KVCacheError, match="already exists"):
        check.reserve_output_directory(output_dir, run_id=run_id)
    with pytest.raises(check.Day14KVCacheError, match="basename"):
        check.reserve_output_directory(
            tmp_path / "wrong-name",
            run_id=run_id,
        )


def test_stage_f_output_preflight_has_no_filesystem_side_effect(tmp_path):
    run_id = "day14-v2-jetson-kv-cache-smoke-20260821T010203Z"
    output_dir = tmp_path / "new-parent" / run_id

    resolved = check.preflight_output_directory(output_dir, run_id=run_id)

    assert resolved == output_dir.resolve()
    assert not output_dir.exists()
    assert not output_dir.parent.exists()


def test_stage_f_atomic_publication_refuses_overwrite(tmp_path):
    path = tmp_path / "result.json"
    check.atomic_write_json_exclusive(path, {"gate": "PASS"})

    assert json.loads(path.read_text(encoding="utf-8")) == {"gate": "PASS"}
    with pytest.raises(check.Day14KVCacheError, match="not be overwritten"):
        check.atomic_write_json_exclusive(path, {"gate": "FAIL"})


def test_stage_f_strict_json_rejects_nonfinite_values():
    with pytest.raises(check.Day14KVCacheError, match="strict finite JSON"):
        check.strict_json_bytes({"value": float("nan")})


def test_stage_f_strict_json_reader_rejects_duplicate_keys(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"gate":"PASS","gate":"FAIL"}', encoding="utf-8")

    with pytest.raises(benchmark.GenerationError, match="duplicate JSON key"):
        benchmark._load_strict_json(path)


def test_stage_f_smoke_run_plan_is_one_complete_pair():
    protocol = check.load_protocol(PROTOCOL_PATH)
    plan = benchmark.build_run_plan(protocol, "smoke", "paired")

    assert len(plan) == 2
    assert [item.sequence_index for item in plan] == [0, 1]
    assert [item.scenario for item in plan] == ["bridge", "bridge"]
    assert [item.phase for item in plan] == ["measured", "measured"]
    assert [item.strategy for item in plan] == [
        benchmark.REFERENCE_STRATEGY,
        benchmark.CACHED_STRATEGY,
    ]
    assert [item.order_index for item in plan] == [0, 1]


def test_stage_f_formal_benchmark_plan_freezes_counts_and_ab_ba_order():
    protocol = check.load_protocol(PROTOCOL_PATH)
    plan = benchmark.build_run_plan(protocol, "benchmark", "paired")

    assert len(plan) == 78
    assert sum(item.phase == "warmup" for item in plan) == 18
    assert sum(item.phase == "measured" for item in plan) == 60
    assert [item.sequence_index for item in plan] == list(range(78))
    for scenario in ("short", "medium", "long"):
        scenario_plan = [item for item in plan if item.scenario == scenario]
        assert len(scenario_plan) == 26
        for pair_index in range(13):
            pair = [
                item
                for item in scenario_plan
                if item.pair_index == pair_index
            ]
            expected = (
                [benchmark.REFERENCE_STRATEGY, benchmark.CACHED_STRATEGY]
                if pair_index % 2 == 0
                else [benchmark.CACHED_STRATEGY, benchmark.REFERENCE_STRATEGY]
            )
            assert [item.strategy for item in pair] == expected
            assert [item.order_index for item in pair] == [0, 1]


def test_stage_f_stability_plan_rotates_30_cached_requests():
    protocol = check.load_protocol(PROTOCOL_PATH)
    plan = benchmark.build_run_plan(
        protocol,
        "stability",
        benchmark.CACHED_STRATEGY,
    )

    assert len(plan) == 30
    assert all(item.strategy == benchmark.CACHED_STRATEGY for item in plan)
    assert all(item.phase == "measured" for item in plan)
    assert [item.scenario for item in plan[:6]] == [
        "short",
        "medium",
        "long",
        "short",
        "medium",
        "long",
    ]
    assert {
        name: sum(item.scenario == name for item in plan)
        for name in ("short", "medium", "long")
    } == {"short": 10, "medium": 10, "long": 10}


def test_stage_f_stability_summary_reports_finite_window_memory_deltas():
    protocol = check.load_protocol(PROTOCOL_PATH)
    scenarios = {
        item["name"]: item
        for item in protocol["prompt_builder"]["scenarios"]
    }
    rows = []
    for spec in benchmark.build_run_plan(
        protocol,
        "stability",
        benchmark.CACHED_STRATEGY,
    ):
        scenario = scenarios[spec.scenario]
        rows.append(
            {
                "scenario": spec.scenario,
                "trace": {
                    "generation": {
                        "token_count": scenario["max_new_tokens"],
                        "all_token_ids_in_range": True,
                        "all_logits_finite": True,
                    },
                    "cache": {
                        "final_cache_length": scenario[
                            "expected_final_cache_length"
                        ]
                    },
                },
                "memory": {
                    "cuda_allocated_after": 100 + spec.sequence_index,
                    "cuda_reserved_after": 200,
                    "mem_available_after": 1_000 - spec.sequence_index,
                    "swap_after": 500,
                },
            }
        )

    summary = benchmark.build_stability_summary(protocol, rows)

    assert summary["sequential_request_count"] == 30
    assert summary["requests_per_scenario"] == {
        "short": 10,
        "medium": 10,
        "long": 10,
    }
    assert summary["cuda_allocated_after"]["delta"] == 29
    assert summary["cuda_allocated_after"][
        "monotonic_growth_pattern_observed"
    ] is True
    assert summary["absolute_memory_leak_freedom_claimed"] is False
    assert summary["continuous_7x24_stability_claimed"] is False


@pytest.mark.parametrize(
    ("mode", "strategy"),
    (
        ("smoke", benchmark.REFERENCE_STRATEGY),
        ("benchmark", benchmark.CACHED_STRATEGY),
        ("stability", benchmark.PAIRED_STRATEGY),
    ),
)
def test_stage_f_run_plan_rejects_nonprotocol_strategy(mode, strategy):
    protocol = check.load_protocol(PROTOCOL_PATH)

    with pytest.raises(benchmark.GenerationError, match="requires"):
        benchmark.build_run_plan(protocol, mode, strategy)


def test_stage_f_summary_statistics_freeze_nearest_rank_p95():
    summary = benchmark.summary_statistics([1.0, 2.0, 3.0, 4.0])

    assert summary == {
        "count": 4,
        "mean": 2.5,
        "median": 2.5,
        "p95": 4.0,
        "minimum": 1.0,
        "maximum": 4.0,
    }


def test_stage_f_tegrastats_parser_records_required_fields():
    payload = (
        "RAM 100/7800MB GR3D_FREQ 37% cpu@44.5C gpu@45.0C "
        "VDD_IN 5200mW/5100mW\n"
        "RAM 110/7800MB GR3D_FREQ 55% cpu@46.0C gpu@47.5C "
        "VDD_IN 5900mW/5400mW\n"
    )

    summary = benchmark.parse_tegrastats(payload)

    assert summary == {
        "sample_count": 2,
        "maximum_cpu_temperature_c": 46.0,
        "maximum_gpu_temperature_c": 47.5,
        "maximum_any_temperature_c": 47.5,
        "maximum_gr3d_percent": 55,
        "maximum_input_power_mw": 5900,
    }


def test_stage_f_stepwise_correctness_reports_required_metrics():
    torch.manual_seed(1450)
    model = GPT(tiny_config()).eval()

    result = benchmark.run_stepwise_correctness(
        model,
        (1, 2, 3, 4),
        max_new_tokens=4,
        rtol=CPU_RTOL,
        atol=CPU_ATOL,
    )

    assert result["pass"] is True
    assert result["comparison_position_count"] == 4
    assert result["generated_token_count"] == 4
    assert result["generated_token_ids_exact_match"] is True
    assert result["generated_length_exact_match"] is True
    assert result["all_logits_finite"] is True
    assert result["all_positions_within_tolerance"] is True
    assert result["legacy_elementwise_allclose_pass"] is True
    assert result["legacy_elementwise_allclose_position_count"] == 4
    assert result["legacy_elementwise_allclose_passing_position_count"] == 4
    assert result["legacy_elementwise_allclose_failing_position_count"] == 0
    assert result["decision_contract_applied"] is False
    assert result["hard_gate_results"] is None
    assert result["parameter_count_stable"] is True
    assert result["state_dict_key_set_stable"] is True
    assert [row["sequence_index"] for row in result["rows"]] == list(range(4))
    assert [row["actual_cache_length"] for row in result["rows"]] == [
        4,
        5,
        6,
        7,
    ]
    assert [row["cache_key_shape"] for row in result["rows"]] == [
        [1, 2, length, 4] for length in (4, 5, 6, 7)
    ]
    assert [row["cache_value_shape"] for row in result["rows"]] == [
        [1, 2, length, 4] for length in (4, 5, 6, 7)
    ]


def test_v2_fp16_decision_contract_allows_descriptive_allclose_failure():
    torch.manual_seed(1452)
    model = GPT(tiny_config()).eval()
    original_forward_cached = model.forward_cached

    def perturbed_forward_cached(input_ids, past_key_values=None):
        output = original_forward_cached(input_ids, past_key_values)
        logits = output.logits.clone()
        indices = torch.argmin(logits, dim=-1, keepdim=True)
        perturbation = torch.full_like(indices, -0.04, dtype=logits.dtype)
        logits.scatter_add_(-1, indices, perturbation)
        return GPTCachedOutput(
            logits=logits,
            past_key_values=output.past_key_values,
        )

    model.forward_cached = perturbed_forward_cached
    protocol = check.load_protocol(PROTOCOL_PATH)
    contract = dict(protocol["correctness"]["jetson_fp16"])
    contract["comparison_position_count"] = 4

    result = benchmark.run_stepwise_correctness(
        model,
        (1, 2, 3, 4),
        max_new_tokens=4,
        rtol=0.01,
        atol=0.01,
        decision_contract=contract,
        context_boundaries_pass=True,
    )

    assert result["legacy_elementwise_allclose_pass"] is False
    assert result["legacy_elementwise_allclose_failing_position_count"] > 0
    assert result["all_argmax_exact_match"] is True
    assert result["minimum_top5_token_set_overlap"] == 5
    assert result["decision_contract_pass"] is True
    assert all(result["hard_gate_results"].values())
    assert result["pass"] is True


def test_v1_observed_fp16_diagnostics_pass_preregistered_v2_hard_gates():
    protocol = check.load_protocol(PROTOCOL_PATH)
    contract = protocol["correctness"]["jetson_fp16"]

    contract_id, results = benchmark._evaluate_decision_contract(
        contract=contract,
        maximum_absolute_error=0.0390625,
        mean_absolute_error=0.0029780296463286504,
        generated_token_ids_exact_match=True,
        generated_length_exact_match=True,
        all_argmax_exact_match=True,
        minimum_top5_token_set_overlap=5,
        all_logits_finite=True,
        nonfinite_count=0,
        parameter_count_stable=True,
        state_dict_key_set_stable=True,
        context_boundaries_pass=True,
        oom_count=0,
    )

    assert contract_id == "jetson_fp16_decision_and_bounded_drift_v2"
    assert all(results.values())


@pytest.mark.parametrize(
    ("maximum_absolute_error", "mean_absolute_error", "failed_gate"),
    (
        (0.0500001, 0.004, "maximum_absolute_error_lte"),
        (0.04, 0.0050001, "mean_absolute_error_lte"),
    ),
)
def test_v2_fp16_decision_contract_rejects_frozen_error_threshold_breach(
    maximum_absolute_error,
    mean_absolute_error,
    failed_gate,
):
    protocol = check.load_protocol(PROTOCOL_PATH)
    contract = protocol["correctness"]["jetson_fp16"]

    contract_id, results = benchmark._evaluate_decision_contract(
        contract=contract,
        maximum_absolute_error=maximum_absolute_error,
        mean_absolute_error=mean_absolute_error,
        generated_token_ids_exact_match=True,
        generated_length_exact_match=True,
        all_argmax_exact_match=True,
        minimum_top5_token_set_overlap=5,
        all_logits_finite=True,
        nonfinite_count=0,
        parameter_count_stable=True,
        state_dict_key_set_stable=True,
        context_boundaries_pass=True,
        oom_count=0,
    )

    assert contract_id == "jetson_fp16_decision_and_bounded_drift_v2"
    assert results[failed_gate] is False
    assert not all(results.values())


def test_stage_f_context_boundary_checks_cover_all_frozen_rejections():
    torch.manual_seed(1451)
    model = GPT(tiny_config()).eval()

    result = benchmark.run_context_boundary_checks(model)

    assert result == {
        "prompt_context_minus_one_generate_one_allowed": True,
        "prompt_context_generate_one_rejected": True,
        "cache_context_then_append_one_rejected": True,
        "inconsistent_layer_lengths_rejected": True,
        "context_length": 16,
        "pass": True,
    }


def _stage_f_synthetic_outputs(tmp_path):
    protocol = check.load_protocol(PROTOCOL_PATH)
    run_id = "day14-v2-jetson-kv-cache-smoke-20260821T010203Z"
    output_dir = check.reserve_output_directory(
        tmp_path / run_id,
        run_id=run_id,
    )
    prompt = (449, 3178, 779)
    generated = tuple([7] * 64)
    returned = prompt + generated
    rows = []
    plan = benchmark.build_run_plan(protocol, "smoke", "paired")
    traces = {}
    for spec in plan:
        cached = spec.strategy == benchmark.CACHED_STRATEGY
        trace = benchmark.GenerationTrace(
            decode_strategy=spec.strategy,
            prompt_token_ids=prompt,
            generated_token_ids=generated,
            returned_token_ids=returned,
            prefill_input_tokens=3,
            final_cache_length=66 if cached else 0,
            cache_layer_count=8 if cached else 0,
            cache_payload_bytes=1_081_344 if cached else 0,
            cache_theoretical_bytes=1_081_344 if cached else 0,
            prompt_preparation_seconds=0.001,
            prefill_seconds=0.01,
            ttft_seconds=0.02,
            decode_seconds=0.63,
            decode_token_count=63,
            request_wall_seconds=0.65,
            per_token_latency_seconds=tuple([0.01] * 64),
            device="cuda:0",
            dtype="torch.float16",
            inference_mode=True,
            all_logits_finite=True,
        )
        traces[spec.strategy] = trace
        rows.append(
            {
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
                "trace": trace.to_dict(),
                "memory": {
                    "cuda_allocated_before": 1,
                    "cuda_reserved_before": 2,
                    "cuda_peak_allocated": 3,
                    "cuda_peak_reserved": 4,
                    "cuda_allocated_after": 1,
                    "cuda_reserved_after": 2,
                    "mem_available_before": 10,
                    "mem_available_after": 9,
                    "swap_before": 8,
                    "swap_after": 8,
                    "cache_theoretical_bytes": trace.cache_theoretical_bytes,
                    "cache_payload_bytes": trace.cache_payload_bytes,
                    "final_cache_length": trace.final_cache_length,
                },
            }
        )
    comparison = benchmark.compare_generation_traces(
        traces[benchmark.REFERENCE_STRATEGY],
        traces[benchmark.CACHED_STRATEGY],
    )
    pair_rows = [
        {
            "pair_sequence_index": 0,
            "scenario": "bridge",
            "phase": "measured",
            "pair_index": 0,
            "comparison": comparison,
            "decode_speedup": 1.0,
        }
    ]
    tegrastats_payload = (
        b"RAM 100/7800MB GR3D_FREQ 55% cpu@46.0C gpu@47.5C "
        b"VDD_IN 5900mW/5400mW\n"
    )
    tegrastats = benchmark.parse_tegrastats(
        tegrastats_payload.decode("utf-8")
    )
    summary = benchmark.build_benchmark_summary(
        protocol=protocol,
        run_id=run_id,
        mode="smoke",
        strategy="paired",
        rows=rows,
        pair_rows=pair_rows,
        tegrastats=tegrastats,
        model_load_seconds=1.0,
        source={
            "branch": EXPECTED_RUNTIME_BRANCH,
            "head": "f" * 40,
            "remote_url": "https://github.com/liuxue-lab/small-gpt.git",
            "worktree_entries": 0,
        },
        artifacts={
            "config": {
                "path": "/frozen/baseline.yaml",
                "bytes": 1_258,
                "sha256": (
                    "ca8524c425e1e5e3a600de5773f9a526"
                    "ef3674741040635bee91fe31f4b24c0e"
                ),
            },
            "checkpoint": {
                "path": "/frozen/step-00004578.pt",
                "bytes": 406_108_827,
                "sha256": (
                    "a39f8378ebe4012afb992be451d355e81"
                    "4b856ffb5e690ac011758f9db614b51"
                ),
            },
            "tokenizer": {
                "path": "/frozen/tokenizer.json",
                "bytes": 1_137_073,
                "sha256": (
                    "b26835e02eebf777a257c4732abdd6f9"
                    "732a115967d2ad839f3a1a00e45ee8c5"
                ),
            },
            "tokenizer_config": {
                "path": "/frozen/tokenizer_config.json",
                "bytes": 2_988,
                "sha256": (
                    "8622711407aab3f299996b7d3009d4f4"
                    "447ae35879ca8e50451b5f0adbdf5141"
                ),
            },
        },
        runtime={
            "device": "cuda:0",
            "precision": "fp16",
            "dtype": "torch.float16",
            "model_training": False,
            "inference_mode": True,
            "parameter_count_before": 33_833_984,
            "parameter_count_after": 33_833_984,
            "parameter_count_stable": True,
            "state_dict_key_count_before": 69,
            "state_dict_key_count_after": 69,
            "state_dict_key_set_stable": True,
        },
    )
    benchmark.publish_success_outputs(
        output_dir=output_dir,
        run_id=run_id,
        mode="smoke",
        summary=summary,
        rows=rows,
        tegrastats_payload=tegrastats_payload,
    )
    return protocol, run_id, output_dir


def test_stage_f_publisher_and_validator_freeze_exact_output_contract(tmp_path):
    protocol, run_id, output_dir = _stage_f_synthetic_outputs(tmp_path)

    validation = benchmark.validate_published_run(
        output_dir=output_dir,
        protocol=protocol,
        mode="smoke",
        strategy="paired",
        run_id=run_id,
    )

    assert validation["gate"] == "PASS"
    assert validation["file_count"] == 4
    assert validation["sample_row_count"] == 2
    assert validation["sequence_index_base"] == 0
    assert validation["pairwise_token_alignment"] is True
    assert validation["manifest_hashes_valid"] is True
    assert {path.name for path in output_dir.iterdir()} == {
        "manifest.json",
        "samples.jsonl",
        "benchmark-summary.json",
        "tegrastats.log",
    }


def test_stage_f_validator_rejects_postpublication_mutation(tmp_path):
    protocol, run_id, output_dir = _stage_f_synthetic_outputs(tmp_path)
    samples_path = output_dir / "samples.jsonl"
    samples_path.write_bytes(samples_path.read_bytes() + b"\n")

    with pytest.raises(benchmark.GenerationError, match="identity mismatch"):
        benchmark.validate_published_run(
            output_dir=output_dir,
            protocol=protocol,
            mode="smoke",
            strategy="paired",
            run_id=run_id,
        )


def test_stage_f_validator_rejects_unexpected_output_file(tmp_path):
    protocol, run_id, output_dir = _stage_f_synthetic_outputs(tmp_path)
    (output_dir / "unexpected.txt").write_text("forbidden", encoding="utf-8")

    with pytest.raises(benchmark.GenerationError, match="file set mismatch"):
        benchmark.validate_published_run(
            output_dir=output_dir,
            protocol=protocol,
            mode="smoke",
            strategy="paired",
            run_id=run_id,
        )


def test_stage_f_validator_rejects_unexpected_output_directory(tmp_path):
    protocol, run_id, output_dir = _stage_f_synthetic_outputs(tmp_path)
    (output_dir / "unexpected-directory").mkdir()

    with pytest.raises(benchmark.GenerationError, match="file set mismatch"):
        benchmark.validate_published_run(
            output_dir=output_dir,
            protocol=protocol,
            mode="smoke",
            strategy="paired",
            run_id=run_id,
        )


@pytest.mark.parametrize(
    ("precision", "legacy_failure_count"),
    (("fp32", 0), ("fp16", 43)),
)
def test_stage_f_correctness_validator_freezes_rows_hash_and_boundary(
    tmp_path,
    precision,
    legacy_failure_count,
):
    protocol = check.load_protocol(PROTOCOL_PATH)
    run_id = "day14-v2-jetson-kv-cache-correctness-20260821T010203Z"
    output_dir = check.reserve_output_directory(
        tmp_path / run_id,
        run_id=run_id,
    )
    is_fp16 = precision == "fp16"
    row_rtol = 1.0e-2 if is_fp16 else 1.0e-4
    row_atol = 1.0e-2 if is_fp16 else 1.0e-5
    payload_bytes_per_position = 16_384 if is_fp16 else 32_768
    rows = [
        {
            "format_name": "small_gpt_day14_kv_cache_comparison",
            "schema_version": 2,
            "sequence_index": index,
            "prefix_length": 16 + index,
            "expected_cache_length": 16 + index,
            "actual_cache_length": 16 + index,
            "cache_key_shape": [1, 8, 16 + index, 64],
            "cache_value_shape": [1, 8, 16 + index, 64],
            "cache_layer_count": 8,
            "cache_payload_bytes": payload_bytes_per_position * (16 + index),
            "maximum_absolute_error": 0.0,
            "mean_absolute_error": 0.0,
            "maximum_error_token_id": 1,
            "reference_argmax": 1,
            "cached_argmax": 1,
            "argmax_exact_match": True,
            "top5_token_set_overlap": 5,
            "finite_count": 32_768,
            "nonfinite_count": 0,
            "rtol": row_rtol,
            "atol": row_atol,
            "within_tolerance": index >= legacy_failure_count,
            "all_finite": True,
        }
        for index in range(64)
    ]
    comparisons_payload = check.strict_jsonl_bytes(rows)
    check.atomic_write_bytes_exclusive(
        output_dir / "comparisons.jsonl",
        comparisons_payload,
    )
    if is_fp16:
        fp16_contract = protocol["correctness"]["jetson_fp16"]
        tolerance_summary = {
            "contract_id": fp16_contract["contract_id"],
            "legacy_elementwise_allclose": dict(
                fp16_contract["legacy_elementwise_allclose"]
            ),
            "hard_gates": dict(fp16_contract["hard_gates"]),
            "relaxed_after_v2_execution": False,
        }
        decision_fields = {
            "decision_contract_id": fp16_contract["contract_id"],
            "decision_contract_applied": True,
            "decision_contract_pass": True,
            "hard_gate_results": {
                key: True for key in fp16_contract["hard_gates"]
            },
        }
    else:
        tolerance_summary = {
            "rtol": row_rtol,
            "atol": row_atol,
            "relaxed_after_failure": False,
        }
        decision_fields = {}
    summary = {
        "format_name": "small_gpt_day14_kv_cache_correctness_summary",
        "schema_version": 2,
        "run_id": run_id,
        "protocol_id": "day14-kv-cache-v2",
        "protocol_fingerprint": check.canonical_sha256(protocol),
        "status": "complete",
        "gate": "PASS",
        "scenario": "short",
        "runtime": {
            "device": "cuda:0",
            "precision": precision,
            "dtype": "torch.float16" if is_fp16 else "torch.float32",
            "model_load_seconds": 1.0,
        },
        "tolerance": tolerance_summary,
        "comparison": {
            "comparison_position_count": 64,
            "generated_token_count": 64,
            "generated_token_ids_exact_match": True,
            "generated_length_exact_match": True,
            "reference_generated_token_ids": [1] * 64,
            "cached_generated_token_ids": [1] * 64,
            "maximum_absolute_error": 0.0,
            "mean_absolute_error": 0.0,
            "maximum_error_index": {
                "sequence_index": 0,
                "token_id": 1,
            },
            "finite_count": 64 * 32_768,
            "nonfinite_count": 0,
            "oom_count": 0,
            "all_logits_finite": True,
            "all_positions_within_tolerance": legacy_failure_count == 0,
            "legacy_elementwise_allclose_pass": legacy_failure_count == 0,
            "legacy_elementwise_allclose_position_count": 64,
            "legacy_elementwise_allclose_passing_position_count": (
                64 - legacy_failure_count
            ),
            "legacy_elementwise_allclose_failing_position_count": (
                legacy_failure_count
            ),
            "all_argmax_exact_match": True,
            "minimum_top5_token_set_overlap": 5,
            "parameter_count_stable": True,
            "state_dict_key_set_stable": True,
            **decision_fields,
            "pass": True,
        },
        "model": {
            "parameters": 33_833_984,
            "state_dict_key_count": 69,
            "training": False,
            "weight_tied": True,
        },
        "checkpoint": {
            "run_id": "baseline-full-300m-20260813-232952",
            "load_mode": "model_only",
            "strict_state_dict_load": True,
            "missing_keys": 0,
            "unexpected_keys": 0,
            "optimizer_state_restored": False,
            "scheduler_state_restored": False,
            "training_resume": False,
        },
        "strategies": {
            "reference": "full_prefix_recompute",
            "cached": "kv_cache",
            "decoding": "greedy",
            "stop_on_eos": False,
        },
        "cache_contract": {
            "layer_count": 8,
            "head_count": 8,
            "head_dimension": 64,
            "dtype": "torch.float16" if is_fp16 else "torch.float32",
            "device": "cuda:0",
        },
        "source": {
            "branch": EXPECTED_RUNTIME_BRANCH,
            "head": "f" * 40,
            "remote_url": "https://github.com/liuxue-lab/small-gpt.git",
            "worktree_entries": 0,
        },
        "artifacts": {
            "config": {
                "bytes": protocol["frozen_artifacts"]["baseline_config"][
                    "bytes"
                ],
                "sha256": protocol["frozen_artifacts"]["baseline_config"][
                    "sha256"
                ],
            },
            "checkpoint": {
                "bytes": protocol["frozen_artifacts"][
                    "control_checkpoint"
                ]["bytes"],
                "sha256": protocol["frozen_artifacts"][
                    "control_checkpoint"
                ]["sha256"],
            },
            "tokenizer": {
                "bytes": protocol["frozen_artifacts"]["tokenizer_json"][
                    "bytes"
                ],
                "sha256": protocol["frozen_artifacts"]["tokenizer_json"][
                    "sha256"
                ],
            },
            "tokenizer_config": {
                "bytes": protocol["frozen_artifacts"][
                    "tokenizer_config"
                ]["bytes"],
                "sha256": protocol["frozen_artifacts"][
                    "tokenizer_config"
                ]["sha256"],
            },
        },
        "context_boundaries": {
            "prompt_context_minus_one_generate_one_allowed": True,
            "prompt_context_generate_one_rejected": True,
            "cache_context_then_append_one_rejected": True,
            "inconsistent_layer_lengths_rejected": True,
            "context_length": 512,
            "pass": True,
        },
        "published_files": {
            "comparisons.jsonl": {
                "bytes": len(comparisons_payload),
                "sha256": check.sha256_bytes(comparisons_payload),
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
    check.atomic_write_json_exclusive(
        output_dir / "correctness-summary.json",
        summary,
    )

    validation = check.validate_correctness_output(
        output_dir=output_dir,
        protocol=protocol,
        run_id=run_id,
    )

    assert validation["gate"] == "PASS"
    assert validation["file_count"] == 2
    assert validation["comparison_row_count"] == 64
    assert validation["sequence_index_base"] == 0
    assert validation["context_boundary_pass"] is True


def test_stage_f_functional_sources_contain_no_training_or_checkpoint_writes():
    for relative_path in (
        "scripts/check_day14_kv_cache.py",
        "scripts/benchmark_day14_kv_cache.py",
    ):
        source = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = [
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
        ]

        assert "backward" not in calls
        assert "step" not in calls
        assert "torch.save" not in source
        assert "torch.optim" not in source
