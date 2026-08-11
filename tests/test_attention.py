import pytest
import torch
from model import CausalSelfAttention, GPTConfig


def tiny_config(**overrides) -> GPTConfig:
    values = {
        "architecture": "decoder_only_gpt",
        "n_layer": 2,
        "n_head": 2,
        "n_embd": 8,
        "ffn_hidden": 32,
        "context_length": 8,
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


def capture_attention_probabilities(
    attention: CausalSelfAttention,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    captured = {}

    def capture(_module, _inputs, output):
        captured["probabilities"] = output.detach().clone()

    handle = attention.attn_dropout.register_forward_hook(capture)
    try:
        output = attention(hidden_states)
    finally:
        handle.remove()

    return output, captured["probabilities"]


def test_projection_shapes_scale_and_bias_contract():
    config = tiny_config()
    attention = CausalSelfAttention(config)

    assert attention.qkv_proj.in_features == config.n_embd
    assert attention.qkv_proj.out_features == 3 * config.n_embd
    assert attention.out_proj.in_features == config.n_embd
    assert attention.out_proj.out_features == config.n_embd
    assert attention.qkv_proj.bias is None
    assert attention.out_proj.bias is None
    assert attention.head_dim == config.n_embd // config.n_head
    assert attention.scale == pytest.approx(config.head_dim**-0.5)


def test_split_and_merge_heads_round_trip():
    config = tiny_config()
    attention = CausalSelfAttention(config)
    hidden_states = torch.randn(3, 5, config.n_embd)

    split = attention._split_heads(hidden_states)
    merged = attention._merge_heads(split)

    assert split.shape == (3, config.n_head, 5, config.head_dim)
    torch.testing.assert_close(merged, hidden_states, rtol=0, atol=0)


def test_attention_scaling_uses_head_dimension():
    config = tiny_config()
    attention = CausalSelfAttention(config).eval()
    query = torch.tensor(
        [
            [
                [[2.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
                [[1.0, 1.0, 0.0, 0.0], [2.0, -1.0, 0.0, 0.0]],
            ]
        ]
    )
    key = query.clone()

    probabilities = attention._attention_probabilities(query, key)
    scores = torch.matmul(query, key.transpose(-2, -1))
    scores = scores * (config.head_dim**-0.5)
    mask = attention.causal_mask[:, :, :2, :2]
    scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    expected = torch.softmax(scores, dim=-1, dtype=torch.float32)

    torch.testing.assert_close(probabilities, expected)


def test_causal_mask_shape_dtype_and_diagonal():
    config = tiny_config()
    attention = CausalSelfAttention(config)
    expected = torch.tril(
        torch.ones(
            config.context_length,
            config.context_length,
            dtype=torch.bool,
        )
    )

    assert attention.causal_mask.shape == (
        1,
        1,
        config.context_length,
        config.context_length,
    )
    assert attention.causal_mask.dtype == torch.bool
    torch.testing.assert_close(attention.causal_mask[0, 0], expected)
    assert torch.diagonal(attention.causal_mask[0, 0]).all()


def test_causal_mask_is_non_persistent_buffer_not_parameter():
    attention = CausalSelfAttention(tiny_config())

    assert "causal_mask" in dict(attention.named_buffers())
    assert "causal_mask" not in dict(attention.named_parameters())
    assert "causal_mask" not in attention.state_dict()


def test_attention_output_shape_dtype_and_finiteness():
    config = tiny_config()
    attention = CausalSelfAttention(config)
    hidden_states = torch.randn(3, 5, config.n_embd)

    output = attention(hidden_states)

    assert output.shape == hidden_states.shape
    assert output.dtype == hidden_states.dtype
    assert torch.isfinite(output).all()


def test_future_probabilities_are_zero_and_rows_sum_to_one():
    config = tiny_config()
    attention = CausalSelfAttention(config).eval()
    hidden_states = torch.randn(2, 6, config.n_embd)

    _, probabilities = capture_attention_probabilities(
        attention,
        hidden_states,
    )
    future_mask = torch.triu(
        torch.ones(6, 6, dtype=torch.bool),
        diagonal=1,
    )
    future_probabilities = probabilities[..., future_mask]

    assert probabilities.shape == (2, config.n_head, 6, 6)
    assert torch.isfinite(probabilities).all()
    assert torch.count_nonzero(future_probabilities) == 0
    torch.testing.assert_close(
        probabilities.sum(dim=-1),
        torch.ones_like(probabilities[..., 0]),
    )


def test_softmax_is_explicitly_computed_in_float32(monkeypatch):
    attention = CausalSelfAttention(tiny_config())
    hidden_states = torch.randn(2, 5, attention.config.n_embd)
    original_softmax = torch.softmax
    observed_dtypes = []

    def recording_softmax(input_tensor, dim, *, dtype=None):
        observed_dtypes.append(dtype)
        return original_softmax(input_tensor, dim=dim, dtype=dtype)

    monkeypatch.setattr(torch, "softmax", recording_softmax)

    attention(hidden_states)

    assert observed_dtypes == [torch.float32]


def test_uniform_scores_average_only_visible_values():
    config = tiny_config(n_embd=4, n_head=2, ffn_hidden=16)
    attention = CausalSelfAttention(config).eval()
    hidden_states = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ]
        ]
    )

    with torch.no_grad():
        attention.qkv_proj.weight.zero_()
        attention.qkv_proj.weight[2 * config.n_embd : 3 * config.n_embd].copy_(
            torch.eye(config.n_embd)
        )
        attention.out_proj.weight.copy_(torch.eye(config.n_embd))

    output = attention(hidden_states)
    expected = torch.stack(
        [
            hidden_states[:, : position + 1].mean(dim=1)
            for position in range(hidden_states.shape[1])
        ],
        dim=1,
    )

    torch.testing.assert_close(output, expected)


def test_sequence_length_one_is_finite():
    config = tiny_config()
    attention = CausalSelfAttention(config)
    hidden_states = torch.randn(2, 1, config.n_embd)

    output, probabilities = capture_attention_probabilities(
        attention,
        hidden_states,
    )

    assert output.shape == hidden_states.shape
    assert torch.isfinite(output).all()
    torch.testing.assert_close(
        probabilities,
        torch.ones_like(probabilities),
    )


def test_full_context_length_is_supported():
    config = tiny_config()
    attention = CausalSelfAttention(config)
    hidden_states = torch.randn(
        2,
        config.context_length,
        config.n_embd,
    )

    output = attention(hidden_states)

    assert output.shape == hidden_states.shape
    assert torch.isfinite(output).all()


def test_changing_future_input_does_not_change_past_output():
    torch.manual_seed(1337)
    config = tiny_config()
    attention = CausalSelfAttention(config).eval()
    original = torch.randn(2, 7, config.n_embd)
    modified = original.clone()
    modified[:, 4:] = torch.randn_like(modified[:, 4:]) * 10.0

    original_output = attention(original)
    modified_output = attention(modified)

    torch.testing.assert_close(
        original_output[:, :4],
        modified_output[:, :4],
        rtol=0,
        atol=0,
    )


def test_backward_reaches_input_and_projection_weights():
    config = tiny_config()
    attention = CausalSelfAttention(config)
    hidden_states = torch.randn(
        2,
        6,
        config.n_embd,
        requires_grad=True,
    )

    loss = attention(hidden_states).square().mean()
    loss.backward()

    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()
    assert torch.count_nonzero(hidden_states.grad) > 0

    for projection in (attention.qkv_proj, attention.out_proj):
        assert projection.weight.grad is not None
        assert torch.isfinite(projection.weight.grad).all()
        assert torch.count_nonzero(projection.weight.grad) > 0


def test_dropout_zero_is_deterministic_in_train_and_eval_modes():
    config = tiny_config()
    attention = CausalSelfAttention(config)
    hidden_states = torch.randn(2, 6, config.n_embd)

    attention.train()
    first = attention(hidden_states)
    second = attention(hidden_states)
    attention.eval()
    third = attention(hidden_states)

    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(first, third, rtol=0, atol=0)


def test_attention_uses_configured_dropout():
    attention = CausalSelfAttention(tiny_config(dropout=0.25))

    assert attention.attn_dropout.p == 0.25
    assert attention.resid_dropout.p == 0.25


def test_forward_does_not_call_scaled_dot_product_attention(monkeypatch):
    def forbidden_call(*_args, **_kwargs):
        raise AssertionError("SDPA must not be called by the handwritten path")

    monkeypatch.setattr(
        torch.nn.functional,
        "scaled_dot_product_attention",
        forbidden_call,
    )
    attention = CausalSelfAttention(tiny_config())

    output = attention(torch.randn(2, 5, attention.config.n_embd))

    assert output.shape == (2, 5, attention.config.n_embd)


def test_rejects_sequence_longer_than_context():
    config = tiny_config()
    attention = CausalSelfAttention(config)
    hidden_states = torch.randn(
        1,
        config.context_length + 1,
        config.n_embd,
    )

    with pytest.raises(ValueError, match="exceeds configured context"):
        attention(hidden_states)


def test_rejects_wrong_rank():
    attention = CausalSelfAttention(tiny_config())

    with pytest.raises(ValueError, match="shape"):
        attention(torch.randn(4, 8))


def test_rejects_wrong_embedding_width():
    attention = CausalSelfAttention(tiny_config())

    with pytest.raises(ValueError, match="n_embd"):
        attention(torch.randn(2, 4, 7))


def test_rejects_non_floating_input():
    attention = CausalSelfAttention(tiny_config())

    with pytest.raises(TypeError, match="floating-point"):
        attention(torch.zeros((2, 4, 8), dtype=torch.long))


@pytest.mark.parametrize("shape", ((0, 4, 8), (2, 0, 8)))
def test_rejects_empty_dimensions(shape):
    attention = CausalSelfAttention(tiny_config())

    with pytest.raises(ValueError, match="dimension must be positive"):
        attention(torch.empty(shape))


def test_rejects_non_tensor_input():
    attention = CausalSelfAttention(tiny_config())

    with pytest.raises(TypeError, match="torch.Tensor"):
        attention([[[0.0] * 8]])
