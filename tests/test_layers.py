import pytest
import torch
from model import MLP, GPTConfig, TokenPositionEmbedding
from torch import nn


def tiny_config(**overrides) -> GPTConfig:
    values = {
        "architecture": "decoder_only_gpt",
        "n_layer": 2,
        "n_head": 4,
        "n_embd": 32,
        "ffn_hidden": 128,
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


def test_token_position_embedding_shape_and_dtype():
    config = tiny_config()
    layer = TokenPositionEmbedding(config)
    input_ids = torch.randint(0, config.vocab_size, (3, 7))

    output = layer(input_ids)

    assert output.shape == (3, 7, config.n_embd)
    assert output.dtype == layer.token_embedding.weight.dtype


def test_position_embeddings_are_shared_across_batch():
    config = tiny_config()
    layer = TokenPositionEmbedding(config)
    input_ids = torch.zeros((2, 5), dtype=torch.long)

    with torch.no_grad():
        layer.token_embedding.weight.zero_()
        positions = torch.arange(
            config.context_length * config.n_embd,
            dtype=layer.position_embedding.weight.dtype,
        ).view(config.context_length, config.n_embd)
        layer.position_embedding.weight.copy_(positions)

    output = layer(input_ids)

    torch.testing.assert_close(output[0], positions[:5])
    torch.testing.assert_close(output[1], positions[:5])


def test_embedding_accepts_full_context_length():
    config = tiny_config()
    layer = TokenPositionEmbedding(config)
    input_ids = torch.zeros(
        (1, config.context_length),
        dtype=torch.long,
    )

    output = layer(input_ids)

    assert output.shape == (1, config.context_length, config.n_embd)


def test_embedding_and_mlp_use_configured_dropout():
    config = tiny_config(dropout=0.25)

    assert TokenPositionEmbedding(config).dropout.p == 0.25
    assert MLP(config).dropout.p == 0.25


def test_embedding_rejects_sequence_longer_than_context():
    config = tiny_config()
    layer = TokenPositionEmbedding(config)
    input_ids = torch.zeros(
        (1, config.context_length + 1),
        dtype=torch.long,
    )

    with pytest.raises(ValueError, match="exceeds configured context"):
        layer(input_ids)


def test_embedding_rejects_wrong_rank():
    layer = TokenPositionEmbedding(tiny_config())

    with pytest.raises(ValueError, match="shape"):
        layer(torch.zeros(8, dtype=torch.long))


def test_embedding_rejects_non_long_ids():
    layer = TokenPositionEmbedding(tiny_config())

    with pytest.raises(TypeError, match="torch.long"):
        layer(torch.zeros((2, 4), dtype=torch.int32))


@pytest.mark.parametrize("invalid_id", (-1, 64))
def test_embedding_rejects_out_of_range_token_ids(invalid_id):
    layer = TokenPositionEmbedding(tiny_config())
    input_ids = torch.zeros((2, 4), dtype=torch.long)
    input_ids[0, 0] = invalid_id

    with pytest.raises(ValueError, match="valid range"):
        layer(input_ids)


@pytest.mark.parametrize("shape", ((0, 4), (2, 0)))
def test_embedding_rejects_empty_dimensions(shape):
    layer = TokenPositionEmbedding(tiny_config())
    input_ids = torch.empty(shape, dtype=torch.long)

    with pytest.raises(ValueError, match="dimension must be positive"):
        layer(input_ids)


def test_mlp_architecture_matches_contract():
    config = tiny_config()
    mlp = MLP(config)

    assert mlp.fc_in.in_features == config.n_embd
    assert mlp.fc_in.out_features == config.ffn_hidden
    assert mlp.fc_out.in_features == config.ffn_hidden
    assert mlp.fc_out.out_features == config.n_embd
    assert mlp.fc_in.bias is None
    assert mlp.fc_out.bias is None
    assert isinstance(mlp.activation, nn.GELU)
    assert mlp.activation.approximate == "tanh"
    assert mlp.dropout.p == 0.0


def test_mlp_preserves_outer_shape():
    config = tiny_config()
    mlp = MLP(config)
    hidden_states = torch.randn(3, 7, config.n_embd)

    output = mlp(hidden_states)

    assert output.shape == hidden_states.shape


def test_mlp_is_deterministic_when_dropout_is_zero():
    config = tiny_config()
    mlp = MLP(config)
    hidden_states = torch.randn(2, 5, config.n_embd)

    mlp.train()
    first = mlp(hidden_states)
    second = mlp(hidden_states)
    mlp.eval()
    third = mlp(hidden_states)

    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(first, third, rtol=0, atol=0)


def test_mlp_backward_reaches_input_and_all_parameters():
    config = tiny_config()
    mlp = MLP(config)
    hidden_states = torch.randn(
        2,
        5,
        config.n_embd,
        requires_grad=True,
    )

    loss = mlp(hidden_states).square().mean()
    loss.backward()

    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()
    assert torch.count_nonzero(hidden_states.grad) > 0

    for parameter in mlp.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


def test_mlp_rejects_wrong_rank():
    mlp = MLP(tiny_config())

    with pytest.raises(ValueError, match="shape"):
        mlp(torch.randn(4, 32))


def test_mlp_rejects_wrong_embedding_width():
    mlp = MLP(tiny_config())

    with pytest.raises(ValueError, match="n_embd"):
        mlp(torch.randn(2, 4, 31))


def test_mlp_rejects_non_floating_input():
    mlp = MLP(tiny_config())

    with pytest.raises(TypeError, match="floating-point"):
        mlp(torch.zeros((2, 4, 32), dtype=torch.long))
