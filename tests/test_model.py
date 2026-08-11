from pathlib import Path

import pytest
import torch
from model import GPT, GPTConfig, GPTOutput, TransformerBlock
from torch import nn
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def test_gpt_module_structure_matches_contract():
    config = tiny_config()
    model = GPT(config)

    assert isinstance(model.token_embedding, nn.Embedding)
    assert model.token_embedding.num_embeddings == config.vocab_size
    assert model.token_embedding.embedding_dim == config.n_embd
    assert isinstance(model.position_embedding, nn.Embedding)
    assert model.position_embedding.num_embeddings == config.context_length
    assert model.position_embedding.embedding_dim == config.n_embd
    assert model.embedding_dropout.p == config.dropout
    assert len(model.blocks) == config.n_layer
    assert all(isinstance(block, TransformerBlock) for block in model.blocks)
    assert isinstance(model.final_norm, nn.LayerNorm)
    assert model.final_norm.normalized_shape == (config.n_embd,)
    assert model.lm_head.in_features == config.n_embd
    assert model.lm_head.out_features == config.vocab_size
    assert model.lm_head.bias is None


def test_forward_without_targets_returns_full_logits_and_no_loss():
    config = tiny_config()
    model = GPT(config)
    input_ids = torch.randint(0, config.vocab_size, (3, 7))

    output = model(input_ids)

    assert isinstance(output, GPTOutput)
    assert output.logits.shape == (3, 7, config.vocab_size)
    assert torch.isfinite(output.logits).all()
    assert output.loss is None


def test_forward_with_targets_returns_full_logits_and_scalar_loss():
    config = tiny_config()
    model = GPT(config)
    input_ids = torch.randint(0, config.vocab_size, (3, 7))
    targets = torch.randint(0, config.vocab_size, (3, 7))

    output = model(input_ids, targets)

    assert output.logits.shape == (3, 7, config.vocab_size)
    assert output.loss is not None
    assert output.loss.ndim == 0
    assert torch.isfinite(output.loss)


def test_loss_matches_unshifted_flattened_cross_entropy():
    config = tiny_config()
    model = GPT(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 6))
    targets = torch.randint(0, config.vocab_size, (2, 6))

    output = model(input_ids, targets)
    expected = F.cross_entropy(
        output.logits.reshape(-1, config.vocab_size),
        targets.reshape(-1),
    )

    assert output.loss is not None
    torch.testing.assert_close(output.loss, expected)


def test_weight_tying_uses_the_same_parameter_and_storage():
    model = GPT(tiny_config())

    assert model.lm_head.weight is model.token_embedding.weight
    assert model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr()

    with torch.no_grad():
        model.token_embedding.weight[0, 0] = 123.0

    assert model.lm_head.weight[0, 0].item() == 123.0


def test_parameter_iteration_deduplicates_tied_weight():
    config = tiny_config()
    model = GPT(config)
    parameters = list(model.parameters())

    assert len({id(parameter) for parameter in parameters}) == len(parameters)
    assert sum(parameter.numel() for parameter in parameters) == config.parameter_count


@pytest.mark.parametrize(
    ("filename", "expected_parameters"),
    (
        ("debug.yaml", 2_508_032),
        ("baseline.yaml", 33_833_984),
    ),
)
def test_formal_model_constructs_with_exact_parameter_count(
    filename,
    expected_parameters,
):
    config = GPTConfig.from_yaml(PROJECT_ROOT / "configs" / filename)
    model = GPT(config)

    actual_parameters = sum(parameter.numel() for parameter in model.parameters())

    assert actual_parameters == expected_parameters
    assert actual_parameters == config.parameter_count
    assert model.lm_head.weight is model.token_embedding.weight


def test_initialization_is_finite_and_layer_norm_is_exact():
    model = GPT(tiny_config())

    for parameter in model.parameters():
        assert torch.isfinite(parameter).all()

    layer_norms = [
        module for module in model.modules() if isinstance(module, nn.LayerNorm)
    ]
    assert len(layer_norms) == 2 * model.config.n_layer + 1
    for layer_norm in layer_norms:
        torch.testing.assert_close(
            layer_norm.weight,
            torch.ones_like(layer_norm.weight),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            layer_norm.bias,
            torch.zeros_like(layer_norm.bias),
            rtol=0,
            atol=0,
        )


def test_residual_projections_use_scaled_initialization():
    torch.manual_seed(42)
    config = tiny_config(
        n_layer=4,
        n_head=4,
        n_embd=128,
        ffn_hidden=512,
        vocab_size=256,
    )
    model = GPT(config)
    block = model.blocks[0]

    assert block.attention.qkv_proj.weight.std().item() == pytest.approx(
        config.init_std,
        rel=0.10,
    )
    assert block.mlp.fc_in.weight.std().item() == pytest.approx(
        config.init_std,
        rel=0.10,
    )
    assert block.attention.out_proj.weight.std().item() == pytest.approx(
        config.residual_init_std,
        rel=0.15,
    )
    assert block.mlp.fc_out.weight.std().item() == pytest.approx(
        config.residual_init_std,
        rel=0.15,
    )


def test_all_linear_layers_follow_bias_free_contract():
    model = GPT(tiny_config())

    linear_layers = [
        module for module in model.modules() if isinstance(module, nn.Linear)
    ]

    assert linear_layers
    assert all(layer.bias is None for layer in linear_layers)


def test_full_context_length_is_supported():
    config = tiny_config()
    model = GPT(config)
    input_ids = torch.randint(
        0,
        config.vocab_size,
        (1, config.context_length),
    )

    output = model(input_ids)

    assert output.logits.shape == (
        1,
        config.context_length,
        config.vocab_size,
    )


def test_rejects_non_tensor_input_ids():
    model = GPT(tiny_config())

    with pytest.raises(TypeError, match="torch.Tensor"):
        model([[0, 1, 2]])


def test_rejects_wrong_rank_input_ids():
    model = GPT(tiny_config())

    with pytest.raises(ValueError, match="shape"):
        model(torch.zeros(8, dtype=torch.long))


def test_rejects_non_long_input_ids():
    model = GPT(tiny_config())

    with pytest.raises(TypeError, match="torch.long"):
        model(torch.zeros((2, 4), dtype=torch.int32))


@pytest.mark.parametrize("shape", ((0, 4), (2, 0)))
def test_rejects_empty_input_dimensions(shape):
    model = GPT(tiny_config())

    with pytest.raises(ValueError, match="dimension must be positive"):
        model(torch.empty(shape, dtype=torch.long))


def test_rejects_sequence_longer_than_context():
    config = tiny_config()
    model = GPT(config)
    input_ids = torch.zeros(
        (1, config.context_length + 1),
        dtype=torch.long,
    )

    with pytest.raises(ValueError, match="exceeds configured context"):
        model(input_ids)


@pytest.mark.parametrize("invalid_id", (-1, 64))
def test_rejects_out_of_range_input_ids(invalid_id):
    model = GPT(tiny_config())
    input_ids = torch.zeros((2, 4), dtype=torch.long)
    input_ids[0, 0] = invalid_id

    with pytest.raises(ValueError, match="valid range"):
        model(input_ids)


def test_rejects_non_tensor_targets():
    model = GPT(tiny_config())
    input_ids = torch.zeros((2, 4), dtype=torch.long)

    with pytest.raises(TypeError, match="targets must be a torch.Tensor"):
        model(input_ids, [[0, 0, 0, 0], [0, 0, 0, 0]])


def test_rejects_target_shape_mismatch():
    model = GPT(tiny_config())
    input_ids = torch.zeros((2, 4), dtype=torch.long)
    targets = torch.zeros((2, 3), dtype=torch.long)

    with pytest.raises(ValueError, match="shape must match"):
        model(input_ids, targets)


def test_rejects_non_long_targets():
    model = GPT(tiny_config())
    input_ids = torch.zeros((2, 4), dtype=torch.long)
    targets = torch.zeros((2, 4), dtype=torch.int32)

    with pytest.raises(TypeError, match="torch.long"):
        model(input_ids, targets)


def test_rejects_target_device_mismatch():
    model = GPT(tiny_config())
    input_ids = torch.zeros((2, 4), dtype=torch.long)
    targets = torch.zeros((2, 4), dtype=torch.long, device="meta")

    with pytest.raises(ValueError, match="same device"):
        model(input_ids, targets)


@pytest.mark.parametrize("invalid_id", (-1, 64))
def test_rejects_out_of_range_targets(invalid_id):
    model = GPT(tiny_config())
    input_ids = torch.zeros((2, 4), dtype=torch.long)
    targets = input_ids.clone()
    targets[0, 0] = invalid_id

    with pytest.raises(ValueError, match="valid range"):
        model(input_ids, targets)


def test_future_tokens_do_not_change_past_logits():
    torch.manual_seed(1337)
    config = tiny_config()
    model = GPT(config).eval()
    original = torch.randint(0, config.vocab_size, (2, 12))
    modified = original.clone()
    modified[:, 7:] = (modified[:, 7:] + 1) % config.vocab_size

    with torch.no_grad():
        original_logits = model(original).logits
        modified_logits = model(modified).logits

    torch.testing.assert_close(
        original_logits[:, :7],
        modified_logits[:, :7],
        rtol=0,
        atol=1.0e-7,
    )
    assert torch.count_nonzero(original_logits[:, 7:] - modified_logits[:, 7:]) > 0


def test_full_model_backward_and_second_pass_have_valid_gradients():
    torch.manual_seed(1337)
    config = tiny_config()
    model = GPT(config)
    input_ids = torch.randint(0, config.vocab_size, (3, 10))
    targets = torch.randint(0, config.vocab_size, (3, 10))

    first_output = model(input_ids, targets)
    assert first_output.loss is not None
    first_output.loss.backward()

    first_gradients = {
        name: parameter.grad for name, parameter in model.named_parameters()
    }
    assert first_gradients
    assert all(gradient is not None for gradient in first_gradients.values())
    assert all(
        bool(torch.isfinite(gradient).all())
        for gradient in first_gradients.values()
        if gradient is not None
    )
    assert all(
        bool(torch.count_nonzero(gradient))
        for gradient in first_gradients.values()
        if gradient is not None
    )

    assert model.token_embedding.weight.grad is not None
    assert model.position_embedding.weight.grad is not None
    assert model.final_norm.weight.grad is not None
    for block in model.blocks:
        assert block.norm1.weight.grad is not None
        assert block.norm2.weight.grad is not None
        assert block.attention.qkv_proj.weight.grad is not None
        assert block.attention.out_proj.weight.grad is not None
        assert block.mlp.fc_in.weight.grad is not None
        assert block.mlp.fc_out.weight.grad is not None

    model.zero_grad(set_to_none=True)
    assert all(parameter.grad is None for parameter in model.parameters())

    second_output = model(input_ids, targets)
    assert second_output.loss is not None
    second_output.loss.backward()

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


def test_tied_parameter_is_not_duplicated_in_optimizer_inputs():
    model = GPT(tiny_config())
    named_parameters = dict(model.named_parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    optimizer_parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]

    assert "token_embedding.weight" in named_parameters
    assert "lm_head.weight" not in named_parameters
    assert len({id(parameter) for parameter in optimizer_parameters}) == len(
        optimizer_parameters
    )
    assert (
        sum(
            parameter is model.token_embedding.weight
            for parameter in optimizer_parameters
        )
        == 1
    )


def test_state_dict_strict_round_trip_preserves_logits_and_tying(tmp_path):
    torch.manual_seed(1337)
    config = tiny_config()
    original_model = GPT(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (2, 9))

    with torch.no_grad():
        reference_logits = original_model(input_ids).logits

    state_path = tmp_path / "model-state.pt"
    torch.save(original_model.state_dict(), state_path)

    loaded_model = GPT(config).eval()
    state_dict = torch.load(
        state_path,
        map_location="cpu",
        weights_only=True,
    )
    incompatible = loaded_model.load_state_dict(state_dict, strict=True)

    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert loaded_model.lm_head.weight is loaded_model.token_embedding.weight
    assert (
        loaded_model.lm_head.weight.data_ptr()
        == loaded_model.token_embedding.weight.data_ptr()
    )

    with torch.no_grad():
        loaded_logits = loaded_model(input_ids).logits

    torch.testing.assert_close(
        loaded_logits,
        reference_logits,
        rtol=0,
        atol=0,
    )


def test_fixed_seed_reproduces_entire_state_dict():
    config = tiny_config()

    torch.manual_seed(2026)
    first_model = GPT(config)
    first_state = first_model.state_dict()

    torch.manual_seed(2026)
    second_model = GPT(config)
    second_state = second_model.state_dict()

    assert first_state.keys() == second_state.keys()
    for key in first_state:
        torch.testing.assert_close(
            first_state[key],
            second_state[key],
            rtol=0,
            atol=0,
        )


def test_causal_masks_are_buffers_excluded_from_state_and_parameters():
    config = tiny_config()
    model = GPT(config)
    named_buffers = dict(model.named_buffers())
    state_keys = set(model.state_dict())
    parameter_names = set(dict(model.named_parameters()))

    mask_names = {
        f"blocks.{index}.attention.causal_mask" for index in range(config.n_layer)
    }

    assert mask_names <= set(named_buffers)
    assert mask_names.isdisjoint(state_keys)
    assert mask_names.isdisjoint(parameter_names)
