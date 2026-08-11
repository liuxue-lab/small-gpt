from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml
from model import GPTConfig, GPTConfigError

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


def test_load_debug_project_config():
    config = GPTConfig.from_yaml(PROJECT_ROOT / "configs" / "debug.yaml")

    assert config.n_layer == 2
    assert config.n_head == 2
    assert config.n_embd == 128
    assert config.head_dim == 64
    assert config.parameter_count == 2_508_032


def test_load_baseline_project_config():
    config = GPTConfig.from_yaml(PROJECT_ROOT / "configs" / "baseline.yaml")

    assert config.n_layer == 8
    assert config.n_head == 8
    assert config.n_embd == 512
    assert config.head_dim == 64
    assert config.parameter_count == 33_833_984


def test_tiny_config_properties():
    config = tiny_config()

    assert config.head_dim == 8
    assert config.residual_init_std == pytest.approx(0.02 / (2 * config.n_layer) ** 0.5)
    assert config.to_dict()["architecture"] == "decoder_only_gpt"


def test_config_is_frozen():
    config = tiny_config()

    with pytest.raises(FrozenInstanceError):
        config.n_layer = 3


@pytest.mark.parametrize(
    "overrides",
    (
        {"n_layer": 0},
        {"n_head": True},
        {"n_embd": -1},
        {"ffn_hidden": 0},
        {"context_length": 0},
        {"vocab_size": 0},
    ),
)
def test_rejects_non_positive_or_boolean_dimensions(overrides):
    with pytest.raises(GPTConfigError, match="positive integer"):
        tiny_config(**overrides)


def test_rejects_incompatible_head_dimensions():
    with pytest.raises(GPTConfigError, match="divisible"):
        tiny_config(n_head=3)


def test_rejects_non_four_x_ffn():
    with pytest.raises(GPTConfigError, match=r"4 \* n_embd"):
        tiny_config(ffn_hidden=96)


@pytest.mark.parametrize("dropout", (-0.1, 1.0, float("inf")))
def test_rejects_invalid_dropout(dropout):
    with pytest.raises(GPTConfigError, match="dropout"):
        tiny_config(dropout=dropout)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    (
        ("layer_norm_eps", 0.0),
        ("layer_norm_eps", float("inf")),
        ("init_std", 0.0),
        ("init_std", float("nan")),
    ),
)
def test_rejects_invalid_positive_finite_scalars(field_name, invalid_value):
    with pytest.raises(GPTConfigError, match=field_name):
        tiny_config(**{field_name: invalid_value})


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    (
        ("architecture", "encoder_decoder"),
        ("normalization", "rmsnorm"),
        ("norm_position", "post"),
        ("activation", "relu"),
        ("gelu_approximate", "none"),
        ("position_encoding", "rope"),
        ("tie_embeddings", False),
        ("linear_bias", True),
        ("lm_head_bias", True),
        ("layer_norm_affine", False),
        ("scale_residual_projections", False),
    ),
)
def test_rejects_frozen_architecture_drift(field_name, invalid_value):
    with pytest.raises(GPTConfigError, match=field_name):
        tiny_config(**{field_name: invalid_value})


def test_from_mapping_rejects_missing_field():
    values = tiny_config().to_dict()
    del values["activation"]

    with pytest.raises(GPTConfigError, match="missing fields"):
        GPTConfig.from_mapping(values)


def test_from_mapping_rejects_unknown_field():
    values = tiny_config().to_dict()
    values["positon_encoding"] = "learned_absolute"

    with pytest.raises(GPTConfigError, match="unknown fields"):
        GPTConfig.from_mapping(values)


def test_project_yaml_rejects_tokenizer_vocab_mismatch(tmp_path):
    config = tiny_config(vocab_size=16_384).to_dict()
    document = {
        "model": config,
        "tokenizer": {"vocab_size": 16_383},
    }
    path = tmp_path / "bad.yaml"
    path.write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(GPTConfigError, match="does not match") as error_info:
        GPTConfig.from_yaml(path)

    assert str(path) in str(error_info.value)


def test_project_yaml_rejects_matching_non_project_vocab(tmp_path):
    config = tiny_config().to_dict()
    document = {
        "model": config,
        "tokenizer": {"vocab_size": config["vocab_size"]},
    }
    path = tmp_path / "wrong-project-vocab.yaml"
    path.write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(GPTConfigError, match="must equal 16384") as error_info:
        GPTConfig.from_yaml(path)

    assert str(path) in str(error_info.value)


def test_non_project_yaml_can_use_tiny_vocab(tmp_path):
    document = {
        "model": tiny_config().to_dict(),
        "tokenizer": {"vocab_size": 64},
    }
    path = tmp_path / "tiny.yaml"
    path.write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )

    config = GPTConfig.from_yaml(
        path,
        require_project_contract=False,
    )

    assert config.vocab_size == 64
