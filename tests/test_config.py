import math
from copy import deepcopy

import pytest

from scripts.check_config import (
    EXPECTED_PARAMETER_COUNTS,
    estimate_parameters,
    load_config,
    validate_config,
)


def test_debug_model_dimensions():
    config = load_config("debug.yaml")
    model = config["model"]

    assert model["n_layer"] == 2
    assert model["n_head"] == 2
    assert model["n_embd"] == 128
    assert model["ffn_hidden"] == 512
    assert model["context_length"] == 128
    assert model["n_embd"] % model["n_head"] == 0


def test_baseline_model_dimensions():
    config = load_config("baseline.yaml")
    model = config["model"]

    assert model["n_layer"] == 8
    assert model["n_head"] == 8
    assert model["n_embd"] == 512
    assert model["ffn_hidden"] == 2048
    assert model["context_length"] == 512
    assert model["n_embd"] % model["n_head"] == 0


@pytest.mark.parametrize("filename", ("debug.yaml", "baseline.yaml"))
def test_model_architecture_contract(filename):
    model = load_config(filename)["model"]

    assert model["architecture"] == "decoder_only_gpt"
    assert model["normalization"] == "layernorm"
    assert model["norm_position"] == "pre"
    assert model["layer_norm_eps"] == pytest.approx(1.0e-5)
    assert model["activation"] == "gelu"
    assert model["gelu_approximate"] == "tanh"
    assert model["position_encoding"] == "learned_absolute"
    assert model["dropout"] == 0.0
    assert model["linear_bias"] is False
    assert model["lm_head_bias"] is False
    assert model["layer_norm_affine"] is True
    assert model["tie_embeddings"] is True
    assert model["init_std"] == pytest.approx(0.02)
    assert model["scale_residual_projections"] is True


@pytest.mark.parametrize(
    ("filename", "expected"),
    tuple(EXPECTED_PARAMETER_COUNTS.items()),
)
def test_exact_parameter_count(filename, expected):
    config = load_config(filename)

    assert estimate_parameters(config) == expected


def test_baseline_parameter_count_is_about_34m():
    config = load_config("baseline.yaml")
    parameter_count = estimate_parameters(config)

    assert 30_000_000 <= parameter_count <= 40_000_000


def test_baseline_dataset_splits_sum_to_one():
    config = load_config("baseline.yaml")
    data = config["data"]

    split_total = (
        data["train_ratio"]
        + data["validation_ratio"]
        + data["test_ratio"]
    )

    assert math.isclose(split_total, 1.0, abs_tol=1e-9)


def test_tokenizer_and_model_vocabulary_match():
    for filename in ("debug.yaml", "baseline.yaml"):
        config = load_config(filename)

        tokenizer_vocab_size = config["tokenizer"]["vocab_size"]
        model_vocab_size = config["model"]["vocab_size"]

        assert tokenizer_vocab_size == model_vocab_size == 16_384


@pytest.mark.parametrize("filename", ("debug.yaml", "baseline.yaml"))
def test_full_config_validation_passes(filename):
    config = load_config(filename)

    validate_config(filename, config)


def test_validation_rejects_incompatible_head_dimensions():
    config = deepcopy(load_config("debug.yaml"))
    config["model"]["n_head"] = 3

    with pytest.raises(ValueError, match="n_embd must be divisible"):
        validate_config("debug.yaml", config)


def test_validation_rejects_unknown_model_field():
    config = deepcopy(load_config("debug.yaml"))
    config["model"]["positon_encoding"] = "learned_absolute"

    with pytest.raises(ValueError, match="unknown fields"):
        validate_config("debug.yaml", config)


def test_validation_rejects_architecture_drift():
    config = deepcopy(load_config("debug.yaml"))
    config["model"]["linear_bias"] = True

    with pytest.raises(ValueError, match="linear_bias"):
        validate_config("debug.yaml", config)
