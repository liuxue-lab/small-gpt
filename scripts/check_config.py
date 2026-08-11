from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIRECTORY = PROJECT_ROOT / "configs"

CONFIG_FILENAMES = ("debug.yaml", "baseline.yaml")

EXPECTED_PARAMETER_COUNTS = {
    "debug.yaml": 2_508_032,
    "baseline.yaml": 33_833_984,
}

MODEL_FIELDS = {
    "architecture",
    "n_layer",
    "n_head",
    "n_embd",
    "ffn_hidden",
    "context_length",
    "vocab_size",
    "dropout",
    "tie_embeddings",
    "normalization",
    "norm_position",
    "layer_norm_eps",
    "activation",
    "gelu_approximate",
    "position_encoding",
    "linear_bias",
    "lm_head_bias",
    "layer_norm_affine",
    "init_std",
    "scale_residual_projections",
}


def load_config(filename: str) -> dict[str, Any]:
    config_path = CONFIG_DIRECTORY / filename

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise TypeError(
            f"{filename} did not produce a configuration dictionary"
        )

    return config


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def validate_config(filename: str, config: dict[str, Any]) -> None:
    required_sections = {
        "project",
        "data",
        "tokenizer",
        "model",
        "training",
    }
    missing_sections = required_sections - set(config)
    _require(
        not missing_sections,
        f"{filename} is missing sections: {sorted(missing_sections)}",
    )

    model = config["model"]
    _require(
        isinstance(model, dict),
        f"{filename}: model must be a mapping",
    )

    missing_model_fields = MODEL_FIELDS - set(model)
    unknown_model_fields = set(model) - MODEL_FIELDS
    _require(
        not missing_model_fields,
        f"{filename}: model is missing fields: "
        f"{sorted(missing_model_fields)}",
    )
    _require(
        not unknown_model_fields,
        f"{filename}: model has unknown fields: "
        f"{sorted(unknown_model_fields)}",
    )

    for field in (
        "n_layer",
        "n_head",
        "n_embd",
        "ffn_hidden",
        "context_length",
        "vocab_size",
    ):
        value = model[field]
        _require(
            _is_plain_int(value) and value > 0,
            f"{filename}: model.{field} must be a positive integer",
        )

    _require(
        model["n_embd"] % model["n_head"] == 0,
        f"{filename}: model.n_embd must be divisible by model.n_head",
    )
    _require(
        model["ffn_hidden"] == 4 * model["n_embd"],
        f"{filename}: model.ffn_hidden must equal 4 * model.n_embd",
    )

    _require(
        _is_finite_number(model["dropout"])
        and 0.0 <= float(model["dropout"]) < 1.0,
        f"{filename}: model.dropout must be finite and in [0, 1)",
    )
    _require(
        _is_finite_number(model["layer_norm_eps"])
        and float(model["layer_norm_eps"]) > 0.0,
        f"{filename}: model.layer_norm_eps must be finite and positive",
    )
    _require(
        _is_finite_number(model["init_std"])
        and float(model["init_std"]) > 0.0,
        f"{filename}: model.init_std must be finite and positive",
    )

    for field in (
        "tie_embeddings",
        "linear_bias",
        "lm_head_bias",
        "layer_norm_affine",
        "scale_residual_projections",
    ):
        _require(
            isinstance(model[field], bool),
            f"{filename}: model.{field} must be a boolean",
        )

    expected_values = {
        "architecture": "decoder_only_gpt",
        "normalization": "layernorm",
        "norm_position": "pre",
        "activation": "gelu",
        "gelu_approximate": "tanh",
        "position_encoding": "learned_absolute",
        "tie_embeddings": True,
        "linear_bias": False,
        "lm_head_bias": False,
        "layer_norm_affine": True,
        "scale_residual_projections": True,
        "vocab_size": 16_384,
    }
    for field, expected in expected_values.items():
        _require(
            model[field] == expected,
            f"{filename}: model.{field} must equal {expected!r}, "
            f"got {model[field]!r}",
        )

    tokenizer = config["tokenizer"]
    _require(
        isinstance(tokenizer, dict),
        f"{filename}: tokenizer must be a mapping",
    )
    _require(
        tokenizer.get("vocab_size") == model["vocab_size"],
        f"{filename}: tokenizer and model vocabulary sizes must match",
    )

    data = config["data"]
    _require(
        isinstance(data, dict),
        f"{filename}: data must be a mapping",
    )
    split_names = (
        "train_ratio",
        "validation_ratio",
        "test_ratio",
    )
    if all(name in data for name in split_names):
        split_total = sum(data[name] for name in split_names)
        _require(
            math.isclose(split_total, 1.0, abs_tol=1e-9),
            f"{filename}: dataset split ratios must sum to 1.0, "
            f"got {split_total}",
        )


def estimate_parameters(config: dict[str, Any]) -> int:
    model = config["model"]

    vocab_size = model["vocab_size"]
    context_length = model["context_length"]
    n_layer = model["n_layer"]
    n_embd = model["n_embd"]
    ffn_hidden = model["ffn_hidden"]

    token_embedding = vocab_size * n_embd
    position_embedding = context_length * n_embd

    qkv_projection = 3 * n_embd * n_embd
    attention_output_projection = n_embd * n_embd
    mlp_projections = 2 * n_embd * ffn_hidden

    if model["linear_bias"]:
        qkv_projection += 3 * n_embd
        attention_output_projection += n_embd
        mlp_projections += ffn_hidden + n_embd

    layer_norms_per_block = 4 * n_embd
    if not model["layer_norm_affine"]:
        layer_norms_per_block = 0

    transformer_blocks = n_layer * (
        qkv_projection
        + attention_output_projection
        + mlp_projections
        + layer_norms_per_block
    )

    final_norm = 2 * n_embd if model["layer_norm_affine"] else 0

    output_head = 0
    if not model["tie_embeddings"]:
        output_head += vocab_size * n_embd
    if model["lm_head_bias"]:
        output_head += vocab_size

    return (
        token_embedding
        + position_embedding
        + transformer_blocks
        + final_norm
        + output_head
    )


def print_config_summary(filename: str, config: dict[str, Any]) -> None:
    model = config["model"]
    head_dimension = model["n_embd"] // model["n_head"]
    parameter_count = estimate_parameters(config)

    print(f"Configuration      : {filename}")
    print(f"Architecture       : {model['architecture']}")
    print(f"Transformer layers : {model['n_layer']}")
    print(f"Attention heads    : {model['n_head']}")
    print(f"Embedding size     : {model['n_embd']}")
    print(f"Head dimension     : {head_dimension}")
    print(f"FFN hidden size    : {model['ffn_hidden']}")
    print(f"Context length     : {model['context_length']}")
    print(f"Vocabulary size    : {model['vocab_size']}")
    print(f"Norm order         : {model['norm_position']}")
    print(f"Position encoding  : {model['position_encoding']}")
    print(f"Linear bias        : {model['linear_bias']}")
    print(f"Tied embeddings    : {model['tie_embeddings']}")
    print(f"Exact parameters   : {parameter_count:,}")
    print("-" * 52)


def main() -> int:
    for config_filename in CONFIG_FILENAMES:
        loaded_config = load_config(config_filename)
        validate_config(config_filename, loaded_config)

        parameter_count = estimate_parameters(loaded_config)
        expected_count = EXPECTED_PARAMETER_COUNTS[config_filename]
        if parameter_count != expected_count:
            raise ValueError(
                f"{config_filename}: exact parameter count must be "
                f"{expected_count:,}, got {parameter_count:,}"
            )

        print_config_summary(config_filename, loaded_config)

    print("All configuration checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
