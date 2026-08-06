from __future__ import annotations

import math
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIRECTORY = PROJECT_ROOT / "configs"


def load_config(filename: str) -> dict:
    config_path = CONFIG_DIRECTORY / filename

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise TypeError(f"{filename} did not produce a configuration dictionary")

    return config


def validate_config(filename: str, config: dict) -> None:
    required_sections = {
        "project",
        "data",
        "tokenizer",
        "model",
        "training",
    }

    missing_sections = required_sections - set(config)

    assert not missing_sections, (
        f"{filename} is missing sections: {sorted(missing_sections)}"
    )

    model = config["model"]

    assert model["n_layer"] > 0, "n_layer must be positive"
    assert model["n_head"] > 0, "n_head must be positive"
    assert model["n_embd"] > 0, "n_embd must be positive"
    assert model["context_length"] > 0, "context_length must be positive"
    assert model["vocab_size"] > 0, "vocab_size must be positive"

    assert model["n_embd"] % model["n_head"] == 0, (
        "n_embd must be divisible by n_head"
    )

    assert model["ffn_hidden"] == 4 * model["n_embd"], (
        "ffn_hidden must equal 4 * n_embd"
    )

    data = config["data"]

    split_names = (
        "train_ratio",
        "validation_ratio",
        "test_ratio",
    )

    if all(name in data for name in split_names):
        split_total = sum(data[name] for name in split_names)

        assert math.isclose(split_total, 1.0, abs_tol=1e-9), (
            f"dataset split ratios must sum to 1.0, got {split_total}"
        )


def estimate_parameters(config: dict) -> int:
    model = config["model"]

    vocab_size = model["vocab_size"]
    context_length = model["context_length"]
    n_layer = model["n_layer"]
    n_embd = model["n_embd"]

    token_embedding = vocab_size * n_embd
    position_embedding = context_length * n_embd
    transformer_blocks = n_layer * 12 * n_embd * n_embd

    if model["tie_embeddings"]:
        output_head = 0
    else:
        output_head = vocab_size * n_embd

    return (
        token_embedding
        + position_embedding
        + transformer_blocks
        + output_head
    )


def print_config_summary(filename: str, config: dict) -> None:
    model = config["model"]

    head_dimension = model["n_embd"] // model["n_head"]
    parameter_count = estimate_parameters(config)

    print(f"Configuration     : {filename}")
    print(f"Transformer layers: {model['n_layer']}")
    print(f"Attention heads   : {model['n_head']}")
    print(f"Embedding size    : {model['n_embd']}")
    print(f"Head dimension    : {head_dimension}")
    print(f"Context length    : {model['context_length']}")
    print(f"Vocabulary size   : {model['vocab_size']}")
    print(f"Approx. parameters: {parameter_count / 1_000_000:.2f}M")
    print("-" * 52)


for config_filename in ("debug.yaml", "baseline.yaml"):
    loaded_config = load_config(config_filename)
    validate_config(config_filename, loaded_config)
    print_config_summary(config_filename, loaded_config)

print("All configuration checks passed.")