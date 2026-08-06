import math
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIRECTORY = PROJECT_ROOT / "configs"


def load_config(filename: str) -> dict:
    config_path = CONFIG_DIRECTORY / filename

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def estimate_parameters(config: dict) -> int:
    model = config["model"]

    token_embedding = model["vocab_size"] * model["n_embd"]
    position_embedding = model["context_length"] * model["n_embd"]
    transformer_blocks = (
        model["n_layer"]
        * 12
        * model["n_embd"]
        * model["n_embd"]
    )

    output_head = 0

    if not model["tie_embeddings"]:
        output_head = model["vocab_size"] * model["n_embd"]

    return (
        token_embedding
        + position_embedding
        + transformer_blocks
        + output_head
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


def test_baseline_model_dimensions_and_size():
    config = load_config("baseline.yaml")
    model = config["model"]

    assert model["n_layer"] == 8
    assert model["n_head"] == 8
    assert model["n_embd"] == 512
    assert model["ffn_hidden"] == 2048
    assert model["context_length"] == 512
    assert model["n_embd"] % model["n_head"] == 0

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

        assert tokenizer_vocab_size == model_vocab_size