import math
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_CONFIG_PATH = PROJECT_ROOT / "configs" / "data_fineweb_edu.yaml"
EXPECTED_FULL_SOURCE_FILES = [
    f"sample/10BT/{index:03d}_00000.parquet" for index in range(14)
]


def load_data_config() -> dict:
    with DATA_CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    assert isinstance(config, dict)
    return config


def test_data_source_is_fixed_and_streamed():
    config = load_data_config()
    dataset = config["dataset"]

    assert dataset["name"] == "HuggingFaceFW/fineweb-edu"
    assert dataset["configuration"] == "sample-10BT"
    assert dataset["split"] == "train"
    assert (
        dataset["revision"]
        == "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
    )
    assert dataset["streaming"] is True


def test_data_split_ratios_sum_to_one():
    splits = load_data_config()["splits"]

    total = (
        splits["train_ratio"]
        + splits["validation_ratio"]
        + splits["test_ratio"]
    )

    assert math.isclose(total, 1.0, abs_tol=1e-9)
    assert splits["seed"] == 42


def test_pilot_and_full_collection_budgets():
    profiles = load_data_config()["profiles"]

    pilot = profiles["pilot"]
    assert pilot["target_provided_tokens"] == 2_000_000
    assert pilot["shard_target_provided_tokens"] == 500_000
    assert pilot["estimated_shards"] == 4

    full = profiles["full"]
    assert full["target_provided_tokens"] == 350_000_000
    assert full["shard_target_provided_tokens"] == 5_000_000
    assert full["estimated_shards"] == 70
    assert full["output_dir"] == "data/processed/fineweb_edu_full"
    assert full["source_files"] == EXPECTED_FULL_SOURCE_FILES


def test_output_and_storage_budget():
    config = load_data_config()
    output = config["output"]
    storage = config["storage"]

    assert output["directory"] == "data/processed/fineweb_edu_corpus"
    assert output["format"] == "jsonl"
    assert output["encoding"] == "utf-8"
    assert output["save_raw_records"] is False
    assert output["manifest_filename"] == "manifest.json"
    assert output["state_filename"] == "state.json"

    assert math.isclose(
        storage["estimated_bytes_per_provided_token"],
        4.99,
        abs_tol=0.01,
    )
    assert storage["reserved_disk_gb"] == 5
