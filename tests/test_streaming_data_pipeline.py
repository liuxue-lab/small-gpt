import hashlib
import json
from pathlib import Path

import pytest
import yaml

from scripts.build_fineweb_edu_corpus import (
    ControlledInterruption,
    CorpusIntegrityError,
    build_corpus,
    config_fingerprint,
    load_run_config,
)
from scripts.prepare_data import SPLIT_NAMES, file_sha256


REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"


def write_config(path: Path, output_dir: Path) -> Path:
    payload = {
        "dataset": {
            "name": "HuggingFaceFW/fineweb-edu",
            "configuration": "sample-10BT",
            "split": "train",
            "revision": REVISION,
            "streaming": True,
        },
        "cleaning": {
            "min_characters": 200,
            "min_language_score": 0.65,
            "min_quality_score": 3,
        },
        "splits": {
            "seed": 42,
            "train_ratio": 0.98,
            "validation_ratio": 0.01,
            "test_ratio": 0.01,
        },
        "output": {
            "directory": str(output_dir),
            "format": "jsonl",
            "encoding": "utf-8",
            "save_raw_records": False,
            "manifest_filename": "manifest.json",
            "state_filename": "state.json",
        },
        "profiles": {
            "pilot": {
                "target_provided_tokens": 2_000_000,
                "shard_target_provided_tokens": 500_000,
                "estimated_shards": 4,
            },
            "full": {
                "target_provided_tokens": 350_000_000,
                "shard_target_provided_tokens": 5_000_000,
                "estimated_shards": 70,
            },
        },
        "storage": {
            "estimated_bytes_per_provided_token": 4.99,
            "reserved_disk_gb": 5,
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def make_record(
    index: int,
    *,
    token_count: int | None = 100,
    language: str = "en",
    text: str | None = None,
) -> dict:
    document_text = text or (
        f"Educational document {index} explains mathematics, science, "
        "language, and careful reasoning with a reproducible example. " * 4
    )
    return {
        "id": f"document-{index}",
        "text": document_text,
        "url": f"https://example.com/{index}",
        "language": language,
        "language_score": 0.95,
        "token_count": token_count,
        "score": 3.4,
        "int_score": 3,
    }


def load_test_config(
    tmp_path: Path,
    *,
    target: int,
    shard_target: int,
):
    config_path = write_config(tmp_path / "data.yaml", tmp_path / "corpus")
    return load_run_config(
        config_path,
        "pilot",
        target_provided_tokens_override=target,
        shard_target_provided_tokens_override=shard_target,
    )


def corpus_snapshot(output_dir: Path) -> dict[str, str]:
    snapshot = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            relative = path.relative_to(output_dir).as_posix()
            snapshot[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return snapshot


def read_all_records(output_dir: Path) -> list[dict]:
    records = []
    for path in sorted((output_dir / "shards").glob("shard-*/*.jsonl")):
        records.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        )
    return records


def test_load_run_config_applies_bounded_overrides(tmp_path):
    config = load_test_config(tmp_path, target=600, shard_target=200)

    assert config.dataset_revision == REVISION
    assert config.target_provided_tokens == 600
    assert config.shard_target_provided_tokens == 200
    assert config.estimated_shards == 3
    assert config.save_raw_records is False
    assert len(config_fingerprint(config)) == 64


def test_build_corpus_filters_deduplicates_and_writes_verified_shards(tmp_path):
    config = load_test_config(tmp_path, target=500, shard_target=200)
    duplicate_text = make_record(0)["text"]
    records = [
        make_record(0),
        make_record(1),
        make_record(10, text=duplicate_text),
        make_record(2, language="fr"),
        make_record(3),
        make_record(4),
        make_record(5),
    ]

    manifest = build_corpus(records, config)

    assert manifest["status"] == "complete"
    assert manifest["statistics"]["input_records"] == 7
    assert manifest["statistics"]["kept_records"] == 5
    assert manifest["statistics"]["kept_provided_tokens"] == 500
    assert manifest["statistics"]["removal_counts"] == {
        "exact_duplicate": 1,
        "non_english": 1,
    }
    assert len(manifest["shards"]) == 3

    cleaned_records = read_all_records(config.output_dir)
    assert len(cleaned_records) == 5
    assert len({record["text_sha256"] for record in cleaned_records}) == 5

    for shard in manifest["shards"]:
        for split_name in SPLIT_NAMES:
            file_info = shard["files"][split_name]
            path = config.output_dir / file_info["path"]
            assert path.is_file()
            assert path.stat().st_size == file_info["bytes"]
            assert file_sha256(path) == file_info["sha256"]


def test_interrupted_run_resumes_and_completed_rerun_is_idempotent(tmp_path):
    config = load_test_config(tmp_path, target=700, shard_target=300)
    records = [make_record(index) for index in range(10)]

    with pytest.raises(ControlledInterruption):
        build_corpus(
            records,
            config,
            interrupt_after_source_records=5,
        )

    completed_before_resume = sorted(
        (config.output_dir / "shards").glob("shard-*"),
    )
    temporary_before_resume = sorted(
        (config.output_dir / "shards").glob(".shard-*.tmp"),
    )
    assert len(completed_before_resume) == 1
    assert len(temporary_before_resume) == 1

    manifest = build_corpus(records, config)
    assert manifest["status"] == "complete"
    assert manifest["statistics"]["input_records"] == 7
    assert manifest["statistics"]["kept_records"] == 7
    assert manifest["statistics"]["kept_provided_tokens"] == 700
    assert len(manifest["shards"]) == 3
    assert not list((config.output_dir / "shards").glob(".shard-*.tmp"))

    first_snapshot = corpus_snapshot(config.output_dir)
    second_manifest = build_corpus(records, config)
    second_snapshot = corpus_snapshot(config.output_dir)

    assert second_manifest == manifest
    assert second_snapshot == first_snapshot


def test_completed_file_corruption_is_detected_before_resume(tmp_path):
    config = load_test_config(tmp_path, target=200, shard_target=200)
    manifest = build_corpus([make_record(0), make_record(1)], config)

    populated_file = next(
        config.output_dir / info["path"]
        for info in manifest["shards"][0]["files"].values()
        if info["records"] > 0
    )
    with populated_file.open("a", encoding="utf-8") as file:
        file.write("{}\n")

    with pytest.raises(CorpusIntegrityError, match="file size mismatch"):
        build_corpus([make_record(0), make_record(1)], config)
