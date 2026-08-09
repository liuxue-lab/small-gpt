from __future__ import annotations

import json
import unicodedata
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

import tokenizer.bpe as bpe_module
from tokenizer.bpe import (
    ArtifactError,
    CorpusFormatError,
    CorpusStats,
    TokenizerConfigError,
    atomic_write_json,
    combine_split_statistics,
    config_fingerprint,
    discover_split_files,
    encode_document,
    encode_text,
    evaluate_split,
    iter_jsonl_records,
    iter_jsonl_texts,
    load_tokenizer,
    load_tokenizer_config,
    percentile,
    save_tokenizer_artifacts,
    sha256_file,
    special_token_ids,
    train_tokenizer,
    validate_installed_tokenizers_version,
    validate_model_configs,
    validate_tokenizer_config,
    validate_trained_tokenizer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_CONFIG_PATH = PROJECT_ROOT / "configs" / "tokenizer.yaml"


def make_record(
    text: str,
    *,
    provided_token_count: int = 5,
    record_id: str = "record-1",
) -> dict[str, Any]:
    return {
        "id": record_id,
        "text": text,
        "url": "https://example.invalid/document",
        "language": "en",
        "language_score": 0.99,
        "quality_score": 4,
        "provided_token_count": provided_token_count,
        "text_sha256": "0" * 64,
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


@pytest.fixture(scope="session")
def project_config() -> dict[str, Any]:
    return load_tokenizer_config(TOKENIZER_CONFIG_PATH)


@pytest.fixture(scope="session")
def small_config(project_config: dict[str, Any]) -> dict[str, Any]:
    config = deepcopy(project_config)
    config["model"]["vocab_size"] = 300
    config["model"]["min_frequency"] = 1
    return config


@pytest.fixture(scope="session")
def training_texts() -> list[str]:
    return [
        "Hello world!",
        "Hello tokenizer!",
        "The quick brown fox jumps over the lazy dog.",
        "café naïve résumé 🤖",
        "中文 byte coverage",
    ] * 20


@pytest.fixture(scope="session")
def trained_tokenizer(small_config: dict[str, Any], training_texts: list[str]):
    return train_tokenizer(
        small_config,
        iter(training_texts),
        length=len(training_texts),
        show_progress=False,
    )


def test_project_tokenizer_config_contract(project_config: dict[str, Any]) -> None:
    assert project_config["library"] == {
        "name": "tokenizers",
        "version": "0.23.1",
    }
    assert project_config["model"]["vocab_size"] == 16384
    assert project_config["normalizer"]["type"] == "nfc"
    assert [
        (item["name"], item["token"], item["id"])
        for item in project_config["special_tokens"]
    ] == [
        ("bos", "<bos>", 0),
        ("eos", "<eos>", 1),
        ("pad", "<pad>", 2),
        ("unk", "<unk>", 3),
    ]


def test_installed_tokenizers_version(project_config: dict[str, Any]) -> None:
    validate_installed_tokenizers_version(project_config)


def test_model_configs_match_tokenizer(project_config: dict[str, Any]) -> None:
    result = validate_model_configs(project_config, PROJECT_ROOT)
    assert result["max_context_length"] == 512
    assert len(result["configs"]) == 2
    assert {item["vocab_size"] for item in result["configs"]} == {16384}


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda config: config["model"].update(vocab_size=300), "vocab_size"),
        (lambda config: config["normalizer"].update(type="nfkc"), "normalizer.type"),
        (lambda config: config["source"].update(split="validation"), "source.split"),
        (
            lambda config: config["special_tokens"][1].update(id=0),
            "special_tokens",
        ),
    ],
)
def test_invalid_config_is_rejected(
    project_config: dict[str, Any],
    mutation,
    message: str,
) -> None:
    config = deepcopy(project_config)
    mutation(config)
    with pytest.raises(TokenizerConfigError, match=message):
        validate_tokenizer_config(config)


def test_config_fingerprint_is_independent_of_mapping_order(
    project_config: dict[str, Any],
) -> None:
    reordered = dict(reversed(list(project_config.items())))
    assert config_fingerprint(reordered) == config_fingerprint(project_config)


def test_discover_split_files_is_sorted_and_split_specific(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    for shard in ("shard-00002", "shard-00000", "shard-00001"):
        write_jsonl(corpus_dir / "shards" / shard / "train.jsonl", [make_record(shard)])
        write_jsonl(
            corpus_dir / "shards" / shard / "validation.jsonl",
            [make_record(f"validation-{shard}")],
        )
        write_jsonl(
            corpus_dir / "shards" / shard / "test.jsonl",
            [make_record(f"test-{shard}")],
        )

    paths = discover_split_files(corpus_dir, "train", expected_shards=3)
    assert [path.parent.name for path in paths] == [
        "shard-00000",
        "shard-00001",
        "shard-00002",
    ]
    assert all(path.name == "train.jsonl" for path in paths)


def test_discover_split_files_checks_expected_shards(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    write_jsonl(
        corpus_dir / "shards" / "shard-00000" / "train.jsonl",
        [make_record("one shard")],
    )
    with pytest.raises(TokenizerConfigError, match="expected 2"):
        discover_split_files(corpus_dir, "train", expected_shards=2)


def test_train_iterator_cannot_receive_validation_or_test_sentinels(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    shard = corpus_dir / "shards" / "shard-00000"
    write_jsonl(shard / "train.jsonl", [make_record("TRAIN_ONLY_SENTINEL")])
    write_jsonl(
        shard / "validation.jsonl",
        [make_record("VALIDATION_MUST_NOT_BE_READ")],
    )
    write_jsonl(shard / "test.jsonl", [make_record("TEST_MUST_NOT_BE_READ")])

    stats = CorpusStats()
    texts = list(
        iter_jsonl_texts(
            discover_split_files(corpus_dir, "train", expected_shards=1),
            stats,
        )
    )
    assert texts == ["TRAIN_ONLY_SENTINEL"]
    assert all("VALIDATION" not in text and "TEST_" not in text for text in texts)


def test_iter_jsonl_records_reads_expected_schema(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    record = make_record("valid text", provided_token_count=7)
    write_jsonl(path, [record])
    assert list(iter_jsonl_records([path])) == [record]


def test_invalid_json_reports_file_and_line(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text("{not-json}\n", encoding="utf-8")
    with pytest.raises(CorpusFormatError) as exc_info:
        list(iter_jsonl_records([path]))
    assert str(path) in str(exc_info.value)
    assert ":1" in str(exc_info.value)


@pytest.mark.parametrize(
    "record",
    [
        {"provided_token_count": 1},
        {"text": "   ", "provided_token_count": 1},
        {"text": "valid", "provided_token_count": -1},
        {"text": "valid", "provided_token_count": "1"},
    ],
)
def test_invalid_record_fields_are_rejected(
    tmp_path: Path,
    record: dict[str, Any],
) -> None:
    path = tmp_path / "train.jsonl"
    write_jsonl(path, [record])
    with pytest.raises(CorpusFormatError):
        list(iter_jsonl_records([path]))


def test_text_iterator_tracks_corpus_statistics(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    records = [
        make_record("hello", provided_token_count=2, record_id="one"),
        make_record("café", provided_token_count=3, record_id="two"),
    ]
    write_jsonl(path, records)
    stats = CorpusStats()
    assert list(iter_jsonl_texts([path], stats)) == ["hello", "café"]
    assert stats.records == 2
    assert stats.provided_tokens == 5
    assert stats.characters == 9
    assert stats.utf8_bytes == 10


def test_small_training_has_expected_vocab_and_special_ids(
    trained_tokenizer,
    small_config: dict[str, Any],
) -> None:
    summary = validate_trained_tokenizer(trained_tokenizer, small_config)
    assert summary["vocab_size"] == 300
    assert summary["special_token_ids"] == {
        "bos": 0,
        "eos": 1,
        "pad": 2,
        "unk": 3,
    }


def test_encode_is_deterministic_and_ids_are_in_range(trained_tokenizer) -> None:
    text = "The same input must always produce the same IDs."
    first = encode_text(trained_tokenizer, text)
    second = encode_text(trained_tokenizer, text)
    assert first == second
    assert first
    assert all(isinstance(token_id, int) for token_id in first)
    assert min(first) >= 0
    assert max(first) < 300


@pytest.mark.parametrize(
    "text",
    [
        "Hello, world!",
        "Don't split contractions incorrectly.",
        "Line one.\nLine two.",
        "café naïve résumé",
        "Emoji test: 🤖🚀",
        "中文只用于验证字节覆盖。",
    ],
)
def test_byte_level_unicode_round_trip_has_no_unknowns(
    trained_tokenizer,
    text: str,
) -> None:
    ids = encode_text(trained_tokenizer, text)
    assert 3 not in ids
    assert trained_tokenizer.decode(ids, skip_special_tokens=True) == unicodedata.normalize(
        "NFC", text
    )


def test_nfc_combines_canonically_equivalent_input(trained_tokenizer) -> None:
    decomposed = "cafe\u0301"
    ids = encode_text(trained_tokenizer, decomposed)
    assert trained_tokenizer.decode(ids) == "café"


def test_empty_text_and_empty_document_contract(trained_tokenizer) -> None:
    assert encode_text(trained_tokenizer, "") == []
    with pytest.raises(CorpusFormatError, match="empty document"):
        encode_document(trained_tokenizer, "   ", eos_id=1)


def test_document_encoding_appends_exactly_one_eos(trained_tokenizer) -> None:
    raw_ids = encode_text(trained_tokenizer, "Document boundary")
    document_ids = encode_document(trained_tokenizer, "Document boundary", eos_id=1)
    assert document_ids == raw_ids + [1]
    assert document_ids[-1] == 1
    assert 0 not in document_ids
    assert 2 not in document_ids


def test_save_reload_and_artifact_hashes(
    tmp_path: Path,
    trained_tokenizer,
    small_config: dict[str, Any],
) -> None:
    output_dir = tmp_path / "artifacts"
    bundle = save_tokenizer_artifacts(
        trained_tokenizer,
        small_config,
        output_dir,
        project_root=tmp_path,
        training_metadata={"records": 100},
        validation_samples=["Hello world!", "café 🤖"],
    )

    assert sorted(path.name for path in output_dir.iterdir()) == [
        "merges.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ]
    reloaded = load_tokenizer(output_dir / "tokenizer.json")
    sample = "Reloaded tokenizer must be identical."
    assert encode_text(reloaded, sample) == encode_text(trained_tokenizer, sample)

    metadata = json.loads((output_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    assert metadata["artifacts"]["tokenizer"]["sha256"] == sha256_file(
        output_dir / "tokenizer.json"
    )
    assert metadata["artifacts"]["vocab"]["sha256"] == sha256_file(
        output_dir / "vocab.json"
    )
    assert metadata["artifacts"]["merges"]["sha256"] == sha256_file(
        output_dir / "merges.txt"
    )
    assert bundle.config.sha256 == sha256_file(output_dir / "tokenizer_config.json")


def test_existing_artifact_directory_is_not_overwritten(
    tmp_path: Path,
    trained_tokenizer,
    small_config: dict[str, Any],
) -> None:
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    sentinel = output_dir / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(ArtifactError, match="refusing to overwrite"):
        save_tokenizer_artifacts(
            trained_tokenizer,
            small_config,
            output_dir,
            project_root=tmp_path,
            training_metadata={},
            validation_samples=["Hello"],
        )
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_failed_reload_validation_does_not_publish_artifacts(
    tmp_path: Path,
    trained_tokenizer,
    small_config: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "artifacts"

    def fail_to_load(_path: str | Path):
        raise ArtifactError("simulated reload failure")

    monkeypatch.setattr(bpe_module, "load_tokenizer", fail_to_load)
    with pytest.raises(ArtifactError, match="simulated reload failure"):
        save_tokenizer_artifacts(
            trained_tokenizer,
            small_config,
            output_dir,
            project_root=tmp_path,
            training_metadata={},
            validation_samples=["Hello"],
        )
    assert not output_dir.exists()
    assert not list(tmp_path.glob(".artifacts.tmp-*"))


def test_evaluate_split_counts_eos_and_tokens(
    tmp_path: Path,
    trained_tokenizer,
    small_config: dict[str, Any],
) -> None:
    path = tmp_path / "train.jsonl"
    write_jsonl(
        path,
        [
            make_record("Hello world!", provided_token_count=4, record_id="one"),
            make_record("café 🤖", provided_token_count=3, record_id="two"),
        ],
    )
    stats = evaluate_split(
        trained_tokenizer,
        [path],
        split="train",
        config=small_config,
        context_length=8,
    )
    assert stats["records"] == 2
    assert stats["provided_tokens"] == 7
    assert stats["eos_tokens"] == 2
    assert stats["total_model_tokens"] == stats["bpe_tokens_without_eos"] + 2
    assert stats["unknown_tokens"] == 0
    assert stats["minimum_token_id"] >= 0
    assert stats["maximum_token_id"] < 300


def test_combine_split_statistics_preserves_totals(
    tmp_path: Path,
    trained_tokenizer,
    small_config: dict[str, Any],
) -> None:
    split_stats: dict[str, dict[str, Any]] = {}
    for index, split in enumerate(("train", "validation", "test"), start=1):
        path = tmp_path / f"{split}.jsonl"
        write_jsonl(
            path,
            [make_record(f"{split} text", provided_token_count=index)],
        )
        split_stats[split] = evaluate_split(
            trained_tokenizer,
            [path],
            split=split,
            config=small_config,
            context_length=512,
        )

    totals = combine_split_statistics(split_stats)
    assert totals["records"] == 3
    assert totals["provided_tokens"] == 6
    assert totals["eos_tokens"] == 3
    assert totals["total_model_tokens"] == (
        totals["bpe_tokens_without_eos"] + totals["eos_tokens"]
    )


def test_percentile_uses_linear_interpolation() -> None:
    values = [1, 2, 3, 4, 5]
    assert percentile(values, 0.0) == 1.0
    assert percentile(values, 0.5) == 3.0
    assert percentile(values, 0.95) == pytest.approx(4.8)
    assert percentile([], 0.5) is None


def test_atomic_json_write_replaces_complete_file(tmp_path: Path) -> None:
    path = tmp_path / "stats.json"
    atomic_write_json(path, {"version": 1})
    assert json.loads(path.read_text(encoding="utf-8")) == {"version": 1}
    atomic_write_json(path, {"version": 2, "complete": True})
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "complete": True,
        "version": 2,
    }
    assert not list(tmp_path.glob(".stats.json.tmp-*"))


def test_tokenized_data_is_git_ignored() -> None:
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "data/tokenized/" in {line.strip() for line in gitignore}
