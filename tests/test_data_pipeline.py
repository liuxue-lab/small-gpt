import json
from collections import Counter

import pytest

from scripts.prepare_data import (
    CleaningConfig,
    SPLIT_NAMES,
    assign_split,
    clean_record,
    normalize_text,
    prepare_dataset,
    text_sha256,
)


def make_valid_record(**overrides):
    record = {
        "id": "document-1",
        "text": (
            "This is a clear English educational document about mathematics, "
            "science, and careful reasoning. "
        )
        * 5,
        "url": "https://example.com/document-1",
        "language": "en",
        "language_score": 0.95,
        "token_count": 120,
        "score": 3.4,
        "int_score": 3,
    }
    record.update(overrides)
    return record


def test_normalize_text_preserves_paragraphs_and_normalizes_unicode():
    raw_text = "  Cafe\u0301\tlesson\r\n\r\n\r\nSecond   paragraph  "

    normalized = normalize_text(raw_text)

    assert normalized == "Caf\u00e9 lesson\n\nSecond paragraph"


def test_clean_record_accepts_a_valid_document():
    cleaned, reason = clean_record(make_valid_record(), CleaningConfig())

    assert reason is None
    assert cleaned is not None
    assert cleaned["language"] == "en"
    assert cleaned["quality_score"] == 3
    assert cleaned["provided_token_count"] == 120
    assert cleaned["text_sha256"] == text_sha256(cleaned["text"])


def test_clean_record_accepts_nested_metadata():
    flat = make_valid_record()
    nested = {
        "text": flat.pop("text"),
        "id": flat.pop("id"),
        "metadata": flat,
    }

    cleaned, reason = clean_record(nested, CleaningConfig())

    assert reason is None
    assert cleaned is not None
    assert cleaned["language_score"] == 0.95


@pytest.mark.parametrize(
    ("overrides", "expected_reason"),
    [
        ({"text": "   "}, "missing_or_empty_text"),
        ({"text": "Too short."}, "short_text"),
        ({"language": "fr"}, "non_english"),
        ({"language_score": 0.64}, "low_or_missing_language_score"),
        ({"int_score": 2}, "low_or_missing_quality_score"),
    ],
)
def test_clean_record_rejects_invalid_documents(overrides, expected_reason):
    cleaned, reason = clean_record(
        make_valid_record(**overrides), CleaningConfig()
    )

    assert cleaned is None
    assert reason == expected_reason


def test_assign_split_is_deterministic_and_close_to_requested_ratios():
    config = CleaningConfig()
    first_digest = text_sha256("deterministic document")

    assert assign_split(first_digest, config) == assign_split(first_digest, config)

    counts = Counter(
        assign_split(text_sha256(f"document-{index}"), config)
        for index in range(10_000)
    )

    assert set(counts) == set(SPLIT_NAMES)
    assert abs(counts["train"] / 10_000 - 0.98) < 0.01
    assert abs(counts["validation"] / 10_000 - 0.01) < 0.005
    assert abs(counts["test"] / 10_000 - 0.01) < 0.005


def test_prepare_dataset_filters_deduplicates_and_writes_splits(tmp_path):
    valid_a = make_valid_record(id="valid-a")
    duplicate_a = make_valid_record(
        id="duplicate-a",
        url="https://example.com/duplicate-a",
    )
    valid_b = make_valid_record(
        id="valid-b",
        text=("A different educational discussion of language models. " * 8),
        url="https://example.com/valid-b",
    )

    records = [
        valid_a,
        duplicate_a,
        {"id": "empty", "text": "   "},
        make_valid_record(id="short", text="Too short."),
        make_valid_record(id="french", language="fr"),
        make_valid_record(id="low-language", language_score=0.64),
        make_valid_record(id="low-quality", int_score=2),
        valid_b,
    ]

    input_path = tmp_path / "sample.jsonl"
    input_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    output_dir = tmp_path / "processed"
    report_path = tmp_path / "report.json"

    report = prepare_dataset(
        input_path,
        output_dir,
        report_path,
        CleaningConfig(),
    )
    statistics = report["statistics"]

    assert statistics["input_records"] == 8
    assert statistics["kept_records"] == 2
    assert statistics["removed_missing_or_empty_text"] == 1
    assert statistics["removed_short_text"] == 1
    assert statistics["removed_non_english"] == 1
    assert statistics["removed_low_or_missing_language_score"] == 1
    assert statistics["removed_low_or_missing_quality_score"] == 1
    assert statistics["removed_exact_duplicate"] == 1

    output_records = []
    for split_name in SPLIT_NAMES:
        split_path = output_dir / f"{split_name}.jsonl"
        assert split_path.is_file()
        output_records.extend(
            json.loads(line)
            for line in split_path.read_text(encoding="utf-8").splitlines()
        )

    assert len(output_records) == 2
    assert len({record["text_sha256"] for record in output_records}) == 2
    assert report_path.is_file()
    assert not list(output_dir.glob("*.tmp"))


def test_prepare_dataset_fails_cleanly_on_invalid_json(tmp_path):
    input_path = tmp_path / "broken.jsonl"
    input_path.write_text(
        json.dumps(make_valid_record()) + "\n{broken json\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "processed"

    with pytest.raises(ValueError, match="line 2"):
        prepare_dataset(
            input_path,
            output_dir,
            tmp_path / "report.json",
            CleaningConfig(),
        )

    assert not (output_dir / "train.jsonl").exists()
    assert not list(output_dir.glob("*.tmp"))


@pytest.mark.parametrize(
    "config",
    [
        CleaningConfig(min_characters=0),
        CleaningConfig(min_language_score=1.1),
        CleaningConfig(train_ratio=0.99, validation_ratio=0.01),
    ],
)
def test_invalid_configuration_is_rejected(config):
    with pytest.raises(ValueError):
        config.validate()