from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("data/raw/fineweb_edu_sample.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/processed")
DEFAULT_REPORT = Path("reports/day-02-cleaning-stats.json")

HORIZONTAL_WHITESPACE_RE = re.compile(r"[^\S\n]+")
SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class CleaningConfig:
    min_characters: int = 200
    min_language_score: float = 0.65
    min_quality_score: int = 3
    seed: int = 42
    train_ratio: float = 0.98
    validation_ratio: float = 0.01

    def validate(self) -> None:
        if self.min_characters < 1:
            raise ValueError("min_characters must be at least 1")
        if not 0.0 <= self.min_language_score <= 1.0:
            raise ValueError("min_language_score must be between 0 and 1")
        if self.min_quality_score < 0:
            raise ValueError("min_quality_score must not be negative")
        if self.train_ratio <= 0.0 or self.validation_ratio < 0.0:
            raise ValueError("split ratios must not be negative")
        if self.train_ratio + self.validation_ratio >= 1.0:
            raise ValueError("train_ratio + validation_ratio must be less than 1")

    @property
    def test_ratio(self) -> float:
        return round(1.0 - self.train_ratio - self.validation_ratio, 12)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean, deduplicate, and split a FineWeb-Edu JSONL sample."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--min-characters", type=int, default=200)
    parser.add_argument("--min-language-score", type=float, default=0.65)
    parser.add_argument("--min-quality-score", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.98)
    parser.add_argument("--validation-ratio", type=float, default=0.01)
    return parser.parse_args()


def get_field(record: dict[str, Any], field: str) -> Any:
    if field in record:
        return record[field]
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        return metadata.get(field)
    return None


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = HORIZONTAL_WHITESPACE_RE.sub(" ", text)

    normalized_lines: list[str] = []
    blank_line_pending = False
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if line:
            if blank_line_pending and normalized_lines:
                normalized_lines.append("")
            normalized_lines.append(line)
            blank_line_pending = False
        else:
            blank_line_pending = True

    return "\n".join(normalized_lines).strip()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric_value(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def clean_record(
    record: dict[str, Any], config: CleaningConfig
) -> tuple[dict[str, Any] | None, str | None]:
    text = get_field(record, "text")
    if not isinstance(text, str) or not text.strip():
        return None, "missing_or_empty_text"

    normalized_text = normalize_text(text)
    if len(normalized_text) < config.min_characters:
        return None, "short_text"

    language = get_field(record, "language")
    if language != "en":
        return None, "non_english"

    language_score = numeric_value(get_field(record, "language_score"))
    if language_score is None or language_score < config.min_language_score:
        return None, "low_or_missing_language_score"

    quality_score = numeric_value(get_field(record, "int_score"))
    if quality_score is None or quality_score < config.min_quality_score:
        return None, "low_or_missing_quality_score"

    supplied_token_count = numeric_value(get_field(record, "token_count"))
    digest = text_sha256(normalized_text)

    cleaned = {
        "id": get_field(record, "id"),
        "text": normalized_text,
        "url": get_field(record, "url"),
        "language": language,
        "language_score": language_score,
        "quality_score": int(quality_score),
        "provided_token_count": (
            int(supplied_token_count) if supplied_token_count is not None else None
        ),
        "text_sha256": digest,
    }
    return cleaned, None


def assign_split(text_digest: str, config: CleaningConfig) -> str:
    split_digest = hashlib.sha256(
        f"{config.seed}:{text_digest}".encode("utf-8")
    ).digest()
    value = int.from_bytes(split_digest[:8], byteorder="big") / 2**64

    if value < config.train_ratio:
        return "train"
    if value < config.train_ratio + config.validation_ratio:
        return "validation"
    return "test"


def prepare_dataset(
    input_path: Path,
    output_dir: Path,
    report_path: Path,
    config: CleaningConfig,
) -> dict[str, Any]:
    config.validate()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input JSONL file not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    final_paths = {name: output_dir / f"{name}.jsonl" for name in SPLIT_NAMES}
    temporary_paths = {
        name: output_dir / f".{name}.jsonl.tmp" for name in SPLIT_NAMES
    }

    statistics_counter: Counter[str] = Counter()
    seen_text_hashes: set[str] = set()
    kept_characters = 0
    kept_provided_tokens = 0

    try:
        with ExitStack() as stack:
            writers = {
                name: stack.enter_context(path.open("w", encoding="utf-8"))
                for name, path in temporary_paths.items()
            }

            with input_path.open("r", encoding="utf-8") as input_file:
                for line_number, line in enumerate(input_file, start=1):
                    statistics_counter["input_records"] += 1
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"Invalid JSON on line {line_number}: {error.msg}"
                        ) from error

                    if not isinstance(record, dict):
                        raise ValueError(
                            f"Expected a JSON object on line {line_number}"
                        )

                    cleaned, removal_reason = clean_record(record, config)
                    if cleaned is None:
                        statistics_counter[f"removed_{removal_reason}"] += 1
                        continue

                    digest = cleaned["text_sha256"]
                    if digest in seen_text_hashes:
                        statistics_counter["removed_exact_duplicate"] += 1
                        continue
                    seen_text_hashes.add(digest)

                    split_name = assign_split(digest, config)
                    writers[split_name].write(
                        json.dumps(cleaned, ensure_ascii=False) + "\n"
                    )

                    statistics_counter["kept_records"] += 1
                    statistics_counter[f"split_{split_name}"] += 1
                    kept_characters += len(cleaned["text"])
                    if cleaned["provided_token_count"] is not None:
                        kept_provided_tokens += cleaned["provided_token_count"]

        for name in SPLIT_NAMES:
            temporary_paths[name].replace(final_paths[name])
    except Exception:
        for path in temporary_paths.values():
            path.unlink(missing_ok=True)
        raise

    input_records = statistics_counter["input_records"]
    kept_records = statistics_counter["kept_records"]
    ordered_statistics = {
        "input_records": input_records,
        "kept_records": kept_records,
        "removed_missing_or_empty_text": statistics_counter[
            "removed_missing_or_empty_text"
        ],
        "removed_short_text": statistics_counter["removed_short_text"],
        "removed_non_english": statistics_counter["removed_non_english"],
        "removed_low_or_missing_language_score": statistics_counter[
            "removed_low_or_missing_language_score"
        ],
        "removed_low_or_missing_quality_score": statistics_counter[
            "removed_low_or_missing_quality_score"
        ],
        "removed_exact_duplicate": statistics_counter["removed_exact_duplicate"],
        "split_train": statistics_counter["split_train"],
        "split_validation": statistics_counter["split_validation"],
        "split_test": statistics_counter["split_test"],
        "kept_characters": kept_characters,
        "kept_provided_tokens": kept_provided_tokens,
    }

    report = {
        "input": {
            "path": str(input_path),
            "sha256": file_sha256(input_path),
        },
        "output_directory": str(output_dir),
        "configuration": {
            **asdict(config),
            "test_ratio": config.test_ratio,
        },
        "statistics": ordered_statistics,
        "retention_rate": round(kept_records / input_records, 6)
        if input_records
        else 0.0,
    }

    temporary_report = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_report.replace(report_path)
    return report


def main() -> None:
    args = parse_args()
    config = CleaningConfig(
        min_characters=args.min_characters,
        min_language_score=args.min_language_score,
        min_quality_score=args.min_quality_score,
        seed=args.seed,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
    )
    report = prepare_dataset(
        input_path=args.input,
        output_dir=args.output_dir,
        report_path=args.report_output,
        config=config,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Processed data saved to: {args.output_dir}")
    print(f"Cleaning report saved to: {args.report_output}")


if __name__ == "__main__":
    main()