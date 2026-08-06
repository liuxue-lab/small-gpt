import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset


DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"
DATASET_SPLIT = "train"

# 固定数据集版本，避免未来数据集更新导致抽样结果变化。
DATASET_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"

EXPECTED_FIELDS = (
    "text",
    "id",
    "dump",
    "url",
    "date",
    "file_path",
    "language",
    "language_score",
    "token_count",
    "score",
    "int_score",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a small streamed sample from FineWeb-Edu."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Maximum number of documents to inspect.",
    )
    parser.add_argument(
        "--sample-output",
        type=Path,
        default=Path("data/raw/fineweb_edu_sample.jsonl"),
        help="Path used to save the streamed raw sample.",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("reports/day-02-inspection.json"),
        help="Path used to save the statistical report.",
    )

    args = parser.parse_args()

    if args.limit <= 0:
        parser.error("--limit must be greater than zero.")

    return args


def get_field(record: dict[str, Any], field: str) -> Any:
    """Read both flat FineWeb fields and nested metadata fields."""
    if field in record:
        return record[field]

    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        return metadata.get(field)

    return None


def numeric_summary(values: list[float | int]) -> dict[str, float | int] | None:
    if not values:
        return None

    return {
        "minimum": min(values),
        "maximum": max(values),
        "mean": round(statistics.fmean(values), 2),
        "median": statistics.median(values),
    }


def main() -> None:
    args = parse_args()

    args.sample_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Dataset : {DATASET_NAME}")
    print(f"Config  : {DATASET_CONFIG}")
    print(f"Revision: {DATASET_REVISION}")
    print(f"Limit   : {args.limit}")
    print("Opening streaming dataset...")

    dataset = load_dataset(
        DATASET_NAME,
        name=DATASET_CONFIG,
        split=DATASET_SPLIT,
        revision=DATASET_REVISION,
        streaming=True,
    )

    missing_fields: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    quality_score_counts: Counter[str] = Counter()

    character_lengths: list[int] = []
    provided_token_counts: list[int] = []
    language_scores: list[float] = []

    seen_text_hashes: set[str] = set()
    seen_ids: set[str] = set()

    empty_texts = 0
    duplicate_texts = 0
    duplicate_ids = 0
    records_seen = 0

    with args.sample_output.open("w", encoding="utf-8") as sample_file:
        for record in dataset:
            if records_seen >= args.limit:
                break

            records_seen += 1

            sample_file.write(
                json.dumps(record, ensure_ascii=False, default=str) + "\n"
            )

            for field in EXPECTED_FIELDS:
                if get_field(record, field) is None:
                    missing_fields[field] += 1

            text = get_field(record, "text")
            if not isinstance(text, str) or not text.strip():
                empty_texts += 1
            else:
                character_lengths.append(len(text))

                text_hash = hashlib.sha256(
                    text.encode("utf-8", errors="replace")
                ).hexdigest()

                if text_hash in seen_text_hashes:
                    duplicate_texts += 1
                else:
                    seen_text_hashes.add(text_hash)

            document_id = get_field(record, "id")
            if isinstance(document_id, str) and document_id:
                if document_id in seen_ids:
                    duplicate_ids += 1
                else:
                    seen_ids.add(document_id)

            language = get_field(record, "language")
            language_counts[str(language)] += 1

            language_score = get_field(record, "language_score")
            if isinstance(language_score, (int, float)):
                language_scores.append(float(language_score))

            token_count = get_field(record, "token_count")
            if isinstance(token_count, int):
                provided_token_counts.append(token_count)

            quality_score = get_field(record, "int_score")
            quality_score_counts[str(quality_score)] += 1

    report = {
        "dataset": {
            "name": DATASET_NAME,
            "configuration": DATASET_CONFIG,
            "split": DATASET_SPLIT,
            "revision": DATASET_REVISION,
        },
        "sample": {
            "requested_records": args.limit,
            "records_seen": records_seen,
            "empty_texts": empty_texts,
            "exact_duplicate_texts": duplicate_texts,
            "duplicate_ids": duplicate_ids,
        },
        "missing_fields": dict(sorted(missing_fields.items())),
        "language_counts": dict(language_counts.most_common()),
        "educational_score_counts": dict(quality_score_counts.most_common()),
        "character_lengths": numeric_summary(character_lengths),
        "provided_token_counts": numeric_summary(provided_token_counts),
        "language_scores": numeric_summary(language_scores),
    }

    args.report_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Raw sample saved to : {args.sample_output}")
    print(f"Report saved to     : {args.report_output}")


if __name__ == "__main__":
    main()