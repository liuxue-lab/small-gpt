"""Evaluate the saved tokenizer and count real Pilot tokens by split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tokenizer.bpe import (  # noqa: E402
    ALLOWED_SPLITS,
    ArtifactError,
    TokenizerPipelineError,
    atomic_write_json,
    combine_split_statistics,
    config_fingerprint,
    discover_split_files,
    evaluate_split,
    load_source_manifest,
    load_tokenizer,
    load_tokenizer_config,
    project_relative_path,
    resolve_project_path,
    sha256_file,
    special_token_ids,
    validate_installed_tokenizers_version,
    validate_model_configs,
    validate_trained_tokenizer,
)


FULL_PROVIDED_TOKEN_TARGET = 350_000_000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load the saved small-gpt tokenizer and stream-count real BPE "
            "tokens for train, validation, and test."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/tokenizer.yaml",
        help="Tokenizer YAML configuration relative to the project root.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional statistics JSON path override.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace statistics created by a different tokenizer artifact.",
    )
    return parser.parse_args(argv)


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ArtifactError(f"{label} does not exist: {path}")
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ArtifactError(f"{label} must contain a JSON object: {path}")
    return loaded


def _verify_artifact_hashes(
    metadata: Mapping[str, Any],
    artifact_dir: Path,
) -> dict[str, str]:
    artifacts = metadata.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ArtifactError("tokenizer_config.json has no artifacts mapping")

    hashes: dict[str, str] = {}
    for name in ("tokenizer", "vocab", "merges"):
        item = artifacts.get(name)
        if not isinstance(item, Mapping):
            raise ArtifactError(f"tokenizer_config.json has no {name} artifact")
        filename = item.get("filename")
        expected_sha256 = item.get("sha256")
        if not isinstance(filename, str) or not isinstance(expected_sha256, str):
            raise ArtifactError(f"invalid {name} artifact metadata")
        path = artifact_dir / filename
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ArtifactError(
                f"{name} artifact SHA-256 mismatch: expected {expected_sha256}, "
                f"found {actual_sha256}"
            )
        hashes[name] = actual_sha256
    return hashes


def _validate_against_manifest_statistics(
    manifest: Mapping[str, Any],
    split_statistics: Mapping[str, Mapping[str, Any]],
) -> None:
    statistics = manifest.get("statistics")
    if not isinstance(statistics, Mapping):
        return

    expected_records = statistics.get("split_records")
    if isinstance(expected_records, Mapping):
        for split, actual in split_statistics.items():
            expected = expected_records.get(split)
            if expected is not None and int(expected) != int(actual["records"]):
                raise ArtifactError(
                    f"{split} records mismatch: manifest={expected}, "
                    f"evaluation={actual['records']}"
                )

    expected_provided_tokens = statistics.get("split_provided_tokens")
    if isinstance(expected_provided_tokens, Mapping):
        for split, actual in split_statistics.items():
            expected = expected_provided_tokens.get(split)
            if expected is not None and int(expected) != int(actual["provided_tokens"]):
                raise ArtifactError(
                    f"{split} provided tokens mismatch: manifest={expected}, "
                    f"evaluation={actual['provided_tokens']}"
                )


def _allow_statistics_write(
    output_path: Path,
    tokenizer_sha256: str,
    overwrite: bool,
) -> None:
    if not output_path.exists() or overwrite:
        return
    existing = _load_json_object(output_path, "existing statistics")
    existing_tokenizer = existing.get("tokenizer")
    existing_sha256 = (
        existing_tokenizer.get("sha256")
        if isinstance(existing_tokenizer, Mapping)
        else None
    )
    if existing_sha256 != tokenizer_sha256:
        raise ArtifactError(
            f"statistics already exist for a different tokenizer: {output_path}; "
            "use --overwrite only after reviewing the existing file"
        )


def run(args: argparse.Namespace) -> int:
    config_path = resolve_project_path(PROJECT_ROOT, args.config)
    config = load_tokenizer_config(config_path)
    validate_installed_tokenizers_version(config)
    model_validation = validate_model_configs(config, PROJECT_ROOT)
    context_length = int(model_validation["max_context_length"])

    source = config["source"]
    corpus_dir = resolve_project_path(PROJECT_ROOT, source["corpus_dir"])
    manifest_path = resolve_project_path(PROJECT_ROOT, source["manifest"])
    manifest = load_source_manifest(manifest_path, int(source["expected_shards"]))

    artifact_dir = resolve_project_path(PROJECT_ROOT, config["artifacts"]["output_dir"])
    tokenizer_path = artifact_dir / config["artifacts"]["tokenizer_file"]
    metadata_path = artifact_dir / config["artifacts"]["config_file"]
    metadata = _load_json_object(metadata_path, "tokenizer metadata")
    if metadata.get("config_fingerprint") != config_fingerprint(config):
        raise ArtifactError(
            "tokenizer artifact config fingerprint does not match configs/tokenizer.yaml"
        )

    artifact_hashes = _verify_artifact_hashes(metadata, artifact_dir)
    tokenizer_sha256 = artifact_hashes["tokenizer"]
    tokenizer = load_tokenizer(tokenizer_path)
    validation_summary = validate_trained_tokenizer(tokenizer, config)

    split_statistics: dict[str, dict[str, Any]] = {}
    for split in ALLOWED_SPLITS:
        paths = discover_split_files(
            corpus_dir,
            split,
            expected_shards=int(source["expected_shards"]),
        )
        stats = evaluate_split(
            tokenizer,
            paths,
            split=split,
            config=config,
            context_length=context_length,
        )
        split_statistics[split] = stats
        print(
            f"{split}: records={stats['records']}, "
            f"provided_tokens={stats['provided_tokens']}, "
            f"model_tokens={stats['total_model_tokens']}, "
            f"unknown_tokens={stats['unknown_tokens']}"
        )

    train_stats = split_statistics["train"]
    if int(train_stats["records"]) != int(source["expected_records"]):
        raise ArtifactError(
            f"train records are {train_stats['records']}; "
            f"expected {source['expected_records']}"
        )
    if int(train_stats["provided_tokens"]) != int(source["expected_provided_tokens"]):
        raise ArtifactError(
            f"train provided tokens are {train_stats['provided_tokens']}; "
            f"expected {source['expected_provided_tokens']}"
        )

    _validate_against_manifest_statistics(manifest, split_statistics)
    totals = combine_split_statistics(split_statistics)
    if int(totals["eos_tokens"]) != int(totals["records"]):
        raise ArtifactError("EOS token count does not equal document count")
    if int(totals["total_model_tokens"]) != (
        int(totals["bpe_tokens_without_eos"]) + int(totals["eos_tokens"])
    ):
        raise ArtifactError("total token conservation check failed")
    if config["validation"]["require_zero_unknown_tokens"] and int(
        totals["unknown_tokens"]
    ) != 0:
        raise ArtifactError(
            f"evaluation found {totals['unknown_tokens']} unknown tokens; expected zero"
        )

    train_ratio = 0.98
    pilot_train_ratio = float(train_stats["model_to_provided_token_ratio"])
    estimated_full_train_model_tokens = round(
        FULL_PROVIDED_TOKEN_TARGET * train_ratio * pilot_train_ratio
    )

    output_path = resolve_project_path(
        PROJECT_ROOT,
        args.output or config["evaluation"]["stats_file"],
    )
    _allow_statistics_write(output_path, tokenizer_sha256, args.overwrite)

    payload = {
        "schema_version": 1,
        "tokenizer": {
            "path": project_relative_path(tokenizer_path, PROJECT_ROOT),
            "sha256": tokenizer_sha256,
            "config_path": project_relative_path(metadata_path, PROJECT_ROOT),
            "config_sha256": sha256_file(metadata_path),
            "vocab_size": validation_summary["vocab_size"],
            "special_token_ids": special_token_ids(config),
            "core_artifact_hashes": artifact_hashes,
        },
        "source": {
            "dataset": source["dataset"],
            "configuration": source["configuration"],
            "revision": source["revision"],
            "manifest": project_relative_path(manifest_path, PROJECT_ROOT),
            "manifest_sha256": sha256_file(manifest_path),
        },
        "context_length": context_length,
        "splits": split_statistics,
        "totals": totals,
        "projection": {
            "full_provided_token_target": FULL_PROVIDED_TOKEN_TARGET,
            "train_split_ratio": train_ratio,
            "pilot_train_model_to_provided_ratio": pilot_train_ratio,
            "estimated_full_train_model_tokens": estimated_full_train_model_tokens,
            "is_estimate_only": True,
        },
        "storage": {
            "minimum_unsigned_dtype": "uint16",
            "uint16_is_sufficient": int(config["model"]["vocab_size"]) <= 65_536,
            "binary_dataset_generated": False,
        },
    }
    atomic_write_json(output_path, payload)

    print("Tokenizer evaluation complete")
    print(f"  total records: {totals['records']}")
    print(f"  total model tokens: {totals['total_model_tokens']}")
    print(f"  unknown tokens: {totals['unknown_tokens']}")
    print(
        "  estimated 350M Full train model tokens: "
        f"{estimated_full_train_model_tokens}"
    )
    print(f"  statistics: {project_relative_path(output_path, PROJECT_ROOT)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        print("Tokenizer evaluation interrupted; statistics were not published.", file=sys.stderr)
        return 130
    except (TokenizerPipelineError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
