"""Train and publish the project's 16,384-token ByteLevel BPE tokenizer."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tokenizer.bpe import (  # noqa: E402
    ArtifactError,
    CorpusStats,
    TokenizerPipelineError,
    discover_split_files,
    iter_jsonl_texts,
    load_source_manifest,
    load_tokenizer_config,
    project_relative_path,
    resolve_project_path,
    save_tokenizer_artifacts,
    sha256_file,
    train_tokenizer,
    validate_installed_tokenizers_version,
    validate_model_configs,
)


VALIDATION_SAMPLES = (
    "Hello, world!",
    "The Transformer predicts the next token.",
    "Don't split contractions incorrectly.",
    "Line one.\nLine two.",
    "café naïve résumé",
    "Emoji test: 🤖🚀",
    "中文只用于验证字节覆盖。",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the small-gpt ByteLevel BPE tokenizer from the existing "
            "FineWeb-Edu Pilot train split."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/tokenizer.yaml",
        help="Tokenizer YAML configuration relative to the project root.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional artifact output directory override.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing artifact directory after validation.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the native tokenizer training progress bar.",
    )
    return parser.parse_args(argv)


def _print_preflight(
    *,
    config_path: Path,
    corpus_dir: Path,
    manifest_path: Path,
    train_files: list[Path],
    output_dir: Path,
    config: dict[str, Any],
    overwrite: bool,
) -> None:
    print("Tokenizer training preflight")
    print(f"  config: {project_relative_path(config_path, PROJECT_ROOT)}")
    print(f"  corpus: {project_relative_path(corpus_dir, PROJECT_ROOT)}")
    print(f"  manifest: {project_relative_path(manifest_path, PROJECT_ROOT)}")
    print(f"  split: {config['source']['split']}")
    print(f"  train files: {len(train_files)}")
    print(f"  expected records: {config['source']['expected_records']}")
    print(
        "  expected provided tokens: "
        f"{config['source']['expected_provided_tokens']}"
    )
    print(f"  vocab size: {config['model']['vocab_size']}")
    print(
        "  special tokens: "
        + ", ".join(
            f"{item['token']}={item['id']}" for item in config["special_tokens"]
        )
    )
    print(f"  output: {project_relative_path(output_dir, PROJECT_ROOT)}")
    print(f"  overwrite: {overwrite}")


def run(args: argparse.Namespace) -> int:
    config_path = resolve_project_path(PROJECT_ROOT, args.config)
    config = load_tokenizer_config(config_path)
    validate_installed_tokenizers_version(config)
    model_validation = validate_model_configs(config, PROJECT_ROOT)

    source = config["source"]
    corpus_dir = resolve_project_path(PROJECT_ROOT, source["corpus_dir"])
    manifest_path = resolve_project_path(PROJECT_ROOT, source["manifest"])
    load_source_manifest(manifest_path, int(source["expected_shards"]))

    train_files = discover_split_files(
        corpus_dir,
        "train",
        expected_shards=int(source["expected_shards"]),
    )

    configured_output = args.output_dir or config["artifacts"]["output_dir"]
    output_dir = resolve_project_path(PROJECT_ROOT, configured_output)
    overwrite = bool(args.overwrite or config["training"]["overwrite_existing"])
    if output_dir.exists() and not overwrite:
        raise ArtifactError(
            f"artifact directory already exists: {output_dir}; refusing to overwrite"
        )

    _print_preflight(
        config_path=config_path,
        corpus_dir=corpus_dir,
        manifest_path=manifest_path,
        train_files=train_files,
        output_dir=output_dir,
        config=config,
        overwrite=overwrite,
    )

    corpus_stats = CorpusStats()
    text_iterator = iter_jsonl_texts(train_files, corpus_stats)
    started = time.perf_counter()
    tokenizer = train_tokenizer(
        config,
        text_iterator,
        length=int(source["expected_records"]),
        show_progress=False if args.no_progress else None,
    )
    elapsed_seconds = time.perf_counter() - started

    expected_records = int(source["expected_records"])
    expected_provided_tokens = int(source["expected_provided_tokens"])
    if corpus_stats.records != expected_records:
        raise ArtifactError(
            f"training consumed {corpus_stats.records} records; expected {expected_records}"
        )
    if corpus_stats.provided_tokens != expected_provided_tokens:
        raise ArtifactError(
            "training consumed "
            f"{corpus_stats.provided_tokens} provided tokens; "
            f"expected {expected_provided_tokens}"
        )

    training_metadata = {
        "dataset": source["dataset"],
        "configuration": source["configuration"],
        "split": "train",
        "revision": source["revision"],
        "manifest": {
            "path": project_relative_path(manifest_path, PROJECT_ROOT),
            "sha256": sha256_file(manifest_path),
        },
        "input_files": [
            project_relative_path(path, PROJECT_ROOT) for path in train_files
        ],
        "corpus": corpus_stats.to_dict(),
        "deterministic_order": source["deterministic_order"],
        "trainer": {
            "vocab_size": config["model"]["vocab_size"],
            "min_frequency": config["model"]["min_frequency"],
            "max_token_length": config["model"]["max_token_length"],
            "dropout": config["model"]["dropout"],
            "initial_alphabet": config["pre_tokenizer"]["initial_alphabet"],
        },
        "model_config_validation": model_validation,
    }

    bundle = save_tokenizer_artifacts(
        tokenizer,
        config,
        output_dir,
        project_root=PROJECT_ROOT,
        training_metadata=training_metadata,
        validation_samples=VALIDATION_SAMPLES,
        overwrite=overwrite,
    )

    print("Tokenizer training complete")
    print(f"  records: {corpus_stats.records}")
    print(f"  provided tokens: {corpus_stats.provided_tokens}")
    print(f"  vocab size: {tokenizer.get_vocab_size(with_added_tokens=True)}")
    print(f"  elapsed seconds: {elapsed_seconds:.3f}")
    print(f"  artifact directory: {project_relative_path(output_dir, PROJECT_ROOT)}")
    print(f"  tokenizer sha256: {bundle.tokenizer.sha256}")
    print(f"  config sha256: {bundle.config.sha256}")
    print("  save/reload validation: passed")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        print("Tokenizer training interrupted; no validated artifact was published.", file=sys.stderr)
        return 130
    except (TokenizerPipelineError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
