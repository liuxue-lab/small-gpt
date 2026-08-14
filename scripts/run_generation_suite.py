"""Run the complete frozen Day 11 text-generation protocol."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval import (  # noqa: E402
    GENERATION_MANIFEST_FILENAME,
    GENERATION_SAMPLES_FILENAME,
    GenerationSuiteError,
    run_generation_suite,
    sha256_file,
)
from model import GPTConfig  # noqa: E402
from train import PrecisionPolicy, validate_run_id  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Strictly load a completed small-gpt checkpoint once and run the "
            "complete frozen prompt/decoding protocol."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "baseline.yaml",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=(
            PROJECT_ROOT / "configs" / "day11_generation_protocol.json"
        ),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=PROJECT_ROOT / "tokenizer" / "artifacts" / "tokenizer.json",
    )
    parser.add_argument("--tokenizer-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16"),
        default="fp32",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def source_identity() -> tuple[str | None, bool]:
    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None, True
    return commit_result.stdout.strip().lower(), bool(status_result.stdout.strip())


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise GenerationSuiteError(
            f"generation suite output already exists: {output_dir}"
        )

    run_id = validate_run_id(args.run_id)
    model_config = GPTConfig.from_yaml(args.config.resolve())
    precision = PrecisionPolicy.resolve(args.device, args.precision)
    source_commit, source_dirty = source_identity()
    published, manifest = run_generation_suite(
        args.checkpoint.resolve(),
        args.tokenizer.resolve(),
        args.protocol.resolve(),
        output_dir,
        model_config=model_config,
        expected_run_id=run_id,
        expected_checkpoint_sha256=args.checkpoint_sha256,
        expected_tokenizer_sha256=args.tokenizer_sha256,
        precision=precision,
        generator_source_commit=source_commit,
        generator_source_dirty=source_dirty,
    )

    manifest_path = published / GENERATION_MANIFEST_FILENAME
    samples_path = published / GENERATION_SAMPLES_FILENAME
    execution = manifest["execution"]
    summary = manifest["summary"]
    print("Frozen generation suite complete")
    print(f"ProtocolID={manifest['protocol']['definition']['protocol_id']}")
    print(f"PromptCount={execution['prompt_count']}")
    print(f"DecodingCount={execution['decoding_count']}")
    print(f"CompletedSamples={execution['completed_samples']}")
    print(f"ArtifactsLoadedOnce={execution['artifacts_loaded_once']}")
    print(f"GeneratedTokens={summary['generated_tokens']}")
    print(
        "StopReasonCounts="
        + json.dumps(summary["stop_reason_counts"], sort_keys=True)
    )
    print(f"WallElapsedSeconds={execution['wall_elapsed_seconds']:.6f}")
    print(f"OutputDirectory={published}")
    print(f"ManifestPath={manifest_path}")
    print(f"ManifestSHA256={sha256_file(manifest_path)}")
    print(f"SamplesPath={samples_path}")
    print(f"SamplesSHA256={sha256_file(samples_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
