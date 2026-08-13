"""Generate one auditable continuation from a completed checkpoint."""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval import (  # noqa: E402
    GenerationError,
    GenerationSettings,
    generate_from_checkpoint,
    publish_generation_result,
    sha256_file,
)
from model import GPTConfig  # noqa: E402
from train import PrecisionPolicy, validate_run_id  # noqa: E402


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "value must be a positive integer"
        ) from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "value must be a non-negative integer"
        ) from error
    if parsed < 0:
        raise argparse.ArgumentTypeError(
            "value must be a non-negative integer"
        )
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "value must be finite and positive"
        ) from error
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError(
            "value must be finite and positive"
        )
    return parsed


def _top_p(value: str) -> float:
    parsed = _positive_float(value)
    if parsed > 1.0:
        raise argparse.ArgumentTypeError("value must be in (0, 1]")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Strictly load a completed small-gpt checkpoint and generate one "
            "auditable continuation."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "baseline.yaml",
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
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--strategy",
        choices=("greedy", "sample"),
        required=True,
    )
    parser.add_argument("--max-new-tokens", type=_positive_int, required=True)
    parser.add_argument("--temperature", type=_positive_float, default=1.0)
    parser.add_argument("--top-k", type=_positive_int)
    parser.add_argument("--top-p", type=_top_p)
    parser.add_argument("--seed", type=_nonnegative_int)
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
    parser.add_argument("--output", type=Path, required=True)
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
    output_path = args.output.resolve()
    if output_path.exists():
        raise GenerationError(
            "generation output already exists and will not be overwritten: "
            f"{output_path}"
        )

    run_id = validate_run_id(args.run_id)
    model_config = GPTConfig.from_yaml(args.config.resolve())
    settings = GenerationSettings(
        strategy=args.strategy,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
    )
    precision = PrecisionPolicy.resolve(args.device, args.precision)
    source_commit, source_dirty = source_identity()
    result = generate_from_checkpoint(
        args.checkpoint.resolve(),
        args.tokenizer.resolve(),
        args.prompt,
        model_config=model_config,
        expected_run_id=run_id,
        expected_checkpoint_sha256=args.checkpoint_sha256,
        expected_tokenizer_sha256=args.tokenizer_sha256,
        settings=settings,
        precision=precision,
        generator_source_commit=source_commit,
        generator_source_dirty=source_dirty,
    )
    published_path = publish_generation_result(output_path, result)
    output_sha256 = sha256_file(published_path)

    generation = result["generation"]
    protocol = result["protocol"]
    print("Text generation complete")
    print(f"Strategy={protocol['strategy']}")
    print(f"Seed={protocol['seed']}")
    print(f"CheckpointSHA256={result['checkpoint']['sha256']}")
    print(f"TokenizerSHA256={result['tokenizer']['sha256']}")
    print(f"PromptTokens={result['prompt']['token_count']}")
    print(f"GeneratedTokens={generation['token_count']}")
    print(f"StopReason={generation['stop_reason']}")
    print(f"ContextCropEvents={generation['context_crop_events']}")
    print(f"ElapsedSeconds={generation['elapsed_seconds']:.6f}")
    print(f"Continuation={generation['continuation_text']!r}")
    print(f"OutputPath={published_path}")
    print(f"OutputSHA256={output_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
