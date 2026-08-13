"""Evaluate one frozen validation or test split from a completed checkpoint."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval import (  # noqa: E402
    FrozenEvaluationError,
    evaluate_frozen_split,
    publish_evaluation_result,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Strictly load a completed small-gpt checkpoint and evaluate one "
            "explicit frozen split."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "baseline.yaml",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        required=True,
    )
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
    parser.add_argument("--max-batches", type=_positive_int)
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
        raise FrozenEvaluationError(
            "evaluation output already exists and will not be overwritten: "
            f"{output_path}"
        )

    run_id = validate_run_id(args.run_id)
    model_config = GPTConfig.from_yaml(args.config.resolve())
    precision = PrecisionPolicy.resolve(args.device, args.precision)
    source_commit, source_dirty = source_identity()
    result = evaluate_frozen_split(
        args.checkpoint.resolve(),
        args.manifest.resolve(),
        model_config=model_config,
        expected_run_id=run_id,
        expected_checkpoint_sha256=args.checkpoint_sha256,
        split=args.split,
        precision=precision,
        max_batches=args.max_batches,
        evaluator_source_commit=source_commit,
        evaluator_source_dirty=source_dirty,
    )
    published_path = publish_evaluation_result(output_path, result)
    output_sha256 = sha256_file(published_path)

    metrics = result["metrics"]
    coverage = result["coverage"]
    perplexity = metrics["perplexity"]
    print("Frozen evaluation complete")
    print(f"Split={result['split']}")
    print(f"FullSplit={coverage['is_full_split']}")
    print(f"CheckpointSHA256={result['checkpoint']['sha256']}")
    print(f"Loss={metrics['loss']:.12f}")
    print(
        "Perplexity="
        + ("inf" if perplexity is None else f"{perplexity:.12f}")
    )
    print(f"EvaluatedTokens={metrics['evaluated_tokens']}")
    print(f"EvaluatedBatches={metrics['evaluated_batches']}")
    print(f"AvailableBatches={coverage['available_batches']}")
    print(f"FullEvaluationTokens={coverage['full_evaluation_tokens']}")
    print(
        "TrailingTokensDiscarded="
        f"{coverage['trailing_tokens_discarded']}"
    )
    print(f"OutputPath={published_path}")
    print(f"OutputSHA256={output_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
