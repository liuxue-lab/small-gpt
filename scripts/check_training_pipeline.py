from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from model import GPT, GPTConfig  # noqa: E402
from train import (  # noqa: E402
    JsonlMetricLogger,
    PrecisionPolicy,
    Trainer,
    TrainerState,
    TrainingConfig,
    TrainingDataStream,
    ValidationDataStream,
    build_optimizer,
    build_scheduler,
    evaluate_model,
    initialize_run_directory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Day 7 Pilot train/eval/logging integration check."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "debug.yaml",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16"),
        default="fp32",
    )
    parser.add_argument("--stop-at-step", type=int, default=3)
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def configure_reproducible_runtime(config: TrainingConfig) -> None:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
        torch.backends.cudnn.allow_tf32 = config.allow_tf32


def relative_identity(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.name


def main() -> int:
    args = parse_args()
    if args.stop_at_step <= 0:
        raise ValueError("--stop-at-step must be positive")
    if args.eval_batches <= 0:
        raise ValueError("--eval-batches must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")

    config_path = args.config.resolve()
    manifest_path = args.manifest.resolve()
    model_config = GPTConfig.from_yaml(config_path)
    training_config = replace(
        TrainingConfig.from_yaml(config_path),
        device=args.device,
        precision=args.precision,
        eval_batches=args.eval_batches,
        num_workers=args.num_workers,
    )
    plan = training_config.resolve()
    if args.stop_at_step > plan.total_updates:
        raise ValueError(
            "--stop-at-step cannot exceed the resolved training horizon"
        )
    precision = PrecisionPolicy.from_config(training_config)
    configure_reproducible_runtime(training_config)

    model = GPT(model_config).to(precision.device)
    optimizer = build_optimizer(model, training_config)
    scheduler = build_scheduler(optimizer, training_config, plan)
    state = TrainerState(run_id=args.run_id)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        state=state,
        config=training_config,
        plan=plan,
        precision=precision,
    )

    with TrainingDataStream(
        manifest_path,
        config=training_config,
        plan=plan,
        state=state,
    ) as train_stream, ValidationDataStream(
        manifest_path,
        plan=plan,
    ) as validation_stream:
        run_paths = initialize_run_directory(
            PROJECT_ROOT / training_config.run_dir,
            run_id=args.run_id,
            resolved_config={
                "schema_version": 1,
                "project_name": training_config.project_name,
                "model": model_config.to_dict(),
                "training": training_config.to_dict(),
                "plan": plan.to_dict(),
                "runtime": precision.to_dict(),
                "operation": {
                    "stop_at_step": args.stop_at_step,
                    "manifest": relative_identity(manifest_path),
                },
            },
            metadata={
                "purpose": "day07-pilot-train-eval-smoke",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "manifest": relative_identity(manifest_path),
                "model_parameters": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
            },
        )

        with JsonlMetricLogger(run_paths.metrics_path) as logger:
            while state.global_step < args.stop_at_step:
                metrics = trainer.run_update(train_stream)
                logger.log_train_update(
                    metrics,
                    state=state,
                    precision=precision,
                )
                print(
                    f"train step={state.global_step} "
                    f"tokens_seen={state.tokens_seen} "
                    f"loss={metrics.raw_token_weighted_loss:.6f} "
                    f"lr={metrics.learning_rate:.12g} "
                    f"grad_norm={metrics.grad_norm_before_clip:.6f}"
                )

            evaluation = evaluate_model(
                model,
                validation_stream,
                precision=precision,
                global_step=state.global_step,
                max_batches=training_config.eval_batches,
            )
            state.record_evaluation(evaluation.validation_loss)
            logger.log_evaluation(evaluation, state=state)
            print(
                f"eval step={state.global_step} "
                f"tokens={evaluation.evaluated_tokens} "
                f"loss={evaluation.validation_loss:.6f} "
                f"perplexity={evaluation.perplexity:.6f}"
            )

    if not train_stream.is_closed or not validation_stream.is_closed:
        raise RuntimeError("token stores were not closed after the integration run")

    metric_events = [
        json.loads(line)
        for line in run_paths.metrics_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    expected_events = args.stop_at_step + 1
    if len(metric_events) != expected_events:
        raise RuntimeError(
            f"expected {expected_events} metric events, got {len(metric_events)}"
        )
    if [event["step"] for event in metric_events[:-1]] != list(
        range(1, args.stop_at_step + 1)
    ):
        raise RuntimeError("train metric steps are not contiguous")
    if metric_events[-1]["event"] != "evaluation":
        raise RuntimeError("final metric event is not evaluation")
    if not all(
        math.isfinite(float(event["train_loss"]))
        for event in metric_events
        if event["event"] == "train_update"
    ):
        raise RuntimeError("a train event contains non-finite loss")

    print(f"Run directory         : {relative_identity(run_paths.run_dir)}")
    print(f"Metric events         : {len(metric_events)}")
    print(f"Final global step     : {state.global_step}")
    print(f"Final tokens seen     : {state.tokens_seen:,}")
    print("Checkpoint written    : False (Stage F scope)")
    print("All Pilot training-pipeline checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
