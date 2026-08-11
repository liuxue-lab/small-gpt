from __future__ import annotations

import argparse
import math
import random
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from model import GPT, GPTConfig  # noqa: E402
from train import (  # noqa: E402
    CheckpointIdentity,
    PrecisionPolicy,
    Trainer,
    TrainerState,
    TrainingConfig,
    TrainingDataStream,
    build_checkpoint_identity,
    build_optimizer,
    build_scheduler,
    load_checkpoint,
    save_checkpoint,
    validate_run_id,
)


@dataclass
class Components:
    model: GPT
    optimizer: torch.optim.Optimizer
    scheduler: Any
    precision: PrecisionPolicy


@dataclass
class Branch:
    components: Components
    state: TrainerState
    trainer: Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare uninterrupted and checkpoint-resumed Debug/Pilot training."
        )
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
    parser.add_argument("--split-step", type=int, default=2)
    parser.add_argument("--final-step", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def configure_runtime(config: TrainingConfig) -> None:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
        torch.backends.cudnn.allow_tf32 = config.allow_tf32


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
    commit = commit_result.stdout.strip().lower()
    return commit, bool(status_result.stdout.strip())


def relative_identity(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.name


def build_components(
    model_config: GPTConfig,
    training_config: TrainingConfig,
) -> Components:
    precision = PrecisionPolicy.from_config(training_config)
    model = GPT(model_config).to(precision.device)
    optimizer = build_optimizer(model, training_config)
    scheduler = build_scheduler(
        optimizer,
        training_config,
        training_config.resolve(),
    )
    return Components(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        precision=precision,
    )


def build_branch(
    components: Components,
    *,
    training_config: TrainingConfig,
    state: TrainerState,
) -> Branch:
    trainer = Trainer(
        model=components.model,
        optimizer=components.optimizer,
        scheduler=components.scheduler,
        state=state,
        config=training_config,
        plan=training_config.resolve(),
        precision=components.precision,
    )
    return Branch(components=components, state=state, trainer=trainer)


def consume_runtime_rng(precision: PrecisionPolicy) -> dict[str, Any]:
    values: dict[str, Any] = {
        "python": random.random(),
        "numpy": float(np.random.random()),
        "torch_cpu": torch.rand(3),
        "torch_cuda": None,
    }
    if precision.device.type == "cuda":
        values["torch_cuda"] = torch.rand(
            3,
            device=precision.device,
        ).cpu()
    return values


def run_until(
    branch: Branch,
    *,
    manifest_path: Path,
    training_config: TrainingConfig,
    final_step: int,
) -> tuple[list[Any], list[dict[str, Any]]]:
    metrics = []
    rng_values = []
    with TrainingDataStream(
        manifest_path,
        config=training_config,
        plan=training_config.resolve(),
        state=branch.state,
    ) as stream:
        while branch.state.global_step < final_step:
            metrics.append(branch.trainer.run_update(stream))
            rng_values.append(consume_runtime_rng(branch.components.precision))
    return metrics, rng_values


def clone_batch(batch: tuple[torch.Tensor, torch.Tensor]):
    return batch[0].clone(), batch[1].clone()


def assert_batch_equal(
    expected: tuple[torch.Tensor, torch.Tensor],
    actual: tuple[torch.Tensor, torch.Tensor],
) -> None:
    if not torch.equal(expected[0], actual[0]) or not torch.equal(
        expected[1],
        actual[1],
    ):
        raise RuntimeError("resumed training stream did not restore the next batch")


def assert_metrics_equal(
    expected: Any,
    actual: Any,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> None:
    exact_fields = (
        "update_index",
        "completed_global_step",
        "learning_rate",
        "micro_steps",
        "samples",
        "tokens",
    )
    mismatches = [
        field
        for field in exact_fields
        if getattr(expected, field) != getattr(actual, field)
    ]
    for field in ("raw_token_weighted_loss", "grad_norm_before_clip"):
        if not math.isclose(
            getattr(expected, field),
            getattr(actual, field),
            rel_tol=relative_tolerance,
            abs_tol=absolute_tolerance,
        ):
            mismatches.append(field)
    if mismatches:
        raise RuntimeError(
            "resumed update metrics exceed the declared tolerance: "
            f"{mismatches}"
        )


def assert_nested_equal(
    expected: Any,
    actual: Any,
    *,
    path: str,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> None:
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor):
            raise RuntimeError(f"{path} type mismatch")
        same_values = torch.equal(expected, actual)
        if expected.is_floating_point() and not same_values:
            same_values = torch.allclose(
                expected,
                actual,
                rtol=relative_tolerance,
                atol=absolute_tolerance,
            )
        if expected.device != actual.device or expected.dtype != actual.dtype:
            same_values = False
        if not same_values:
            raise RuntimeError(f"{path} tensor mismatch")
        return
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or expected.keys() != actual.keys():
            raise RuntimeError(f"{path} mapping mismatch")
        for key in expected:
            assert_nested_equal(
                expected[key],
                actual[key],
                path=f"{path}.{key}",
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
            )
        return
    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        if not isinstance(actual, type(expected)) or len(expected) != len(actual):
            raise RuntimeError(f"{path} sequence mismatch")
        for index, (expected_value, actual_value) in enumerate(
            zip(expected, actual, strict=True)
        ):
            assert_nested_equal(
                expected_value,
                actual_value,
                path=f"{path}[{index}]",
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
            )
        return
    if expected != actual:
        raise RuntimeError(f"{path} value mismatch")


def assert_rng_equal(
    expected: dict[str, Any],
    actual: dict[str, Any],
) -> None:
    if expected["python"] != actual["python"]:
        raise RuntimeError("Python RNG continuation is not exact")
    if expected["numpy"] != actual["numpy"]:
        raise RuntimeError("NumPy RNG continuation is not exact")
    if not torch.equal(expected["torch_cpu"], actual["torch_cpu"]):
        raise RuntimeError("Torch CPU RNG continuation is not exact")
    expected_cuda = expected["torch_cuda"]
    actual_cuda = actual["torch_cuda"]
    if (expected_cuda is None) != (actual_cuda is None):
        raise RuntimeError("Torch CUDA RNG presence is not exact")
    if expected_cuda is not None and not torch.equal(expected_cuda, actual_cuda):
        raise RuntimeError("Torch CUDA RNG continuation is not exact")


def state_without_save_marker(state: TrainerState) -> dict[str, Any]:
    payload = state.state_dict()
    payload.pop("last_save_step")
    return payload


def resolved_snapshot(
    *,
    model_config: GPTConfig,
    training_config: TrainingConfig,
    precision: PrecisionPolicy,
    manifest_path: Path,
    split_step: int,
    final_step: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "project_name": training_config.project_name,
        "model": model_config.to_dict(),
        "training": training_config.to_dict(),
        "plan": training_config.resolve().to_dict(),
        "runtime": precision.to_dict(),
        "operation": {
            "manifest": relative_identity(manifest_path),
            "split_step": split_step,
            "final_step": final_step,
        },
    }


def main() -> int:
    args = parse_args()
    run_id = validate_run_id(args.run_id)
    if args.split_step <= 0:
        raise ValueError("--split-step must be positive")
    if args.final_step <= args.split_step:
        raise ValueError("--final-step must be greater than --split-step")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")

    config_path = args.config.resolve()
    manifest_path = args.manifest.resolve()
    model_config = GPTConfig.from_yaml(config_path)
    training_config = replace(
        TrainingConfig.from_yaml(config_path),
        device=args.device,
        precision=args.precision,
        num_workers=args.num_workers,
    )
    plan = training_config.resolve()
    if args.final_step > plan.total_updates:
        raise ValueError(
            "--final-step cannot exceed the resolved training horizon"
        )
    if args.device == "cuda" and args.precision == "bf16":
        relative_tolerance, absolute_tolerance = 5.0e-4, 5.0e-5
    elif args.device == "cuda":
        relative_tolerance, absolute_tolerance = 1.0e-6, 1.0e-7
    else:
        relative_tolerance, absolute_tolerance = 0.0, 0.0

    source_commit, source_dirty = source_identity()
    identity: CheckpointIdentity = build_checkpoint_identity(
        model_config.to_dict(),
        manifest_path,
        source_commit=source_commit,
        source_dirty=source_dirty,
    )
    checkpoint_run_dir = (
        PROJECT_ROOT / training_config.checkpoint_dir / run_id
    ).resolve()
    checkpoint_path = checkpoint_run_dir / (
        f"step-{args.split_step:08d}.pt"
    )
    if checkpoint_run_dir.exists():
        raise RuntimeError(
            "checkpoint verification directory already exists and will not be "
            f"overwritten: {checkpoint_run_dir}"
        )

    configure_runtime(training_config)
    continuous_components = build_components(model_config, training_config)
    continuous = build_branch(
        continuous_components,
        training_config=training_config,
        state=TrainerState(run_id=run_id),
    )
    continuous_metrics, continuous_rng = run_until(
        continuous,
        manifest_path=manifest_path,
        training_config=training_config,
        final_step=args.final_step,
    )
    continuous_probe = consume_runtime_rng(continuous.components.precision)

    configure_runtime(training_config)
    interrupted_components = build_components(model_config, training_config)
    interrupted = build_branch(
        interrupted_components,
        training_config=training_config,
        state=TrainerState(run_id=run_id),
    )
    interrupted_metrics = []
    interrupted_rng = []
    with TrainingDataStream(
        manifest_path,
        config=training_config,
        plan=plan,
        state=interrupted.state,
    ) as interrupted_stream:
        while interrupted.state.global_step < args.split_step:
            interrupted_metrics.append(
                interrupted.trainer.run_update(interrupted_stream)
            )
            interrupted_rng.append(
                consume_runtime_rng(interrupted.components.precision)
            )

        snapshot = resolved_snapshot(
            model_config=model_config,
            training_config=training_config,
            precision=interrupted.components.precision,
            manifest_path=manifest_path,
            split_step=args.split_step,
            final_step=args.final_step,
        )
        record = save_checkpoint(
            checkpoint_path,
            model=interrupted.components.model,
            optimizer=interrupted.components.optimizer,
            scheduler=interrupted.components.scheduler,
            state=interrupted.state,
            plan=plan,
            resolved_config=snapshot,
            identity=identity,
        )
        expected_next_batch = clone_batch(next(interrupted_stream))

    for expected, actual in zip(
        continuous_metrics[: args.split_step],
        interrupted_metrics,
        strict=True,
    ):
        assert_metrics_equal(
            expected,
            actual,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )
    for expected, actual in zip(
        continuous_rng[: args.split_step],
        interrupted_rng,
        strict=True,
    ):
        assert_rng_equal(expected, actual)

    resumed_components = build_components(model_config, training_config)
    loaded = load_checkpoint(
        checkpoint_path,
        model=resumed_components.model,
        optimizer=resumed_components.optimizer,
        scheduler=resumed_components.scheduler,
        plan=plan,
        resolved_config=snapshot,
        identity=identity,
        expected_run_id=run_id,
    )
    resumed = build_branch(
        resumed_components,
        training_config=training_config,
        state=loaded.state,
    )
    expected_next_lr = continuous_metrics[args.split_step].learning_rate
    actual_next_lr = resumed.components.scheduler.lr_for_update(args.split_step)
    if actual_next_lr != expected_next_lr:
        raise RuntimeError(
            "restored scheduler produced the wrong next learning rate"
        )

    with TrainingDataStream(
        manifest_path,
        config=training_config,
        plan=plan,
        state=resumed.state,
    ) as probe_stream:
        actual_next_batch = clone_batch(next(probe_stream))
    assert_batch_equal(expected_next_batch, actual_next_batch)

    resumed_metrics, resumed_rng = run_until(
        resumed,
        manifest_path=manifest_path,
        training_config=training_config,
        final_step=args.final_step,
    )
    resumed_probe = consume_runtime_rng(resumed.components.precision)

    for expected, actual in zip(
        continuous_metrics[args.split_step :],
        resumed_metrics,
        strict=True,
    ):
        assert_metrics_equal(
            expected,
            actual,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )
    for expected, actual in zip(
        continuous_rng[args.split_step :],
        resumed_rng,
        strict=True,
    ):
        assert_rng_equal(expected, actual)
    assert_rng_equal(continuous_probe, resumed_probe)

    assert_nested_equal(
        continuous.components.model.state_dict(),
        resumed.components.model.state_dict(),
        path="model",
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )
    assert_nested_equal(
        continuous.components.optimizer.state_dict(),
        resumed.components.optimizer.state_dict(),
        path="optimizer",
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )
    assert_nested_equal(
        continuous.components.scheduler.state_dict(),
        resumed.components.scheduler.state_dict(),
        path="scheduler",
        relative_tolerance=0.0,
        absolute_tolerance=0.0,
    )
    if state_without_save_marker(continuous.state) != state_without_save_marker(
        resumed.state
    ):
        raise RuntimeError("restored TrainerState continuation is not exact")

    print(f"Resolved device        : {resumed.components.precision.device}")
    print(f"Precision              : {resumed.components.precision.precision}")
    print(f"Source commit          : {identity.source_commit}")
    print(f"Source dirty           : {identity.source_dirty}")
    print(f"Checkpoint path        : {relative_identity(record.path)}")
    print(f"Checkpoint bytes       : {record.file_size:,}")
    print(f"Saved global step      : {record.global_step}")
    print(f"Saved tokens seen      : {record.tokens_seen:,}")
    print(f"Restored global step   : {loaded.record.global_step}")
    print(f"Next learning rate     : {actual_next_lr:.12g}")
    print("Next training batch    : exact")
    print(
        "Numeric tolerance      : "
        f"rtol={relative_tolerance:g}, atol={absolute_tolerance:g}"
    )
    print("Continuation metrics   : within tolerance")
    print("Model parameters       : within tolerance")
    print("Optimizer state        : within tolerance")
    print("Scheduler state        : exact")
    print("Python/NumPy/Torch RNG : exact")
    print(f"Final global step      : {resumed.state.global_step}")
    print(f"Final tokens seen      : {resumed.state.tokens_seen:,}")
    print("All checkpoint/resume checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
