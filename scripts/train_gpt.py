from __future__ import annotations

import argparse
import hashlib
import os
import random
import subprocess
import sys
from contextlib import ExitStack
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from model import GPT, GPTConfig  # noqa: E402
from train import (  # noqa: E402
    CheckpointIdentity,
    JsonlMetricLogger,
    PrecisionPolicy,
    Trainer,
    TrainerState,
    TrainingConfig,
    TrainingDataStream,
    ValidationDataStream,
    build_checkpoint_identity,
    build_optimizer,
    build_scheduler,
    initialize_run_directory,
    load_checkpoint,
    open_existing_run_directory,
    partition_parameters,
    read_metric_events,
    run_training_loop,
    save_checkpoint,
    validate_run_id,
)


class TrainingEntryError(RuntimeError):
    """Raised when the formal CLI cannot start without violating safety."""


class SyntheticTrainingBatches(
    Iterator[tuple[torch.Tensor, torch.Tensor]]
):
    """Index-seeded synthetic micro-batches for bounded CLI smoke checks."""

    def __init__(
        self,
        *,
        state: TrainerState,
        config: TrainingConfig,
    ) -> None:
        plan = config.resolve()
        state.validate_for_plan(plan)
        self._seed = config.seed
        self._vocab_size = config.vocab_size
        self._batch_size = plan.micro_batch_size
        self._context_length = plan.context_length
        self._index = state.micro_steps_seen
        self._limit = plan.total_updates * plan.gradient_accumulation_steps

    def __iter__(self) -> SyntheticTrainingBatches:
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._index >= self._limit:
            raise StopIteration
        tokens = _synthetic_tokens(
            base_seed=self._seed,
            stream_name="train",
            batch_index=self._index,
            batch_size=self._batch_size,
            context_length=self._context_length,
            vocab_size=self._vocab_size,
        )
        self._index += 1
        return tokens[:, :-1].clone(), tokens[:, 1:].clone()


class SyntheticValidationBatches:
    """Fixed synthetic validation batches that repeat exactly per evaluation."""

    def __init__(self, *, config: TrainingConfig) -> None:
        plan = config.resolve()
        self._seed = config.seed
        self._vocab_size = config.vocab_size
        self._batch_size = plan.micro_batch_size
        self._context_length = plan.context_length
        self._batches = config.eval_batches if config.eval_batches is not None else 1

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for batch_index in range(self._batches):
            tokens = _synthetic_tokens(
                base_seed=self._seed,
                stream_name="validation",
                batch_index=batch_index,
                batch_size=self._batch_size,
                context_length=self._context_length,
                vocab_size=self._vocab_size,
            )
            yield tokens[:, :-1].clone(), tokens[:, 1:].clone()


def _synthetic_tokens(
    *,
    base_seed: int,
    stream_name: str,
    batch_index: int,
    batch_size: int,
    context_length: int,
    vocab_size: int,
) -> torch.Tensor:
    digest = hashlib.sha256(
        f"small-gpt-synthetic:{stream_name}:{base_seed}:{batch_index}".encode(
            "ascii"
        )
    ).digest()
    seed = int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, context_length + 1),
        dtype=torch.long,
        generator=generator,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the Day 7 decoder-only GPT with strict logging and resume."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "debug.yaml",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--precision", choices=("fp32", "bf16"))
    parser.add_argument(
        "--batch-source",
        choices=("pilot", "synthetic"),
        default="pilot",
    )
    parser.add_argument("--stop-at-step", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def configure_runtime(config: TrainingConfig) -> None:
    if config.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(config.deterministic)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
        torch.backends.cudnn.allow_tf32 = config.allow_tf32
        torch.backends.cudnn.deterministic = config.deterministic
        torch.backends.cudnn.benchmark = False


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


def relative_identity(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.name


def resolve_training_config(
    config_path: Path,
    *,
    device: str | None,
    precision: str | None,
    num_workers: int | None,
) -> TrainingConfig:
    config = TrainingConfig.from_yaml(config_path)
    overrides: dict[str, Any] = {}
    if device is not None:
        overrides["device"] = device
    if precision is not None:
        overrides["precision"] = precision
    if num_workers is not None:
        if num_workers < 0:
            raise TrainingEntryError("--num-workers must be non-negative")
        overrides["num_workers"] = num_workers
    return replace(config, **overrides)


def resolved_snapshot(
    *,
    model_config: GPTConfig,
    training_config: TrainingConfig,
    precision: PrecisionPolicy,
    batch_source: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "project_name": training_config.project_name,
        "model": model_config.to_dict(),
        "training": training_config.to_dict(),
        "plan": training_config.resolve().to_dict(),
        "runtime": precision.to_dict(),
        "inputs": {"batch_source": batch_source},
    }


def _validate_stop_at_step(value: int | None, *, total_updates: int) -> int:
    if value is None:
        return total_updates
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TrainingEntryError("--stop-at-step must be a positive integer")
    if value > total_updates:
        raise TrainingEntryError(
            "--stop-at-step cannot exceed the resolved training horizon"
        )
    return value


def _validate_resume_log(events: Sequence[dict[str, Any]], *, step: int) -> None:
    future = [
        event
        for event in events
        if event.get("step") is not None and event["step"] > step
    ]
    if future:
        raise TrainingEntryError(
            "run metrics contain events newer than the requested checkpoint"
        )


def _print_preflight(
    *,
    model: GPT,
    training_config: TrainingConfig,
    precision: PrecisionPolicy,
    identity: CheckpointIdentity,
    batch_source: str,
    stop_at_step: int,
    run_dir: Path,
    checkpoint_dir: Path,
    resume_path: Path | None,
    state: TrainerState,
    next_learning_rate: float | None,
    sample_batch: tuple[torch.Tensor, torch.Tensor] | None,
) -> None:
    plan = training_config.resolve()
    groups = partition_parameters(model)
    print(f"Project                : {training_config.project_name}")
    print(f"Resolved device        : {precision.device}")
    print(f"Precision              : {precision.precision}")
    print(f"Autocast enabled       : {precision.uses_autocast}")
    print(f"GradScaler enabled     : {precision.uses_grad_scaler}")
    print(f"Batch source           : {batch_source}")
    print(f"Model parameters       : {groups.total_numel:,}")
    print(
        "Optimizer decay       : "
        f"{len(groups.decay_parameters)} tensors / {groups.decay_numel:,} params"
    )
    print(
        "Optimizer no-decay    : "
        f"{len(groups.no_decay_parameters)} tensors / "
        f"{groups.no_decay_numel:,} params"
    )
    print(f"Tokens/update          : {plan.tokens_per_update:,}")
    print(f"Total updates          : {plan.total_updates:,}")
    print(f"Warmup updates         : {plan.warmup_updates:,}")
    print(f"Process stop step      : {stop_at_step:,}")
    print(f"Restored global step   : {state.global_step:,}")
    print(f"Restored tokens seen   : {state.tokens_seen:,}")
    print(f"Data batches consumed  : {state.batches_consumed_in_epoch:,}")
    print(f"Data samples consumed  : {state.samples_consumed:,}")
    print(
        "Next learning rate     : "
        + (
            "none (training complete)"
            if next_learning_rate is None
            else f"{next_learning_rate:.12g}"
        )
    )
    print(f"Tokenizer SHA-256      : {identity.tokenizer_sha256}")
    print(f"Manifest SHA-256       : {identity.dataset_manifest_sha256}")
    print(f"Dataset fingerprint    : {identity.dataset_config_fingerprint}")
    print(f"Source commit          : {identity.source_commit}")
    print(f"Source dirty           : {identity.source_dirty}")
    print(f"Run directory          : {relative_identity(run_dir)}")
    print(f"Checkpoint directory   : {relative_identity(checkpoint_dir)}")
    if resume_path is not None:
        print(f"Resume checkpoint      : {relative_identity(resume_path)}")
    if sample_batch is not None:
        input_ids, targets = sample_batch
        shift_exact = torch.equal(targets[:, :-1], input_ids[:, 1:])
        print(f"Sample input shape     : {tuple(input_ids.shape)}")
        print(f"Sample target shape    : {tuple(targets.shape)}")
        print(f"Sample causal shift    : {shift_exact}")


def _build_data_streams(
    stack: ExitStack,
    *,
    batch_source: str,
    manifest_path: Path,
    training_config: TrainingConfig,
    state: TrainerState,
) -> tuple[Iterator[tuple[torch.Tensor, torch.Tensor]], Any]:
    plan = training_config.resolve()
    if batch_source == "pilot":
        training = stack.enter_context(
            TrainingDataStream(
                manifest_path,
                config=training_config,
                plan=plan,
                state=state,
            )
        )
        validation = stack.enter_context(
            ValidationDataStream(manifest_path, plan=plan)
        )
        return training, validation
    if batch_source == "synthetic":
        return (
            SyntheticTrainingBatches(state=state, config=training_config),
            SyntheticValidationBatches(config=training_config),
        )
    raise TrainingEntryError(f"unsupported batch source {batch_source!r}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_id = validate_run_id(args.run_id)
    config_path = args.config.resolve()
    manifest_path = args.manifest.resolve()
    model_config = GPTConfig.from_yaml(config_path)
    training_config = resolve_training_config(
        config_path,
        device=args.device,
        precision=args.precision,
        num_workers=args.num_workers,
    )
    plan = training_config.resolve()
    stop_at_step = _validate_stop_at_step(
        args.stop_at_step,
        total_updates=plan.total_updates,
    )
    configure_runtime(training_config)
    precision = PrecisionPolicy.from_config(training_config)
    source_commit, source_dirty = source_identity()
    identity = build_checkpoint_identity(
        model_config.to_dict(),
        manifest_path,
        source_commit=source_commit,
        source_dirty=source_dirty,
    )
    snapshot = resolved_snapshot(
        model_config=model_config,
        training_config=training_config,
        precision=precision,
        batch_source=args.batch_source,
    )

    runs_root = (PROJECT_ROOT / training_config.run_dir).resolve()
    checkpoint_root = (PROJECT_ROOT / training_config.checkpoint_dir).resolve()
    run_dir = runs_root / run_id
    checkpoint_run_dir = checkpoint_root / run_id
    resume_path = args.resume.resolve() if args.resume is not None else None

    if resume_path is None:
        if run_dir.exists():
            raise TrainingEntryError(
                f"run directory already exists and will not be overwritten: {run_dir}"
            )
        if checkpoint_run_dir.exists():
            raise TrainingEntryError(
                "checkpoint directory already exists and will not be overwritten: "
                f"{checkpoint_run_dir}"
            )
    elif not resume_path.is_file():
        raise TrainingEntryError(f"resume checkpoint does not exist: {resume_path}")

    model = GPT(model_config).to(precision.device)
    optimizer = build_optimizer(model, training_config)
    scheduler = build_scheduler(optimizer, training_config, plan)

    loaded = None
    if resume_path is None:
        state = TrainerState(run_id=run_id)
    else:
        loaded = load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            plan=plan,
            resolved_config=snapshot,
            identity=identity,
            expected_run_id=run_id,
        )
        state = loaded.state
        run_paths = open_existing_run_directory(
            runs_root,
            run_id=run_id,
            expected_resolved_config=snapshot,
        )
        _validate_resume_log(
            read_metric_events(run_paths.metrics_path),
            step=state.global_step,
        )

    if not args.dry_run and stop_at_step <= state.global_step:
        raise TrainingEntryError(
            "--stop-at-step must be greater than the restored checkpoint step"
        )

    trainer = None
    if state.global_step < plan.total_updates:
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            state=state,
            config=training_config,
            plan=plan,
            precision=precision,
        )

    with ExitStack() as stack:
        sample_batch = None
        training_batches = None
        validation_batches = None
        if state.global_step < plan.total_updates:
            training_batches, validation_batches = _build_data_streams(
                stack,
                batch_source=args.batch_source,
                manifest_path=manifest_path,
                training_config=training_config,
                state=state,
            )
            if args.dry_run:
                sample_batch = next(training_batches)

        _print_preflight(
            model=model,
            training_config=training_config,
            precision=precision,
            identity=identity,
            batch_source=args.batch_source,
            stop_at_step=stop_at_step,
            run_dir=run_dir,
            checkpoint_dir=checkpoint_run_dir,
            resume_path=resume_path,
            state=state,
            next_learning_rate=(
                None
                if state.global_step >= plan.total_updates
                else scheduler.lr_for_update(state.global_step)
            ),
            sample_batch=sample_batch,
        )
        if args.dry_run:
            print("Dry run complete        : no update, log, or checkpoint written")
            return 0

        assert trainer is not None
        assert training_batches is not None
        assert validation_batches is not None
        if resume_path is None:
            run_paths = initialize_run_directory(
                runs_root,
                run_id=run_id,
                resolved_config=snapshot,
                metadata={
                    "purpose": "day07-formal-training",
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "manifest": relative_identity(manifest_path),
                    "batch_source": args.batch_source,
                    "initial_stop_at_step": stop_at_step,
                    "model_parameters": sum(
                        parameter.numel() for parameter in model.parameters()
                    ),
                    "identity": identity.to_dict(),
                },
            )

        def write_checkpoint(step: int):
            checkpoint_path = checkpoint_run_dir / f"step-{step:08d}.pt"
            if checkpoint_path.exists():
                raise TrainingEntryError(
                    "checkpoint target already exists and will not be overwritten: "
                    f"{checkpoint_path}"
                )
            return save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                state=state,
                plan=plan,
                resolved_config=snapshot,
                identity=identity,
            )

        with JsonlMetricLogger(run_paths.metrics_path, append=True) as logger:
            logger.write_event(
                {
                    "event": "resume_start" if loaded is not None else "run_start",
                    "step": state.global_step,
                    "tokens_seen": state.tokens_seen,
                    "stop_at_step": stop_at_step,
                    "resume_checkpoint": (
                        None
                        if resume_path is None
                        else relative_identity(resume_path)
                    ),
                }
            )
            result = run_training_loop(
                trainer,
                training_batches,
                validation_batches,
                logger=logger,
                stop_at_step=stop_at_step,
                checkpoint_writer=write_checkpoint,
                checkpoint_path_formatter=relative_identity,
            )

    final_checkpoint = result.final_checkpoint
    print(f"Run directory          : {relative_identity(run_paths.run_dir)}")
    print(f"Final global step      : {state.global_step:,}")
    print(f"Final tokens seen      : {state.tokens_seen:,}")
    print(f"Evaluations this run   : {len(result.evaluation_steps)}")
    print(f"Checkpoints this run   : {len(result.checkpoint_records)}")
    print(f"Final checkpoint       : {relative_identity(final_checkpoint.path)}")
    print("Training process complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
