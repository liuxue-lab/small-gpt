from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from train import (
    CheckpointRecord,
    JsonlMetricLogger,
    PrecisionPolicy,
    Trainer,
    TrainerState,
    TrainingConfig,
    TrainingLoopError,
    build_optimizer,
    build_scheduler,
    run_training_loop,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEBUG_PATH = PROJECT_ROOT / "configs" / "debug.yaml"


class TinyLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.projection = nn.Linear(8, 16)

    def forward(self, input_ids, targets=None):
        logits = self.projection(self.embedding(input_ids))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
            )
        return SimpleNamespace(logits=logits, loss=loss)


def loop_config(**overrides) -> TrainingConfig:
    values = {
        "project_name": "loop-test",
        "context_length": 4,
        "vocab_size": 16,
        "device": "cpu",
        "precision": "fp32",
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 1,
        "max_steps": 4,
        "target_tokens": None,
        "warmup_steps": 1,
        "warmup_ratio": None,
        "log_interval": 2,
        "eval_interval": 2,
        "eval_batches": 1,
        "save_interval": 3,
        "num_workers": 0,
        "pin_memory": False,
    }
    values.update(overrides)
    return replace(TrainingConfig.from_yaml(DEBUG_PATH), **values)


def fixed_batches(count: int = 4):
    base = torch.arange(8, dtype=torch.long).reshape(2, 4)
    return [
        ((base + offset) % 16, (base + offset + 1) % 16)
        for offset in range(count)
    ]


def build_trainer(config: TrainingConfig):
    torch.manual_seed(11)
    plan = config.resolve()
    precision = PrecisionPolicy.from_config(config)
    model = TinyLanguageModel().to(precision.device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, plan)
    state = TrainerState(run_id="loop-test")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        state=state,
        config=config,
        plan=plan,
        precision=precision,
    )
    return trainer


def checkpoint_writer(tmp_path: Path, trainer: Trainer, calls: list[int]):
    def write(step: int) -> CheckpointRecord:
        calls.append(step)
        path = tmp_path / f"step-{step:08d}.pt"
        path.write_bytes(f"checkpoint-{step}".encode("ascii"))
        trainer.state.record_checkpoint()
        return CheckpointRecord(
            path=path,
            file_size=path.stat().st_size,
            global_step=trainer.state.global_step,
            tokens_seen=trainer.state.tokens_seen,
        )

    return write


def test_loop_schedules_train_eval_periodic_and_final_checkpoint(tmp_path):
    trainer = build_trainer(loop_config())
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    checkpoint_calls: list[int] = []
    output: list[str] = []

    with JsonlMetricLogger(metrics_path) as logger:
        result = run_training_loop(
            trainer,
            fixed_batches(),
            fixed_batches(1),
            logger=logger,
            stop_at_step=4,
            checkpoint_writer=checkpoint_writer(
                tmp_path,
                trainer,
                checkpoint_calls,
            ),
            checkpoint_path_formatter=lambda path: path.name,
            emit=output.append,
        )

    events = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    assert result.start_step == 0
    assert result.stop_step == 4
    assert result.updates_completed == 4
    assert result.evaluation_steps == (2, 4)
    assert checkpoint_calls == [3, 4]
    assert [record.global_step for record in result.checkpoint_records] == [3, 4]
    assert result.final_checkpoint.global_step == 4
    assert trainer.state.global_step == 4
    assert trainer.state.tokens_seen == 32
    assert trainer.state.last_eval_step == 4
    assert trainer.state.last_save_step == 4
    assert [event["event"] for event in events] == [
        "train_update",
        "train_update",
        "evaluation",
        "train_update",
        "checkpoint",
        "train_update",
        "evaluation",
        "checkpoint",
    ]
    assert [event["step"] for event in events] == [1, 2, 2, 3, 3, 4, 4, 4]
    assert any(line.startswith("train step=1 ") for line in output)
    assert any(line.startswith("eval step=2 ") for line in output)


def test_interval_checkpoint_at_stop_is_not_written_twice(tmp_path):
    config = loop_config(save_interval=2, eval_interval=3)
    trainer = build_trainer(config)
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    calls: list[int] = []

    with JsonlMetricLogger(metrics_path) as logger:
        result = run_training_loop(
            trainer,
            fixed_batches(2),
            fixed_batches(1),
            logger=logger,
            stop_at_step=2,
            checkpoint_writer=checkpoint_writer(tmp_path, trainer, calls),
            emit=lambda _line: None,
        )

    assert calls == [2]
    assert len(result.checkpoint_records) == 1


@pytest.mark.parametrize("stop_at_step", (0, -1, True, 5))
def test_loop_rejects_invalid_stop_without_updating(tmp_path, stop_at_step):
    trainer = build_trainer(loop_config())
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")

    with JsonlMetricLogger(metrics_path) as logger:
        with pytest.raises(TrainingLoopError):
            run_training_loop(
                trainer,
                fixed_batches(),
                fixed_batches(1),
                logger=logger,
                stop_at_step=stop_at_step,
                checkpoint_writer=lambda _step: None,
            )

    assert trainer.state.global_step == 0
    assert metrics_path.read_text(encoding="utf-8") == ""


def test_checkpoint_writer_must_commit_and_return_matching_record(tmp_path):
    trainer = build_trainer(loop_config())
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")

    def wrong_writer(_step: int) -> CheckpointRecord:
        return CheckpointRecord(
            path=tmp_path / "wrong.pt",
            file_size=1,
            global_step=99,
            tokens_seen=trainer.state.tokens_seen,
        )

    with JsonlMetricLogger(metrics_path) as logger:
        with pytest.raises(TrainingLoopError, match="wrong global step"):
            run_training_loop(
                trainer,
                fixed_batches(1),
                fixed_batches(1),
                logger=logger,
                stop_at_step=1,
                checkpoint_writer=wrong_writer,
                emit=lambda _line: None,
            )

    assert trainer.state.global_step == 1
    assert trainer.state.last_save_step == 0
