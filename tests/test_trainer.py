from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from train import (
    BatchContractError,
    NonFiniteTrainingError,
    PrecisionPolicy,
    Trainer,
    TrainerState,
    TrainingConfig,
    TrainingStepError,
    build_optimizer,
    build_scheduler,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEBUG_PATH = PROJECT_ROOT / "configs" / "debug.yaml"
VOCAB_SIZE = 32


class ToyLanguageModel(nn.Module):
    def __init__(self, vocab_size: int = VOCAB_SIZE, embedding_size: int = 12):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.projection = nn.Linear(embedding_size, vocab_size)
        self.forward_calls = 0

    def forward(self, input_ids, targets=None):
        self.forward_calls += 1
        logits = self.projection(self.embedding(input_ids))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
            )
        return SimpleNamespace(logits=logits, loss=loss)


class NonFiniteLossModel(ToyLanguageModel):
    def forward(self, input_ids, targets=None):
        output = super().forward(input_ids, targets)
        assert output.loss is not None
        return SimpleNamespace(
            logits=output.logits,
            loss=output.loss * torch.tensor(float("nan")),
        )


class MissingLossModel(ToyLanguageModel):
    def forward(self, input_ids, targets=None):
        output = super().forward(input_ids, None)
        return SimpleNamespace(logits=output.logits, loss=None)


class UnusedParameterModel(ToyLanguageModel):
    def __init__(self):
        super().__init__()
        self.unused = nn.Parameter(torch.ones(3))


def training_config(
    *,
    micro_batch_size: int = 2,
    accumulation_steps: int = 2,
    gradient_clip: float = 1.0e9,
    weight_decay: float = 0.0,
) -> TrainingConfig:
    return replace(
        TrainingConfig.from_yaml(DEBUG_PATH),
        context_length=4,
        vocab_size=VOCAB_SIZE,
        device="cpu",
        precision="fp32",
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=accumulation_steps,
        max_steps=8,
        target_tokens=None,
        learning_rate=1.0e-2,
        min_learning_rate=1.0e-3,
        weight_decay=weight_decay,
        warmup_steps=2,
        warmup_ratio=None,
        gradient_clip=gradient_clip,
        num_workers=0,
        pin_memory=False,
    )


def shifted_batch(
    batch_size: int,
    *,
    seed: int,
    context_length: int = 4,
):
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(
        0,
        VOCAB_SIZE,
        (batch_size, context_length + 1),
        generator=generator,
        dtype=torch.long,
    )
    return tokens[:, :-1], tokens[:, 1:]


def build_harness(
    config: TrainingConfig,
    *,
    model: nn.Module | None = None,
    state: TrainerState | None = None,
):
    if model is None:
        torch.manual_seed(123)
        model = ToyLanguageModel()
    if state is None:
        state = TrainerState(run_id="trainer-test")
    plan = config.resolve()
    precision = PrecisionPolicy.from_config(config)
    model.to(precision.device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, plan)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        state=state,
        config=config,
        plan=plan,
        precision=precision,
    )
    return trainer, model, optimizer, scheduler, state, plan


def test_one_accumulated_update_commits_exact_metrics_and_state():
    config = training_config()
    trainer, model, _, scheduler, state, plan = build_harness(config)
    before = model.projection.weight.detach().clone()

    metrics = trainer.run_update(
        [shifted_batch(2, seed=1), shifted_batch(2, seed=2)]
    )

    assert metrics.update_index == 0
    assert metrics.completed_global_step == 1
    assert math.isfinite(metrics.raw_token_weighted_loss)
    assert metrics.learning_rate == pytest.approx(config.learning_rate / 2)
    assert math.isfinite(metrics.grad_norm_before_clip)
    assert metrics.micro_steps == 2
    assert metrics.samples == 4
    assert metrics.tokens == 16
    assert metrics.elapsed_seconds > 0.0
    assert metrics.tokens_per_second > 0.0
    assert metrics.to_dict()["tokens"] == 16
    assert state.global_step == 1
    assert state.micro_steps_seen == 2
    assert state.tokens_seen == 16
    assert state.samples_consumed == 4
    assert scheduler.next_update_index == 1
    assert not torch.equal(model.projection.weight, before)
    state.validate_for_plan(plan)


def test_second_update_uses_next_lr_and_keeps_counters_consistent():
    config = training_config()
    trainer, _, _, scheduler, state, plan = build_harness(config)
    trainer.run_update(
        [shifted_batch(2, seed=1), shifted_batch(2, seed=2)]
    )

    metrics = trainer.run_update(
        [shifted_batch(2, seed=3), shifted_batch(2, seed=4)]
    )

    assert metrics.update_index == 1
    assert metrics.completed_global_step == 2
    assert metrics.learning_rate == pytest.approx(config.learning_rate)
    assert state.global_step == 2
    assert state.micro_steps_seen == 4
    assert state.tokens_seen == 32
    assert state.samples_consumed == 8
    assert scheduler.next_update_index == 2
    state.validate_for_plan(plan)


def test_update_operations_have_exact_accumulation_call_counts(monkeypatch):
    config = training_config(accumulation_steps=3)
    trainer, model, optimizer, scheduler, _, _ = build_harness(config)
    optimizer.zero_grad = Mock(wraps=optimizer.zero_grad)
    optimizer.step = Mock(wraps=optimizer.step)
    scheduler.apply_for_update = Mock(wraps=scheduler.apply_for_update)
    original_clip = torch.nn.utils.clip_grad_norm_
    clip_spy = Mock(wraps=original_clip)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip_spy)

    trainer.run_update(
        [
            shifted_batch(2, seed=1),
            shifted_batch(2, seed=2),
            shifted_batch(2, seed=3),
        ]
    )

    assert model.forward_calls == 3
    assert optimizer.zero_grad.call_count == 1
    assert optimizer.step.call_count == 1
    assert scheduler.apply_for_update.call_count == 1
    assert clip_spy.call_count == 1


def test_accumulation_matches_one_large_batch_update():
    full_batch = shifted_batch(4, seed=17)
    config_large = training_config(micro_batch_size=4, accumulation_steps=1)
    config_accumulated = training_config(
        micro_batch_size=2,
        accumulation_steps=2,
    )
    torch.manual_seed(999)
    initial_model = ToyLanguageModel()
    large_model = ToyLanguageModel()
    accumulated_model = ToyLanguageModel()
    large_model.load_state_dict(initial_model.state_dict())
    accumulated_model.load_state_dict(initial_model.state_dict())
    large_trainer, _, _, _, large_state, _ = build_harness(
        config_large,
        model=large_model,
    )
    accumulated_trainer, _, _, _, accumulated_state, _ = build_harness(
        config_accumulated,
        model=accumulated_model,
    )

    large_metrics = large_trainer.run_update([full_batch])
    accumulated_metrics = accumulated_trainer.run_update(
        [
            (full_batch[0][:2], full_batch[1][:2]),
            (full_batch[0][2:], full_batch[1][2:]),
        ]
    )

    assert accumulated_metrics.raw_token_weighted_loss == pytest.approx(
        large_metrics.raw_token_weighted_loss,
        rel=1.0e-7,
        abs=1.0e-8,
    )
    for large_parameter, accumulated_parameter in zip(
        large_model.parameters(),
        accumulated_model.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(
            accumulated_parameter,
            large_parameter,
            rtol=2.0e-6,
            atol=2.0e-7,
        )
    assert large_state.tokens_seen == accumulated_state.tokens_seen == 16
    assert large_state.samples_consumed == accumulated_state.samples_consumed == 4
    assert large_state.global_step == accumulated_state.global_step == 1
    assert large_state.micro_steps_seen == 1
    assert accumulated_state.micro_steps_seen == 2


def test_reported_loss_is_token_weighted_across_micro_batches():
    config = training_config()
    trainer, model, _, _, _, _ = build_harness(config)
    batches = [shifted_batch(2, seed=31), shifted_batch(2, seed=32)]
    with torch.no_grad():
        losses = [
            float(model(inputs, targets).loss.detach().item())
            for inputs, targets in batches
        ]
    expected = sum(loss * targets.numel() for loss, (_, targets) in zip(
        losses,
        batches,
        strict=True,
    )) / sum(targets.numel() for _, targets in batches)

    metrics = trainer.run_update(batches)

    assert metrics.raw_token_weighted_loss == pytest.approx(expected)


def test_gradient_clipping_reports_pre_clip_norm_and_limits_gradients():
    config = training_config(gradient_clip=1.0e-3)
    trainer, model, _, _, _, _ = build_harness(config)

    metrics = trainer.run_update(
        [shifted_batch(2, seed=1), shifted_batch(2, seed=2)]
    )
    post_clip_norm = torch.linalg.vector_norm(
        torch.stack(
            [
                parameter.grad.detach().norm()
                for parameter in model.parameters()
                if parameter.grad is not None
            ]
        )
    )

    assert metrics.grad_norm_before_clip > config.gradient_clip
    assert post_clip_norm.item() <= config.gradient_clip * 1.001


def test_short_micro_batch_source_rolls_back_scheduler_and_state():
    config = training_config(accumulation_steps=2)
    trainer, _, optimizer, scheduler, state, _ = build_harness(config)
    optimizer.step = Mock(wraps=optimizer.step)
    initial_learning_rates = [group["lr"] for group in optimizer.param_groups]

    with pytest.raises(BatchContractError, match="ended before"):
        trainer.run_update([shifted_batch(2, seed=1)])

    assert optimizer.step.call_count == 0
    assert state.global_step == 0
    assert state.tokens_seen == 0
    assert scheduler.next_update_index == 0
    assert scheduler.last_learning_rate is None
    assert [group["lr"] for group in optimizer.param_groups] == (
        initial_learning_rates
    )
    assert all(parameter.grad is None for parameter in trainer.model.parameters())


@pytest.mark.parametrize(
    "bad_batch",
    (
        torch.zeros((2, 4), dtype=torch.long),
        (torch.zeros((2, 4), dtype=torch.long),),
        (
            torch.zeros((1, 4), dtype=torch.long),
            torch.zeros((1, 4), dtype=torch.long),
        ),
        (
            torch.zeros((2, 4), dtype=torch.float32),
            torch.zeros((2, 4), dtype=torch.long),
        ),
    ),
)
def test_bad_micro_batch_rolls_back_without_update(bad_batch):
    trainer, _, optimizer, scheduler, state, _ = build_harness(training_config())
    optimizer.step = Mock(wraps=optimizer.step)

    with pytest.raises(BatchContractError):
        trainer.run_update([bad_batch, shifted_batch(2, seed=2)])

    assert optimizer.step.call_count == 0
    assert state.global_step == 0
    assert scheduler.next_update_index == 0


def test_non_iterable_batch_source_is_rejected_before_scheduler_update():
    trainer, _, _, scheduler, state, _ = build_harness(training_config())

    with pytest.raises(BatchContractError, match="iterable"):
        trainer.run_update(None)

    assert state.global_step == 0
    assert scheduler.next_update_index == 0


def test_nonfinite_loss_stops_before_backward_step_and_state_commit():
    model = NonFiniteLossModel()
    trainer, _, optimizer, scheduler, state, _ = build_harness(
        training_config(),
        model=model,
    )
    optimizer.step = Mock(wraps=optimizer.step)
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    with pytest.raises(NonFiniteTrainingError, match="non-finite loss"):
        trainer.run_update(
            [shifted_batch(2, seed=1), shifted_batch(2, seed=2)]
        )

    assert optimizer.step.call_count == 0
    assert state.global_step == 0
    assert state.tokens_seen == 0
    assert scheduler.next_update_index == 0
    assert all(parameter.grad is None for parameter in model.parameters())
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)


def test_nonfinite_gradient_stops_before_optimizer_and_state_commit():
    trainer, model, optimizer, scheduler, state, _ = build_harness(
        training_config()
    )
    optimizer.step = Mock(wraps=optimizer.step)
    first_parameter = next(model.parameters())
    hook = first_parameter.register_hook(
        lambda gradient: torch.full_like(gradient, float("inf"))
    )

    try:
        with pytest.raises(NonFiniteTrainingError, match="gradient norm"):
            trainer.run_update(
                [shifted_batch(2, seed=1), shifted_batch(2, seed=2)]
            )
    finally:
        hook.remove()

    assert optimizer.step.call_count == 0
    assert state.global_step == 0
    assert scheduler.next_update_index == 0
    assert all(parameter.grad is None for parameter in model.parameters())


def test_missing_model_loss_is_rejected_without_state_commit():
    trainer, _, optimizer, scheduler, state, _ = build_harness(
        training_config(),
        model=MissingLossModel(),
    )
    optimizer.step = Mock(wraps=optimizer.step)

    with pytest.raises(TrainingStepError, match="Tensor loss"):
        trainer.run_update(
            [shifted_batch(2, seed=1), shifted_batch(2, seed=2)]
        )

    assert optimizer.step.call_count == 0
    assert state.global_step == 0
    assert scheduler.next_update_index == 0


def test_missing_gradient_is_rejected_without_optimizer_step():
    trainer, _, optimizer, scheduler, state, _ = build_harness(
        training_config(),
        model=UnusedParameterModel(),
    )
    optimizer.step = Mock(wraps=optimizer.step)

    with pytest.raises(TrainingStepError, match="missing gradients"):
        trainer.run_update(
            [shifted_batch(2, seed=1), shifted_batch(2, seed=2)]
        )

    assert optimizer.step.call_count == 0
    assert state.global_step == 0
    assert scheduler.next_update_index == 0


def test_constructor_rejects_scheduler_state_misalignment():
    config = training_config()
    plan = config.resolve()
    state = TrainerState(
        run_id="misaligned",
        global_step=1,
        micro_steps_seen=plan.gradient_accumulation_steps,
        tokens_seen=plan.tokens_per_update,
        samples_consumed=(
            plan.micro_batch_size * plan.gradient_accumulation_steps
        ),
    )

    with pytest.raises(TrainingStepError, match="disagree"):
        build_harness(config, state=state)


def test_constructor_rejects_completed_training_horizon():
    config = training_config()
    plan = config.resolve()
    state = TrainerState(
        run_id="complete",
        global_step=plan.total_updates,
        micro_steps_seen=(
            plan.total_updates * plan.gradient_accumulation_steps
        ),
        tokens_seen=plan.total_updates * plan.tokens_per_update,
        samples_consumed=(
            plan.total_updates
            * plan.micro_batch_size
            * plan.gradient_accumulation_steps
        ),
    )

    with pytest.raises(TrainingStepError, match="update horizon"):
        build_harness(config, state=state)


def test_constructor_rejects_precision_policy_mismatch(monkeypatch):
    config = training_config()
    plan = config.resolve()
    model = ToyLanguageModel()
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, plan)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    with pytest.raises(TrainingStepError, match="precision policy"):
        Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            state=TrainerState(run_id="mismatch"),
            config=config,
            plan=plan,
            precision=PrecisionPolicy(torch.device("cuda"), "fp32"),
        )


def test_constructor_rejects_optimizer_parameter_coverage_mismatch():
    config = training_config()
    plan = config.resolve()
    model = ToyLanguageModel()
    unrelated_model = nn.Linear(3, 2)
    optimizer = build_optimizer(unrelated_model, config)
    scheduler = build_scheduler(optimizer, config, plan)

    with pytest.raises(TrainingStepError, match="exactly cover"):
        Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            state=TrainerState(run_id="mismatch"),
            config=config,
            plan=plan,
            precision=PrecisionPolicy.from_config(config),
        )
