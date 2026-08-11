from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from train import (
    EvaluationError,
    PrecisionPolicy,
    evaluate_model,
)


class ControlledLossModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(1.0))
        self.forward_training_modes: list[bool] = []
        self.grad_enabled_values: list[bool] = []

    def forward(self, input_ids, targets=None):
        self.forward_training_modes.append(self.training)
        self.grad_enabled_values.append(torch.is_grad_enabled())
        assert targets is not None
        loss = self.anchor * 0.0 + targets[0, 0].float()
        logits = torch.zeros(
            (*input_ids.shape, 2),
            device=input_ids.device,
        )
        return SimpleNamespace(logits=logits, loss=loss)


class NonFiniteEvaluationModel(ControlledLossModel):
    def forward(self, input_ids, targets=None):
        output = super().forward(input_ids, targets)
        return SimpleNamespace(
            logits=output.logits,
            loss=output.loss * torch.tensor(float("nan")),
        )


def batch(loss_value: int, batch_size: int, sequence_length: int = 4):
    input_ids = torch.zeros(
        (batch_size, sequence_length),
        dtype=torch.long,
    )
    targets = torch.full_like(input_ids, loss_value)
    return input_ids, targets


def cpu_policy() -> PrecisionPolicy:
    return PrecisionPolicy.resolve("cpu", "fp32")


def test_evaluation_uses_token_weighted_loss_for_uneven_batches():
    model = ControlledLossModel()
    batches = [batch(2, 2), batch(10, 1)]

    metrics = evaluate_model(
        model,
        batches,
        precision=cpu_policy(),
        global_step=7,
        max_batches=None,
    )

    expected_loss = (2.0 * 8 + 10.0 * 4) / 12
    assert metrics.global_step == 7
    assert metrics.validation_loss == pytest.approx(expected_loss)
    assert metrics.perplexity == pytest.approx(math.exp(expected_loss))
    assert metrics.evaluated_batches == 2
    assert metrics.evaluated_tokens == 12
    assert metrics.elapsed_seconds > 0.0
    assert metrics.to_dict()["validation_loss"] == pytest.approx(expected_loss)


def test_evaluation_enters_eval_no_grad_and_restores_train_mode():
    model = ControlledLossModel()
    model.train()

    evaluate_model(
        model,
        [batch(2, 2)],
        precision=cpu_policy(),
        global_step=0,
        max_batches=None,
    )

    assert model.training is True
    assert model.forward_training_modes == [False]
    assert model.grad_enabled_values == [False]
    assert model.anchor.grad is None


def test_evaluation_preserves_existing_eval_mode_and_parameter_gradients():
    model = ControlledLossModel()
    model.eval()
    model.anchor.grad = torch.tensor(3.0)

    evaluate_model(
        model,
        [batch(2, 2)],
        precision=cpu_policy(),
        global_step=0,
        max_batches=None,
    )

    assert model.training is False
    torch.testing.assert_close(model.anchor.grad, torch.tensor(3.0))


def test_max_batches_stops_without_consuming_an_extra_batch():
    model = ControlledLossModel()

    metrics = evaluate_model(
        model,
        iter([batch(2, 2), batch(10, 1)]),
        precision=cpu_policy(),
        global_step=0,
        max_batches=1,
    )

    assert metrics.validation_loss == pytest.approx(2.0)
    assert metrics.evaluated_batches == 1
    assert metrics.evaluated_tokens == 8
    assert len(model.forward_training_modes) == 1


def test_repeated_evaluation_is_exact_for_fixed_batches():
    model = ControlledLossModel()
    batches = [batch(2, 2), batch(10, 1)]

    first = evaluate_model(
        model,
        batches,
        precision=cpu_policy(),
        global_step=3,
        max_batches=None,
    )
    second = evaluate_model(
        model,
        batches,
        precision=cpu_policy(),
        global_step=3,
        max_batches=None,
    )

    assert first.validation_loss == second.validation_loss
    assert first.perplexity == second.perplexity
    assert first.evaluated_tokens == second.evaluated_tokens


def test_large_finite_loss_reports_infinite_perplexity_without_crashing():
    metrics = evaluate_model(
        ControlledLossModel(),
        [batch(1_000, 1)],
        precision=cpu_policy(),
        global_step=0,
        max_batches=None,
    )

    assert metrics.validation_loss == pytest.approx(1_000.0)
    assert metrics.perplexity == math.inf


def test_nonfinite_loss_restores_mode_and_fails_clearly():
    model = NonFiniteEvaluationModel()
    model.train()

    with pytest.raises(EvaluationError, match="non-finite"):
        evaluate_model(
            model,
            [batch(2, 2)],
            precision=cpu_policy(),
            global_step=0,
            max_batches=None,
        )

    assert model.training is True


@pytest.mark.parametrize("empty_batches", ([], iter(())))
def test_empty_evaluation_is_rejected(empty_batches):
    with pytest.raises(EvaluationError, match="no complete batches"):
        evaluate_model(
            ControlledLossModel(),
            empty_batches,
            precision=cpu_policy(),
            global_step=0,
            max_batches=None,
        )


@pytest.mark.parametrize("max_batches", (0, -1, True, 1.5))
def test_rejects_invalid_max_batches(max_batches):
    with pytest.raises(EvaluationError, match="max_batches"):
        evaluate_model(
            ControlledLossModel(),
            [batch(2, 2)],
            precision=cpu_policy(),
            global_step=0,
            max_batches=max_batches,
        )


@pytest.mark.parametrize("global_step", (-1, True, 1.5))
def test_rejects_invalid_global_step(global_step):
    with pytest.raises(EvaluationError, match="global_step"):
        evaluate_model(
            ControlledLossModel(),
            [batch(2, 2)],
            precision=cpu_policy(),
            global_step=global_step,
            max_batches=None,
        )


@pytest.mark.parametrize(
    "bad_batch",
    (
        torch.zeros((2, 4), dtype=torch.long),
        (torch.zeros((2, 4), dtype=torch.long),),
        (
            torch.zeros((2, 4), dtype=torch.float32),
            torch.zeros((2, 4), dtype=torch.long),
        ),
        (
            torch.zeros((2, 4), dtype=torch.long),
            torch.zeros((1, 4), dtype=torch.long),
        ),
    ),
)
def test_rejects_invalid_evaluation_batch(bad_batch):
    with pytest.raises(EvaluationError, match="batch"):
        evaluate_model(
            ControlledLossModel(),
            [bad_batch],
            precision=cpu_policy(),
            global_step=0,
            max_batches=None,
        )


def test_rejects_non_iterable_batch_source():
    with pytest.raises(EvaluationError, match="iterable"):
        evaluate_model(
            ControlledLossModel(),
            None,
            precision=cpu_policy(),
            global_step=0,
            max_batches=None,
        )
