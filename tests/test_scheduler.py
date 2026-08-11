import math
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from train import (
    SchedulerConfigError,
    SchedulerStateError,
    TrainingConfig,
    WarmupCosineScheduler,
    build_scheduler,
    warmup_cosine_lr,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEBUG_PATH = PROJECT_ROOT / "configs" / "debug.yaml"
MAX_LR = 3.0e-4
MIN_LR = 3.0e-5


def lr_at(update_index, **overrides):
    values = {
        "total_updates": 200,
        "warmup_updates": 20,
        "max_learning_rate": MAX_LR,
        "min_learning_rate": MIN_LR,
    }
    values.update(overrides)
    return warmup_cosine_lr(update_index, **values)


def toy_optimizer():
    first = torch.nn.Parameter(torch.tensor([1.0]))
    second = torch.nn.Parameter(torch.tensor([2.0]))
    optimizer = torch.optim.AdamW(
        [
            {"params": [first], "weight_decay": 0.1},
            {"params": [second], "weight_decay": 0.0},
        ],
        lr=MAX_LR,
    )
    return optimizer


def toy_scheduler(optimizer=None):
    if optimizer is None:
        optimizer = toy_optimizer()
    return WarmupCosineScheduler(
        optimizer,
        total_updates=200,
        warmup_updates=20,
        max_learning_rate=MAX_LR,
        min_learning_rate=MIN_LR,
    )


def test_debug_schedule_exact_boundary_learning_rates():
    assert lr_at(0) == pytest.approx(MAX_LR / 20)
    assert lr_at(9) == pytest.approx(MAX_LR / 2)
    assert lr_at(19) == pytest.approx(MAX_LR)
    assert lr_at(20) == pytest.approx(MAX_LR)
    assert lr_at(199) == pytest.approx(MIN_LR)
    assert lr_at(200) == pytest.approx(MIN_LR)
    assert lr_at(10_000) == pytest.approx(MIN_LR)


def test_cosine_midpoint_matches_frozen_formula():
    update_index = 110
    progress = (update_index - 20) / (180 - 1)
    expected = MIN_LR + 0.5 * (MAX_LR - MIN_LR) * (
        1.0 + math.cos(math.pi * progress)
    )

    assert lr_at(update_index) == pytest.approx(expected)


def test_full_debug_curve_is_finite_bounded_and_monotonic_by_phase():
    learning_rates = [lr_at(update_index) for update_index in range(200)]

    assert all(math.isfinite(value) for value in learning_rates)
    assert all(0.0 < value <= MAX_LR for value in learning_rates[:20])
    assert all(MIN_LR <= value <= MAX_LR for value in learning_rates[20:])
    assert learning_rates[:20] == sorted(learning_rates[:20])
    assert learning_rates[20:] == sorted(learning_rates[20:], reverse=True)


def test_zero_warmup_starts_at_peak_and_ends_at_minimum():
    values = [
        lr_at(
            update_index,
            total_updates=5,
            warmup_updates=0,
        )
        for update_index in range(5)
    ]

    assert values[0] == pytest.approx(MAX_LR)
    assert values[-1] == pytest.approx(MIN_LR)
    assert values == sorted(values, reverse=True)


def test_single_cosine_update_uses_terminal_minimum():
    assert lr_at(
        0,
        total_updates=1,
        warmup_updates=0,
    ) == pytest.approx(MIN_LR)
    assert lr_at(
        1,
        total_updates=2,
        warmup_updates=1,
    ) == pytest.approx(MIN_LR)


@pytest.mark.parametrize(
    "overrides",
    (
        {"total_updates": 0},
        {"total_updates": True},
        {"warmup_updates": -1},
        {"warmup_updates": True},
        {"warmup_updates": 200},
        {"max_learning_rate": 0.0},
        {"max_learning_rate": float("nan")},
        {"min_learning_rate": -1.0e-5},
        {"min_learning_rate": 4.0e-4},
    ),
)
def test_rejects_invalid_schedule_definition(overrides):
    with pytest.raises(SchedulerConfigError):
        lr_at(0, **overrides)


@pytest.mark.parametrize("update_index", (-1, True, 1.5, "1"))
def test_rejects_invalid_update_index(update_index):
    with pytest.raises(SchedulerConfigError, match="update_index"):
        lr_at(update_index)


def test_apply_sets_every_optimizer_group_and_advances_once():
    optimizer = toy_optimizer()
    scheduler = toy_scheduler(optimizer)

    learning_rate = scheduler.apply_for_update(0)

    assert learning_rate == pytest.approx(MAX_LR / 20)
    assert scheduler.last_applied_update == 0
    assert scheduler.next_update_index == 1
    assert all(
        group["lr"] == pytest.approx(learning_rate)
        for group in optimizer.param_groups
    )


def test_apply_rejects_duplicate_skipped_or_out_of_horizon_update():
    scheduler = toy_scheduler()
    scheduler.apply_for_update(0)

    with pytest.raises(SchedulerStateError, match="exactly once"):
        scheduler.apply_for_update(0)
    with pytest.raises(SchedulerStateError, match="exactly once"):
        scheduler.apply_for_update(2)

    terminal = toy_scheduler()
    terminal.last_applied_update = 198
    terminal.last_learning_rate = terminal.lr_for_update(198)
    terminal.apply_for_update(199)
    with pytest.raises(SchedulerStateError, match="horizon"):
        terminal.apply_for_update(200)


def test_scheduler_state_round_trip_preserves_the_next_learning_rate():
    continuous = toy_scheduler()
    continuous.apply_for_update(0)
    continuous.apply_for_update(1)
    payload = continuous.state_dict()

    resumed_optimizer = toy_optimizer()
    resumed = toy_scheduler(resumed_optimizer)
    resumed.load_state_dict(payload)

    assert resumed.last_applied_update == 1
    assert resumed.next_update_index == 2
    assert all(
        group["lr"] == pytest.approx(payload["last_learning_rate"])
        for group in resumed_optimizer.param_groups
    )
    assert resumed.apply_for_update(2) == pytest.approx(
        continuous.apply_for_update(2)
    )


def test_pristine_scheduler_state_round_trip():
    original = toy_scheduler()
    restored = toy_scheduler()

    restored.load_state_dict(original.state_dict())

    assert restored.next_update_index == 0
    assert restored.last_learning_rate is None


@pytest.mark.parametrize("field_name", ("total_updates", "last_learning_rate"))
def test_scheduler_state_rejects_missing_fields(field_name):
    payload = toy_scheduler().state_dict()
    del payload[field_name]

    with pytest.raises(SchedulerStateError, match="missing fields"):
        toy_scheduler().load_state_dict(payload)


def test_scheduler_state_rejects_unknown_fields():
    payload = toy_scheduler().state_dict()
    payload["global_step"] = 0

    with pytest.raises(SchedulerStateError, match="unknown fields"):
        toy_scheduler().load_state_dict(payload)


@pytest.mark.parametrize("schema_version", (0, 2, True))
def test_scheduler_state_rejects_bad_schema_version(schema_version):
    payload = toy_scheduler().state_dict()
    payload["schema_version"] = schema_version

    with pytest.raises(SchedulerStateError, match="schema version"):
        toy_scheduler().load_state_dict(payload)


def test_scheduler_state_rejects_schedule_identity_mismatch():
    payload = toy_scheduler().state_dict()
    payload["total_updates"] = 201

    with pytest.raises(SchedulerStateError, match="active plan"):
        toy_scheduler().load_state_dict(payload)


def test_scheduler_state_rejects_learning_rate_mismatch():
    scheduler = toy_scheduler()
    scheduler.apply_for_update(0)
    payload = scheduler.state_dict()
    payload["last_learning_rate"] *= 2

    with pytest.raises(SchedulerStateError, match="inconsistent"):
        toy_scheduler().load_state_dict(payload)


def test_build_scheduler_uses_resolved_debug_horizon():
    config = TrainingConfig.from_yaml(DEBUG_PATH)
    plan = config.resolve()
    scheduler = build_scheduler(toy_optimizer(), config, plan)

    assert scheduler.total_updates == 200
    assert scheduler.warmup_updates == 20
    assert scheduler.lr_for_update(0) == pytest.approx(MAX_LR / 20)
    assert scheduler.lr_for_update(199) == pytest.approx(MIN_LR)


def test_build_scheduler_rejects_mismatched_project_identity():
    config = TrainingConfig.from_yaml(DEBUG_PATH)
    wrong_plan = replace(config.resolve(), project_name="different-project")

    with pytest.raises(SchedulerConfigError, match="different projects"):
        build_scheduler(toy_optimizer(), config, wrong_plan)


def test_build_scheduler_rejects_mismatched_execution_plan():
    config = TrainingConfig.from_yaml(DEBUG_PATH)
    wrong_plan = replace(config.resolve(), total_updates=201)

    with pytest.raises(SchedulerConfigError, match="same execution"):
        build_scheduler(toy_optimizer(), config, wrong_plan)


def test_build_scheduler_rejects_wrong_types():
    config = TrainingConfig.from_yaml(DEBUG_PATH)
    plan = config.resolve()

    with pytest.raises(TypeError, match="TrainingConfig"):
        build_scheduler(toy_optimizer(), object(), plan)
    with pytest.raises(TypeError, match="ResolvedTrainingPlan"):
        build_scheduler(toy_optimizer(), config, object())


def test_loading_state_does_not_retain_mutable_payload_reference():
    scheduler = toy_scheduler()
    scheduler.apply_for_update(0)
    payload = deepcopy(scheduler.state_dict())
    restored = toy_scheduler()

    restored.load_state_dict(payload)
    payload["last_applied_update"] = 99

    assert restored.last_applied_update == 0
