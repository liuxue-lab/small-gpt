from copy import deepcopy
from pathlib import Path

import pytest

from train import (
    TRAINER_STATE_SCHEMA_VERSION,
    TrainerState,
    TrainerStateError,
    TrainingConfig,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEBUG_PATH = PROJECT_ROOT / "configs" / "debug.yaml"


def debug_plan():
    return TrainingConfig.from_yaml(DEBUG_PATH).resolve()


def test_new_state_is_valid_for_debug_plan():
    state = TrainerState(run_id="day07-debug")

    state.validate_for_plan(debug_plan())

    assert state.schema_version == TRAINER_STATE_SCHEMA_VERSION
    assert state.global_step == 0
    assert state.micro_steps_seen == 0
    assert state.tokens_seen == 0
    assert state.samples_consumed == 0


def test_record_update_commits_all_training_counters_together():
    state = TrainerState(run_id="day07-debug")

    state.record_update(micro_steps=1, tokens=512, samples=4)
    state.record_update(micro_steps=1, tokens=512, samples=4)

    assert state.global_step == 2
    assert state.micro_steps_seen == 2
    assert state.tokens_seen == 1_024
    assert state.samples_consumed == 8
    assert state.batches_consumed_in_epoch == 2
    state.validate_for_plan(debug_plan())


def test_evaluation_and_checkpoint_metadata_follow_completed_step():
    state = TrainerState(run_id="day07-debug")
    state.record_update(micro_steps=1, tokens=512, samples=4)

    state.record_evaluation(9.5)
    state.record_evaluation(9.7)
    state.record_checkpoint()

    assert state.best_validation_loss == pytest.approx(9.5)
    assert state.last_validation_loss == pytest.approx(9.7)
    assert state.last_eval_step == 1
    assert state.last_save_step == 1


def test_better_evaluation_updates_best_loss():
    state = TrainerState(run_id="day07-debug")

    state.record_evaluation(9.5)
    state.record_evaluation(8.25)

    assert state.best_validation_loss == pytest.approx(8.25)
    assert state.last_validation_loss == pytest.approx(8.25)


def test_state_dict_round_trip_is_exact_and_independent():
    original = TrainerState(run_id="day07-debug")
    original.record_update(micro_steps=1, tokens=512, samples=4)
    original.record_evaluation(9.5)
    original.record_checkpoint()

    payload = original.state_dict()
    restored = TrainerState.from_state_dict(payload)
    payload["global_step"] = 99

    assert restored.state_dict() == original.state_dict()
    assert restored.global_step == 1


@pytest.mark.parametrize("field_name", ("global_step", "tokens_seen"))
def test_state_dict_rejects_missing_fields(field_name):
    payload = TrainerState(run_id="day07-debug").state_dict()
    del payload[field_name]

    with pytest.raises(TrainerStateError, match="missing fields"):
        TrainerState.from_state_dict(payload)


def test_state_dict_rejects_unknown_fields():
    payload = TrainerState(run_id="day07-debug").state_dict()
    payload["micro_step"] = 0

    with pytest.raises(TrainerStateError, match="unknown fields"):
        TrainerState.from_state_dict(payload)


@pytest.mark.parametrize(
    "invalid_value",
    (-1, True, 1.5, "1"),
)
def test_rejects_invalid_counter_values(invalid_value):
    with pytest.raises(TrainerStateError, match="global_step"):
        TrainerState(run_id="day07-debug", global_step=invalid_value)


@pytest.mark.parametrize("run_id", ("", "   ", None, 42))
def test_rejects_invalid_run_id(run_id):
    with pytest.raises(TrainerStateError, match="run_id"):
        TrainerState(run_id=run_id)


@pytest.mark.parametrize("schema_version", (0, 2, True))
def test_rejects_unsupported_or_non_integer_schema(schema_version):
    with pytest.raises(TrainerStateError, match="schema_version|schema version"):
        TrainerState(run_id="day07-debug", schema_version=schema_version)


@pytest.mark.parametrize("field_name", ("last_eval_step", "last_save_step"))
def test_rejects_event_step_ahead_of_global_step(field_name):
    with pytest.raises(TrainerStateError, match=field_name):
        TrainerState(run_id="day07-debug", **{field_name: 1})


@pytest.mark.parametrize(
    "invalid_loss",
    (-0.1, float("nan"), float("inf"), True, None),
)
def test_rejects_invalid_validation_loss(invalid_loss):
    state = TrainerState(run_id="day07-debug")

    with pytest.raises(TrainerStateError, match="loss"):
        state.record_evaluation(invalid_loss)


def test_rejects_partial_validation_metadata():
    with pytest.raises(TrainerStateError, match="must either both"):
        TrainerState(
            run_id="day07-debug",
            best_validation_loss=9.0,
        )


def test_rejects_best_loss_worse_than_last_loss():
    with pytest.raises(TrainerStateError, match="cannot exceed"):
        TrainerState(
            run_id="day07-debug",
            best_validation_loss=10.0,
            last_validation_loss=9.0,
        )


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    (
        ("micro_steps_seen", 2, "micro_steps_seen"),
        ("tokens_seen", 511, "tokens_seen"),
        ("samples_consumed", 3, "samples_consumed"),
    ),
)
def test_plan_validation_rejects_counter_drift(field_name, value, message):
    state = TrainerState(run_id="day07-debug")
    state.record_update(micro_steps=1, tokens=512, samples=4)
    payload = state.state_dict()
    payload[field_name] = value
    drifted = TrainerState.from_state_dict(payload)

    with pytest.raises(TrainerStateError, match=message):
        drifted.validate_for_plan(debug_plan())


def test_plan_validation_rejects_step_beyond_horizon():
    plan = debug_plan()
    state = TrainerState(
        run_id="day07-debug",
        global_step=plan.total_updates + 1,
        micro_steps_seen=plan.total_updates + 1,
        tokens_seen=(plan.total_updates + 1) * plan.tokens_per_update,
        samples_consumed=(plan.total_updates + 1) * plan.micro_batch_size,
    )

    with pytest.raises(TrainerStateError, match="total updates"):
        state.validate_for_plan(plan)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    (
        ("micro_steps", 0),
        ("tokens", True),
        ("samples", -1),
    ),
)
def test_record_update_rejects_invalid_delta(field_name, invalid_value):
    state = TrainerState(run_id="day07-debug")
    arguments = {"micro_steps": 1, "tokens": 512, "samples": 4}
    arguments[field_name] = invalid_value
    before = deepcopy(state.state_dict())

    with pytest.raises(TrainerStateError, match=field_name):
        state.record_update(**arguments)

    assert state.state_dict() == before


def test_validate_for_plan_rejects_wrong_object_type():
    state = TrainerState(run_id="day07-debug")

    with pytest.raises(TypeError, match="ResolvedTrainingPlan"):
        state.validate_for_plan({})
