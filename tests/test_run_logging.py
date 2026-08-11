from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from train import (
    CheckpointRecord,
    EvaluationMetrics,
    JsonlMetricLogger,
    MetricLoggingError,
    PrecisionPolicy,
    RunDirectoryError,
    TrainerState,
    UpdateMetrics,
    initialize_run_directory,
    open_existing_run_directory,
    read_metric_events,
    validate_run_id,
)


def initialize_fixture(tmp_path: Path, run_id: str = "day07-test"):
    return initialize_run_directory(
        tmp_path / "runs",
        run_id=run_id,
        resolved_config={
            "project": {"name": "small-gpt-debug"},
            "training": {"total_updates": 200},
        },
        metadata={"purpose": "unit-test"},
    )


def train_metrics(step: int = 1) -> UpdateMetrics:
    return UpdateMetrics(
        update_index=step - 1,
        completed_global_step=step,
        raw_token_weighted_loss=9.5,
        learning_rate=1.5e-5,
        grad_norm_before_clip=1.25,
        micro_steps=1,
        samples=2,
        tokens=8,
        elapsed_seconds=0.1,
        tokens_per_second=80.0,
    )


def evaluation_metrics(step: int = 1, perplexity: float = 100.0):
    return EvaluationMetrics(
        global_step=step,
        validation_loss=4.5,
        perplexity=perplexity,
        evaluated_batches=2,
        evaluated_tokens=12,
        elapsed_seconds=0.2,
    )


def state_at_step_one() -> TrainerState:
    state = TrainerState(run_id="day07-test")
    state.record_update(micro_steps=1, tokens=8, samples=2)
    return state


def test_initialize_run_directory_writes_stable_metadata_files(tmp_path):
    paths = initialize_fixture(tmp_path)

    assert paths.run_id == "day07-test"
    assert paths.run_dir == (tmp_path / "runs" / "day07-test").resolve()
    assert paths.resolved_config_path.is_file()
    assert paths.metadata_path.is_file()
    assert paths.metrics_path.is_file()
    assert paths.metrics_path.read_text(encoding="utf-8") == ""
    resolved = yaml.safe_load(
        paths.resolved_config_path.read_text(encoding="utf-8")
    )
    metadata = json.loads(paths.metadata_path.read_text(encoding="utf-8"))
    assert resolved["training"]["total_updates"] == 200
    assert metadata == {
        "purpose": "unit-test",
        "run_id": "day07-test",
        "schema_version": 1,
    }


def test_existing_run_directory_is_never_overwritten(tmp_path):
    paths = initialize_fixture(tmp_path)
    original_metadata = paths.metadata_path.read_bytes()

    with pytest.raises(RunDirectoryError, match="will not be overwritten"):
        initialize_fixture(tmp_path)

    assert paths.metadata_path.read_bytes() == original_metadata


@pytest.mark.parametrize(
    "run_id",
    (
        "",
        ".",
        "..",
        "has space",
        "../escape",
        "nested/run",
        "nested\\run",
        "-starts-with-symbol",
        "a" * 129,
    ),
)
def test_rejects_unsafe_run_id(run_id):
    with pytest.raises(RunDirectoryError, match="run_id"):
        validate_run_id(run_id)


@pytest.mark.parametrize("run_id", ("a", "day07-pilot-step3", "run.001_test"))
def test_accepts_portable_run_id(run_id):
    assert validate_run_id(run_id) == run_id


def test_rejects_conflicting_reserved_metadata(tmp_path):
    with pytest.raises(RunDirectoryError, match="conflicts"):
        initialize_run_directory(
            tmp_path / "runs",
            run_id="correct-id",
            resolved_config={},
            metadata={"run_id": "wrong-id"},
        )


def test_train_and_evaluation_events_are_flushed_as_strict_jsonl(tmp_path):
    paths = initialize_fixture(tmp_path)
    state = state_at_step_one()
    policy = PrecisionPolicy.resolve("cpu", "fp32")

    with JsonlMetricLogger(paths.metrics_path) as logger:
        logger.log_train_update(
            train_metrics(),
            state=state,
            precision=policy,
        )
        first_visible_lines = paths.metrics_path.read_text(
            encoding="utf-8"
        ).splitlines()
        assert len(first_visible_lines) == 1

        metrics = evaluation_metrics()
        state.record_evaluation(metrics.validation_loss)
        logger.log_evaluation(metrics, state=state)
        assert logger.is_closed is False

    assert logger.is_closed is True
    lines = paths.metrics_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    events = [json.loads(line) for line in lines]
    assert [event["event"] for event in events] == [
        "train_update",
        "evaluation",
    ]
    assert events[0] == {
        "device": "cpu",
        "elapsed_seconds": 0.1,
        "event": "train_update",
        "grad_norm": 1.25,
        "learning_rate": 1.5e-5,
        "micro_steps": 1,
        "precision": "fp32",
        "samples": 2,
        "step": 1,
        "tokens": 8,
        "tokens_per_second": 80.0,
        "tokens_seen": 8,
        "train_loss": 9.5,
    }
    assert events[1]["step"] == 1
    assert events[1]["tokens_seen"] == 8
    assert events[1]["validation_loss"] == 4.5
    assert events[1]["evaluated_tokens"] == 12


def test_infinite_perplexity_is_logged_as_json_null(tmp_path):
    paths = initialize_fixture(tmp_path)
    state = state_at_step_one()
    metrics = evaluation_metrics(perplexity=float("inf"))
    state.record_evaluation(metrics.validation_loss)

    with JsonlMetricLogger(paths.metrics_path) as logger:
        logger.log_evaluation(metrics, state=state)

    event = json.loads(paths.metrics_path.read_text(encoding="utf-8"))
    assert event["perplexity"] is None


@pytest.mark.parametrize(
    "event",
    (
        {},
        {"event": ""},
        {"event": "bad", "value": float("nan")},
        {"event": "bad", "value": object()},
    ),
)
def test_rejects_invalid_or_non_json_event(tmp_path, event):
    paths = initialize_fixture(tmp_path)

    with JsonlMetricLogger(paths.metrics_path) as logger:
        with pytest.raises(MetricLoggingError):
            logger.write_event(event)

    assert paths.metrics_path.read_text(encoding="utf-8") == ""


def test_rejects_step_mismatch_between_metrics_and_state(tmp_path):
    paths = initialize_fixture(tmp_path)

    with JsonlMetricLogger(paths.metrics_path) as logger:
        with pytest.raises(MetricLoggingError, match="does not match"):
            logger.log_train_update(
                train_metrics(step=2),
                state=state_at_step_one(),
                precision=PrecisionPolicy.resolve("cpu", "fp32"),
            )


def test_evaluation_must_be_recorded_in_state_before_logging(tmp_path):
    paths = initialize_fixture(tmp_path)

    with JsonlMetricLogger(paths.metrics_path) as logger:
        with pytest.raises(MetricLoggingError, match="record evaluation"):
            logger.log_evaluation(
                evaluation_metrics(),
                state=state_at_step_one(),
            )


def test_closed_logger_rejects_more_events(tmp_path):
    paths = initialize_fixture(tmp_path)
    logger = JsonlMetricLogger(paths.metrics_path)
    logger.close()
    logger.close()

    with pytest.raises(MetricLoggingError, match="closed"):
        logger.write_event({"event": "late"})


def test_append_mode_preserves_existing_complete_lines(tmp_path):
    paths = initialize_fixture(tmp_path)
    with JsonlMetricLogger(paths.metrics_path) as logger:
        logger.write_event({"event": "first", "step": 1})
    with JsonlMetricLogger(paths.metrics_path, append=True) as logger:
        logger.write_event({"event": "second", "step": 2})

    events = [
        json.loads(line)
        for line in paths.metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event"] for event in events] == ["first", "second"]


def test_checkpoint_event_is_flushed_after_state_save_commit(tmp_path):
    paths = initialize_fixture(tmp_path)
    state = state_at_step_one()
    state.record_checkpoint()
    record = CheckpointRecord(
        path=tmp_path / "checkpoints" / "step-00000001.pt",
        file_size=1234,
        global_step=1,
        tokens_seen=8,
    )

    with JsonlMetricLogger(paths.metrics_path) as logger:
        logger.log_checkpoint(
            record,
            state=state,
            checkpoint_path="checkpoints/day07-test/step-00000001.pt",
            elapsed_seconds=0.25,
        )

    event = json.loads(paths.metrics_path.read_text(encoding="utf-8"))
    assert event == {
        "checkpoint_bytes": 1234,
        "checkpoint_path": "checkpoints/day07-test/step-00000001.pt",
        "event": "checkpoint",
        "save_elapsed_seconds": 0.25,
        "step": 1,
        "tokens_seen": 8,
    }


def test_checkpoint_event_rejects_uncommitted_or_mismatched_state(tmp_path):
    paths = initialize_fixture(tmp_path)
    state = state_at_step_one()
    record = CheckpointRecord(
        path=tmp_path / "step.pt",
        file_size=10,
        global_step=1,
        tokens_seen=8,
    )

    with JsonlMetricLogger(paths.metrics_path) as logger:
        with pytest.raises(MetricLoggingError, match="record checkpoint"):
            logger.log_checkpoint(
                record,
                state=state,
                checkpoint_path="step.pt",
                elapsed_seconds=0.1,
            )


def test_open_existing_run_validates_identity_without_modifying_files(tmp_path):
    paths = initialize_fixture(tmp_path)
    with JsonlMetricLogger(paths.metrics_path) as logger:
        logger.write_event({"event": "run_start", "step": 0})
        logger.write_event({"event": "train_update", "step": 1})
    before = {
        path: path.read_bytes()
        for path in (
            paths.resolved_config_path,
            paths.metadata_path,
            paths.metrics_path,
        )
    }

    opened = open_existing_run_directory(
        tmp_path / "runs",
        run_id="day07-test",
        expected_resolved_config={
            "project": {"name": "small-gpt-debug"},
            "training": {"total_updates": 200},
        },
    )

    assert opened == paths
    assert [event["step"] for event in read_metric_events(paths.metrics_path)] == [
        0,
        1,
    ]
    assert all(path.read_bytes() == contents for path, contents in before.items())


def test_open_existing_run_rejects_config_mismatch(tmp_path):
    paths = initialize_fixture(tmp_path)

    with pytest.raises(RunDirectoryError, match="does not match"):
        open_existing_run_directory(
            tmp_path / "runs",
            run_id="day07-test",
            expected_resolved_config={"different": True},
        )

    assert paths.metadata_path.is_file()


@pytest.mark.parametrize(
    "contents, message",
    (
        ('{"event":"ok","step":1}\nnot-json\n', "strict JSON"),
        ('{"event":"ok","step":2}\n{"event":"back","step":1}\n', "backward"),
        ('{"event":"bad","step":NaN}\n', "strict JSON"),
        ('{"event":"ok","step":0}\n\n', "blank"),
        ('{"event":"unterminated","step":0}', "newline-terminated"),
    ),
)
def test_metric_reader_rejects_corrupt_or_non_monotonic_jsonl(
    tmp_path,
    contents,
    message,
):
    path = tmp_path / "metrics.jsonl"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(RunDirectoryError, match=message):
        read_metric_events(path)
