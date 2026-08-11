from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from scripts import train_gpt
from train import CheckpointIdentity, TrainerState, TrainingConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEBUG_PATH = PROJECT_ROOT / "configs" / "debug.yaml"


def synthetic_config(**overrides) -> TrainingConfig:
    values = {
        "project_name": "entry-test",
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
        "num_workers": 0,
        "pin_memory": False,
    }
    values.update(overrides)
    return replace(TrainingConfig.from_yaml(DEBUG_PATH), **values)


def identity() -> CheckpointIdentity:
    return CheckpointIdentity(
        model_config_sha256="1" * 64,
        tokenizer_sha256="2" * 64,
        dataset_manifest_sha256="3" * 64,
        dataset_config_fingerprint="4" * 64,
        source_commit="5" * 40,
        source_dirty=True,
    )


def write_tiny_cli_config(tmp_path: Path) -> Path:
    document = yaml.safe_load(DEBUG_PATH.read_text(encoding="utf-8"))
    document["model"].update(
        {
            "n_layer": 1,
            "n_head": 2,
            "n_embd": 8,
            "ffn_hidden": 32,
            "context_length": 4,
        }
    )
    document["training"].update(
        {
            "device": "cpu",
            "precision": "fp32",
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "max_steps": 4,
            "target_tokens": None,
            "warmup_steps": 1,
            "warmup_ratio": None,
            "log_interval": 1,
            "eval_interval": 2,
            "eval_batches": 1,
            "save_interval": 100,
            "num_workers": 0,
            "pin_memory": False,
            "run_dir": str(tmp_path / "runs"),
            "checkpoint_dir": str(tmp_path / "checkpoints"),
        }
    )
    path = tmp_path / "tiny-debug.yaml"
    path.write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )
    return path


def test_parser_exposes_only_the_frozen_operational_overrides():
    parser = train_gpt.build_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }

    assert option_strings == {
        "-h",
        "--help",
        "--config",
        "--manifest",
        "--run-id",
        "--device",
        "--precision",
        "--batch-source",
        "--stop-at-step",
        "--num-workers",
        "--resume",
        "--dry-run",
    }
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--manifest",
                "manifest.json",
                "--run-id",
                "run",
                "--learning-rate",
                "0.1",
            ]
        )


def test_training_config_overrides_do_not_change_scheduler_math(tmp_path):
    config = train_gpt.resolve_training_config(
        DEBUG_PATH,
        device="cpu",
        precision="fp32",
        num_workers=2,
    )

    assert config.device == "cpu"
    assert config.precision == "fp32"
    assert config.num_workers == 2
    assert config.max_steps == 200
    assert config.learning_rate == 3.0e-4
    assert config.warmup_steps == 20
    with pytest.raises(train_gpt.TrainingEntryError, match="non-negative"):
        train_gpt.resolve_training_config(
            DEBUG_PATH,
            device=None,
            precision=None,
            num_workers=-1,
        )


def test_stop_gate_never_changes_the_resolved_horizon():
    assert train_gpt._validate_stop_at_step(None, total_updates=200) == 200
    assert train_gpt._validate_stop_at_step(5, total_updates=200) == 5
    for value in (0, -1, True, 201):
        with pytest.raises(train_gpt.TrainingEntryError):
            train_gpt._validate_stop_at_step(value, total_updates=200)


def test_synthetic_stream_resume_reconstructs_exact_next_batch():
    config = synthetic_config()
    continuous = train_gpt.SyntheticTrainingBatches(
        state=TrainerState(run_id="continuous"),
        config=config,
    )
    first = next(continuous)
    expected_next = next(continuous)

    resumed_state = TrainerState(run_id="resumed")
    resumed_state.record_update(micro_steps=1, tokens=8, samples=2)
    resumed = train_gpt.SyntheticTrainingBatches(
        state=resumed_state,
        config=config,
    )
    actual_next = next(resumed)

    assert torch.equal(first[1][:, :-1], first[0][:, 1:])
    assert torch.equal(actual_next[0], expected_next[0])
    assert torch.equal(actual_next[1], expected_next[1])


def test_synthetic_validation_repeats_without_global_rng_side_effects():
    config = synthetic_config(eval_batches=2)
    batches = train_gpt.SyntheticValidationBatches(config=config)
    torch.manual_seed(9876)
    before = torch.get_rng_state().clone()

    first = list(batches)
    second = list(batches)

    assert torch.equal(torch.get_rng_state(), before)
    assert len(first) == len(second) == 2
    for expected, actual in zip(first, second, strict=True):
        assert torch.equal(expected[0], actual[0])
        assert torch.equal(expected[1], actual[1])


def test_resume_log_rejects_events_beyond_checkpoint_step():
    train_gpt._validate_resume_log(
        [
            {"event": "metadata"},
            {"event": "train_update", "step": 2},
        ],
        step=2,
    )
    with pytest.raises(train_gpt.TrainingEntryError, match="newer"):
        train_gpt._validate_resume_log(
            [{"event": "train_update", "step": 3}],
            step=2,
        )


def test_dry_run_builds_real_debug_components_without_writing_outputs(
    tmp_path,
    monkeypatch,
    capsys,
):
    document = yaml.safe_load(DEBUG_PATH.read_text(encoding="utf-8"))
    document["training"]["device"] = "cpu"
    document["training"]["precision"] = "fp32"
    document["training"]["run_dir"] = str(tmp_path / "runs")
    document["training"]["checkpoint_dir"] = str(tmp_path / "checkpoints")
    config_path = tmp_path / "debug.yaml"
    config_path.write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        train_gpt,
        "build_checkpoint_identity",
        lambda *_args, **_kwargs: identity(),
    )
    monkeypatch.setattr(
        train_gpt,
        "source_identity",
        lambda: ("5" * 40, True),
    )

    try:
        exit_code = train_gpt.main(
            [
                "--config",
                str(config_path),
                "--manifest",
                str(manifest_path),
                "--run-id",
                "dry-run-test",
                "--batch-source",
                "synthetic",
                "--dry-run",
            ]
        )
    finally:
        torch.use_deterministic_algorithms(False)

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Model parameters       : 2,508,032" in output
    assert "Sample causal shift    : True" in output
    assert "no update, log, or checkpoint written" in output
    assert not (tmp_path / "runs").exists()
    assert not (tmp_path / "checkpoints").exists()


def test_formal_entry_runs_then_resumes_same_run_with_new_stop_gate(
    tmp_path,
    monkeypatch,
):
    config_path = write_tiny_cli_config(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        train_gpt,
        "build_checkpoint_identity",
        lambda *_args, **_kwargs: identity(),
    )
    monkeypatch.setattr(
        train_gpt,
        "source_identity",
        lambda: ("5" * 40, True),
    )
    common = [
        "--config",
        str(config_path),
        "--manifest",
        str(manifest_path),
        "--run-id",
        "entry-resume",
        "--device",
        "cpu",
        "--precision",
        "fp32",
        "--batch-source",
        "synthetic",
        "--num-workers",
        "0",
    ]

    try:
        assert train_gpt.main([*common, "--stop-at-step", "1"]) == 0
        first_checkpoint = (
            tmp_path
            / "checkpoints"
            / "entry-resume"
            / "step-00000001.pt"
        )
        assert first_checkpoint.is_file()

        assert train_gpt.main(
            [
                *common,
                "--stop-at-step",
                "2",
                "--resume",
                str(first_checkpoint),
            ]
        ) == 0
    finally:
        torch.use_deterministic_algorithms(False)

    second_checkpoint = (
        tmp_path / "checkpoints" / "entry-resume" / "step-00000002.pt"
    )
    assert second_checkpoint.is_file()
    metrics_path = tmp_path / "runs" / "entry-resume" / "metrics.jsonl"
    events = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event"] for event in events] == [
        "run_start",
        "train_update",
        "checkpoint",
        "resume_start",
        "train_update",
        "evaluation",
        "checkpoint",
    ]
    assert [event["step"] for event in events] == [0, 1, 1, 1, 2, 2, 2]
    assert events[-1]["tokens_seen"] == 16
