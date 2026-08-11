from __future__ import annotations

import json
import random
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from train import (
    CHECKPOINT_FORMAT_NAME,
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointCompatibilityError,
    CheckpointIdentity,
    CheckpointIdentityError,
    CheckpointLoadError,
    CheckpointSaveError,
    PrecisionPolicy,
    Trainer,
    TrainerState,
    TrainingConfig,
    build_checkpoint_identity,
    build_optimizer,
    build_scheduler,
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEBUG_PATH = PROJECT_ROOT / "configs" / "debug.yaml"


class TinyDropoutLanguageModel(nn.Module):
    def __init__(self, vocab_size: int = 32, embedding_size: int = 8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.dropout = nn.Dropout(p=0.25)
        self.projection = nn.Linear(embedding_size, vocab_size)

    def forward(self, input_ids, targets=None):
        hidden = self.dropout(self.embedding(input_ids))
        logits = self.projection(hidden)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
            )
        return SimpleNamespace(logits=logits, loss=loss)


def checkpoint_config() -> TrainingConfig:
    return replace(
        TrainingConfig.from_yaml(DEBUG_PATH),
        project_name="checkpoint-test",
        context_length=4,
        vocab_size=32,
        device="cpu",
        precision="fp32",
        micro_batch_size=2,
        gradient_accumulation_steps=1,
        max_steps=4,
        target_tokens=None,
        warmup_steps=1,
        warmup_ratio=None,
        num_workers=0,
        pin_memory=False,
    )


def checkpoint_identity() -> CheckpointIdentity:
    return CheckpointIdentity(
        model_config_sha256="1" * 64,
        tokenizer_sha256="2" * 64,
        dataset_manifest_sha256="3" * 64,
        dataset_config_fingerprint="4" * 64,
        source_commit="5" * 40,
        source_dirty=True,
    )


def resolved_config(config: TrainingConfig) -> dict:
    return {
        "schema_version": 1,
        "model": {"architecture": "tiny-dropout-lm"},
        "training": config.to_dict(),
        "plan": config.resolve().to_dict(),
        "runtime": {"device": "cpu", "precision": "fp32"},
    }


def fixed_batches() -> list[tuple[torch.Tensor, torch.Tensor]]:
    batches = []
    base = torch.arange(8, dtype=torch.long).reshape(2, 4)
    for offset in range(4):
        input_ids = (base + offset * 3) % 32
        targets = (input_ids + 1) % 32
        batches.append((input_ids, targets))
    return batches


@dataclass
class Harness:
    config: TrainingConfig
    model: TinyDropoutLanguageModel
    optimizer: torch.optim.Optimizer
    scheduler: object
    state: TrainerState
    precision: PrecisionPolicy
    trainer: Trainer


def build_components(config: TrainingConfig):
    plan = config.resolve()
    precision = PrecisionPolicy.from_config(config)
    model = TinyDropoutLanguageModel(vocab_size=config.vocab_size)
    model.to(precision.device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, plan)
    return model, optimizer, scheduler, precision


def build_harness(config: TrainingConfig, run_id: str = "checkpoint-run") -> Harness:
    model, optimizer, scheduler, precision = build_components(config)
    state = TrainerState(run_id=run_id)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        state=state,
        config=config,
        plan=config.resolve(),
        precision=precision,
    )
    return Harness(
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        state=state,
        precision=precision,
        trainer=trainer,
    )


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def consume_external_rng() -> tuple[float, float, torch.Tensor]:
    return (
        random.random(),
        float(np.random.random()),
        torch.rand(3),
    )


def run_one_update(harness: Harness, batch):
    metrics = harness.trainer.run_update(iter([batch]))
    rng_values = consume_external_rng()
    return metrics, rng_values


def assert_nested_equal(first, second, path: str = "root") -> None:
    if isinstance(first, torch.Tensor):
        assert isinstance(second, torch.Tensor), path
        assert first.device == second.device, path
        assert first.dtype == second.dtype, path
        assert torch.equal(first, second), path
        return
    if isinstance(first, dict):
        assert isinstance(second, dict), path
        assert first.keys() == second.keys(), path
        for key in first:
            assert_nested_equal(first[key], second[key], f"{path}.{key}")
        return
    if isinstance(first, (list, tuple)):
        assert isinstance(second, type(first)), path
        assert len(first) == len(second), path
        for index, (first_value, second_value) in enumerate(
            zip(first, second, strict=True)
        ):
            assert_nested_equal(
                first_value,
                second_value,
                f"{path}[{index}]",
            )
        return
    assert first == second, path


def assert_update_equivalent(first, second) -> None:
    assert first.update_index == second.update_index
    assert first.completed_global_step == second.completed_global_step
    assert first.raw_token_weighted_loss == second.raw_token_weighted_loss
    assert first.learning_rate == second.learning_rate
    assert first.grad_norm_before_clip == second.grad_norm_before_clip
    assert first.micro_steps == second.micro_steps
    assert first.samples == second.samples
    assert first.tokens == second.tokens


def save_harness(harness: Harness, path: Path):
    return save_checkpoint(
        path,
        model=harness.model,
        optimizer=harness.optimizer,
        scheduler=harness.scheduler,
        state=harness.state,
        plan=harness.config.resolve(),
        resolved_config=resolved_config(harness.config),
        identity=checkpoint_identity(),
    )


def load_into_new_components(
    config: TrainingConfig,
    path: Path,
    *,
    identity: CheckpointIdentity | None = None,
    active_config: dict | None = None,
    run_id: str = "checkpoint-run",
):
    model, optimizer, scheduler, precision = build_components(config)
    loaded = load_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        plan=config.resolve(),
        resolved_config=(
            resolved_config(config) if active_config is None else active_config
        ),
        identity=checkpoint_identity() if identity is None else identity,
        expected_run_id=run_id,
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        state=loaded.state,
        config=config,
        plan=config.resolve(),
        precision=precision,
    )
    return loaded, Harness(
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        state=loaded.state,
        precision=precision,
        trainer=trainer,
    )


def test_rng_capture_and_restore_is_exact_for_cpu_sources():
    seed_all(2468)
    captured = capture_rng_state(include_cuda=False)
    expected = consume_external_rng()

    _ = [consume_external_rng() for _ in range(3)]
    restore_rng_state(captured, require_cuda=False)
    actual = consume_external_rng()

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_atomic_save_writes_complete_payload_and_commits_save_step(tmp_path):
    seed_all(11)
    config = checkpoint_config()
    harness = build_harness(config)
    run_one_update(harness, fixed_batches()[0])
    path = tmp_path / "checkpoints" / "step-00000001.pt"

    record = save_harness(harness, path)

    assert record.path == path.resolve()
    assert record.file_size == path.stat().st_size > 0
    assert record.global_step == 1
    assert record.tokens_seen == 8
    assert harness.state.last_save_step == 1
    assert list(path.parent.glob(".*.tmp")) == []
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["format_name"] == CHECKPOINT_FORMAT_NAME
    assert payload["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert payload["scaler_state_dict"] is None
    assert payload["trainer_state"]["last_save_step"] == 1
    assert payload["resolved_config"]["plan"] == config.resolve().to_dict()


def test_failed_atomic_save_preserves_previous_file_and_state(tmp_path, monkeypatch):
    seed_all(12)
    config = checkpoint_config()
    harness = build_harness(config)
    batches = fixed_batches()
    run_one_update(harness, batches[0])
    path = tmp_path / "latest.pt"
    save_harness(harness, path)
    previous_bytes = path.read_bytes()

    run_one_update(harness, batches[1])
    assert harness.state.global_step == 2
    assert harness.state.last_save_step == 1

    def controlled_failure(*_args, **_kwargs):
        raise OSError("controlled write failure")

    monkeypatch.setattr(torch, "save", controlled_failure)
    with pytest.raises(CheckpointSaveError, match="atomically publish"):
        save_harness(harness, path)

    assert path.read_bytes() == previous_bytes
    assert harness.state.last_save_step == 1
    assert list(tmp_path.glob(".*.tmp")) == []


def test_continuous_and_resumed_training_are_exact(tmp_path):
    config = checkpoint_config()
    batches = fixed_batches()

    seed_all(2026)
    continuous = build_harness(config)
    continuous_metrics = []
    continuous_rng_values = []
    for batch in batches:
        metrics, rng_values = run_one_update(continuous, batch)
        continuous_metrics.append(metrics)
        continuous_rng_values.append(rng_values)
    continuous_probe = consume_external_rng()

    seed_all(2026)
    interrupted = build_harness(config)
    interrupted_metrics = []
    interrupted_rng_values = []
    for batch in batches[:2]:
        metrics, rng_values = run_one_update(interrupted, batch)
        interrupted_metrics.append(metrics)
        interrupted_rng_values.append(rng_values)

    path = tmp_path / "step-00000002.pt"
    save_harness(interrupted, path)
    loaded, resumed = load_into_new_components(config, path)
    resumed_metrics = []
    resumed_rng_values = []
    for batch in batches[2:]:
        metrics, rng_values = run_one_update(resumed, batch)
        resumed_metrics.append(metrics)
        resumed_rng_values.append(rng_values)
    resumed_probe = consume_external_rng()

    assert loaded.state.last_save_step == 2
    assert loaded.record.global_step == 2
    for interrupted_value, continuous_value in zip(
        interrupted_metrics,
        continuous_metrics[:2],
        strict=True,
    ):
        assert_update_equivalent(interrupted_value, continuous_value)
    for resumed_value, continuous_value in zip(
        resumed_metrics,
        continuous_metrics[2:],
        strict=True,
    ):
        assert_update_equivalent(resumed_value, continuous_value)
    for interrupted_values, continuous_values in zip(
        interrupted_rng_values,
        continuous_rng_values[:2],
        strict=True,
    ):
        assert interrupted_values[:2] == continuous_values[:2]
        assert torch.equal(interrupted_values[2], continuous_values[2])
    for resumed_values, continuous_values in zip(
        resumed_rng_values,
        continuous_rng_values[2:],
        strict=True,
    ):
        assert resumed_values[:2] == continuous_values[:2]
        assert torch.equal(resumed_values[2], continuous_values[2])
    assert resumed_probe[:2] == continuous_probe[:2]
    assert torch.equal(resumed_probe[2], continuous_probe[2])

    assert resumed.state.global_step == continuous.state.global_step == 4
    assert resumed.state.tokens_seen == continuous.state.tokens_seen == 32
    assert resumed.state.samples_consumed == continuous.state.samples_consumed == 8
    assert resumed.state.micro_steps_seen == continuous.state.micro_steps_seen == 4
    assert resumed.scheduler.state_dict() == continuous.scheduler.state_dict()
    assert_nested_equal(
        resumed.model.state_dict(),
        continuous.model.state_dict(),
        "model",
    )
    assert_nested_equal(
        resumed.optimizer.state_dict(),
        continuous.optimizer.state_dict(),
        "optimizer",
    )


def test_identity_mismatch_fails_before_model_mutation(tmp_path):
    seed_all(13)
    config = checkpoint_config()
    harness = build_harness(config)
    run_one_update(harness, fixed_batches()[0])
    path = tmp_path / "identity.pt"
    save_harness(harness, path)

    model, optimizer, scheduler, _precision = build_components(config)
    before = deepcopy(model.state_dict())
    wrong_identity = replace(
        checkpoint_identity(),
        dataset_manifest_sha256="9" * 64,
    )
    with pytest.raises(CheckpointCompatibilityError, match="identity"):
        load_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            plan=config.resolve(),
            resolved_config=resolved_config(config),
            identity=wrong_identity,
            expected_run_id="checkpoint-run",
        )

    assert_nested_equal(model.state_dict(), before, "unmodified_model")


def test_resolved_config_mismatch_is_rejected(tmp_path):
    seed_all(14)
    config = checkpoint_config()
    harness = build_harness(config)
    run_one_update(harness, fixed_batches()[0])
    path = tmp_path / "config.pt"
    save_harness(harness, path)
    drifted = deepcopy(resolved_config(config))
    drifted["runtime"]["precision"] = "bf16"

    with pytest.raises(CheckpointCompatibilityError, match="resolved config"):
        load_into_new_components(config, path, active_config=drifted)


def test_run_id_mismatch_is_rejected(tmp_path):
    seed_all(15)
    config = checkpoint_config()
    harness = build_harness(config)
    run_one_update(harness, fixed_batches()[0])
    path = tmp_path / "run-id.pt"
    save_harness(harness, path)

    with pytest.raises(CheckpointCompatibilityError, match="run_id"):
        load_into_new_components(config, path, run_id="another-run")


def test_corrupt_checkpoint_fails_with_clear_error(tmp_path):
    path = tmp_path / "corrupt.pt"
    path.write_bytes(b"not a torch checkpoint")
    config = checkpoint_config()

    with pytest.raises(CheckpointLoadError, match="could not decode"):
        load_into_new_components(config, path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda payload: payload.pop("rng_state"), "missing fields"),
        (lambda payload: payload.update({"unexpected": 1}), "unknown fields"),
        (
            lambda payload: payload.update({"schema_version": 999}),
            "schema version",
        ),
    ),
)
def test_checkpoint_root_contract_is_strict(tmp_path, mutation, message):
    seed_all(16)
    config = checkpoint_config()
    harness = build_harness(config)
    run_one_update(harness, fixed_batches()[0])
    valid_path = tmp_path / "valid.pt"
    save_harness(harness, valid_path)
    payload = torch.load(valid_path, map_location="cpu", weights_only=True)
    mutation(payload)
    invalid_path = tmp_path / "invalid.pt"
    torch.save(payload, invalid_path)

    with pytest.raises(CheckpointLoadError, match=message):
        load_into_new_components(config, invalid_path)


def test_invalid_rng_payload_is_rejected_before_restoration(tmp_path):
    seed_all(17)
    config = checkpoint_config()
    harness = build_harness(config)
    run_one_update(harness, fixed_batches()[0])
    valid_path = tmp_path / "valid-rng.pt"
    save_harness(harness, valid_path)
    payload = torch.load(valid_path, map_location="cpu", weights_only=True)
    payload["rng_state"]["torch_cpu"] = torch.ones(8, dtype=torch.float32)
    invalid_path = tmp_path / "invalid-rng.pt"
    torch.save(payload, invalid_path)

    with pytest.raises(CheckpointLoadError, match="RNG state"):
        load_into_new_components(config, invalid_path)


def test_save_rejects_non_boundary_data_cursor(tmp_path):
    seed_all(18)
    config = checkpoint_config()
    harness = build_harness(config)
    run_one_update(harness, fixed_batches()[0])
    harness.state.batches_consumed_in_epoch = 0

    with pytest.raises(CheckpointSaveError, match="update boundary"):
        save_harness(harness, tmp_path / "bad-boundary.pt")


def test_save_rejects_resolved_plan_drift(tmp_path):
    seed_all(19)
    config = checkpoint_config()
    harness = build_harness(config)
    active = resolved_config(config)
    active["plan"]["total_updates"] = 99

    with pytest.raises(CheckpointSaveError, match="resolved_config.plan"):
        save_checkpoint(
            tmp_path / "bad-plan.pt",
            model=harness.model,
            optimizer=harness.optimizer,
            scheduler=harness.scheduler,
            state=harness.state,
            plan=config.resolve(),
            resolved_config=active,
            identity=checkpoint_identity(),
        )


def test_build_identity_fingerprints_model_tokenizer_manifest_and_source(tmp_path):
    manifest = {
        "status": "complete",
        "config_fingerprint": "a" * 64,
        "tokenizer": {"sha256": "b" * 64},
        "splits": {"train": {"model_tokens": 100}},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_commit = "c" * 40

    identity = build_checkpoint_identity(
        {"n_layer": 2, "n_embd": 8},
        manifest_path,
        source_commit=source_commit,
        source_dirty=False,
    )
    changed_model = build_checkpoint_identity(
        {"n_layer": 3, "n_embd": 8},
        manifest_path,
        source_commit=source_commit,
        source_dirty=False,
    )

    assert identity.tokenizer_sha256 == "b" * 64
    assert identity.dataset_config_fingerprint == "a" * 64
    assert identity.source_commit == source_commit
    assert identity.source_dirty is False
    assert identity.model_config_sha256 != changed_model.model_config_sha256
    assert len(identity.dataset_manifest_sha256) == 64
    assert CheckpointIdentity.from_mapping(identity.to_dict()) == identity


@pytest.mark.parametrize(
    "invalid_source",
    ("short", "G" * 40, "a" * 41, 123),
)
def test_identity_rejects_invalid_full_source_commit(invalid_source):
    with pytest.raises(CheckpointIdentityError, match="source_commit"):
        replace(checkpoint_identity(), source_commit=invalid_source)


def test_identity_builder_rejects_incomplete_manifest(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "status": "building",
                "config_fingerprint": "a" * 64,
                "tokenizer": {"sha256": "b" * 64},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(CheckpointIdentityError, match="status"):
        build_checkpoint_identity(
            {"n_layer": 2},
            path,
            source_commit=None,
            source_dirty=True,
        )
