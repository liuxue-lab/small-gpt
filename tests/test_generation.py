from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from eval import (
    GENERATION_FORMAT_NAME,
    GenerationError,
    GenerationSettings,
    generate_from_checkpoint,
    generate_token_ids,
    generate_with_session,
    load_generation_session,
    publish_generation_result,
    sha256_file,
)
from model import GPT, GPTConfig, GPTOutput
from scripts import generate_text
from tokenizer import load_tokenizer
from train import (
    CheckpointIdentity,
    PrecisionPolicy,
    TrainerState,
    TrainingConfig,
    build_optimizer,
    build_scheduler,
    save_checkpoint,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEBUG_CONFIG = PROJECT_ROOT / "configs" / "debug.yaml"
PROJECT_TOKENIZER = PROJECT_ROOT / "tokenizer" / "artifacts" / "tokenizer.json"
RUN_ID = "generation-test"


class ScheduledLogitModel(nn.Module):
    def __init__(
        self,
        schedules: list[torch.Tensor],
        *,
        context_length: int = 4,
    ) -> None:
        super().__init__()
        self.config = type(
            "Config",
            (),
            {
                "context_length": context_length,
                "vocab_size": int(schedules[0].numel()),
            },
        )()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.schedules = [item.detach().clone().float() for item in schedules]
        self.inputs: list[torch.Tensor] = []
        self.calls = 0

    def forward(self, input_ids: torch.Tensor) -> GPTOutput:
        self.inputs.append(input_ids.detach().cpu().clone())
        logits = torch.zeros(
            input_ids.shape[0],
            input_ids.shape[1],
            self.config.vocab_size,
            device=input_ids.device,
        )
        schedule = self.schedules[min(self.calls, len(self.schedules) - 1)]
        logits[:, -1, :] = schedule.to(input_ids.device)
        self.calls += 1
        return GPTOutput(logits=logits, loss=None)


def _logits(*, vocab_size: int = 8, winner: int, runner_up: int = 0) -> torch.Tensor:
    values = torch.full((vocab_size,), -10.0)
    values[runner_up] = 3.0
    values[winner] = 5.0
    return values


@dataclass(frozen=True)
class GenerationFixture:
    config_path: Path
    checkpoint_path: Path
    tokenizer_path: Path
    model_config: GPTConfig


def _write_config(tmp_path: Path) -> Path:
    document = yaml.safe_load(DEBUG_CONFIG.read_text(encoding="utf-8"))
    document["project"].update(name="generation-test", seed=29)
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
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "max_steps": 4,
            "target_tokens": None,
            "warmup_steps": 1,
            "warmup_ratio": None,
            "num_workers": 0,
            "pin_memory": False,
        }
    )
    path = tmp_path / "generation.yaml"
    path.write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )
    return path


def _write_checkpoint(
    tmp_path: Path,
    config_path: Path,
    tokenizer_path: Path,
) -> tuple[Path, GPTConfig]:
    model_config = GPTConfig.from_yaml(config_path)
    training_config = TrainingConfig.from_yaml(config_path)
    plan = training_config.resolve()
    model = GPT(model_config)
    optimizer = build_optimizer(model, training_config)
    scheduler = build_scheduler(optimizer, training_config, plan)
    state = TrainerState(run_id=RUN_ID)
    identity = CheckpointIdentity(
        model_config_sha256=hashlib.sha256(
            json.dumps(
                model_config.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
        tokenizer_sha256=sha256_file(tokenizer_path),
        dataset_manifest_sha256="b" * 64,
        dataset_config_fingerprint="c" * 64,
        source_commit="d" * 40,
        source_dirty=False,
    )
    resolved_config = {
        "schema_version": 1,
        "project_name": training_config.project_name,
        "model": model_config.to_dict(),
        "training": training_config.to_dict(),
        "plan": plan.to_dict(),
        "runtime": {"device": "cpu", "precision": "fp32"},
    }
    checkpoint_path = tmp_path / "step-00000000.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        state=state,
        plan=plan,
        resolved_config=resolved_config,
        identity=identity,
    )
    return checkpoint_path, model_config


def write_generation_fixture(tmp_path: Path) -> GenerationFixture:
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_bytes(PROJECT_TOKENIZER.read_bytes())
    config_path = _write_config(tmp_path)
    checkpoint_path, model_config = _write_checkpoint(
        tmp_path,
        config_path,
        tokenizer_path,
    )
    return GenerationFixture(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        model_config=model_config,
    )


def test_parser_exposes_only_generation_controls():
    parser = generate_text.build_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }

    assert option_strings == {
        "-h",
        "--help",
        "--config",
        "--checkpoint",
        "--checkpoint-sha256",
        "--tokenizer",
        "--tokenizer-sha256",
        "--run-id",
        "--prompt",
        "--strategy",
        "--max-new-tokens",
        "--temperature",
        "--top-k",
        "--top-p",
        "--seed",
        "--device",
        "--precision",
        "--output",
    }
    with pytest.raises(SystemExit):
        generate_text.parse_args(
            [
                "--checkpoint",
                "checkpoint.pt",
                "--checkpoint-sha256",
                "0" * 64,
                "--tokenizer-sha256",
                "1" * 64,
                "--run-id",
                RUN_ID,
                "--prompt",
                "Hello",
                "--strategy",
                "beam",
                "--max-new-tokens",
                "8",
                "--output",
                "result.json",
            ]
        )


@pytest.mark.parametrize(
    "kwargs,match",
    (
        ({"strategy": "beam", "max_new_tokens": 4}, "strategy"),
        ({"strategy": "greedy", "max_new_tokens": 0}, "max_new_tokens"),
        (
            {
                "strategy": "greedy",
                "max_new_tokens": 4,
                "temperature": 0.8,
            },
            "temperature=1.0",
        ),
        (
            {"strategy": "greedy", "max_new_tokens": 4, "top_k": 2},
            "does not accept top_k",
        ),
        (
            {"strategy": "greedy", "max_new_tokens": 4, "seed": 3},
            "does not accept a seed",
        ),
        ({"strategy": "sample", "max_new_tokens": 4}, "explicit"),
        (
            {"strategy": "sample", "max_new_tokens": 4, "seed": -1},
            "seed",
        ),
        (
            {
                "strategy": "sample",
                "max_new_tokens": 4,
                "seed": 1,
                "top_p": 1.2,
            },
            "top_p",
        ),
    ),
)
def test_settings_reject_ambiguous_or_invalid_protocol(kwargs, match):
    with pytest.raises(GenerationError, match=match):
        GenerationSettings(**kwargs)


def test_greedy_stops_at_eos_and_restores_training_mode():
    model = ScheduledLogitModel(
        [
            _logits(winner=4, runner_up=3),
            _logits(winner=1, runner_up=5),
            _logits(winner=6, runner_up=7),
        ]
    )
    model.train()
    trace = generate_token_ids(
        model,
        [2, 3],
        settings=GenerationSettings(
            strategy="greedy",
            max_new_tokens=5,
        ),
        precision=PrecisionPolicy.resolve("cpu", "fp32"),
    )

    assert trace.prompt_token_ids == (2, 3)
    assert trace.generated_token_ids == (4, 1)
    assert trace.full_token_ids == (2, 3, 4, 1)
    assert trace.stop_reason == "eos"
    assert trace.forward_passes == 2
    assert model.calls == 2
    assert model.training is True


def test_context_is_left_cropped_while_trace_keeps_all_tokens():
    model = ScheduledLogitModel(
        [
            _logits(winner=6, runner_up=5),
            _logits(winner=7, runner_up=6),
        ],
        context_length=4,
    )
    trace = generate_token_ids(
        model,
        [2, 3, 4, 5, 6],
        settings=GenerationSettings(
            strategy="greedy",
            max_new_tokens=2,
        ),
        precision=PrecisionPolicy.resolve("cpu", "fp32"),
    )

    assert model.inputs[0].tolist() == [[3, 4, 5, 6]]
    assert model.inputs[1].tolist() == [[4, 5, 6, 6]]
    assert trace.initial_prompt_tokens_discarded == 1
    assert trace.context_crop_events == 2
    assert trace.full_token_ids == (2, 3, 4, 5, 6, 6, 7)
    assert trace.stop_reason == "max_new_tokens"


@pytest.mark.parametrize(
    "top_k,top_p",
    (
        (None, None),
        (3, None),
        (None, 0.8),
        (4, 0.9),
    ),
)
def test_sampling_is_seeded_and_does_not_mutate_global_rng(top_k, top_p):
    schedule = torch.tensor([0.0, -10.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    settings = GenerationSettings(
        strategy="sample",
        max_new_tokens=6,
        temperature=0.8,
        top_k=top_k,
        top_p=top_p,
        seed=20260814,
    )

    torch.manual_seed(20260814)
    rng_before = torch.get_rng_state().clone()
    first = generate_token_ids(
        ScheduledLogitModel([schedule]),
        [2, 3],
        settings=settings,
        precision=PrecisionPolicy.resolve("cpu", "fp32"),
        eos_token_id=1,
    )
    rng_after = torch.get_rng_state().clone()
    second = generate_token_ids(
        ScheduledLogitModel([schedule]),
        [2, 3],
        settings=settings,
        precision=PrecisionPolicy.resolve("cpu", "fp32"),
        eos_token_id=1,
    )

    assert first.generated_token_ids == second.generated_token_ids
    assert first.generated_token_ids
    assert torch.equal(rng_before, rng_after)


def test_top_k_one_sampling_matches_greedy():
    schedule = _logits(winner=6, runner_up=5)
    greedy = generate_token_ids(
        ScheduledLogitModel([schedule]),
        [2],
        settings=GenerationSettings(
            strategy="greedy",
            max_new_tokens=4,
        ),
        precision=PrecisionPolicy.resolve("cpu", "fp32"),
    )
    sampled = generate_token_ids(
        ScheduledLogitModel([schedule]),
        [2],
        settings=GenerationSettings(
            strategy="sample",
            max_new_tokens=4,
            top_k=1,
            seed=9,
        ),
        precision=PrecisionPolicy.resolve("cpu", "fp32"),
    )

    assert sampled.generated_token_ids == greedy.generated_token_ids


def test_core_rejects_bad_prompt_and_top_k_before_forward():
    model = ScheduledLogitModel([_logits(winner=4)])
    precision = PrecisionPolicy.resolve("cpu", "fp32")
    with pytest.raises(GenerationError, match="must not be empty"):
        generate_token_ids(
            model,
            [],
            settings=GenerationSettings(
                strategy="greedy",
                max_new_tokens=1,
            ),
            precision=precision,
        )
    with pytest.raises(GenerationError, match="top_k exceeds"):
        generate_token_ids(
            model,
            [2],
            settings=GenerationSettings(
                strategy="sample",
                max_new_tokens=1,
                top_k=9,
                seed=1,
            ),
            precision=precision,
        )
    assert model.calls == 0


def _generate(fixture: GenerationFixture, *, prompt: str = "Once upon a time"):
    return generate_from_checkpoint(
        fixture.checkpoint_path,
        fixture.tokenizer_path,
        prompt,
        model_config=fixture.model_config,
        expected_run_id=RUN_ID,
        expected_checkpoint_sha256=sha256_file(fixture.checkpoint_path),
        expected_tokenizer_sha256=sha256_file(fixture.tokenizer_path),
        settings=GenerationSettings(
            strategy="greedy",
            max_new_tokens=2,
        ),
        precision=PrecisionPolicy.resolve("cpu", "fp32"),
        generator_source_commit="e" * 40,
        generator_source_dirty=False,
    )


def test_checkpoint_generation_emits_strict_identity_and_raw_tokens(tmp_path):
    fixture = write_generation_fixture(tmp_path)
    tokenizer = load_tokenizer(fixture.tokenizer_path)
    torch.manual_seed(20260814)
    rng_before = torch.get_rng_state().clone()
    deterministic_before = torch.are_deterministic_algorithms_enabled()

    result = _generate(fixture)

    assert set(result) == {
        "format_name",
        "schema_version",
        "created_at_utc",
        "run_id",
        "checkpoint",
        "generator",
        "model",
        "tokenizer",
        "protocol",
        "prompt",
        "generation",
        "runtime",
    }
    assert result["format_name"] == GENERATION_FORMAT_NAME
    assert result["run_id"] == RUN_ID
    assert result["checkpoint"]["sha256"] == sha256_file(
        fixture.checkpoint_path
    )
    assert result["tokenizer"]["sha256"] == sha256_file(
        fixture.tokenizer_path
    )
    assert result["checkpoint"]["identity"]["tokenizer_sha256"] == (
        result["tokenizer"]["sha256"]
    )
    assert result["tokenizer"]["special_token_ids"] == {
        "bos": 0,
        "eos": 1,
        "pad": 2,
        "unk": 3,
    }
    assert result["model"]["weight_tying_verified"] is True
    assert result["protocol"]["add_bos"] is False
    assert result["protocol"]["append_eos_to_prompt"] is False
    assert result["protocol"]["kv_cache"] is False
    assert result["prompt"]["token_ids"] == tokenizer.encode(
        "Once upon a time",
        add_special_tokens=False,
    ).ids
    assert result["generation"]["full_token_ids"] == (
        result["prompt"]["token_ids"]
        + result["generation"]["token_ids"]
    )
    assert result["generation"]["token_count"] <= 2
    assert result["generator"] == {
        "source_commit": "e" * 40,
        "source_dirty": False,
    }
    assert torch.equal(rng_before, torch.get_rng_state())
    assert torch.are_deterministic_algorithms_enabled() is deterministic_before
    json.dumps(result, allow_nan=False)


def test_loaded_session_reuses_one_strict_artifact_load(tmp_path, monkeypatch):
    fixture = write_generation_fixture(tmp_path)
    session = load_generation_session(
        fixture.checkpoint_path,
        fixture.tokenizer_path,
        model_config=fixture.model_config,
        expected_run_id=RUN_ID,
        expected_checkpoint_sha256=sha256_file(fixture.checkpoint_path),
        expected_tokenizer_sha256=sha256_file(fixture.tokenizer_path),
        precision=PrecisionPolicy.resolve("cpu", "fp32"),
    )

    def forbidden_reload(*args, **kwargs):
        raise AssertionError("loaded generation session must not reload artifacts")

    monkeypatch.setattr(
        "eval.generation.load_model_checkpoint",
        forbidden_reload,
    )
    settings = GenerationSettings(
        strategy="sample",
        max_new_tokens=2,
        top_k=16,
        seed=1337,
    )
    first = generate_with_session(
        session,
        "Hello",
        settings=settings,
        generator_source_commit="e" * 40,
        generator_source_dirty=False,
    )
    second = generate_with_session(
        session,
        "Hello",
        settings=settings,
        generator_source_commit="e" * 40,
        generator_source_dirty=False,
    )

    assert first["generation"]["token_ids"] == second["generation"]["token_ids"]
    assert first["checkpoint"] == second["checkpoint"]
    assert first["tokenizer"] == second["tokenizer"]


def test_checkpoint_and_tokenizer_hashes_are_required_before_load(
    tmp_path,
    monkeypatch,
):
    fixture = write_generation_fixture(tmp_path)
    called = False

    def forbidden_load(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("load_model_checkpoint should not be called")

    monkeypatch.setattr("eval.generation.load_model_checkpoint", forbidden_load)
    with pytest.raises(GenerationError, match="checkpoint SHA-256 mismatch"):
        generate_from_checkpoint(
            fixture.checkpoint_path,
            fixture.tokenizer_path,
            "Hello",
            model_config=fixture.model_config,
            expected_run_id=RUN_ID,
            expected_checkpoint_sha256="0" * 64,
            expected_tokenizer_sha256=sha256_file(fixture.tokenizer_path),
            settings=GenerationSettings(
                strategy="greedy",
                max_new_tokens=1,
            ),
            precision=PrecisionPolicy.resolve("cpu", "fp32"),
        )
    assert called is False


def test_checkpoint_identity_rejects_wrong_tokenizer(tmp_path):
    fixture = write_generation_fixture(tmp_path)
    tokenizer_document = json.loads(
        fixture.tokenizer_path.read_text(encoding="utf-8")
    )
    tokenizer_document["added_tokens"][0]["special"] = False
    other_tokenizer = tmp_path / "other-tokenizer.json"
    other_tokenizer.write_text(
        json.dumps(tokenizer_document),
        encoding="utf-8",
    )

    with pytest.raises(GenerationError, match="does not match checkpoint identity"):
        generate_from_checkpoint(
            fixture.checkpoint_path,
            other_tokenizer,
            "Hello",
            model_config=fixture.model_config,
            expected_run_id=RUN_ID,
            expected_checkpoint_sha256=sha256_file(fixture.checkpoint_path),
            expected_tokenizer_sha256=sha256_file(other_tokenizer),
            settings=GenerationSettings(
                strategy="greedy",
                max_new_tokens=1,
            ),
            precision=PrecisionPolicy.resolve("cpu", "fp32"),
        )


def test_empty_or_tokenless_prompt_is_rejected(tmp_path):
    fixture = write_generation_fixture(tmp_path)
    with pytest.raises(GenerationError, match="non-empty"):
        _generate(fixture, prompt="   ")


def test_publish_is_strict_atomic_and_never_overwrites(tmp_path):
    output_path = tmp_path / "nested" / "generation.json"
    result = {"format_name": GENERATION_FORMAT_NAME, "value": 1}

    published = publish_generation_result(output_path, result)

    assert published == output_path.resolve()
    assert json.loads(output_path.read_text(encoding="utf-8")) == result
    assert not list(output_path.parent.glob("*.tmp"))
    with pytest.raises(GenerationError, match="will not be overwritten"):
        publish_generation_result(output_path, {"value": 2})
    assert json.loads(output_path.read_text(encoding="utf-8")) == result
    with pytest.raises(GenerationError, match="strict JSON"):
        publish_generation_result(
            tmp_path / "nan.json",
            {"value": float("nan")},
        )


def test_cli_main_publishes_once_and_reports_output(tmp_path, capsys):
    fixture = write_generation_fixture(tmp_path)
    output_path = tmp_path / "generation.json"

    exit_code = generate_text.main(
        [
            "--config",
            str(fixture.config_path),
            "--checkpoint",
            str(fixture.checkpoint_path),
            "--checkpoint-sha256",
            sha256_file(fixture.checkpoint_path),
            "--tokenizer",
            str(fixture.tokenizer_path),
            "--tokenizer-sha256",
            sha256_file(fixture.tokenizer_path),
            "--run-id",
            RUN_ID,
            "--prompt",
            "Hello",
            "--strategy",
            "sample",
            "--max-new-tokens",
            "2",
            "--temperature",
            "0.8",
            "--top-k",
            "16",
            "--top-p",
            "0.9",
            "--seed",
            "42",
            "--device",
            "cpu",
            "--precision",
            "fp32",
            "--output",
            str(output_path),
        ]
    )

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "Text generation complete" in stdout
    assert "Strategy=sample" in stdout
    assert "Seed=42" in stdout
    assert "StopReason=" in stdout
    assert f"OutputPath={output_path.resolve()}" in stdout
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["protocol"]["temperature"] == 0.8
    assert result["protocol"]["top_k"] == 16
    assert result["protocol"]["top_p"] == 0.9
    assert result["protocol"]["seed"] == 42

    with pytest.raises(GenerationError, match="will not be overwritten"):
        generate_text.main(
            [
                "--checkpoint",
                str(fixture.checkpoint_path),
                "--checkpoint-sha256",
                sha256_file(fixture.checkpoint_path),
                "--tokenizer-sha256",
                sha256_file(fixture.tokenizer_path),
                "--run-id",
                RUN_ID,
                "--prompt",
                "Hello",
                "--strategy",
                "greedy",
                "--max-new-tokens",
                "1",
                "--output",
                str(output_path),
            ]
        )
