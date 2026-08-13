from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pytest
import torch
import yaml

from data_pipeline import DatasetContractError, TokenShardWriter
from eval import (
    FROZEN_EVALUATION_FORMAT_NAME,
    FrozenEvaluationError,
    evaluate_frozen_split,
    publish_evaluation_result,
    sha256_file,
)
from model import GPT, GPTConfig
from scripts import evaluate_checkpoint
from train import (
    CheckpointIdentity,
    PrecisionPolicy,
    TrainerState,
    TrainingConfig,
    build_checkpoint_identity,
    build_optimizer,
    build_scheduler,
    save_checkpoint,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEBUG_CONFIG = PROJECT_ROOT / "configs" / "debug.yaml"
SPECIAL_IDS = {"bos": 0, "eos": 1, "pad": 2, "unk": 3}
RUN_ID = "frozen-evaluation-test"


@dataclass(frozen=True)
class FrozenFixture:
    config_path: Path
    checkpoint_path: Path
    manifest_path: Path
    model_config: GPTConfig
    identity: CheckpointIdentity
    binary_paths: dict[str, Path]


def _document_hash(split: str) -> bytes:
    return hashlib.sha256(f"frozen:{split}".encode("ascii")).digest()


def _write_config(tmp_path: Path) -> Path:
    document = yaml.safe_load(DEBUG_CONFIG.read_text(encoding="utf-8"))
    document["project"].update(name="frozen-evaluation-test", seed=17)
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
            "num_workers": 0,
            "pin_memory": False,
        }
    )
    path = tmp_path / "evaluation.yaml"
    path.write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )
    return path


def _write_manifest(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    corpus_root = tmp_path / "tokenized"
    documents = {
        "train": (10, 11, 12, 13, 14, 15, 16, 17, 1),
        "validation": (
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            1,
        ),
        "test": (40, 41, 42, 43, 44, 45, 46, 47, 48, 1),
    }
    splits = {}
    binary_paths: dict[str, Path] = {}
    for split, token_ids in documents.items():
        writer = TokenShardWriter(
            staging_root=corpus_root,
            split=split,
            shard_index=0,
            global_token_start=0,
            global_document_start=0,
            vocab_size=16_384,
            special_token_ids=SPECIAL_IDS,
        )
        writer.append_document(
            token_ids,
            text_sha256=_document_hash(split),
            provided_tokens=len(token_ids) - 1,
        )
        metadata = writer.finalize()
        binary_paths[split] = corpus_root / metadata["binary"]["path"]
        splits[split] = {
            "records": 1,
            "model_tokens": len(token_ids),
            "storage_shards": 1,
            "shards": [metadata],
        }

    manifest = {
        "schema_version": 1,
        "format_name": "small_gpt_tokenized_corpus",
        "status": "complete",
        "profile": "pilot",
        "config_fingerprint": "a" * 64,
        "source": {"split_order": ["train", "validation", "test"]},
        "tokenizer": {
            "sha256": "b" * 64,
            "vocab_size": 16_384,
            "special_token_ids": SPECIAL_IDS,
        },
        "splits": splits,
    }
    manifest_path = corpus_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path, binary_paths


def _write_checkpoint(
    tmp_path: Path,
    config_path: Path,
    manifest_path: Path,
) -> tuple[Path, GPTConfig, CheckpointIdentity]:
    model_config = GPTConfig.from_yaml(config_path)
    training_config = TrainingConfig.from_yaml(config_path)
    plan = training_config.resolve()
    model = GPT(model_config)
    optimizer = build_optimizer(model, training_config)
    scheduler = build_scheduler(optimizer, training_config, plan)
    state = TrainerState(run_id=RUN_ID)
    identity = build_checkpoint_identity(
        model_config.to_dict(),
        manifest_path,
        source_commit="5" * 40,
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
    return checkpoint_path, model_config, identity


def write_frozen_fixture(tmp_path: Path) -> FrozenFixture:
    config_path = _write_config(tmp_path)
    manifest_path, binary_paths = _write_manifest(tmp_path)
    checkpoint_path, model_config, identity = _write_checkpoint(
        tmp_path,
        config_path,
        manifest_path,
    )
    return FrozenFixture(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        manifest_path=manifest_path,
        model_config=model_config,
        identity=identity,
        binary_paths=binary_paths,
    )


def _evaluate(
    fixture: FrozenFixture,
    *,
    split: str,
    max_batches: int | None = None,
):
    return evaluate_frozen_split(
        fixture.checkpoint_path,
        fixture.manifest_path,
        model_config=fixture.model_config,
        expected_run_id=RUN_ID,
        split=split,
        precision=PrecisionPolicy.resolve("cpu", "fp32"),
        expected_checkpoint_sha256=sha256_file(fixture.checkpoint_path),
        max_batches=max_batches,
        evaluator_source_commit="6" * 40,
        evaluator_source_dirty=False,
    )


def test_parser_exposes_only_frozen_evaluation_controls():
    parser = evaluate_checkpoint.build_parser()
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
        "--manifest",
        "--run-id",
        "--split",
        "--device",
        "--precision",
        "--max-batches",
        "--output",
    }
    with pytest.raises(SystemExit):
        evaluate_checkpoint.parse_args(
            [
                "--checkpoint",
                "checkpoint.pt",
                "--checkpoint-sha256",
                "0" * 64,
                "--manifest",
                "manifest.json",
                "--run-id",
                RUN_ID,
                "--split",
                "train",
                "--output",
                "result.json",
            ]
        )


@pytest.mark.parametrize("max_batches", (0, -1, True, 1.5))
def test_core_rejects_invalid_max_batches_before_checkpoint_read(
    tmp_path,
    max_batches,
):
    config_path = _write_config(tmp_path)

    with pytest.raises(FrozenEvaluationError, match="max_batches"):
        evaluate_frozen_split(
            tmp_path / "missing-checkpoint.pt",
            tmp_path / "missing-manifest.json",
            model_config=GPTConfig.from_yaml(config_path),
            expected_run_id=RUN_ID,
            expected_checkpoint_sha256="0" * 64,
            split="test",
            precision=PrecisionPolicy.resolve("cpu", "fp32"),
            max_batches=max_batches,
        )


def test_full_test_split_uses_split_neutral_strict_evidence(tmp_path):
    fixture = write_frozen_fixture(tmp_path)
    torch.manual_seed(20260814)
    rng_before = torch.get_rng_state().clone()
    deterministic_before = torch.are_deterministic_algorithms_enabled()

    result = _evaluate(fixture, split="test")

    assert set(result) == {
        "format_name",
        "schema_version",
        "created_at_utc",
        "run_id",
        "split",
        "checkpoint",
        "evaluator",
        "model",
        "data",
        "coverage",
        "metrics",
        "runtime",
    }
    assert result["format_name"] == FROZEN_EVALUATION_FORMAT_NAME
    assert result["schema_version"] == 1
    datetime.fromisoformat(result["created_at_utc"])
    assert result["run_id"] == RUN_ID
    assert result["split"] == "test"
    assert result["checkpoint"]["identity"] == fixture.identity.to_dict()
    assert result["checkpoint"]["sha256"] == sha256_file(
        fixture.checkpoint_path
    )
    assert result["checkpoint"]["global_step"] == 0
    assert result["checkpoint"]["tokens_seen"] == 0
    assert result["evaluator"] == {
        "source_commit": "6" * 40,
        "source_dirty": False,
    }
    assert result["model"]["parameters"] == fixture.model_config.parameter_count
    assert result["model"]["weight_tying_verified"] is True
    assert result["data"]["manifest_sha256"] == fixture.identity.dataset_manifest_sha256
    assert result["data"]["tokenizer_sha256"] == "b" * 64
    assert result["data"]["config_fingerprint"] == "a" * 64
    assert result["data"]["split_model_tokens"] == 10
    assert result["coverage"] == {
        "requested_max_batches": None,
        "available_batches": 1,
        "total_windows": 2,
        "full_evaluation_tokens": 8,
        "trailing_tokens_discarded": 1,
        "is_full_split": True,
    }
    assert math.isfinite(result["metrics"]["loss"])
    assert result["metrics"]["perplexity_is_finite"] is True
    assert result["metrics"]["perplexity"] == pytest.approx(
        math.exp(result["metrics"]["loss"])
    )
    assert result["metrics"]["evaluated_batches"] == 1
    assert result["metrics"]["evaluated_tokens"] == 8
    assert result["metrics"]["elapsed_seconds"] > 0.0
    assert result["runtime"]["deterministic_algorithms"] is True
    assert result["runtime"]["configured_allow_tf32"] is False
    assert "validation_loss" not in json.dumps(result, sort_keys=True)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert (
        torch.are_deterministic_algorithms_enabled()
        is deterministic_before
    )


def test_checkpoint_hash_mismatch_is_rejected_before_load(tmp_path):
    fixture = write_frozen_fixture(tmp_path)

    with pytest.raises(FrozenEvaluationError, match="checkpoint SHA-256 mismatch"):
        evaluate_frozen_split(
            fixture.checkpoint_path,
            fixture.manifest_path,
            model_config=fixture.model_config,
            expected_run_id=RUN_ID,
            expected_checkpoint_sha256="0" * 64,
            split="test",
            precision=PrecisionPolicy.resolve("cpu", "fp32"),
        )


def test_bounded_validation_is_never_reported_as_full_split(tmp_path):
    fixture = write_frozen_fixture(tmp_path)

    result = _evaluate(
        fixture,
        split="validation",
        max_batches=1,
    )

    assert result["split"] == "validation"
    assert result["coverage"] == {
        "requested_max_batches": 1,
        "available_batches": 2,
        "total_windows": 3,
        "full_evaluation_tokens": 12,
        "trailing_tokens_discarded": 1,
        "is_full_split": False,
    }
    assert result["metrics"]["evaluated_batches"] == 1
    assert result["metrics"]["evaluated_tokens"] == 8


def test_manifest_identity_drift_is_rejected(tmp_path):
    fixture = write_frozen_fixture(tmp_path)
    manifest = json.loads(fixture.manifest_path.read_text(encoding="utf-8"))
    manifest["config_fingerprint"] = "c" * 64
    fixture.manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(FrozenEvaluationError, match="evaluation inputs"):
        _evaluate(fixture, split="test")


def test_selected_shard_hash_drift_is_rejected_when_requested(tmp_path):
    fixture = write_frozen_fixture(tmp_path)
    binary_path = fixture.binary_paths["validation"]
    payload = bytearray(binary_path.read_bytes())
    payload[-1] ^= 1
    binary_path.write_bytes(payload)

    with pytest.raises(DatasetContractError, match="SHA-256 mismatch"):
        _evaluate(fixture, split="validation")


def test_publish_is_strict_atomic_and_refuses_overwrite(tmp_path):
    output_path = tmp_path / "evidence" / "result.json"
    result = {
        "format_name": FROZEN_EVALUATION_FORMAT_NAME,
        "schema_version": 1,
        "loss": 1.25,
    }

    published = publish_evaluation_result(output_path, result)
    original = published.read_bytes()

    assert published == output_path.resolve()
    assert original.endswith(b"\n")
    assert json.loads(original) == result
    assert len(sha256_file(published)) == 64
    assert list(output_path.parent.glob(".*.tmp")) == []
    with pytest.raises(FrozenEvaluationError, match="will not be overwritten"):
        publish_evaluation_result(output_path, {"different": True})
    assert published.read_bytes() == original


def test_publish_rejects_nonfinite_json_without_creating_output(tmp_path):
    output_path = tmp_path / "nonfinite.json"

    with pytest.raises(FrozenEvaluationError, match="strict JSON"):
        publish_evaluation_result(output_path, {"loss": float("nan")})

    assert not output_path.exists()


def test_cli_evaluates_once_writes_evidence_and_refuses_reuse(
    tmp_path,
    monkeypatch,
    capsys,
):
    fixture = write_frozen_fixture(tmp_path)
    output_path = tmp_path / "test-evaluation.json"
    monkeypatch.setattr(
        evaluate_checkpoint,
        "source_identity",
        lambda: ("7" * 40, False),
    )
    arguments = [
        "--config",
        str(fixture.config_path),
        "--checkpoint",
        str(fixture.checkpoint_path),
        "--checkpoint-sha256",
        sha256_file(fixture.checkpoint_path),
        "--manifest",
        str(fixture.manifest_path),
        "--run-id",
        RUN_ID,
        "--split",
        "test",
        "--device",
        "cpu",
        "--precision",
        "fp32",
        "--output",
        str(output_path),
    ]

    assert evaluate_checkpoint.main(arguments) == 0

    output = capsys.readouterr().out
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert "Frozen evaluation complete" in output
    assert "Split=test" in output
    assert "FullSplit=True" in output
    assert "EvaluatedTokens=8" in output
    assert "EvaluatedBatches=1" in output
    assert f"OutputSHA256={sha256_file(output_path)}" in output
    assert result["evaluator"] == {
        "source_commit": "7" * 40,
        "source_dirty": False,
    }
    with pytest.raises(FrozenEvaluationError, match="will not be overwritten"):
        evaluate_checkpoint.main(arguments)
