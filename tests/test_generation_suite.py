from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from eval import (
    GENERATION_MANIFEST_FILENAME,
    GENERATION_PROTOCOL_FORMAT_NAME,
    GENERATION_SAMPLE_FORMAT_NAME,
    GENERATION_SAMPLES_FILENAME,
    GENERATION_SUITE_FORMAT_NAME,
    GenerationSuiteError,
    generation_protocol_fingerprint,
    load_generation_protocol,
    run_generation_suite,
    sha256_file,
)
from model import GPTConfig
from scripts import run_generation_suite as suite_cli
from train import PrecisionPolicy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT_ROOT / "configs" / "day11_generation_protocol.json"
EXPECTED_FINGERPRINT = (
    "e60f3fb381b3efd8f00bd3f3fc3071c11645c78977dc7c6c40e0fd124b6d1ed0"
)
RUN_ID = "generation-suite-test"


def _protocol_document() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def _write_protocol(tmp_path: Path, document: dict) -> Path:
    path = tmp_path / "protocol.json"
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _model_config() -> GPTConfig:
    return GPTConfig.from_mapping(
        {
            "architecture": "decoder_only_gpt",
            "n_layer": 1,
            "n_head": 2,
            "n_embd": 8,
            "ffn_hidden": 32,
            "context_length": 16,
            "vocab_size": 16_384,
            "dropout": 0.0,
            "tie_embeddings": True,
            "normalization": "layernorm",
            "norm_position": "pre",
            "layer_norm_eps": 1.0e-5,
            "activation": "gelu",
            "gelu_approximate": "tanh",
            "position_encoding": "learned_absolute",
            "linear_bias": False,
            "lm_head_bias": False,
            "layer_norm_affine": True,
            "init_std": 0.02,
            "scale_residual_projections": True,
        }
    )


def _fake_common() -> dict:
    return {
        "run_id": RUN_ID,
        "checkpoint": {
            "path": "/checkpoint.pt",
            "bytes": 123,
            "sha256": "a" * 64,
            "global_step": 4,
            "tokens_seen": 64,
            "identity": {
                "model_config_sha256": "b" * 64,
                "tokenizer_sha256": "c" * 64,
                "dataset_manifest_sha256": "d" * 64,
                "dataset_config_fingerprint": "e" * 64,
                "source_commit": "f" * 40,
                "source_dirty": False,
            },
        },
        "generator": {
            "source_commit": "1" * 40,
            "source_dirty": False,
        },
        "model": {
            "config": _model_config().to_dict(),
            "parameters": 1_000,
            "weight_tying_verified": True,
        },
        "tokenizer": {
            "path": "/tokenizer.json",
            "bytes": 456,
            "sha256": "c" * 64,
            "library": "tokenizers",
            "library_version": "0.23.1",
            "vocab_size": 16_384,
            "special_token_ids": {
                "bos": 0,
                "eos": 1,
                "pad": 2,
                "unk": 3,
            },
        },
    }


def _fake_generation_result(prompt: str, settings) -> dict:
    common = _fake_common()
    prompt_ids = [10, 11]
    generated_ids = [12, 13]
    return {
        "format_name": "small_gpt_text_generation",
        "schema_version": 1,
        "created_at_utc": "2026-08-14T00:00:00+00:00",
        **common,
        "protocol": {
            **settings.to_dict(),
            "batch_size": 1,
            "add_bos": False,
            "append_eos_to_prompt": False,
            "eos_token_id": 1,
            "context_policy": "left_crop_conditioning_window",
            "kv_cache": False,
            "filter_order": ["temperature", "top_k", "top_p"],
        },
        "prompt": {
            "text": prompt,
            "decoded_text": prompt,
            "token_ids": prompt_ids,
            "token_count": len(prompt_ids),
            "initial_conditioning_token_ids": prompt_ids,
            "initial_conditioning_token_count": len(prompt_ids),
            "initial_tokens_discarded": 0,
        },
        "generation": {
            "token_ids": generated_ids,
            "token_count": len(generated_ids),
            "full_token_ids": prompt_ids + generated_ids,
            "continuation_text": " example",
            "full_text": prompt + " example",
            "stop_reason": "max_new_tokens",
            "eos_generated": False,
            "context_crop_events": 0,
            "forward_passes": len(generated_ids),
            "elapsed_seconds": 0.01,
        },
        "runtime": {
            "device": "cpu",
            "precision": "fp32",
            "uses_autocast": False,
        },
    }


def _run_fake_suite(tmp_path: Path, monkeypatch):
    load_calls = []
    generation_calls = []

    def fake_load(*args, **kwargs):
        load_calls.append((args, kwargs))
        return SimpleNamespace(name="one-session")

    def fake_generate(
        session,
        prompt,
        *,
        settings,
        generator_source_commit,
        generator_source_dirty,
    ):
        assert session.name == "one-session"
        assert generator_source_commit == "1" * 40
        assert generator_source_dirty is False
        generation_calls.append((prompt, settings))
        return _fake_generation_result(prompt, settings)

    monkeypatch.setattr(
        "eval.generation_suite.load_generation_session",
        fake_load,
    )
    monkeypatch.setattr(
        "eval.generation_suite.generate_with_session",
        fake_generate,
    )
    output_dir = tmp_path / "suite"
    published, manifest = run_generation_suite(
        tmp_path / "checkpoint.pt",
        tmp_path / "tokenizer.json",
        PROTOCOL_PATH,
        output_dir,
        model_config=_model_config(),
        expected_run_id=RUN_ID,
        expected_checkpoint_sha256="a" * 64,
        expected_tokenizer_sha256="c" * 64,
        precision=PrecisionPolicy.resolve("cpu", "fp32"),
        generator_source_commit="1" * 40,
        generator_source_dirty=False,
    )
    return published, manifest, load_calls, generation_calls


def test_checked_in_protocol_is_frozen_and_complete():
    protocol = load_generation_protocol(PROTOCOL_PATH)

    assert generation_protocol_fingerprint(protocol) == EXPECTED_FINGERPRINT
    assert protocol.to_dict()["format_name"] == GENERATION_PROTOCOL_FORMAT_NAME
    assert protocol.protocol_id == "day11-baseline-generation-v1"
    assert protocol.ordering == "prompt_then_decoding"
    assert protocol.max_new_tokens == 64
    assert protocol.stochastic_seed == 1337
    assert len(protocol.prompts) == 6
    assert len(protocol.decodings) == 5
    assert protocol.sample_count == 30
    assert [prompt.prompt_id for prompt in protocol.prompts] == [
        "story",
        "science",
        "technology",
        "history",
        "explanation",
        "instruction",
    ]
    assert [decoding.role for decoding in protocol.decodings] == [
        "greedy",
        "sample_temperature_1",
        "lower_temperature",
        "top_k",
        "top_p",
    ]
    assert protocol.decodings[0].settings.seed is None
    assert {
        decoding.settings.seed
        for decoding in protocol.decodings[1:]
    } == {1337}


def test_cli_parser_exposes_only_frozen_suite_controls():
    parser = suite_cli.build_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }

    assert option_strings == {
        "-h",
        "--help",
        "--config",
        "--protocol",
        "--checkpoint",
        "--checkpoint-sha256",
        "--tokenizer",
        "--tokenizer-sha256",
        "--run-id",
        "--device",
        "--precision",
        "--output-dir",
    }
    assert "--prompt" not in option_strings
    assert "--temperature" not in option_strings
    assert "--top-k" not in option_strings
    assert "--top-p" not in option_strings
    assert "--seed" not in option_strings


def test_protocol_fingerprint_ignores_json_formatting(tmp_path):
    compact_path = tmp_path / "compact.json"
    compact_path.write_text(
        json.dumps(_protocol_document(), separators=(",", ":")),
        encoding="utf-8",
    )
    compact = load_generation_protocol(compact_path)
    frozen = load_generation_protocol(PROTOCOL_PATH)

    assert sha256_file(compact_path) != sha256_file(PROTOCOL_PATH)
    assert generation_protocol_fingerprint(compact) == (
        generation_protocol_fingerprint(frozen)
    )


def test_duplicate_json_key_is_rejected(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text(
        '{"format_name":"small_gpt_generation_protocol",'
        '"format_name":"duplicate"}',
        encoding="utf-8",
    )

    with pytest.raises(GenerationSuiteError, match="duplicate JSON key"):
        load_generation_protocol(path)


@pytest.mark.parametrize(
    "mutation,match",
    (
        (lambda document: document.update(schema_version=2), "schema_version"),
        (lambda document: document.update(ordering="decoding_then_prompt"), "ordering"),
        (lambda document: document.update(max_new_tokens=0), "max_new_tokens"),
        (lambda document: document.update(stochastic_seed=-1), "stochastic_seed"),
        (lambda document: document.update(extra=True), "unknown fields"),
    ),
)
def test_root_protocol_drift_is_rejected(tmp_path, mutation, match):
    document = _protocol_document()
    mutation(document)
    path = _write_protocol(tmp_path, document)

    with pytest.raises(GenerationSuiteError, match=match):
        load_generation_protocol(path)


@pytest.mark.parametrize(
    "mutation,match",
    (
        (
            lambda document: document["prompts"][1].update(
                id=document["prompts"][0]["id"]
            ),
            "prompt IDs must be unique",
        ),
        (
            lambda document: document["prompts"][1].update(
                text=document["prompts"][0]["text"]
            ),
            "prompt texts must be unique",
        ),
        (
            lambda document: document["prompts"][0].update(text="<eos>"),
            "literal special tokens",
        ),
        (
            lambda document: document.update(prompts=document["prompts"][:4]),
            "between 5 and 12",
        ),
    ),
)
def test_prompt_suite_drift_is_rejected(tmp_path, mutation, match):
    document = _protocol_document()
    mutation(document)
    path = _write_protocol(tmp_path, document)

    with pytest.raises(GenerationSuiteError, match=match):
        load_generation_protocol(path)


@pytest.mark.parametrize(
    "mutation,match",
    (
        (
            lambda document: document["decodings"][1].update(seed=7),
            "stochastic decoding seeds",
        ),
        (
            lambda document: document["decodings"][2].update(temperature=1.0),
            "lower_temperature",
        ),
        (
            lambda document: document["decodings"][3].update(top_p=0.9),
            "top_k role",
        ),
        (
            lambda document: document["decodings"][4].update(top_k=50),
            "top_p role",
        ),
        (
            lambda document: document["decodings"][0].update(
                role="sample_temperature_1"
            ),
            "required role exactly once",
        ),
    ),
)
def test_decoding_contract_drift_is_rejected(tmp_path, mutation, match):
    document = _protocol_document()
    mutation(document)
    path = _write_protocol(tmp_path, document)

    with pytest.raises(GenerationSuiteError, match=match):
        load_generation_protocol(path)


def test_suite_loads_artifacts_once_and_publishes_ordered_jsonl(
    tmp_path,
    monkeypatch,
):
    published, manifest, load_calls, generation_calls = _run_fake_suite(
        tmp_path,
        monkeypatch,
    )

    assert published == (tmp_path / "suite").resolve()
    assert len(load_calls) == 1
    assert len(generation_calls) == 30
    assert manifest["format_name"] == GENERATION_SUITE_FORMAT_NAME
    assert manifest["status"] == "complete"
    assert manifest["execution"]["prompt_count"] == 6
    assert manifest["execution"]["decoding_count"] == 5
    assert manifest["execution"]["expected_samples"] == 30
    assert manifest["execution"]["completed_samples"] == 30
    assert manifest["execution"]["artifacts_loaded_once"] is True
    assert manifest["execution"]["model_loads"] == 1
    assert manifest["samples"]["records"] == 30
    assert manifest["summary"]["stop_reason_counts"] == {
        "eos": 0,
        "max_new_tokens": 30,
    }
    assert manifest["summary"]["sample_prompt_tokens"] == 60
    assert manifest["summary"]["generated_tokens"] == 60
    assert manifest["summary"]["forward_passes"] == 60

    manifest_path = published / GENERATION_MANIFEST_FILENAME
    samples_path = published / GENERATION_SAMPLES_FILENAME
    disk_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_lines = samples_path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in raw_lines]

    assert disk_manifest == manifest
    assert len(records) == 30
    assert records[0]["format_name"] == GENERATION_SAMPLE_FORMAT_NAME
    assert [record["sample_index"] for record in records] == list(range(30))
    assert [record["sample_key"] for record in records[:5]] == [
        "story/greedy",
        "story/sample-temperature-1",
        "story/sample-temperature-0-7",
        "story/sample-top-k-50",
        "story/sample-top-p-0-9",
    ]
    assert records[5]["sample_key"] == "science/greedy"
    assert records[-1]["sample_key"] == "instruction/sample-top-p-0-9"
    assert manifest["samples"]["bytes"] == samples_path.stat().st_size
    assert manifest["samples"]["sha256"] == sha256_file(samples_path)
    assert not list(published.parent.glob(f".{published.name}.*.tmp"))


def test_existing_output_is_rejected_before_artifact_load(tmp_path, monkeypatch):
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    called = False

    def forbidden_load(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("artifact load must not occur")

    monkeypatch.setattr(
        "eval.generation_suite.load_generation_session",
        forbidden_load,
    )
    with pytest.raises(GenerationSuiteError, match="already exists"):
        run_generation_suite(
            tmp_path / "checkpoint.pt",
            tmp_path / "tokenizer.json",
            PROTOCOL_PATH,
            output_dir,
            model_config=_model_config(),
            expected_run_id=RUN_ID,
            expected_checkpoint_sha256="a" * 64,
            expected_tokenizer_sha256="c" * 64,
            precision=PrecisionPolicy.resolve("cpu", "fp32"),
        )
    assert called is False


def test_generation_failure_does_not_publish_partial_suite(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "eval.generation_suite.load_generation_session",
        lambda *args, **kwargs: SimpleNamespace(name="session"),
    )

    def fail_generation(*args, **kwargs):
        raise GenerationSuiteError("injected generation failure")

    monkeypatch.setattr(
        "eval.generation_suite.generate_with_session",
        fail_generation,
    )
    output_dir = tmp_path / "failed-suite"
    with pytest.raises(GenerationSuiteError, match="injected"):
        run_generation_suite(
            tmp_path / "checkpoint.pt",
            tmp_path / "tokenizer.json",
            PROTOCOL_PATH,
            output_dir,
            model_config=_model_config(),
            expected_run_id=RUN_ID,
            expected_checkpoint_sha256="a" * 64,
            expected_tokenizer_sha256="c" * 64,
            precision=PrecisionPolicy.resolve("cpu", "fp32"),
        )
    assert not output_dir.exists()


@pytest.mark.parametrize(
    "commit,dirty,exception",
    (
        ("short", False, GenerationSuiteError),
        (None, "false", TypeError),
    ),
)
def test_source_identity_is_validated_before_artifact_load(
    tmp_path,
    monkeypatch,
    commit,
    dirty,
    exception,
):
    called = False

    def forbidden_load(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(
        "eval.generation_suite.load_generation_session",
        forbidden_load,
    )
    with pytest.raises(exception):
        run_generation_suite(
            tmp_path / "checkpoint.pt",
            tmp_path / "tokenizer.json",
            PROTOCOL_PATH,
            tmp_path / "suite",
            model_config=_model_config(),
            expected_run_id=RUN_ID,
            expected_checkpoint_sha256="a" * 64,
            expected_tokenizer_sha256="c" * 64,
            precision=PrecisionPolicy.resolve("cpu", "fp32"),
            generator_source_commit=commit,
            generator_source_dirty=dirty,
        )
    assert called is False


def test_cli_main_reports_published_bundle(tmp_path, monkeypatch, capsys):
    output_dir = tmp_path / "cli-suite"
    output_dir.mkdir()
    samples_path = output_dir / GENERATION_SAMPLES_FILENAME
    manifest_path = output_dir / GENERATION_MANIFEST_FILENAME
    samples_path.write_text('{"sample":1}\n', encoding="utf-8")
    manifest_path.write_text('{"manifest":1}\n', encoding="utf-8")
    fake_manifest = {
        "protocol": {"definition": {"protocol_id": "frozen-v1"}},
        "execution": {
            "prompt_count": 6,
            "decoding_count": 5,
            "completed_samples": 30,
            "artifacts_loaded_once": True,
            "wall_elapsed_seconds": 1.25,
        },
        "summary": {
            "generated_tokens": 1_920,
            "stop_reason_counts": {"eos": 2, "max_new_tokens": 28},
        },
    }

    monkeypatch.setattr(
        suite_cli,
        "run_generation_suite",
        lambda *args, **kwargs: (output_dir.resolve(), fake_manifest),
    )
    monkeypatch.setattr(suite_cli, "source_identity", lambda: ("1" * 40, False))

    exit_code = suite_cli.main(
        [
            "--config",
            str(PROJECT_ROOT / "configs" / "debug.yaml"),
            "--protocol",
            str(PROTOCOL_PATH),
            "--checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--checkpoint-sha256",
            "a" * 64,
            "--tokenizer",
            str(tmp_path / "tokenizer.json"),
            "--tokenizer-sha256",
            "c" * 64,
            "--run-id",
            RUN_ID,
            "--device",
            "cpu",
            "--precision",
            "fp32",
            "--output-dir",
            str(tmp_path / "requested-suite"),
        ]
    )

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "Frozen generation suite complete" in stdout
    assert "ProtocolID=frozen-v1" in stdout
    assert "CompletedSamples=30" in stdout
    assert "ArtifactsLoadedOnce=True" in stdout
    assert f"ManifestSHA256={sha256_file(manifest_path)}" in stdout
    assert f"SamplesSHA256={sha256_file(samples_path)}" in stdout
