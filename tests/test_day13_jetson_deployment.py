from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from model import GPTOutput
from scripts import benchmark_jetson_inference as benchmark
from scripts import check_jetson_deployment as deployment


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    PROJECT_ROOT / "configs" / "day13_jetson_deployment_protocol.json"
)


def _protocol() -> dict:
    document = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def _write_protocol(tmp_path: Path, document: dict) -> Path:
    path = tmp_path / "protocol.json"
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def _mutated_protocol(tmp_path: Path, mutate) -> Path:
    document = _protocol()
    mutate(document)
    return _write_protocol(tmp_path, document)


class ProbeModel(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int = 8,
        dtype: torch.dtype = torch.float32,
        fill: float = 0.0,
    ) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros((), dtype=dtype))
        self.vocab_size = vocab_size
        self.dtype = dtype
        self.fill = fill
        self.inference_mode_seen = False

    def forward(self, input_ids: torch.Tensor) -> GPTOutput:
        self.inference_mode_seen = bool(torch.is_inference_mode_enabled())
        logits = torch.full(
            (input_ids.shape[0], input_ids.shape[1], self.vocab_size),
            self.fill,
            dtype=self.dtype,
            device=input_ids.device,
        )
        return GPTOutput(logits=logits, loss=None)


class TiedRuntimeModel(nn.Module):
    def __init__(self, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(8, 4)
        self.dropout = nn.Dropout(dropout)
        self.lm_head = nn.Linear(4, 8, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: torch.Tensor) -> GPTOutput:
        hidden = self.dropout(self.token_embedding(input_ids))
        return GPTOutput(logits=self.lm_head(hidden), loss=None)


def _runtime_protocol(*, parameters: int = 32, dropout: float = 0.0) -> dict:
    return {
        "model": {
            "parameters": parameters,
            "dropout": dropout,
            "context_length": 8,
        },
        "tokenizer": {"vocab_size": 8},
    }


def _runtime_config(*, dropout: float = 0.0):
    return SimpleNamespace(dropout=dropout, vocab_size=8, context_length=8)


def _fake_sample(spec: benchmark.RunSpec, *, seconds: float) -> dict:
    return {
        "phase": spec.phase,
        "prompt": {"prompt_id": spec.prompt_id},
        "generation": {
            "max_new_tokens": spec.max_new_tokens,
            "token_count": spec.max_new_tokens,
        },
        "timing": {
            "ttft_seconds": seconds / 4.0,
            "decode_tokens_per_second": 10.0 / seconds,
            "end_to_end_tokens_per_second": 8.0 / seconds,
            "end_to_end_seconds": seconds,
        },
        "memory": {
            "cuda_after": {
                "peak_allocated_bytes": 100 + spec.sequence_index,
                "peak_reserved_bytes": 200 + spec.sequence_index,
            },
            "system_before": {"mem_available_bytes": 1_000},
            "system_after": {"mem_available_bytes": 900},
        },
    }


def _fake_summary_session(tmp_path: Path, *, dirty: bool = False):
    identity = deployment.FileIdentity(tmp_path / "artifact", 10, "a" * 64)
    model = nn.Linear(1, 1, bias=False)
    loaded = SimpleNamespace(state=SimpleNamespace(run_id=deployment.CONTROL_RUN_ID))
    return SimpleNamespace(
        protocol=_protocol(),
        protocol_fingerprint="b" * 64,
        source=deployment.SourceIdentity("c" * 40, dirty, "main"),
        config_identity=identity,
        checkpoint_identity=identity,
        tokenizer_identity=identity,
        loaded_checkpoint=loaded,
        model=model,
        model_config=SimpleNamespace(
            context_length=512,
            vocab_size=16_384,
            dropout=0.0,
        ),
        runtime={"device": "cuda:0"},
        precision="fp32",
        weight_dtype=torch.float32,
        compute_dtype=torch.float32,
        model_load_seconds=1.25,
    )


def test_checked_in_protocol_is_frozen_and_complete():
    document = deployment.load_protocol(PROTOCOL_PATH)

    assert document["protocol_id"] == deployment.PROTOCOL_ID
    assert document["model_role"] == "control"
    assert document["checkpoint"]["load_mode"] == "model_only"
    assert document["model"]["parameters"] == 33_833_984
    assert document["model"]["dropout"] == 0.0
    assert document["smoke"]["precisions"] == ["fp32", "fp16"]
    assert [item["prompt_id"] for item in document["smoke"]["prompts"]] == [
        "prompt_01",
        "prompt_02",
        "prompt_03",
    ]
    assert document["safety"]["formal_test_access"] is False
    assert document["safety"]["training_allowed"] is False


def test_protocol_fingerprint_is_stable():
    first = deployment.canonical_sha256(deployment.load_protocol(PROTOCOL_PATH))
    second = deployment.canonical_sha256(deployment.load_protocol(PROTOCOL_PATH))

    assert first == second
    assert len(first) == 64


def test_duplicate_protocol_key_is_rejected(tmp_path):
    raw = PROTOCOL_PATH.read_text(encoding="utf-8")
    path = tmp_path / "duplicate.json"
    path.write_text(
        raw.replace(
            '"schema_version": 1,',
            '"schema_version": 1,\n  "schema_version": 1,',
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(deployment.ProtocolError, match="duplicate JSON key"):
        deployment.load_protocol(path)


def test_missing_protocol_field_is_rejected(tmp_path):
    path = _mutated_protocol(tmp_path, lambda document: document.pop("model_role"))

    with pytest.raises(deployment.ProtocolError, match="missing fields"):
        deployment.load_protocol(path)


def test_unknown_protocol_field_is_rejected(tmp_path):
    path = _mutated_protocol(
        tmp_path,
        lambda document: document.update({"unapproved": True}),
    )

    with pytest.raises(deployment.ProtocolError, match="unknown fields"):
        deployment.load_protocol(path)


def test_protocol_id_drift_is_rejected(tmp_path):
    path = _mutated_protocol(
        tmp_path,
        lambda document: document.update({"protocol_id": "changed"}),
    )

    with pytest.raises(deployment.ProtocolError, match="protocol_id"):
        deployment.load_protocol(path)


def test_checkpoint_sha_drift_is_rejected(tmp_path):
    path = _mutated_protocol(
        tmp_path,
        lambda document: document["checkpoint"].update({"sha256": "0" * 64}),
    )

    with pytest.raises(deployment.ProtocolError, match="checkpoint.sha256"):
        deployment.load_protocol(path)


def test_tokenizer_sha_drift_is_rejected(tmp_path):
    path = _mutated_protocol(
        tmp_path,
        lambda document: document["tokenizer"].update({"sha256": "0" * 64}),
    )

    with pytest.raises(deployment.ProtocolError, match="tokenizer.sha256"):
        deployment.load_protocol(path)


def test_baseline_config_sha_drift_is_rejected(tmp_path):
    path = _mutated_protocol(
        tmp_path,
        lambda document: document["config"].update({"sha256": "0" * 64}),
    )

    with pytest.raises(deployment.ProtocolError, match="config.sha256"):
        deployment.load_protocol(path)


def test_treatment_role_is_rejected(tmp_path):
    path = _mutated_protocol(
        tmp_path,
        lambda document: document.update({"model_role": "treatment"}),
    )

    with pytest.raises(deployment.ProtocolError, match="model_role"):
        deployment.load_protocol(path)


def test_dropout_point_one_is_rejected(tmp_path):
    path = _mutated_protocol(
        tmp_path,
        lambda document: document["model"].update({"dropout": 0.1}),
    )

    with pytest.raises(deployment.ProtocolError, match="model.dropout"):
        deployment.load_protocol(path)


def test_parameter_count_drift_is_rejected(tmp_path):
    path = _mutated_protocol(
        tmp_path,
        lambda document: document["model"].update({"parameters": 1}),
    )

    with pytest.raises(deployment.ProtocolError, match="model.parameters"):
        deployment.load_protocol(path)


def test_special_token_id_drift_is_rejected(tmp_path):
    path = _mutated_protocol(
        tmp_path,
        lambda document: document["tokenizer"]["special_token_ids"].update(
            {"eos": 7}
        ),
    )

    with pytest.raises(deployment.ProtocolError, match="special_token_ids"):
        deployment.load_protocol(path)


def test_formal_test_path_is_rejected():
    with pytest.raises(deployment.ProtocolError, match="formal test path"):
        deployment.assert_no_formal_test_paths(
            {"formal_test_path": "data/test/shard-00000.bin"}
        )


def test_credential_or_host_field_is_rejected():
    with pytest.raises(deployment.ProtocolError, match="credential or host"):
        deployment.assert_no_credentials({"device_ip_address": "192.0.2.1"})


def test_file_identity_accepts_exact_bytes_and_hash(tmp_path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"day13")
    expected_sha = deployment.sha256_file(path)

    identity = deployment.verify_file_identity(
        path,
        expected_bytes=5,
        expected_sha256=expected_sha,
        label="fixture",
    )

    assert identity.bytes == 5
    assert identity.sha256 == expected_sha


def test_file_identity_rejects_wrong_hash(tmp_path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"day13")

    with pytest.raises(deployment.ArtifactIdentityError, match="SHA-256 mismatch"):
        deployment.verify_file_identity(
            path,
            expected_bytes=5,
            expected_sha256="0" * 64,
            label="fixture",
        )


def test_file_identity_rejects_empty_path():
    with pytest.raises(deployment.ArtifactIdentityError, match="path must be non-empty"):
        deployment.verify_file_identity(
            "",
            expected_bytes=1,
            expected_sha256="0" * 64,
            label="fixture",
        )


@pytest.mark.parametrize("requested", ("cuda", "cuda:0"))
def test_cuda_and_cuda_zero_resolve_to_cuda_zero(monkeypatch, requested):
    monkeypatch.setattr(deployment.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(deployment.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(deployment.torch.cuda, "device_count", lambda: 1)

    assert deployment.resolve_cuda_device(requested) == torch.device("cuda:0")


def test_cpu_fallback_cannot_report_cuda_pass(monkeypatch):
    monkeypatch.setattr(deployment.torch.cuda, "is_available", lambda: False)

    with pytest.raises(deployment.CudaDeviceError, match="CPU fallback is forbidden"):
        deployment.resolve_cuda_device("cuda:0")

    with pytest.raises(deployment.CudaDeviceError, match="CPU fallback is forbidden"):
        deployment.resolve_cuda_device("cpu")


def test_out_of_range_cuda_index_is_rejected(monkeypatch):
    monkeypatch.setattr(deployment.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(deployment.torch.cuda, "device_count", lambda: 1)

    with pytest.raises(deployment.CudaDeviceError, match="visible range"):
        deployment.resolve_cuda_device("cuda:1")


@pytest.mark.parametrize(
    ("precision", "expected"),
    (("fp32", torch.float32), ("fp16", torch.float16)),
)
def test_precision_dtype_is_explicit(precision, expected):
    assert deployment.precision_dtype(precision) == expected


def test_unsupported_precision_is_rejected():
    with pytest.raises(deployment.DeploymentError, match="precision"):
        deployment.precision_dtype("bf16")


def test_runtime_model_contract_accepts_exact_eval_model():
    model = TiedRuntimeModel(dropout=0.0).eval()

    result = deployment.validate_model_runtime_contract(
        model,
        model_config=_runtime_config(),
        protocol=_runtime_protocol(),
        device=torch.device("cpu"),
        expected_dtype=torch.float32,
    )

    assert result["parameters"] == 32
    assert result["training"] is False
    assert result["weight_tying_verified"] is True


def test_runtime_parameter_count_mismatch_is_rejected():
    model = TiedRuntimeModel(dropout=0.0).eval()

    with pytest.raises(deployment.DeploymentError, match="parameter count"):
        deployment.validate_model_runtime_contract(
            model,
            model_config=_runtime_config(),
            protocol=_runtime_protocol(parameters=31),
            device=torch.device("cpu"),
            expected_dtype=torch.float32,
        )


def test_model_training_true_is_rejected():
    model = TiedRuntimeModel(dropout=0.0).train()

    with pytest.raises(deployment.DeploymentError, match="model.training"):
        deployment.validate_model_runtime_contract(
            model,
            model_config=_runtime_config(),
            protocol=_runtime_protocol(),
            device=torch.device("cpu"),
            expected_dtype=torch.float32,
        )


def test_runtime_dropout_point_one_is_rejected():
    model = TiedRuntimeModel(dropout=0.1).eval()

    with pytest.raises(deployment.DeploymentError, match="dropout"):
        deployment.validate_model_runtime_contract(
            model,
            model_config=_runtime_config(dropout=0.1),
            protocol=_runtime_protocol(dropout=0.0),
            device=torch.device("cpu"),
            expected_dtype=torch.float32,
        )


def test_checked_forward_uses_inference_mode():
    model = ProbeModel().eval()

    result = deployment.checked_forward(
        model,
        [2, 4],
        device=torch.device("cpu"),
        precision="fp32",
        vocab_size=8,
    )

    assert model.inference_mode_seen is True
    assert result["inference_mode"] is True
    assert result["logits_finite"] is True


def test_checked_forward_uses_fp16_logits_path():
    model = ProbeModel(dtype=torch.float16).eval()

    result = deployment.checked_forward(
        model,
        [2, 4],
        device=torch.device("cpu"),
        precision="fp16",
        vocab_size=8,
    )

    assert result["logits_dtype"] == "torch.float16"


@pytest.mark.parametrize("fill", (float("nan"), float("inf")))
def test_non_finite_logits_are_rejected(fill):
    model = ProbeModel(fill=fill).eval()

    with pytest.raises(deployment.DeploymentError, match="NaN or infinity"):
        deployment.checked_forward(
            model,
            [2, 4],
            device=torch.device("cpu"),
            precision="fp32",
            vocab_size=8,
        )


@pytest.mark.parametrize("token_id", (-1, 8))
def test_out_of_range_token_id_is_rejected(token_id):
    with pytest.raises(deployment.DeploymentError, match="outside"):
        deployment.validate_token_id(token_id, vocab_size=8)


def test_model_only_wrapper_uses_audited_loader(monkeypatch):
    calls = []
    sentinel = object()

    def fake_loader(path, **kwargs):
        calls.append((path, kwargs))
        return sentinel

    monkeypatch.setattr(deployment, "load_model_checkpoint", fake_loader)
    config = SimpleNamespace(to_dict=lambda: {"vocab_size": 8})
    model = object()

    result = deployment.model_only_load(
        "checkpoint.pt",
        model=model,
        model_config=config,
        expected_run_id="control",
    )

    assert result is sentinel
    assert deployment.STRICT_STATE_DICT_LOAD is True
    assert calls[0][1] == {
        "model": model,
        "expected_model_config": {"vocab_size": 8},
        "expected_run_id": "control",
    }


def test_tokenizer_runtime_contract_accepts_frozen_ids(monkeypatch):
    class FakeTokenizer:
        def get_vocab_size(self, *, with_added_tokens):
            assert with_added_tokens is True
            return 16_384

        def token_to_id(self, text):
            return {"<bos>": 0, "<eos>": 1, "<pad>": 2, "<unk>": 3}[text]

        def decode(self, ids, *, skip_special_tokens):
            assert skip_special_tokens is True
            return "Once upon a time"

    monkeypatch.setattr(deployment, "encode_text", lambda tokenizer, text: [9, 10])

    result = deployment.validate_tokenizer_contract(FakeTokenizer())

    assert result["vocab_size"] == 16_384
    assert result["special_token_ids"] == {"bos": 0, "eos": 1, "pad": 2, "unk": 3}
    assert result["probe_unknown_token_count"] == 0


def test_tokenizer_special_id_mismatch_is_rejected(monkeypatch):
    class FakeTokenizer:
        def get_vocab_size(self, *, with_added_tokens):
            return 16_384

        def token_to_id(self, text):
            return 7

    monkeypatch.setattr(deployment, "encode_text", lambda tokenizer, text: [9])

    with pytest.raises(deployment.DeploymentError, match="expected"):
        deployment.validate_tokenizer_contract(FakeTokenizer())


def test_output_directory_existing_is_rejected(tmp_path):
    output_dir = tmp_path / "existing"
    output_dir.mkdir()

    with pytest.raises(benchmark.BenchmarkError, match="will not be overwritten"):
        benchmark.reserve_output_directory(output_dir)


def test_smoke_sample_count_is_conserved():
    plan = benchmark.build_run_plan(_protocol(), "smoke")

    assert len(plan) == 3
    assert [spec.prompt_id for spec in plan] == [
        "prompt_01",
        "prompt_02",
        "prompt_03",
    ]
    assert sum(spec.max_new_tokens for spec in plan) == 192


def test_benchmark_warmups_and_measurements_are_separate():
    plan = benchmark.build_run_plan(_protocol(), "benchmark")

    assert len(plan) == 13
    assert sum(spec.phase == "warmup" for spec in plan) == 3
    assert sum(spec.phase == "measured" for spec in plan) == 10
    assert [spec.phase_index for spec in plan if spec.phase == "measured"] == list(
        range(10)
    )


def test_stability_request_count_and_tokens_are_conserved():
    plan = benchmark.build_run_plan(_protocol(), "stability")

    assert len(plan) == 10
    assert all(spec.phase == "measured" for spec in plan)
    assert sum(spec.max_new_tokens for spec in plan) == 640


def test_zero_benchmark_count_is_rejected():
    document = _protocol()
    document["benchmark"]["warmup_runs"] = 0

    with pytest.raises(benchmark.BenchmarkError, match="positive integer"):
        benchmark.build_run_plan(document, "benchmark")


def test_summary_excludes_warmups_and_records_dirty_source(tmp_path, monkeypatch):
    plan = benchmark.build_run_plan(_protocol(), "benchmark")
    samples = [
        _fake_sample(spec, seconds=1.0 + spec.sequence_index / 100.0)
        for spec in plan
    ]
    session = _fake_summary_session(tmp_path, dirty=True)
    monkeypatch.setattr(
        benchmark,
        "_power_mode",
        lambda: {"query_exit": 0, "output": "MAXN_SUPER", "error": None},
    )

    summary = benchmark.build_summary(
        session,
        mode="benchmark",
        run_id="day13-test",
        plan=plan,
        samples=samples,
    )

    assert summary["counts"]["completed_warmup_runs"] == 3
    assert summary["counts"]["completed_measured_runs"] == 10
    assert summary["counts"]["measured_generated_tokens"] == 640
    assert summary["counts"]["all_generated_tokens"] == 832
    assert summary["counts"]["warmup_excluded_from_measured_summary"] is True
    assert summary["identity"]["source"]["dirty"] is True


def test_summary_statistics_rejects_empty_set():
    with pytest.raises(benchmark.BenchmarkError, match="empty"):
        benchmark.summary_statistics([])


def test_summary_statistics_are_finite_and_complete():
    summary = benchmark.summary_statistics([1.0, 2.0, 3.0, 4.0])

    assert set(summary) == {"mean", "median", "min", "max", "p95"}
    assert summary["mean"] == pytest.approx(2.5)
    assert summary["p95"] == pytest.approx(4.0)


def test_cli_does_not_expose_training_or_resume_controls():
    deployment_options = {
        option
        for action in deployment.build_parser()._actions
        for option in action.option_strings
    }
    benchmark_options = {
        option
        for action in benchmark.build_parser()._actions
        for option in action.option_strings
    }

    for forbidden in ("--train", "--training", "--resume", "--optimizer"):
        assert forbidden not in deployment_options
        assert forbidden not in benchmark_options
    assert "--precision" in benchmark_options
    assert "--mode" in benchmark_options
    assert "--output-dir" in benchmark_options


def test_dry_run_reads_no_artifacts_and_writes_no_output(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run attempted to read an artifact")

    monkeypatch.setattr(deployment, "verify_file_identity", forbidden)

    exit_code = deployment.main(
        ["--protocol", str(PROTOCOL_PATH), "--mode", "dry-run"]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["dry_run"] is True
    assert payload["checkpoint_read"] is False
    assert payload["output_written"] is False
    assert captured.err == ""


def test_dry_run_rejects_output_path(tmp_path, capsys):
    output = tmp_path / "forbidden.json"

    exit_code = deployment.main(
        [
            "--protocol",
            str(PROTOCOL_PATH),
            "--mode",
            "dry-run",
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "forbids --output" in captured.err
    assert not output.exists()


def test_atomic_json_write_does_not_overwrite(tmp_path):
    path = tmp_path / "result.json"
    deployment.atomic_write_json_exclusive(path, {"status": "first"})

    with pytest.raises(deployment.DeploymentError, match="will not be overwritten"):
        deployment.atomic_write_json_exclusive(path, {"status": "second"})

    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "first"}


def test_strict_json_rejects_non_finite_number():
    with pytest.raises(deployment.DeploymentError, match="strict finite JSON"):
        deployment.strict_json_bytes({"value": float("nan")})


def test_samples_jsonl_rejects_empty_rows():
    with pytest.raises(benchmark.BenchmarkError, match="at least one row"):
        benchmark.strict_jsonl_bytes([])


def test_samples_jsonl_rejects_non_finite_number():
    with pytest.raises(benchmark.BenchmarkError, match="strict finite JSON"):
        benchmark.strict_jsonl_bytes([{"value": float("inf")}])


def test_manifest_is_published_last_and_records_source_dirty(tmp_path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    summary = {
        "run_id": "day13-test",
        "mode": "smoke",
        "identity": {"source": {"head": "a" * 40, "dirty": True, "branch": "main"}},
        "protocol": {"deployment_protocol_id": deployment.PROTOCOL_ID},
        "runtime": {"precision": "fp32"},
    }

    manifest_path = benchmark.publish_success_outputs(
        output_dir,
        samples=[{"sample": 1}],
        summary=summary,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["manifest_published_last"] is True
    assert manifest["source"]["dirty"] is True
    assert (output_dir / benchmark.SAMPLES_FILENAME).is_file()
    assert (output_dir / benchmark.SUMMARY_FILENAME).is_file()


def test_atomic_manifest_failure_leaves_no_formal_manifest(tmp_path, monkeypatch):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    summary = {
        "run_id": "day13-test",
        "mode": "smoke",
        "identity": {"source": {"head": "a" * 40, "dirty": False, "branch": "main"}},
        "protocol": {"deployment_protocol_id": deployment.PROTOCOL_ID},
        "runtime": {"precision": "fp32"},
    }
    original = benchmark.atomic_write_json_exclusive

    def fail_manifest(path, value):
        if Path(path).name == benchmark.MANIFEST_FILENAME:
            raise deployment.DeploymentError("injected manifest failure")
        return original(path, value)

    monkeypatch.setattr(benchmark, "atomic_write_json_exclusive", fail_manifest)

    with pytest.raises(deployment.DeploymentError, match="injected"):
        benchmark.publish_success_outputs(
            output_dir,
            samples=[{"sample": 1}],
            summary=summary,
        )

    assert not (output_dir / benchmark.MANIFEST_FILENAME).exists()
    assert (output_dir / benchmark.SAMPLES_FILENAME).is_file()
    assert (output_dir / benchmark.SUMMARY_FILENAME).is_file()
