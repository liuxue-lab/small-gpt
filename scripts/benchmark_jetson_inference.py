"""Auditable FP32/FP16 greedy inference on the frozen Day 13 Jetson target."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from scripts.check_jetson_deployment import (  # noqa: E402
    DeploymentError,
    DeploymentSession,
    atomic_write_bytes_exclusive,
    atomic_write_json_exclusive,
    load_deployment_session,
    load_protocol,
    sha256_file,
    validate_token_id,
)
from tokenizer import encode_text  # noqa: E402


MANIFEST_FILENAME = "manifest.json"
SAMPLES_FILENAME = "samples.jsonl"
SUMMARY_FILENAME = "benchmark-summary.json"
FAILURE_FILENAME = "failure.json"
_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class BenchmarkError(DeploymentError):
    """Raised when a benchmark request or publication is invalid."""


@dataclass(frozen=True, slots=True)
class RunSpec:
    sequence_index: int
    phase: str
    phase_index: int
    prompt_id: str
    prompt_text: str
    max_new_tokens: int


def _is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_run_id(value: str) -> str:
    if not isinstance(value, str) or _RUN_ID_PATTERN.fullmatch(value) is None:
        raise BenchmarkError(
            "run_id must be 1-128 characters using letters, digits, '.', '_' or '-'"
        )
    return value


def reserve_output_directory(path: str | Path) -> Path:
    if not isinstance(path, (str, Path)) or not str(path).strip():
        raise BenchmarkError("output directory path must be non-empty")
    output_dir = Path(path).resolve()
    if output_dir.exists():
        raise BenchmarkError(
            f"output directory already exists and will not be overwritten: {output_dir}"
        )
    try:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir()
    except FileExistsError as error:
        raise BenchmarkError(
            f"output directory already exists and will not be overwritten: {output_dir}"
        ) from error
    except OSError as error:
        raise BenchmarkError(f"could not create output directory: {error}") from error
    return output_dir


def _prompt_map(protocol: Mapping[str, Any]) -> dict[str, str]:
    prompts = protocol["smoke"]["prompts"]
    result = {item["prompt_id"]: item["text"] for item in prompts}
    if not result or len(result) != len(prompts):
        raise BenchmarkError("protocol prompt set must be non-empty and unique")
    return result


def _require_positive_count(value: object, *, field: str) -> int:
    if not _is_plain_int(value) or value <= 0:
        raise BenchmarkError(f"{field} must be a positive integer")
    return int(value)


def build_run_plan(protocol: Mapping[str, Any], mode: str) -> tuple[RunSpec, ...]:
    prompts = _prompt_map(protocol)
    specs: list[RunSpec] = []
    if mode == "smoke":
        max_new_tokens = _require_positive_count(
            protocol["smoke"]["max_new_tokens"],
            field="smoke.max_new_tokens",
        )
        for index, item in enumerate(protocol["smoke"]["prompts"]):
            specs.append(
                RunSpec(
                    sequence_index=index,
                    phase="measured",
                    phase_index=index,
                    prompt_id=item["prompt_id"],
                    prompt_text=item["text"],
                    max_new_tokens=max_new_tokens,
                )
            )
    elif mode == "benchmark":
        config = protocol["benchmark"]
        warmup_runs = _require_positive_count(
            config["warmup_runs"], field="benchmark.warmup_runs"
        )
        measured_runs = _require_positive_count(
            config["measured_runs"], field="benchmark.measured_runs"
        )
        max_new_tokens = _require_positive_count(
            config["max_new_tokens"], field="benchmark.max_new_tokens"
        )
        prompt_id = config["prompt_id"]
        if prompt_id not in prompts:
            raise BenchmarkError(f"benchmark prompt ID is unknown: {prompt_id!r}")
        for phase, count in (("warmup", warmup_runs), ("measured", measured_runs)):
            for phase_index in range(count):
                specs.append(
                    RunSpec(
                        sequence_index=len(specs),
                        phase=phase,
                        phase_index=phase_index,
                        prompt_id=prompt_id,
                        prompt_text=prompts[prompt_id],
                        max_new_tokens=max_new_tokens,
                    )
                )
    elif mode == "stability":
        config = protocol["stability"]
        request_count = _require_positive_count(
            config["sequential_requests"],
            field="stability.sequential_requests",
        )
        max_new_tokens = _require_positive_count(
            config["max_new_tokens"], field="stability.max_new_tokens"
        )
        prompt_id = config["prompt_id"]
        if prompt_id not in prompts:
            raise BenchmarkError(f"stability prompt ID is unknown: {prompt_id!r}")
        for phase_index in range(request_count):
            specs.append(
                RunSpec(
                    sequence_index=len(specs),
                    phase="measured",
                    phase_index=phase_index,
                    prompt_id=prompt_id,
                    prompt_text=prompts[prompt_id],
                    max_new_tokens=max_new_tokens,
                )
            )
    else:
        raise BenchmarkError(
            f"mode must be one of ['smoke', 'benchmark', 'stability'], got {mode!r}"
        )
    if not specs:
        raise BenchmarkError("run plan must contain at least one request")
    return tuple(specs)


def system_memory_snapshot() -> dict[str, int | None]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            name, raw = line.split(":", 1)
            fields = raw.strip().split()
            if not fields:
                continue
            amount = int(fields[0])
            multiplier = 1024 if len(fields) > 1 and fields[1] == "kB" else 1
            values[name] = amount * multiplier
    except (OSError, UnicodeDecodeError, ValueError):
        return {
            "mem_total_bytes": None,
            "mem_available_bytes": None,
            "swap_total_bytes": None,
            "swap_free_bytes": None,
        }
    return {
        "mem_total_bytes": values.get("MemTotal"),
        "mem_available_bytes": values.get("MemAvailable"),
        "swap_total_bytes": values.get("SwapTotal"),
        "swap_free_bytes": values.get("SwapFree"),
    }


def cuda_memory_snapshot(device: torch.device) -> dict[str, int]:
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _power_mode() -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["nvpmodel", "-q"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"query_exit": None, "output": None, "error": str(error)}
    return {
        "query_exit": completed.returncode,
        "output": completed.stdout.strip() or None,
        "error": completed.stderr.strip() or None,
    }


def _validate_logits(
    logits: object,
    *,
    expected_shape: tuple[int, int, int],
    expected_device: torch.device,
    expected_dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(logits, torch.Tensor):
        raise BenchmarkError("model output logits must be a Tensor")
    if tuple(logits.shape) != expected_shape:
        raise BenchmarkError(
            f"logits shape mismatch: expected {expected_shape}, found {tuple(logits.shape)}"
        )
    if logits.device != expected_device:
        raise BenchmarkError(
            f"logits device mismatch: expected {expected_device}, found {logits.device}"
        )
    if logits.dtype != expected_dtype:
        raise BenchmarkError(
            f"logits dtype mismatch: expected {expected_dtype}, found {logits.dtype}"
        )
    if not bool(torch.isfinite(logits).all().item()):
        raise BenchmarkError("logits contain NaN or infinity")
    return logits


def run_greedy_request(
    session: DeploymentSession,
    spec: RunSpec,
    *,
    run_id: str,
    mode: str,
) -> dict[str, Any]:
    if session.model.training:
        raise BenchmarkError("model.training is True before inference")
    try:
        prompt_ids = list(encode_text(session.tokenizer, spec.prompt_text))
    except Exception as error:
        raise BenchmarkError(f"could not tokenize {spec.prompt_id}: {error}") from error
    vocab_size = int(session.model_config.vocab_size)
    prompt_ids = [validate_token_id(value, vocab_size=vocab_size) for value in prompt_ids]
    if not prompt_ids:
        raise BenchmarkError(f"prompt {spec.prompt_id} produced no token IDs")

    device = session.device
    expected_dtype = session.compute_dtype
    context_length = int(session.model_config.context_length)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    system_before = system_memory_snapshot()
    cuda_before = cuda_memory_snapshot(device)
    request_started = time.perf_counter()
    first_token_seconds: float | None = None
    full_sequence = list(prompt_ids)
    generated_ids: list[int] = []
    context_crop_events = 0
    inference_mode_observed = False

    with torch.inference_mode():
        inference_mode_observed = bool(torch.is_inference_mode_enabled())
        if not inference_mode_observed:
            raise BenchmarkError("torch inference mode did not activate")
        for token_index in range(spec.max_new_tokens):
            if len(full_sequence) > context_length:
                context_crop_events += 1
            conditioning = full_sequence[-context_length:]
            input_ids = torch.tensor(
                [conditioning],
                dtype=torch.long,
                device=device,
            )
            output = session.model(input_ids)
            logits = _validate_logits(
                getattr(output, "logits", None),
                expected_shape=(1, len(conditioning), vocab_size),
                expected_device=device,
                expected_dtype=expected_dtype,
            )
            next_id = validate_token_id(
                int(torch.argmax(logits[0, -1].float()).item()),
                vocab_size=vocab_size,
            )
            generated_ids.append(next_id)
            full_sequence.append(next_id)
            if token_index == 0:
                torch.cuda.synchronize(device)
                first_token_seconds = max(
                    time.perf_counter() - request_started,
                    1.0e-12,
                )

    torch.cuda.synchronize(device)
    total_seconds = max(time.perf_counter() - request_started, 1.0e-12)
    if first_token_seconds is None:
        raise BenchmarkError("request produced no first-token timing")
    if len(generated_ids) != spec.max_new_tokens:
        raise BenchmarkError(
            f"generated token count mismatch: {len(generated_ids)} != {spec.max_new_tokens}"
        )
    if total_seconds < first_token_seconds:
        raise BenchmarkError("total request time is less than first-token time")
    decode_token_count = max(0, len(generated_ids) - 1)
    decode_seconds = max(total_seconds - first_token_seconds, 0.0)
    decode_tokens_per_second: float | None
    if decode_token_count == 0 or decode_seconds <= 0.0:
        decode_tokens_per_second = None
    else:
        decode_tokens_per_second = decode_token_count / decode_seconds

    try:
        continuation = session.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )
        full_text = session.tokenizer.decode(
            full_sequence,
            skip_special_tokens=True,
        )
    except Exception as error:
        raise BenchmarkError(f"could not decode generated token IDs: {error}") from error
    cuda_after = cuda_memory_snapshot(device)
    system_after = system_memory_snapshot()
    return {
        "format_name": "small_gpt_jetson_inference_sample",
        "schema_version": 1,
        "run_id": run_id,
        "mode": mode,
        "sequence_index": spec.sequence_index,
        "phase": spec.phase,
        "phase_index": spec.phase_index,
        "prompt": {
            "prompt_id": spec.prompt_id,
            "text": spec.prompt_text,
            "token_ids": prompt_ids,
            "token_count": len(prompt_ids),
        },
        "generation": {
            "decoding": "greedy",
            "stop_on_eos": False,
            "max_new_tokens": spec.max_new_tokens,
            "token_ids": generated_ids,
            "token_count": len(generated_ids),
            "continuation_text": continuation,
            "full_text": full_text,
            "eos_generated": 1 in generated_ids,
            "stop_reason": "fixed_max_new_tokens",
            "context_crop_events": context_crop_events,
            "all_token_ids_in_range": True,
            "all_logits_finite": True,
        },
        "timing": {
            "ttft_available": True,
            "ttft_seconds": first_token_seconds,
            "decode_only_timing_available": decode_tokens_per_second is not None,
            "decode_token_count": decode_token_count,
            "decode_seconds": decode_seconds,
            "decode_tokens_per_second": decode_tokens_per_second,
            "end_to_end_seconds": total_seconds,
            "end_to_end_tokens_per_second": len(generated_ids) / total_seconds,
        },
        "memory": {
            "system_before": system_before,
            "system_after": system_after,
            "cuda_before": cuda_before,
            "cuda_after": cuda_after,
        },
        "runtime": {
            "device": str(device),
            "precision": session.precision,
            "weight_dtype": str(session.weight_dtype),
            "compute_dtype": str(session.compute_dtype),
            "model_training": False,
            "inference_mode": inference_mode_observed,
            "kv_cache_enabled": False,
            "decode_implementation": "full_prefix_recompute",
        },
    }


def summary_statistics(values: Sequence[float]) -> dict[str, float]:
    normalized = [float(value) for value in values]
    if not normalized:
        raise BenchmarkError("cannot summarize an empty measurement set")
    if any(not math.isfinite(value) or value < 0.0 for value in normalized):
        raise BenchmarkError("measurement set contains a negative or non-finite value")
    ordered = sorted(normalized)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "p95": ordered[p95_index],
    }


def _count_by_phase(plan: Sequence[RunSpec], phase: str) -> int:
    return sum(spec.phase == phase for spec in plan)


def build_summary(
    session: DeploymentSession,
    *,
    mode: str,
    run_id: str,
    plan: Sequence[RunSpec],
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not plan or not samples:
        raise BenchmarkError("run plan and completed samples must be non-empty")
    if len(plan) != len(samples):
        raise BenchmarkError(
            f"completed sample count mismatch: {len(samples)} != {len(plan)}"
        )
    planned_warmups = _count_by_phase(plan, "warmup")
    planned_measured = len(plan) - planned_warmups
    warmups = [sample for sample in samples if sample["phase"] == "warmup"]
    measured = [sample for sample in samples if sample["phase"] != "warmup"]
    if len(warmups) != planned_warmups:
        raise BenchmarkError("completed warmup count does not match the run plan")
    if len(measured) != planned_measured:
        raise BenchmarkError("completed measured count does not match the run plan")
    if not measured:
        raise BenchmarkError("measured sample set must not be empty")

    max_new_tokens = {sample["generation"]["max_new_tokens"] for sample in measured}
    if len(max_new_tokens) != 1:
        raise BenchmarkError("measured requests do not share max_new_tokens")
    expected_per_request = next(iter(max_new_tokens))
    if any(
        sample["generation"]["token_count"] != expected_per_request
        for sample in measured
    ):
        raise BenchmarkError("one or more measured requests has an incomplete token count")
    measured_generated_tokens = sum(
        int(sample["generation"]["token_count"]) for sample in measured
    )
    all_generated_tokens = sum(
        int(sample["generation"]["token_count"]) for sample in samples
    )
    ttft_values = [float(sample["timing"]["ttft_seconds"]) for sample in measured]
    end_to_end_values = [
        float(sample["timing"]["end_to_end_tokens_per_second"])
        for sample in measured
    ]
    decode_values = [
        float(sample["timing"]["decode_tokens_per_second"])
        for sample in measured
        if sample["timing"]["decode_tokens_per_second"] is not None
    ]
    if not decode_values:
        raise BenchmarkError("no measured request exposed decode-only throughput")
    peak_allocated = max(
        int(sample["memory"]["cuda_after"]["peak_allocated_bytes"])
        for sample in samples
    )
    peak_reserved = max(
        int(sample["memory"]["cuda_after"]["peak_reserved_bytes"])
        for sample in samples
    )
    prompt_ids = list(dict.fromkeys(sample["prompt"]["prompt_id"] for sample in measured))
    return {
        "format_name": "small_gpt_jetson_inference_summary",
        "schema_version": 1,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "mode": mode,
        "protocol": {
            "deployment_protocol_id": session.protocol["protocol_id"],
            "deployment_protocol_fingerprint": session.protocol_fingerprint,
            "smoke_protocol_id": session.protocol["smoke"]["protocol_id"],
            "benchmark_protocol_id": session.protocol["benchmark"]["protocol_id"],
            "stability_protocol_id": session.protocol["stability"]["protocol_id"],
        },
        "identity": {
            "source": session.source.to_dict(),
            "config": session.config_identity.to_dict(),
            "checkpoint": {
                **session.checkpoint_identity.to_dict(),
                "load_mode": "model_only",
            },
            "tokenizer": session.tokenizer_identity.to_dict(),
            "control_run_id": session.loaded_checkpoint.state.run_id,
        },
        "model": {
            "parameters": sum(parameter.numel() for parameter in session.model.parameters()),
            "context_length": int(session.model_config.context_length),
            "vocab_size": int(session.model_config.vocab_size),
            "dropout": float(session.model_config.dropout),
            "training": False,
            "artifacts_loaded_once": True,
        },
        "runtime": {
            **session.runtime,
            "precision": session.precision,
            "weight_dtype": str(session.weight_dtype),
            "compute_dtype": str(session.compute_dtype),
            "model_load_seconds": session.model_load_seconds,
            "kv_cache_enabled": False,
            "decode_implementation": "full_prefix_recompute",
            "power_mode": _power_mode(),
            "power_mode_changed_during_benchmark": False,
            "jetson_clocks": False,
        },
        "counts": {
            "prompt_ids": prompt_ids,
            "prompt_count": len(prompt_ids),
            "planned_warmup_runs": planned_warmups,
            "completed_warmup_runs": len(warmups),
            "planned_measured_runs": planned_measured,
            "completed_measured_runs": len(measured),
            "completed_total_requests": len(samples),
            "max_new_tokens_per_request": expected_per_request,
            "measured_generated_tokens": measured_generated_tokens,
            "all_generated_tokens": all_generated_tokens,
            "warmup_excluded_from_measured_summary": True,
        },
        "performance": {
            "model_load_seconds": session.model_load_seconds,
            "first_request_wall_seconds": float(
                samples[0]["timing"]["end_to_end_seconds"]
            ),
            "ttft_seconds": summary_statistics(ttft_values),
            "decode_tokens_per_second": summary_statistics(decode_values),
            "end_to_end_tokens_per_second": summary_statistics(end_to_end_values),
            "primary_throughput_metric": "end_to_end_tokens_per_second",
            "ttft_available": True,
            "decode_only_timing_available": True,
        },
        "memory": {
            "cuda_peak_allocated_bytes": peak_allocated,
            "cuda_peak_reserved_bytes": peak_reserved,
            "system_before_first_request": samples[0]["memory"]["system_before"],
            "system_after_last_request": samples[-1]["memory"]["system_after"],
        },
        "reliability": {
            "failed_requests": 0,
            "oom_count": 0,
            "non_finite_count": 0,
            "all_logits_finite": True,
            "all_token_ids_in_range": True,
            "output_overwrite": False,
        },
        "claims": {
            "cross_machine_bitwise": False,
            "performance_scope": "same_device_descriptive",
            "text_quality_gate": False,
        },
    }


def strict_jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise BenchmarkError("samples JSONL must contain at least one row")
    encoded: list[str] = []
    for index, row in enumerate(rows):
        try:
            encoded.append(
                json.dumps(
                    dict(row),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
        except (TypeError, ValueError) as error:
            raise BenchmarkError(f"sample {index} is not strict finite JSON: {error}") from error
    return ("\n".join(encoded) + "\n").encode("utf-8")


def _published_file_identity(path: Path) -> dict[str, Any]:
    return {
        "filename": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def publish_success_outputs(
    output_dir: Path,
    *,
    samples: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> Path:
    samples_path = output_dir / SAMPLES_FILENAME
    summary_path = output_dir / SUMMARY_FILENAME
    manifest_path = output_dir / MANIFEST_FILENAME
    atomic_write_bytes_exclusive(samples_path, strict_jsonl_bytes(samples))
    atomic_write_json_exclusive(summary_path, summary)
    manifest = {
        "format_name": "small_gpt_jetson_inference_manifest",
        "schema_version": 1,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": summary["run_id"],
        "mode": summary["mode"],
        "source": summary["identity"]["source"],
        "protocol": summary["protocol"],
        "precision": summary["runtime"]["precision"],
        "files": [
            _published_file_identity(samples_path),
            _published_file_identity(summary_path),
        ],
        "completed_sample_count": len(samples),
        "manifest_published_last": True,
        "gate": "PASS",
    }
    atomic_write_json_exclusive(manifest_path, manifest)
    return manifest_path


def publish_failure(output_dir: Path, *, error: BaseException, args: argparse.Namespace) -> None:
    manifest_path = output_dir / MANIFEST_FILENAME
    if manifest_path.exists():
        return
    failure_path = output_dir / FAILURE_FILENAME
    if failure_path.exists():
        return
    payload = {
        "format_name": "small_gpt_jetson_inference_failure",
        "schema_version": 1,
        "status": "failed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": getattr(args, "run_id", None),
        "mode": getattr(args, "mode", None),
        "precision": getattr(args, "precision", None),
        "error_type": type(error).__name__,
        "error": str(error),
        "manifest_published": False,
    }
    try:
        atomic_write_json_exclusive(failure_path, payload)
    except Exception:
        pass


def run_benchmark(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    run_id = validate_run_id(args.run_id)
    protocol = load_protocol(args.protocol)
    plan = build_run_plan(protocol, args.mode)
    output_dir = reserve_output_directory(args.output_dir)
    try:
        session = load_deployment_session(
            protocol_path=args.protocol,
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            tokenizer_path=args.tokenizer,
            tokenizer_config_path=None,
            requested_device=args.device,
            precision=args.precision,
            project_root=PROJECT_ROOT,
            enforce_source=True,
            perform_forward_probe=False,
        )
        samples: list[dict[str, Any]] = []
        for spec in plan:
            samples.append(
                run_greedy_request(
                    session,
                    spec,
                    run_id=run_id,
                    mode=args.mode,
                )
            )
        summary = build_summary(
            session,
            mode=args.mode,
            run_id=run_id,
            plan=plan,
            samples=samples,
        )
        manifest_path = publish_success_outputs(
            output_dir,
            samples=samples,
            summary=summary,
        )
        return summary, manifest_path
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            publish_failure(output_dir, error=error, args=args)
            raise
        publish_failure(output_dir, error=error, args=args)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run frozen smoke, steady-state benchmark, or sequential stability "
            "requests on one CUDA Jetson device."
        )
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "configs" / "day13_jetson_deployment_protocol.json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "baseline.yaml",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=PROJECT_ROOT / "tokenizer" / "artifacts" / "tokenizer.json",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp32", "fp16"), required=True)
    parser.add_argument(
        "--mode",
        choices=("smoke", "benchmark", "stability"),
        required=True,
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args: argparse.Namespace | None = None
    try:
        args = parse_args(argv)
        summary, manifest_path = run_benchmark(args)
        payload = {
            "Day13JetsonInference": "PASS",
            "RunID": summary["run_id"],
            "Mode": summary["mode"],
            "Precision": summary["runtime"]["precision"],
            "CompletedSamples": summary["counts"]["completed_total_requests"],
            "MeasuredSamples": summary["counts"]["completed_measured_runs"],
            "GeneratedTokens": summary["counts"]["measured_generated_tokens"],
            "Manifest": manifest_path.as_posix(),
        }
        print(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
        return 0
    except (DeploymentError, OSError, RuntimeError, ValueError, TypeError) as error:
        payload = {
            "Day13JetsonInference": "FAIL",
            "RunID": None if args is None else getattr(args, "run_id", None),
            "Mode": None if args is None else getattr(args, "mode", None),
            "ErrorType": type(error).__name__,
            "Error": str(error),
            "ManifestPublished": False,
        }
        print(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
