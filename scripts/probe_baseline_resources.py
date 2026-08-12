from __future__ import annotations

import argparse
import math
import os
import platform
import random
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import replace
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from model import GPT, GPTConfig  # noqa: E402
from train import (  # noqa: E402
    PrecisionPolicy,
    Trainer,
    TrainerState,
    TrainingConfig,
    TrainingDataStream,
    build_optimizer,
    build_scheduler,
    partition_parameters,
)
from train.resource_probe import (  # noqa: E402
    RESOURCE_FIELDS,
    RESOURCE_PROBE_SCHEMA_VERSION,
    ProbeCandidate,
    ProbeSettings,
    ResourceProbeError,
    atomic_write_json,
    build_loader_candidates,
    build_micro_batch_candidates,
    canonical_sha256,
    is_cuda_oom,
    load_json_object,
    propose_accumulation,
    recommend_loader,
    recommend_micro_batch,
    resolve_candidate_config,
    sha256_file,
    tail_text,
    validate_candidate_result,
)


PROBE_KIND = "small_gpt_baseline_resource_probe"
EXPECTED_BASELINE_PARAMETERS = 33_833_984
RUNTIME_CONTRACT_FIELDS = (
    "python_version",
    "platform",
    "torch_version",
    "torch_cuda_version",
    "cudnn_version",
    "cuda_device_name",
    "cuda_capability",
    "total_device_memory_bytes",
    "bf16_supported",
)


class ProbeEntryError(RuntimeError):
    """Raised when the probe CLI cannot proceed safely."""


def _positive_int_arg(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _non_negative_int_arg(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _fraction_arg(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("must be in (0, 1)")
    return parsed


def _positive_float_arg(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run isolated RTX resource candidates for the frozen Small GPT "
            "Baseline without editing configs/baseline.yaml."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "baseline.yaml",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--phase",
        choices=("microbatch", "loader", "single"),
        default="microbatch",
    )
    parser.add_argument(
        "--micro-batch-sizes",
        type=_positive_int_arg,
        nargs="+",
    )
    parser.add_argument("--micro-batch-size", type=_positive_int_arg)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=_positive_int_arg,
        default=1,
    )
    parser.add_argument(
        "--num-workers",
        type=_non_negative_int_arg,
        nargs="+",
        default=[0],
    )
    parser.add_argument(
        "--pin-memory",
        choices=("false", "true", "both"),
        default="true",
    )
    parser.add_argument(
        "--warmup-updates",
        type=_positive_int_arg,
        default=1,
    )
    parser.add_argument(
        "--measured-updates",
        type=_positive_int_arg,
        default=3,
    )
    parser.add_argument(
        "--candidate-timeout-seconds",
        type=_positive_int_arg,
        default=900,
    )
    parser.add_argument(
        "--max-reserved-fraction",
        type=_fraction_arg,
        default=0.85,
    )
    parser.add_argument(
        "--target-tokens-per-update",
        type=_positive_int_arg,
        help=(
            "Optional arithmetic target used to propose accumulation; it is "
            "never written back to the Baseline config."
        ),
    )
    parser.add_argument(
        "--expected-device-name",
        default="RTX 5090",
        help="Required substring of the CUDA device name.",
    )
    parser.add_argument(
        "--minimum-device-memory-gib",
        type=_positive_float_arg,
        default=30.0,
    )
    parser.add_argument("--verify-hashes", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted report with the exact same request.",
    )
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--_worker-result",
        type=Path,
        help=argparse.SUPPRESS,
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_status() -> tuple[str, tuple[str, ...]]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip().lower()
        status_output = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ProbeEntryError(
            "resource probing requires a readable Git worktree"
        ) from error
    return commit, tuple(
        line for line in status_output.splitlines() if line.strip()
    )


def validate_source_status(output_path: Path) -> str:
    commit, status_lines = source_status()
    allowed_path: str | None = None
    try:
        allowed_path = output_path.resolve().relative_to(
            PROJECT_ROOT
        ).as_posix()
    except ValueError:
        pass
    unexpected: list[str] = []
    for line in status_lines:
        raw_path = line[3:] if len(line) >= 4 else ""
        if " -> " in raw_path or raw_path != allowed_path:
            unexpected.append(line)
    if unexpected:
        raise ProbeEntryError(
            "resource probing requires a clean Git worktree except for its "
            f"own output file; found {unexpected[:5]}"
        )
    return commit


def relative_identity(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.name


def pin_memory_values(raw: str) -> tuple[bool, ...]:
    if raw == "false":
        return (False,)
    if raw == "true":
        return (True,)
    if raw == "both":
        return (False, True)
    raise ProbeEntryError(f"unsupported --pin-memory value {raw!r}")


def candidates_from_args(args: argparse.Namespace) -> tuple[ProbeCandidate, ...]:
    pin_values = pin_memory_values(args.pin_memory)
    workers = tuple(args.num_workers)
    if len(set(workers)) != len(workers):
        raise ProbeEntryError("--num-workers values must be unique")

    if args.phase == "microbatch":
        sizes = args.micro_batch_sizes
        if sizes is None:
            raise ProbeEntryError(
                "microbatch phase requires --micro-batch-sizes"
            )
        if args.micro_batch_size is not None:
            raise ProbeEntryError(
                "microbatch phase does not accept --micro-batch-size"
            )
        if sizes != sorted(set(sizes)):
            raise ProbeEntryError(
                "--micro-batch-sizes must be unique and strictly increasing"
            )
        if len(workers) != 1 or len(pin_values) != 1:
            raise ProbeEntryError(
                "microbatch phase keeps one workers/pin-memory tuple fixed"
            )
        return build_micro_batch_candidates(
            sizes,
            gradient_accumulation_steps=(
                args.gradient_accumulation_steps
            ),
            num_workers=workers[0],
            pin_memory=pin_values[0],
        )

    if args.micro_batch_sizes is not None:
        raise ProbeEntryError(
            f"{args.phase} phase does not accept --micro-batch-sizes"
        )
    if args.micro_batch_size is None:
        raise ProbeEntryError(
            f"{args.phase} phase requires --micro-batch-size"
        )
    if args.phase == "loader":
        return build_loader_candidates(
            micro_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=(
                args.gradient_accumulation_steps
            ),
            num_workers_values=workers,
            pin_memory_values=pin_values,
        )
    if len(workers) != 1 or len(pin_values) != 1:
        raise ProbeEntryError(
            "single phase requires exactly one workers/pin-memory tuple"
        )
    return (
        ProbeCandidate(
            micro_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=(
                args.gradient_accumulation_steps
            ),
            num_workers=workers[0],
            pin_memory=pin_values[0],
        ),
    )


def configure_runtime(config: TrainingConfig) -> None:
    if config.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(config.deterministic)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
    torch.backends.cudnn.allow_tf32 = config.allow_tf32
    torch.backends.cudnn.deterministic = config.deterministic
    torch.backends.cudnn.benchmark = False


def runtime_snapshot(device: torch.device) -> dict[str, Any]:
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    cudnn_version = torch.backends.cudnn.version()
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": cudnn_version,
        "cuda_device_index": index,
        "cuda_device_name": properties.name,
        "cuda_capability": list(torch.cuda.get_device_capability(index)),
        "total_device_memory_bytes": int(properties.total_memory),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
    }


def failure_result(
    candidate: ProbeCandidate,
    *,
    status: str,
    started_at_utc: str,
    error_type: str,
    error_message: str,
    traceback_tail: str = "",
) -> dict[str, Any]:
    return {
        "schema_version": RESOURCE_PROBE_SCHEMA_VERSION,
        "candidate_key": candidate.key,
        "candidate": candidate.to_dict(),
        "status": status,
        "started_at_utc": started_at_utc,
        "finished_at_utc": utc_now(),
        "error_type": error_type,
        "error_message": error_message,
        "traceback_tail": tail_text(traceback_tail),
    }


def worker_candidate(args: argparse.Namespace) -> ProbeCandidate:
    if args.micro_batch_size is None:
        raise ProbeEntryError("worker requires --micro-batch-size")
    workers = tuple(args.num_workers)
    pins = pin_memory_values(args.pin_memory)
    if len(workers) != 1 or len(pins) != 1:
        raise ProbeEntryError(
            "worker requires one workers/pin-memory tuple"
        )
    return ProbeCandidate(
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_workers=workers[0],
        pin_memory=pins[0],
    )


def run_worker(args: argparse.Namespace) -> int:
    if args._worker_result is None:
        raise ProbeEntryError("worker requires --_worker-result")
    candidate = worker_candidate(args)
    started_at_utc = utc_now()
    try:
        config_path = args.config.resolve()
        manifest_path = args.manifest.resolve()
        base_config = TrainingConfig.from_yaml(config_path)
        model_config = GPTConfig.from_yaml(config_path)
        config, plan = resolve_candidate_config(base_config, candidate)
        settings = ProbeSettings(
            warmup_updates=args.warmup_updates,
            measured_updates=args.measured_updates,
            candidate_timeout_seconds=args.candidate_timeout_seconds,
            max_reserved_fraction=args.max_reserved_fraction,
            verify_hashes=args.verify_hashes,
        )
        precision = PrecisionPolicy.from_config(config)
        if precision.device.type != "cuda":
            raise ProbeEntryError("worker did not resolve a CUDA device")
        configure_runtime(config)
        runtime = runtime_snapshot(precision.device)
        expected_name = args.expected_device_name.strip()
        if not expected_name:
            raise ProbeEntryError("--expected-device-name must not be empty")
        if expected_name.lower() not in runtime["cuda_device_name"].lower():
            raise ProbeEntryError(
                "CUDA device name does not match the explicit gate: "
                f"expected substring {expected_name!r}, "
                f"found {runtime['cuda_device_name']!r}"
            )
        minimum_memory_bytes = int(
            args.minimum_device_memory_gib * (1024**3)
        )
        if runtime["total_device_memory_bytes"] < minimum_memory_bytes:
            raise ProbeEntryError(
                "CUDA device memory is below the explicit gate: "
                f"required {args.minimum_device_memory_gib:.2f} GiB, "
                f"found "
                f"{runtime['total_device_memory_bytes'] / (1024**3):.2f} GiB"
            )

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(precision.device)
        free_before, total_memory = torch.cuda.mem_get_info(precision.device)

        model = GPT(model_config).to(precision.device)
        groups = partition_parameters(model)
        optimizer = build_optimizer(model, config)
        scheduler = build_scheduler(optimizer, config, plan)
        state = TrainerState(run_id=f"resource-probe-{candidate.key}")
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            state=state,
            config=config,
            plan=plan,
            precision=precision,
        )
        allocated_after_setup = int(
            torch.cuda.memory_allocated(precision.device)
        )
        reserved_after_setup = int(
            torch.cuda.memory_reserved(precision.device)
        )

        measured_metrics: list[dict[str, Any]] = []
        with TrainingDataStream(
            manifest_path,
            config=config,
            plan=plan,
            state=state,
            verify_hashes=settings.verify_hashes,
        ) as batches:
            for _ in range(settings.warmup_updates):
                trainer.run_update(
                    islice(
                        batches,
                        candidate.gradient_accumulation_steps,
                    )
                )
            torch.cuda.synchronize(precision.device)
            torch.cuda.reset_peak_memory_stats(precision.device)
            measured_start = time.perf_counter()
            for _ in range(settings.measured_updates):
                metrics = trainer.run_update(
                    islice(
                        batches,
                        candidate.gradient_accumulation_steps,
                    )
                )
                measured_metrics.append(metrics.to_dict())
            torch.cuda.synchronize(precision.device)
            elapsed_seconds = time.perf_counter() - measured_start

        measured_tokens = sum(
            int(metrics["tokens"]) for metrics in measured_metrics
        )
        peak_allocated = int(
            torch.cuda.max_memory_allocated(precision.device)
        )
        peak_reserved = int(
            torch.cuda.max_memory_reserved(precision.device)
        )
        allocated_after_measurement = int(
            torch.cuda.memory_allocated(precision.device)
        )
        reserved_after_measurement = int(
            torch.cuda.memory_reserved(precision.device)
        )
        free_after, total_memory_after = torch.cuda.mem_get_info(
            precision.device
        )
        if int(total_memory_after) != int(total_memory):
            raise ProbeEntryError("CUDA total memory changed during candidate")
        losses = [
            float(metrics["raw_token_weighted_loss"])
            for metrics in measured_metrics
        ]
        grad_norms = [
            float(metrics["grad_norm_before_clip"])
            for metrics in measured_metrics
        ]
        result = {
            "schema_version": RESOURCE_PROBE_SCHEMA_VERSION,
            "candidate_key": candidate.key,
            "candidate": candidate.to_dict(),
            "status": "ok",
            "started_at_utc": started_at_utc,
            "finished_at_utc": utc_now(),
            "config_path": relative_identity(config_path),
            "manifest_path": relative_identity(manifest_path),
            "runtime": runtime,
            "plan": plan.to_dict(),
            "model_parameters": groups.total_numel,
            "warmup_updates_completed": settings.warmup_updates,
            "measured_updates_completed": len(measured_metrics),
            "measured_tokens": measured_tokens,
            "elapsed_seconds": elapsed_seconds,
            "tokens_per_second": measured_tokens / elapsed_seconds,
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "loss_mean": sum(losses) / len(losses),
            "grad_norm_max": max(grad_norms),
            "learning_rate_first": measured_metrics[0]["learning_rate"],
            "learning_rate_last": measured_metrics[-1]["learning_rate"],
            "allocated_after_setup_bytes": allocated_after_setup,
            "reserved_after_setup_bytes": reserved_after_setup,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "peak_reserved_fraction": peak_reserved / int(total_memory),
            "allocated_after_measurement_bytes": allocated_after_measurement,
            "reserved_after_measurement_bytes": reserved_after_measurement,
            "free_device_memory_before_bytes": int(free_before),
            "free_device_memory_after_bytes": int(free_after),
            "total_device_memory_bytes": int(total_memory),
            "measured_update_metrics": measured_metrics,
        }
    except Exception as error:
        status = "oom" if is_cuda_oom(error) else "error"
        result = failure_result(
            candidate,
            status=status,
            started_at_utc=started_at_utc,
            error_type=type(error).__name__,
            error_message=str(error) or repr(error),
            traceback_tail=traceback.format_exc(),
        )

    atomic_write_json(args._worker_result, result)
    return 0


def validate_base_inputs(
    config_path: Path,
    manifest_path: Path,
) -> tuple[TrainingConfig, GPTConfig]:
    if not config_path.is_file():
        raise ProbeEntryError(f"config does not exist: {config_path}")
    if not manifest_path.is_file():
        raise ProbeEntryError(f"manifest does not exist: {manifest_path}")
    base_config = TrainingConfig.from_yaml(config_path)
    model_config = GPTConfig.from_yaml(config_path)
    if base_config.device != "cuda" or base_config.precision != "bf16":
        raise ProbeEntryError(
            "Baseline probe requires device=cuda and precision=bf16"
        )
    if base_config.project_name != "small-gpt-baseline":
        raise ProbeEntryError(
            "resource probe requires project.name='small-gpt-baseline'"
        )
    if base_config.context_length != model_config.context_length:
        raise ProbeEntryError("training/model context lengths disagree")
    if base_config.vocab_size != model_config.vocab_size:
        raise ProbeEntryError("training/model vocabulary sizes disagree")
    if model_config.context_length != 512 or model_config.vocab_size != 16_384:
        raise ProbeEntryError(
            "Baseline resource probe requires context=512 and vocab=16384"
        )
    if model_config.parameter_count != EXPECTED_BASELINE_PARAMETERS:
        raise ProbeEntryError(
            "Baseline parameter contract changed: "
            f"expected {EXPECTED_BASELINE_PARAMETERS:,}, "
            f"found {model_config.parameter_count:,}"
        )
    if (
        base_config.max_steps is not None
        or base_config.target_tokens != 300_000_000
        or base_config.warmup_steps is not None
        or base_config.warmup_ratio != 0.02
    ):
        raise ProbeEntryError(
            "Baseline budget must remain target_tokens=300000000 with "
            "warmup_ratio=0.02"
        )
    if base_config.unresolved_fields != RESOURCE_FIELDS:
        raise ProbeEntryError(
            "Baseline must keep exactly the four resource fields unresolved "
            f"before probing; found {list(base_config.unresolved_fields)}"
        )
    return base_config, model_config


def request_payload(
    *,
    args: argparse.Namespace,
    config_path: Path,
    manifest_path: Path,
    base_config: TrainingConfig,
    model_config: GPTConfig,
    candidates: tuple[ProbeCandidate, ...],
    settings: ProbeSettings,
    source_commit: str,
) -> dict[str, Any]:
    return {
        "phase": args.phase,
        "source": {
            "commit": source_commit,
            "dirty_excluding_probe_output": False,
            "entry_sha256": sha256_file(Path(__file__).resolve()),
            "core_sha256": sha256_file(
                PROJECT_ROOT / "train" / "resource_probe.py"
            ),
        },
        "config": {
            "path": relative_identity(config_path),
            "sha256": sha256_file(config_path),
            "unresolved_fields_before_probe": list(
                base_config.unresolved_fields
            ),
        },
        "manifest": {
            "path": relative_identity(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "model": model_config.to_dict(),
        "baseline_training": base_config.to_dict(),
        "candidates": [candidate.to_dict() for candidate in candidates],
        "settings": settings.to_dict(),
        "device_gate": {
            "expected_device_name": args.expected_device_name,
            "minimum_device_memory_gib": args.minimum_device_memory_gib,
        },
        "target_tokens_per_update": args.target_tokens_per_update,
    }


def initial_report(
    *,
    request: dict[str, Any],
    fingerprint: str,
) -> dict[str, Any]:
    created_at = utc_now()
    return {
        "schema_version": RESOURCE_PROBE_SCHEMA_VERSION,
        "kind": PROBE_KIND,
        "probe_status": "running",
        "created_at_utc": created_at,
        "updated_at_utc": created_at,
        "source": dict(request["source"]),
        "request_fingerprint": fingerprint,
        "request": request,
        "results": [],
        "recommendation": None,
        "stopped_reason": None,
        "unattempted_candidate_keys": [],
        "runtime_contract": None,
    }


def load_resume_report(
    output_path: Path,
    *,
    fingerprint: str,
) -> dict[str, Any]:
    report = load_json_object(output_path)
    if report.get("schema_version") != RESOURCE_PROBE_SCHEMA_VERSION:
        raise ProbeEntryError("resume report schema_version does not match")
    if report.get("kind") != PROBE_KIND:
        raise ProbeEntryError("resume report kind does not match")
    if report.get("request_fingerprint") != fingerprint:
        raise ProbeEntryError(
            "resume request does not exactly match the existing report"
        )
    results = report.get("results")
    if not isinstance(results, list):
        raise ProbeEntryError("resume report results must be a list")
    seen: set[str] = set()
    runtime_contract = report.get("runtime_contract")
    for result in results:
        validated = validate_candidate_result(result)
        key = validated["candidate_key"]
        if key in seen:
            raise ProbeEntryError(
                f"resume report contains duplicate result {key!r}"
            )
        seen.add(key)
        if validated["status"] == "ok":
            observed = runtime_contract_from_result(validated)
            if runtime_contract is None:
                runtime_contract = observed
            elif observed != runtime_contract:
                raise ProbeEntryError(
                    "resume report mixes incompatible worker runtimes"
                )
    report["runtime_contract"] = runtime_contract
    report["probe_status"] = "running"
    report["stopped_reason"] = None
    report["unattempted_candidate_keys"] = []
    report["updated_at_utc"] = utc_now()
    return report


def subprocess_failure(
    candidate: ProbeCandidate,
    *,
    started_at_utc: str,
    status: str,
    error_type: str,
    error_message: str,
    stdout: str = "",
    stderr: str = "",
    returncode: int | None = None,
) -> dict[str, Any]:
    result = failure_result(
        candidate,
        status=status,
        started_at_utc=started_at_utc,
        error_type=error_type,
        error_message=error_message,
    )
    result["worker_returncode"] = returncode
    result["worker_stdout_tail"] = tail_text(stdout)
    result["worker_stderr_tail"] = tail_text(stderr)
    return result


def terminate_worker_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        if process.poll() is None:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except OSError:
                pass
            process.wait()


def subprocess_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def runtime_contract_from_result(result: dict[str, Any]) -> dict[str, Any]:
    runtime = result.get("runtime")
    if not isinstance(runtime, dict):
        raise ProbeEntryError(
            "successful worker result is missing its runtime identity"
        )
    missing = [field for field in RUNTIME_CONTRACT_FIELDS if field not in runtime]
    if missing:
        raise ProbeEntryError(
            f"worker runtime identity is missing fields: {missing}"
        )
    return {field: runtime[field] for field in RUNTIME_CONTRACT_FIELDS}


def enforce_runtime_contract(
    report: dict[str, Any],
    result: dict[str, Any],
    candidate: ProbeCandidate,
) -> dict[str, Any]:
    if result["status"] != "ok":
        return result
    observed = runtime_contract_from_result(result)
    expected = report.get("runtime_contract")
    if expected is None:
        report["runtime_contract"] = observed
        return result
    if observed == expected:
        return result
    failure = failure_result(
        candidate,
        status="error",
        started_at_utc=result["started_at_utc"],
        error_type="RuntimeIdentityMismatch",
        error_message=(
            "candidate worker runtime differs from earlier successful workers"
        ),
    )
    failure["expected_runtime_contract"] = expected
    failure["observed_runtime_contract"] = observed
    failure["worker_returncode"] = result.get("worker_returncode")
    failure["worker_stdout_tail"] = result.get("worker_stdout_tail", "")
    failure["worker_stderr_tail"] = result.get("worker_stderr_tail", "")
    return failure


def run_candidate_subprocess(
    *,
    args: argparse.Namespace,
    candidate: ProbeCandidate,
    config_path: Path,
    manifest_path: Path,
    settings: ProbeSettings,
    temporary_directory: Path,
) -> dict[str, Any]:
    result_path = temporary_directory / f"{candidate.key}.json"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_worker",
        "--config",
        str(config_path),
        "--manifest",
        str(manifest_path),
        "--micro-batch-size",
        str(candidate.micro_batch_size),
        "--gradient-accumulation-steps",
        str(candidate.gradient_accumulation_steps),
        "--num-workers",
        str(candidate.num_workers),
        "--pin-memory",
        "true" if candidate.pin_memory else "false",
        "--warmup-updates",
        str(settings.warmup_updates),
        "--measured-updates",
        str(settings.measured_updates),
        "--candidate-timeout-seconds",
        str(settings.candidate_timeout_seconds),
        "--max-reserved-fraction",
        str(settings.max_reserved_fraction),
        "--expected-device-name",
        args.expected_device_name,
        "--minimum-device-memory-gib",
        str(args.minimum_device_memory_gib),
        "--_worker-result",
        str(result_path),
    ]
    if settings.verify_hashes:
        command.append("--verify-hashes")
    started_at_utc = utc_now()
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    process: subprocess.Popen[str] | None = None
    try:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=(os.name == "posix"),
        )
        stdout, stderr = process.communicate(
            timeout=settings.candidate_timeout_seconds
        )
    except subprocess.TimeoutExpired as error:
        assert process is not None
        terminate_worker_process(process)
        remaining_stdout, remaining_stderr = process.communicate()
        return subprocess_failure(
            candidate,
            started_at_utc=started_at_utc,
            status="timeout",
            error_type="CandidateTimeout",
            error_message=(
                "candidate exceeded "
                f"{settings.candidate_timeout_seconds} seconds"
            ),
            stdout=(
                subprocess_text(remaining_stdout)
                or subprocess_text(error.stdout)
            ),
            stderr=(
                subprocess_text(remaining_stderr)
                or subprocess_text(error.stderr)
            ),
        )
    except KeyboardInterrupt:
        if process is not None:
            terminate_worker_process(process)
        raise
    except OSError as error:
        return subprocess_failure(
            candidate,
            started_at_utc=started_at_utc,
            status="error",
            error_type=type(error).__name__,
            error_message=str(error),
        )

    assert process is not None
    returncode = process.returncode
    assert returncode is not None
    if returncode != 0:
        return subprocess_failure(
            candidate,
            started_at_utc=started_at_utc,
            status="error",
            error_type="WorkerProcessError",
            error_message=(
                f"candidate worker exited with code {returncode}"
            ),
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
        )
    if not result_path.is_file():
        return subprocess_failure(
            candidate,
            started_at_utc=started_at_utc,
            status="error",
            error_type="MissingWorkerResult",
            error_message="candidate worker did not write its result file",
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
        )
    try:
        result = validate_candidate_result(
            load_json_object(result_path),
            expected_candidate=candidate,
        )
    except ResourceProbeError as error:
        return subprocess_failure(
            candidate,
            started_at_utc=started_at_utc,
            status="error",
            error_type="InvalidWorkerResult",
            error_message=str(error),
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
        )
    if result["status"] == "ok":
        if result.get("model_parameters") != EXPECTED_BASELINE_PARAMETERS:
            return subprocess_failure(
                candidate,
                started_at_utc=started_at_utc,
                status="error",
                error_type="InvalidWorkerResult",
                error_message="worker model parameter count does not match Baseline",
                stdout=stdout,
                stderr=stderr,
                returncode=returncode,
            )
        plan = result.get("plan")
        if not isinstance(plan, dict) or any(
            plan.get(field) != getattr(candidate, field)
            for field in RESOURCE_FIELDS
        ):
            return subprocess_failure(
                candidate,
                started_at_utc=started_at_utc,
                status="error",
                error_type="InvalidWorkerResult",
                error_message="worker resolved plan does not match candidate",
                stdout=stdout,
                stderr=stderr,
                returncode=returncode,
            )
        if result.get("warmup_updates_completed") != settings.warmup_updates:
            return subprocess_failure(
                candidate,
                started_at_utc=started_at_utc,
                status="error",
                error_type="InvalidWorkerResult",
                error_message="worker warmup update count does not match request",
                stdout=stdout,
                stderr=stderr,
                returncode=returncode,
            )
        if result.get("measured_updates_completed") != settings.measured_updates:
            return subprocess_failure(
                candidate,
                started_at_utc=started_at_utc,
                status="error",
                error_type="InvalidWorkerResult",
                error_message="worker measured update count does not match request",
                stdout=stdout,
                stderr=stderr,
                returncode=returncode,
            )
    result["worker_returncode"] = returncode
    result["worker_stdout_tail"] = tail_text(stdout)
    result["worker_stderr_tail"] = tail_text(stderr)
    return result


def update_recommendation(
    report: dict[str, Any],
    *,
    phase: str,
    settings: ProbeSettings,
    target_tokens_per_update: int | None,
    base_config: TrainingConfig,
) -> None:
    results = report["results"]
    if phase == "microbatch":
        recommendation = recommend_micro_batch(
            results,
            max_reserved_fraction=settings.max_reserved_fraction,
        )
        if recommendation is not None and target_tokens_per_update is not None:
            selected = ProbeCandidate.from_mapping(
                recommendation["candidate"]
            )
            proposal = propose_accumulation(
                micro_batch_size=selected.micro_batch_size,
                context_length=base_config.context_length,
                requested_tokens_per_update=target_tokens_per_update,
            )
            proposed_candidate = replace(
                selected,
                gradient_accumulation_steps=(
                    proposal.gradient_accumulation_steps
                ),
            )
            _, proposed_plan = resolve_candidate_config(
                base_config,
                proposed_candidate,
            )
            recommendation["accumulation_proposal"] = proposal.to_dict()
            recommendation["resolved_math_if_accepted"] = (
                proposed_plan.to_dict()
            )
        report["recommendation"] = recommendation
    elif phase == "loader":
        report["recommendation"] = recommend_loader(results)
    else:
        report["recommendation"] = None


def print_candidate_result(result: dict[str, Any]) -> None:
    key = result["candidate_key"]
    status = result["status"]
    if status == "ok":
        throughput = float(result["tokens_per_second"])
        allocated_gib = float(result["peak_allocated_bytes"]) / (1024**3)
        reserved_gib = float(result["peak_reserved_bytes"]) / (1024**3)
        reserved_percent = float(result["peak_reserved_fraction"]) * 100.0
        print(
            f"{key}: ok | {throughput:,.0f} tok/s | "
            f"peak allocated {allocated_gib:.2f} GiB | "
            f"reserved {reserved_gib:.2f} GiB ({reserved_percent:.1f}%)",
            flush=True,
        )
    else:
        print(
            f"{key}: {status} | {result['error_type']}: "
            f"{result['error_message']}",
            flush=True,
        )


def run_parent(args: argparse.Namespace) -> int:
    if args.output is None:
        raise ProbeEntryError("parent probe requires --output")
    config_path = args.config.resolve()
    manifest_path = args.manifest.resolve()
    output_path = args.output.resolve()
    if output_path in {config_path, manifest_path}:
        raise ProbeEntryError("--output must not overwrite an input file")
    candidates = candidates_from_args(args)
    settings = ProbeSettings(
        warmup_updates=args.warmup_updates,
        measured_updates=args.measured_updates,
        candidate_timeout_seconds=args.candidate_timeout_seconds,
        max_reserved_fraction=args.max_reserved_fraction,
        verify_hashes=args.verify_hashes,
    )
    base_config, model_config = validate_base_inputs(
        config_path,
        manifest_path,
    )
    source_commit = validate_source_status(output_path)
    request = request_payload(
        args=args,
        config_path=config_path,
        manifest_path=manifest_path,
        base_config=base_config,
        model_config=model_config,
        candidates=candidates,
        settings=settings,
        source_commit=source_commit,
    )
    fingerprint = canonical_sha256(request)

    if output_path.exists():
        if not args.resume:
            raise ProbeEntryError(
                f"output already exists; use a new path or --resume: {output_path}"
            )
        report = load_resume_report(
            output_path,
            fingerprint=fingerprint,
        )
    else:
        if args.resume:
            raise ProbeEntryError(
                f"--resume requires an existing output: {output_path}"
            )
        report = initial_report(request=request, fingerprint=fingerprint)
    atomic_write_json(output_path, report)

    existing = {
        result["candidate_key"]: result
        for result in report["results"]
    }
    stopped_reason: str | None = None
    with tempfile.TemporaryDirectory(
        prefix="small-gpt-resource-probe-"
    ) as temporary:
        temporary_directory = Path(temporary)
        for candidate in candidates:
            current_commit = validate_source_status(output_path)
            if current_commit != source_commit:
                raise ProbeEntryError(
                    "Git commit changed while the resource probe was running"
                )
            if sha256_file(config_path) != request["config"]["sha256"]:
                raise ProbeEntryError(
                    "config changed while the resource probe was running"
                )
            if sha256_file(manifest_path) != request["manifest"]["sha256"]:
                raise ProbeEntryError(
                    "manifest changed while the resource probe was running"
                )
            prior = existing.get(candidate.key)
            if prior is not None and prior["status"] in {"ok", "oom"}:
                print(
                    f"{candidate.key}: resume keeps prior {prior['status']} result",
                    flush=True,
                )
                if prior["status"] == "oom":
                    stopped_reason = "prior_oom_boundary"
                    break
                continue

            print(f"{candidate.key}: starting isolated worker", flush=True)
            result = run_candidate_subprocess(
                args=args,
                candidate=candidate,
                config_path=config_path,
                manifest_path=manifest_path,
                settings=settings,
                temporary_directory=temporary_directory,
            )
            result = enforce_runtime_contract(report, result, candidate)
            if prior is None:
                report["results"].append(result)
            else:
                index = next(
                    index
                    for index, item in enumerate(report["results"])
                    if item["candidate_key"] == candidate.key
                )
                report["results"][index] = result
            existing[candidate.key] = result
            report["updated_at_utc"] = utc_now()
            update_recommendation(
                report,
                phase=args.phase,
                settings=settings,
                target_tokens_per_update=args.target_tokens_per_update,
                base_config=base_config,
            )
            atomic_write_json(output_path, report)
            print_candidate_result(result)

            if result["status"] == "oom":
                stopped_reason = "first_oom_boundary"
                break
            if result["status"] != "ok":
                stopped_reason = "candidate_failure"
                break

    attempted = set(existing)
    report["unattempted_candidate_keys"] = [
        candidate.key
        for candidate in candidates
        if candidate.key not in attempted
    ]
    update_recommendation(
        report,
        phase=args.phase,
        settings=settings,
        target_tokens_per_update=args.target_tokens_per_update,
        base_config=base_config,
    )
    report["stopped_reason"] = stopped_reason
    if stopped_reason in {"first_oom_boundary", "prior_oom_boundary"}:
        report["probe_status"] = "completed_at_oom_boundary"
    elif stopped_reason is not None:
        report["probe_status"] = "failed"
    elif report["unattempted_candidate_keys"]:
        report["probe_status"] = "interrupted"
    else:
        report["probe_status"] = "complete"
    report["updated_at_utc"] = utc_now()
    atomic_write_json(output_path, report)

    print(f"Report: {output_path}")
    recommendation = report["recommendation"]
    if recommendation is not None:
        print(
            "Preliminary recommendation: "
            f"{recommendation['candidate_key']}"
        )
    else:
        print("Preliminary recommendation: none")
    if report["probe_status"] == "failed":
        return 2
    if args.phase in {"microbatch", "loader"} and recommendation is None:
        return 2
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args._worker:
            return run_worker(args)
        return run_parent(args)
    except (ProbeEntryError, ResourceProbeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
