from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from scripts import probe_baseline_resources
from train import TrainingConfig
from train.resource_probe import (
    RESOURCE_PROBE_SCHEMA_VERSION,
    ProbeCandidate,
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
    tail_text,
    validate_candidate_result,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = PROJECT_ROOT / "configs" / "baseline.yaml"


def candidate(
    batch_size: int,
    *,
    accumulation: int = 1,
    workers: int = 0,
    pin_memory: bool = True,
) -> ProbeCandidate:
    return ProbeCandidate(
        micro_batch_size=batch_size,
        gradient_accumulation_steps=accumulation,
        num_workers=workers,
        pin_memory=pin_memory,
    )


def ok_result(
    value: ProbeCandidate,
    *,
    throughput: float,
    peak_reserved_bytes: int,
    total_memory_bytes: int = 1_000,
) -> dict:
    return {
        "schema_version": RESOURCE_PROBE_SCHEMA_VERSION,
        "candidate_key": value.key,
        "candidate": value.to_dict(),
        "status": "ok",
        "started_at_utc": "2026-08-12T00:00:00+00:00",
        "finished_at_utc": "2026-08-12T00:00:01+00:00",
        "measured_updates_completed": 3,
        "measured_tokens": 12_288,
        "elapsed_seconds": 1.0,
        "tokens_per_second": throughput,
        "peak_reserved_bytes": peak_reserved_bytes,
        "total_device_memory_bytes": total_memory_bytes,
        "peak_reserved_fraction": (
            peak_reserved_bytes / total_memory_bytes
        ),
    }


def failure_result(value: ProbeCandidate, status: str) -> dict:
    return {
        "schema_version": RESOURCE_PROBE_SCHEMA_VERSION,
        "candidate_key": value.key,
        "candidate": value.to_dict(),
        "status": status,
        "started_at_utc": "2026-08-12T00:00:00+00:00",
        "finished_at_utc": "2026-08-12T00:00:01+00:00",
        "error_type": "SyntheticFailure",
        "error_message": "expected test failure",
    }


def test_candidate_key_and_mapping_round_trip_are_exact():
    original = candidate(8, accumulation=4, workers=2, pin_memory=False)

    assert original.key == "b8-a4-w2-p0"
    assert ProbeCandidate.from_mapping(original.to_dict()) == original


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("micro_batch_size", 0),
        ("micro_batch_size", True),
        ("gradient_accumulation_steps", -1),
        ("num_workers", -1),
        ("num_workers", False),
        ("pin_memory", 1),
    ),
)
def test_candidate_rejects_invalid_resource_values(field, value):
    payload = candidate(1).to_dict()
    payload[field] = value

    with pytest.raises(ResourceProbeError, match=field):
        ProbeCandidate.from_mapping(payload)


def test_micro_batch_grid_preserves_strict_requested_order():
    candidates = build_micro_batch_candidates(
        [1, 2, 4, 8],
        gradient_accumulation_steps=1,
        num_workers=0,
        pin_memory=True,
    )

    assert [value.key for value in candidates] == [
        "b1-a1-w0-p1",
        "b2-a1-w0-p1",
        "b4-a1-w0-p1",
        "b8-a1-w0-p1",
    ]
    with pytest.raises(ResourceProbeError, match="duplicate"):
        build_micro_batch_candidates(
            [1, 1],
            gradient_accumulation_steps=1,
            num_workers=0,
            pin_memory=True,
        )


def test_loader_grid_is_the_exact_cartesian_product():
    candidates = build_loader_candidates(
        micro_batch_size=8,
        gradient_accumulation_steps=2,
        num_workers_values=[0, 2],
        pin_memory_values=[False, True],
    )

    assert [value.key for value in candidates] == [
        "b8-a2-w0-p0",
        "b8-a2-w0-p1",
        "b8-a2-w2-p0",
        "b8-a2-w2-p1",
    ]


def test_baseline_candidate_resolution_changes_only_resource_fields():
    baseline = TrainingConfig.from_yaml(BASELINE_PATH)
    resolved_config, plan = resolve_candidate_config(
        baseline,
        candidate(2, accumulation=4, workers=2, pin_memory=True),
    )

    before = baseline.to_dict()
    after = resolved_config.to_dict()
    changed = {key for key in before if before[key] != after[key]}
    assert changed == {
        "micro_batch_size",
        "gradient_accumulation_steps",
        "num_workers",
        "pin_memory",
    }
    assert baseline.unresolved_fields == (
        "micro_batch_size",
        "gradient_accumulation_steps",
        "num_workers",
        "pin_memory",
    )
    assert plan.tokens_per_micro_step == 2 * 512
    assert plan.tokens_per_update == 2 * 512 * 4
    assert plan.total_updates == math.ceil(300_000_000 / 4_096)
    assert plan.warmup_updates == math.ceil(plan.total_updates * 0.02)
    assert plan.planned_tokens - 300_000_000 == plan.token_overshoot


def test_candidate_resolution_requires_baseline_cuda_bf16_contract():
    baseline = TrainingConfig.from_yaml(BASELINE_PATH)
    cpu = replace(baseline, device="cpu", precision="fp32")

    with pytest.raises(ResourceProbeError, match="device='cuda'"):
        resolve_candidate_config(cpu, candidate(1))


def test_accumulation_proposal_rounds_up_without_mutating_a_config():
    proposal = propose_accumulation(
        micro_batch_size=6,
        context_length=512,
        requested_tokens_per_update=32_768,
    )

    assert proposal.tokens_per_micro_step == 3_072
    assert proposal.gradient_accumulation_steps == 11
    assert proposal.tokens_per_update == 33_792
    assert proposal.proposal_overshoot_tokens == 1_024


def test_atomic_json_write_round_trip_and_replacement(tmp_path):
    output = tmp_path / "nested" / "probe.json"

    atomic_write_json(output, {"schema_version": 1, "value": 1})
    atomic_write_json(output, {"schema_version": 1, "value": 2})

    assert load_json_object(output) == {"schema_version": 1, "value": 2}
    assert json.loads(output.read_text(encoding="utf-8"))["value"] == 2
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))


def test_canonical_fingerprint_is_order_independent_but_value_sensitive():
    first = canonical_sha256({"a": 1, "b": {"x": 2, "y": 3}})
    reordered = canonical_sha256({"b": {"y": 3, "x": 2}, "a": 1})
    changed = canonical_sha256({"a": 1, "b": {"x": 2, "y": 4}})

    assert first == reordered
    assert first != changed


def test_result_validation_rejects_wrong_candidate_and_memory_fraction():
    requested = candidate(2)
    payload = ok_result(
        requested,
        throughput=10_000.0,
        peak_reserved_bytes=800,
    )

    validate_candidate_result(payload, expected_candidate=requested)
    with pytest.raises(ResourceProbeError, match="does not match"):
        validate_candidate_result(
            payload,
            expected_candidate=candidate(4),
        )
    payload["peak_reserved_fraction"] = 0.7
    with pytest.raises(ResourceProbeError, match="fraction"):
        validate_candidate_result(payload)


def test_micro_batch_recommendation_requires_reserved_memory_headroom():
    small = candidate(2)
    fast_but_too_full = candidate(4)
    oom = candidate(8)
    results = [
        ok_result(small, throughput=10_000.0, peak_reserved_bytes=700),
        ok_result(
            fast_but_too_full,
            throughput=20_000.0,
            peak_reserved_bytes=900,
        ),
        failure_result(oom, "oom"),
    ]

    recommendation = recommend_micro_batch(
        results,
        max_reserved_fraction=0.85,
    )

    assert recommendation is not None
    assert recommendation["candidate_key"] == small.key
    assert recommendation["status"] == "preliminary"


def test_loader_recommendation_uses_end_to_end_throughput():
    slow = candidate(8, workers=0, pin_memory=False)
    fast = candidate(8, workers=2, pin_memory=True)

    recommendation = recommend_loader(
        [
            ok_result(slow, throughput=11_000.0, peak_reserved_bytes=500),
            ok_result(fast, throughput=14_000.0, peak_reserved_bytes=500),
        ]
    )

    assert recommendation is not None
    assert recommendation["candidate_key"] == fast.key


@pytest.mark.parametrize(
    "message",
    (
        "CUDA out of memory. Tried to allocate 1 GiB",
        "CUDA error: out of memory",
        "CUBLAS_STATUS_ALLOC_FAILED",
    ),
)
def test_cuda_oom_detection_covers_expected_pytorch_messages(message):
    assert is_cuda_oom(RuntimeError(message)) is True


def test_tail_text_is_bounded_to_the_requested_suffix():
    assert tail_text("abcdefgh", limit=4) == "efgh"


def test_cli_builds_microbatch_and_loader_candidates_without_cuda():
    micro_args = probe_baseline_resources.parse_args(
        [
            "--manifest",
            "manifest.json",
            "--output",
            "probe.json",
            "--phase",
            "microbatch",
            "--micro-batch-sizes",
            "1",
            "2",
            "4",
            "--num-workers",
            "0",
            "--pin-memory",
            "true",
        ]
    )
    loader_args = probe_baseline_resources.parse_args(
        [
            "--manifest",
            "manifest.json",
            "--output",
            "probe.json",
            "--phase",
            "loader",
            "--micro-batch-size",
            "4",
            "--num-workers",
            "0",
            "2",
            "4",
            "--pin-memory",
            "both",
        ]
    )

    assert len(probe_baseline_resources.candidates_from_args(micro_args)) == 3
    assert len(probe_baseline_resources.candidates_from_args(loader_args)) == 6


def test_cli_rejects_non_increasing_microbatch_probe_order():
    args = probe_baseline_resources.parse_args(
        [
            "--manifest",
            "manifest.json",
            "--output",
            "probe.json",
            "--phase",
            "microbatch",
            "--micro-batch-sizes",
            "1",
            "4",
            "2",
        ]
    )

    with pytest.raises(
        probe_baseline_resources.ProbeEntryError,
        match="strictly increasing",
    ):
        probe_baseline_resources.candidates_from_args(args)
