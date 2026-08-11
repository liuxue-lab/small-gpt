from dataclasses import replace
from pathlib import Path

import pytest
import torch

from train import (
    DeviceResolutionError,
    PrecisionConfigError,
    PrecisionPolicy,
    TrainingConfig,
    resolve_device,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEBUG_PATH = PROJECT_ROOT / "configs" / "debug.yaml"


def test_auto_resolves_cpu_when_cuda_is_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert resolve_device("auto") == torch.device("cpu")


def test_auto_resolves_cuda_when_cuda_is_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)

    assert resolve_device("auto") == torch.device("cuda:2")


def test_explicit_cuda_resolves_current_device_index(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)

    assert resolve_device("cuda") == torch.device("cuda:1")


def test_explicit_cpu_does_not_depend_on_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert resolve_device("cpu") == torch.device("cpu")


def test_explicit_cuda_rejects_unavailable_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(DeviceResolutionError, match="CUDA was requested"):
        resolve_device("cuda")


@pytest.mark.parametrize("requested_device", ("gpu", "cuda:0", "", None, True))
def test_rejects_unknown_device_request(requested_device):
    with pytest.raises(DeviceResolutionError, match="requested_device"):
        resolve_device(requested_device)


def test_cpu_fp32_policy_disables_autocast_and_grad_scaler():
    policy = PrecisionPolicy.resolve("cpu", "fp32")

    with policy.autocast_context():
        result = torch.ones(2, dtype=torch.float32) + 1.0

    assert policy.device == torch.device("cpu")
    assert policy.uses_autocast is False
    assert policy.autocast_dtype is None
    assert policy.uses_grad_scaler is False
    assert result.dtype == torch.float32


def test_cpu_bf16_is_explicitly_rejected():
    with pytest.raises(PrecisionConfigError, match="bf16 requires CUDA"):
        PrecisionPolicy.resolve("cpu", "bf16")


@pytest.mark.parametrize("precision", ("fp16", "float32", "", None, True))
def test_rejects_unsupported_precision(precision):
    with pytest.raises(PrecisionConfigError, match="precision"):
        PrecisionPolicy.resolve("cpu", precision)


def test_cuda_bf16_policy_requires_and_records_native_support(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    policy = PrecisionPolicy.resolve("cuda", "bf16")

    assert policy.device == torch.device("cuda:0")
    assert policy.uses_autocast is True
    assert policy.autocast_dtype == torch.bfloat16
    assert policy.uses_grad_scaler is False


def test_cuda_bf16_rejects_missing_native_support(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)

    with pytest.raises(PrecisionConfigError, match="bfloat16 support"):
        PrecisionPolicy.resolve("cuda", "bf16")


def test_direct_cuda_policy_canonicalizes_current_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)

    policy = PrecisionPolicy(torch.device("cuda"), "fp32")

    assert policy.device == torch.device("cuda:3")


def test_direct_cuda_policy_still_rejects_unavailable_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(DeviceResolutionError, match="CUDA is unavailable"):
        PrecisionPolicy(torch.device("cuda"), "bf16")


def test_policy_rejects_non_device_and_unsupported_device_type():
    with pytest.raises(PrecisionConfigError, match="torch.device"):
        PrecisionPolicy("cpu", "fp32")
    with pytest.raises(PrecisionConfigError, match="device type"):
        PrecisionPolicy(torch.device("meta"), "fp32")


def test_from_config_uses_the_strict_training_fields():
    config = replace(
        TrainingConfig.from_yaml(DEBUG_PATH),
        device="cpu",
        precision="fp32",
    )

    policy = PrecisionPolicy.from_config(config)

    assert policy.to_dict() == {
        "device": "cpu",
        "precision": "fp32",
        "uses_autocast": False,
        "autocast_dtype": None,
        "uses_grad_scaler": False,
    }


def test_from_config_rejects_wrong_type():
    with pytest.raises(TypeError, match="TrainingConfig"):
        PrecisionPolicy.from_config(object())
