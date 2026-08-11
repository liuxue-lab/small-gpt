from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any

import torch

from .config import TrainingConfig


class DeviceResolutionError(RuntimeError):
    """Raised when the requested training device is unavailable."""


class PrecisionConfigError(ValueError):
    """Raised when the requested precision is unsupported on a device."""


def resolve_device(requested_device: str) -> torch.device:
    if not isinstance(requested_device, str):
        raise DeviceResolutionError(
            "requested_device must be one of ['auto', 'cpu', 'cuda'], "
            f"got {requested_device!r}"
        )
    if requested_device not in {"auto", "cpu", "cuda"}:
        raise DeviceResolutionError(
            "requested_device must be one of ['auto', 'cpu', 'cuda'], "
            f"got {requested_device!r}"
        )

    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise DeviceResolutionError(
            "CUDA was requested but torch.cuda.is_available() is False"
        )
    return torch.device(requested_device)


@dataclass(frozen=True, slots=True)
class PrecisionPolicy:
    """Resolved FP32 or CUDA-BF16 execution policy for model forward passes."""

    device: torch.device
    precision: str

    def __post_init__(self) -> None:
        if not isinstance(self.device, torch.device):
            raise PrecisionConfigError(
                f"device must be a torch.device, got {type(self.device)!r}"
            )
        if self.device.type not in {"cpu", "cuda"}:
            raise PrecisionConfigError(
                f"device type must be cpu or cuda, got {self.device.type!r}"
            )
        if self.precision not in {"fp32", "bf16"}:
            raise PrecisionConfigError(
                "precision must be one of ['fp32', 'bf16'], "
                f"got {self.precision!r}"
            )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise DeviceResolutionError(
                "CUDA policy was constructed, but CUDA is unavailable"
            )
        if self.precision == "bf16":
            if self.device.type != "cuda":
                raise PrecisionConfigError(
                    "bf16 requires CUDA in the Day 7 training contract"
                )
            if not torch.cuda.is_bf16_supported():
                raise PrecisionConfigError(
                    "CUDA device does not report native bfloat16 support"
                )

    @property
    def uses_autocast(self) -> bool:
        return self.precision == "bf16"

    @property
    def autocast_dtype(self) -> torch.dtype | None:
        return torch.bfloat16 if self.uses_autocast else None

    @property
    def uses_grad_scaler(self) -> bool:
        return False

    def autocast_context(self) -> AbstractContextManager[Any]:
        if not self.uses_autocast:
            return nullcontext()
        return torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=True,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "device": str(self.device),
            "precision": self.precision,
            "uses_autocast": self.uses_autocast,
            "autocast_dtype": (
                None if self.autocast_dtype is None else str(self.autocast_dtype)
            ),
            "uses_grad_scaler": self.uses_grad_scaler,
        }

    @classmethod
    def resolve(
        cls,
        requested_device: str,
        precision: str,
    ) -> PrecisionPolicy:
        return cls(
            device=resolve_device(requested_device),
            precision=precision,
        )

    @classmethod
    def from_config(cls, config: TrainingConfig) -> PrecisionPolicy:
        if not isinstance(config, TrainingConfig):
            raise TypeError(
                f"config must be a TrainingConfig, got {type(config)!r}"
            )
        return cls.resolve(config.device, config.precision)
