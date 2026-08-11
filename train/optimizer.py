from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import TrainingConfig


class OptimizerConfigError(ValueError):
    """Raised when optimizer parameter coverage or configuration is invalid."""


@dataclass(frozen=True, slots=True)
class OptimizerParameterGroups:
    """Identity-deduplicated trainable parameters and their audit metadata."""

    decay_parameters: tuple[nn.Parameter, ...]
    no_decay_parameters: tuple[nn.Parameter, ...]
    decay_names: tuple[str, ...]
    no_decay_names: tuple[str, ...]
    aliases: tuple[tuple[str, tuple[str, ...]], ...]

    @property
    def unique_parameter_tensors(self) -> int:
        return len(self.decay_parameters) + len(self.no_decay_parameters)

    @property
    def decay_numel(self) -> int:
        return sum(parameter.numel() for parameter in self.decay_parameters)

    @property
    def no_decay_numel(self) -> int:
        return sum(parameter.numel() for parameter in self.no_decay_parameters)

    @property
    def total_numel(self) -> int:
        return self.decay_numel + self.no_decay_numel

    def aliases_for(self, canonical_name: str) -> tuple[str, ...]:
        for name, aliases in self.aliases:
            if name == canonical_name:
                return aliases
        raise KeyError(canonical_name)


def partition_parameters(model: nn.Module) -> OptimizerParameterGroups:
    """Partition every unique trainable Parameter by ndim without duplication."""

    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model)!r}")

    unique_parameters: dict[int, nn.Parameter] = {}
    canonical_names: dict[int, str] = {}
    aliases_by_id: dict[int, list[str]] = {}

    for name, parameter in model.named_parameters(remove_duplicate=False):
        if not parameter.requires_grad:
            continue

        parameter_id = id(parameter)
        aliases_by_id.setdefault(parameter_id, []).append(name)
        if parameter_id not in unique_parameters:
            unique_parameters[parameter_id] = parameter
            canonical_names[parameter_id] = name

    if not unique_parameters:
        raise OptimizerConfigError("model has no trainable parameters")

    decay_parameters: list[nn.Parameter] = []
    no_decay_parameters: list[nn.Parameter] = []
    decay_names: list[str] = []
    no_decay_names: list[str] = []

    for parameter_id, parameter in unique_parameters.items():
        name = canonical_names[parameter_id]
        if parameter.ndim >= 2:
            decay_parameters.append(parameter)
            decay_names.append(name)
        else:
            no_decay_parameters.append(parameter)
            no_decay_names.append(name)

    grouped_ids = {
        id(parameter)
        for parameter in (*decay_parameters, *no_decay_parameters)
    }
    decay_ids = {id(parameter) for parameter in decay_parameters}
    no_decay_ids = {id(parameter) for parameter in no_decay_parameters}
    expected_ids = set(unique_parameters)

    if decay_ids & no_decay_ids:
        raise OptimizerConfigError(
            "decay and no-decay optimizer groups overlap by Parameter identity"
        )
    if grouped_ids != expected_ids:
        missing = len(expected_ids - grouped_ids)
        unexpected = len(grouped_ids - expected_ids)
        raise OptimizerConfigError(
            "optimizer parameter groups do not exactly cover trainable parameters: "
            f"missing={missing}, unexpected={unexpected}"
        )

    aliases = tuple(
        (
            canonical_names[parameter_id],
            tuple(aliases_by_id[parameter_id]),
        )
        for parameter_id in unique_parameters
    )

    return OptimizerParameterGroups(
        decay_parameters=tuple(decay_parameters),
        no_decay_parameters=tuple(no_decay_parameters),
        decay_names=tuple(decay_names),
        no_decay_names=tuple(no_decay_names),
        aliases=aliases,
    )


def build_optimizer(
    model: nn.Module,
    config: TrainingConfig,
) -> torch.optim.AdamW:
    """Build the Day 7 AdamW optimizer from the strict training config."""

    if not isinstance(config, TrainingConfig):
        raise TypeError(
            f"config must be a TrainingConfig, got {type(config)!r}"
        )

    groups = partition_parameters(model)
    parameter_groups: list[dict[str, object]] = []
    if groups.decay_parameters:
        parameter_groups.append(
            {
                "params": list(groups.decay_parameters),
                "weight_decay": config.weight_decay,
                "group_name": "decay",
            }
        )
    if groups.no_decay_parameters:
        parameter_groups.append(
            {
                "params": list(groups.no_decay_parameters),
                "weight_decay": 0.0,
                "group_name": "no_decay",
            }
        )

    optimizer = torch.optim.AdamW(
        parameter_groups,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.adam_eps,
    )

    optimizer_ids = [
        id(parameter)
        for parameter_group in optimizer.param_groups
        for parameter in parameter_group["params"]
    ]
    if len(optimizer_ids) != len(set(optimizer_ids)):
        raise OptimizerConfigError(
            "a trainable Parameter appears more than once in the optimizer"
        )
    if set(optimizer_ids) != {
        id(parameter)
        for parameter in (*groups.decay_parameters, *groups.no_decay_parameters)
    }:
        raise OptimizerConfigError(
            "optimizer does not exactly cover the partitioned parameters"
        )

    return optimizer
