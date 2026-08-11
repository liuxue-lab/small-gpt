from pathlib import Path

import pytest
import torch
from model import GPT, GPTConfig
from torch import nn

from train import (
    OptimizerConfigError,
    TrainingConfig,
    build_optimizer,
    partition_parameters,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEBUG_PATH = PROJECT_ROOT / "configs" / "debug.yaml"


def debug_model() -> GPT:
    return GPT(GPTConfig.from_yaml(DEBUG_PATH))


def debug_training_config() -> TrainingConfig:
    return TrainingConfig.from_yaml(DEBUG_PATH)


def test_debug_gpt_partition_exactly_covers_unique_trainable_parameters():
    model = debug_model()
    groups = partition_parameters(model)
    expected_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    grouped_parameters = [
        *groups.decay_parameters,
        *groups.no_decay_parameters,
    ]

    assert groups.unique_parameter_tensors == 20
    assert len(groups.decay_parameters) == 10
    assert len(groups.no_decay_parameters) == 10
    assert {id(parameter) for parameter in grouped_parameters} == {
        id(parameter) for parameter in expected_parameters
    }
    assert len({id(parameter) for parameter in grouped_parameters}) == len(
        grouped_parameters
    )


def test_debug_gpt_partition_preserves_exact_parameter_count():
    groups = partition_parameters(debug_model())

    assert groups.decay_numel == 2_506_752
    assert groups.no_decay_numel == 1_280
    assert groups.total_numel == 2_508_032


def test_partition_uses_parameter_ndim_as_the_only_decay_rule():
    groups = partition_parameters(debug_model())

    assert all(parameter.ndim >= 2 for parameter in groups.decay_parameters)
    assert all(parameter.ndim < 2 for parameter in groups.no_decay_parameters)
    assert all("norm" not in name for name in groups.decay_names)
    assert all("norm" in name for name in groups.no_decay_names)


def test_tied_embedding_and_lm_head_appear_once_with_both_aliases():
    model = debug_model()
    groups = partition_parameters(model)

    aliases = groups.aliases_for("token_embedding.weight")
    grouped_ids = [
        id(parameter)
        for parameter in (*groups.decay_parameters, *groups.no_decay_parameters)
    ]

    assert aliases == ("token_embedding.weight", "lm_head.weight")
    assert grouped_ids.count(id(model.token_embedding.weight)) == 1
    assert model.lm_head.weight is model.token_embedding.weight


def test_frozen_parameters_are_excluded():
    model = nn.Linear(4, 2)
    model.weight.requires_grad_(False)

    groups = partition_parameters(model)

    assert groups.decay_parameters == ()
    assert groups.no_decay_parameters == (model.bias,)
    assert groups.no_decay_names == ("bias",)


def test_rejects_model_without_trainable_parameters():
    model = nn.Linear(4, 2)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    with pytest.raises(OptimizerConfigError, match="no trainable parameters"):
        partition_parameters(model)


def test_partition_rejects_non_module():
    with pytest.raises(TypeError, match="nn.Module"):
        partition_parameters(object())


def test_build_optimizer_uses_frozen_adamw_hyperparameters():
    config = debug_training_config()
    optimizer = build_optimizer(debug_model(), config)
    groups_by_name = {
        parameter_group["group_name"]: parameter_group
        for parameter_group in optimizer.param_groups
    }

    assert isinstance(optimizer, torch.optim.AdamW)
    assert set(groups_by_name) == {"decay", "no_decay"}
    assert groups_by_name["decay"]["weight_decay"] == pytest.approx(0.1)
    assert groups_by_name["no_decay"]["weight_decay"] == pytest.approx(0.0)
    for parameter_group in optimizer.param_groups:
        assert parameter_group["lr"] == pytest.approx(config.learning_rate)
        assert parameter_group["betas"] == pytest.approx(
            (config.beta1, config.beta2)
        )
        assert parameter_group["eps"] == pytest.approx(config.adam_eps)


def test_built_optimizer_contains_no_duplicate_parameter_identity():
    model = debug_model()
    optimizer = build_optimizer(model, debug_training_config())
    optimizer_parameters = [
        parameter
        for parameter_group in optimizer.param_groups
        for parameter in parameter_group["params"]
    ]

    assert len(optimizer_parameters) == 20
    assert len({id(parameter) for parameter in optimizer_parameters}) == 20
    assert sum(parameter.numel() for parameter in optimizer_parameters) == 2_508_032


def test_adamw_step_changes_parameters_and_creates_finite_state():
    torch.manual_seed(7)
    model = nn.Linear(4, 2)
    optimizer = build_optimizer(model, debug_training_config())
    inputs = torch.randn(3, 4)
    before = model.weight.detach().clone()

    loss = model(inputs).square().mean()
    loss.backward()
    optimizer.step()

    assert not torch.equal(model.weight, before)
    assert torch.isfinite(model.weight).all()
    assert len(optimizer.state) == 2
    assert all(
        torch.isfinite(value).all()
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )


def test_adamw_state_dict_round_trip():
    torch.manual_seed(11)
    model = nn.Linear(4, 2)
    optimizer = build_optimizer(model, debug_training_config())
    model(torch.randn(3, 4)).square().mean().backward()
    optimizer.step()
    payload = optimizer.state_dict()

    restored_model = nn.Linear(4, 2)
    restored_optimizer = build_optimizer(
        restored_model,
        debug_training_config(),
    )
    restored_optimizer.load_state_dict(payload)

    assert len(restored_optimizer.state) == len(optimizer.state) == 2
    assert restored_optimizer.state_dict()["param_groups"] == payload["param_groups"]


def test_build_optimizer_rejects_wrong_config_type():
    with pytest.raises(TypeError, match="TrainingConfig"):
        build_optimizer(nn.Linear(4, 2), object())
