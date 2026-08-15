from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest
import torch
from torch import nn
import yaml

from model import GPT, GPTConfig
from scripts.check_ablation_contract import (
    AblationContractError,
    CONTROL_CONFIG_FINGERPRINT,
    CONTROL_MODEL_CONFIG_FINGERPRINT,
    EXPECTED_PARAMETER_COUNT,
    GENERATION_PROTOCOL_FINGERPRINT,
    PROTOCOL_ID,
    TREATMENT_CONFIG_FINGERPRINT,
    TREATMENT_MODEL_CONFIG_FINGERPRINT,
    canonical_sha256,
    main,
    strict_json_bytes,
    validate_ablation_contract,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = PROJECT_ROOT / "configs" / "day12_ablation_contract.json"
CONTROL_PATH = PROJECT_ROOT / "configs" / "baseline.yaml"
TREATMENT_PATH = PROJECT_ROOT / "configs" / "ablation_dropout_01.yaml"
GENERATION_PROTOCOL_PATH = (
    PROJECT_ROOT / "configs" / "day11_generation_protocol.json"
)


def _load_json(path: Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def _write_json(path: Path, document: dict) -> None:
    path.write_text(
        json.dumps(
            document,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _load_yaml(path: Path) -> dict:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def _write_yaml(path: Path, document: dict) -> None:
    path.write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
        newline="\n",
    )


def _workspace(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "project"
    configs = root / "configs"
    configs.mkdir(parents=True)
    shutil.copyfile(CONTROL_PATH, configs / CONTROL_PATH.name)
    shutil.copyfile(TREATMENT_PATH, configs / TREATMENT_PATH.name)
    shutil.copyfile(
        GENERATION_PROTOCOL_PATH,
        configs / GENERATION_PROTOCOL_PATH.name,
    )
    contract_path = configs / CONTRACT_PATH.name
    shutil.copyfile(CONTRACT_PATH, contract_path)
    return root, contract_path


def _mutate_yaml(root: Path, filename: str, mutate) -> None:
    path = root / "configs" / filename
    document = _load_yaml(path)
    mutate(document)
    _write_yaml(path, document)


def _mutate_contract(contract_path: Path, mutate) -> None:
    document = _load_json(contract_path)
    mutate(document)
    _write_json(contract_path, document)


def _tiny_model(dropout: float) -> GPT:
    values = GPTConfig.from_yaml(CONTROL_PATH).to_dict()
    values.update(
        {
            "n_layer": 1,
            "n_head": 2,
            "n_embd": 16,
            "ffn_hidden": 64,
            "context_length": 8,
            "vocab_size": 64,
            "dropout": dropout,
        }
    )
    return GPT(GPTConfig.from_mapping(values))


def test_frozen_contract_passes_with_exact_derived_values():
    report = validate_ablation_contract(CONTRACT_PATH)

    assert report.protocol_id == PROTOCOL_ID
    assert report.control_config_valid is True
    assert report.treatment_config_valid is True
    assert report.observed_experimental_diff_count == 1
    assert report.observed_experimental_diff_paths == ("model.dropout",)
    assert report.control_dropout == 0.0
    assert report.treatment_dropout == 0.1
    assert report.control_parameters == EXPECTED_PARAMETER_COUNT
    assert report.treatment_parameters == EXPECTED_PARAMETER_COUNT
    assert report.parameter_delta == 0
    assert report.control_tokens_per_update == 65_536
    assert report.treatment_tokens_per_update == 65_536
    assert report.control_total_updates == 4_578
    assert report.treatment_total_updates == 4_578
    assert report.control_planned_tokens == 300_023_808
    assert report.treatment_planned_tokens == 300_023_808
    assert report.held_constants_match is True
    assert (
        report.generation_protocol_fingerprint
        == GENERATION_PROTOCOL_FINGERPRINT
    )


def test_frozen_config_fingerprints_are_exact():
    control = _load_yaml(CONTROL_PATH)
    treatment = _load_yaml(TREATMENT_PATH)

    assert canonical_sha256(control) == CONTROL_CONFIG_FINGERPRINT
    assert canonical_sha256(control["model"]) == CONTROL_MODEL_CONFIG_FINGERPRINT
    assert canonical_sha256(treatment) == TREATMENT_CONFIG_FINGERPRINT
    assert (
        canonical_sha256(treatment["model"])
        == TREATMENT_MODEL_CONFIG_FINGERPRINT
    )


def test_validator_cli_prints_machine_readable_pass(capsys):
    exit_code = main(["--contract", str(CONTRACT_PATH)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "ProtocolID=day12-dropout-01-ablation-v1" in captured.out
    assert "ObservedExperimentalDiffCount=1" in captured.out
    assert "ObservedExperimentalDiffPaths=model.dropout" in captured.out
    assert "ParameterDelta=0" in captured.out
    assert "Day12AblationContract=PASS" in captured.out
    assert captured.err == ""


def test_validation_report_is_strict_finite_json():
    report = validate_ablation_contract(CONTRACT_PATH)

    payload = strict_json_bytes(report.to_dict())
    decoded = json.loads(payload)

    assert decoded["protocol_id"] == PROTOCOL_ID
    assert decoded["parameter_delta"] == 0
    assert b"NaN" not in payload
    assert b"Infinity" not in payload


def test_contract_fingerprint_is_stable():
    first = validate_ablation_contract(CONTRACT_PATH)
    second = validate_ablation_contract(CONTRACT_PATH)

    assert first.contract_fingerprint == second.contract_fingerprint
    assert len(first.contract_fingerprint) == 64


def test_missing_contract_field_is_rejected(tmp_path):
    root, contract_path = _workspace(tmp_path)
    _mutate_contract(contract_path, lambda document: document.pop("changed_field"))

    with pytest.raises(AblationContractError, match="missing fields"):
        validate_ablation_contract(contract_path, project_root=root)


def test_unknown_contract_field_is_rejected(tmp_path):
    root, contract_path = _workspace(tmp_path)
    _mutate_contract(
        contract_path,
        lambda document: document.update({"extra_field": True}),
    )

    with pytest.raises(AblationContractError, match="unknown fields"):
        validate_ablation_contract(contract_path, project_root=root)


def test_duplicate_contract_json_key_is_rejected(tmp_path):
    root, contract_path = _workspace(tmp_path)
    raw = contract_path.read_text(encoding="utf-8")
    raw = raw.replace(
        '"schema_version": 1,',
        '"schema_version": 1,\n  "schema_version": 1,',
        1,
    )
    contract_path.write_text(raw, encoding="utf-8", newline="\n")

    with pytest.raises(AblationContractError, match="duplicate JSON key"):
        validate_ablation_contract(contract_path, project_root=root)


def test_non_finite_contract_number_is_rejected(tmp_path):
    root, contract_path = _workspace(tmp_path)
    raw = contract_path.read_text(encoding="utf-8")
    raw = raw.replace('"control_value": 0.0', '"control_value": NaN', 1)
    contract_path.write_text(raw, encoding="utf-8", newline="\n")

    with pytest.raises(AblationContractError, match="non-finite"):
        validate_ablation_contract(contract_path, project_root=root)


def test_invalid_control_config_is_rejected(tmp_path):
    root, contract_path = _workspace(tmp_path)
    _mutate_yaml(
        root,
        "baseline.yaml",
        lambda document: document["model"].update({"n_head": 7}),
    )

    with pytest.raises(AblationContractError, match="control config is invalid"):
        validate_ablation_contract(contract_path, project_root=root)


def test_invalid_treatment_config_is_rejected(tmp_path):
    root, contract_path = _workspace(tmp_path)
    _mutate_yaml(
        root,
        "ablation_dropout_01.yaml",
        lambda document: document["model"].update({"dropout": 1.0}),
    )

    with pytest.raises(AblationContractError, match="treatment config is invalid"):
        validate_ablation_contract(contract_path, project_root=root)


def test_changed_field_other_than_model_dropout_is_rejected(tmp_path):
    root, contract_path = _workspace(tmp_path)
    _mutate_contract(
        contract_path,
        lambda document: document.update(
            {"changed_field": "training.weight_decay"}
        ),
    )

    with pytest.raises(AblationContractError, match="difference paths"):
        validate_ablation_contract(contract_path, project_root=root)


def test_zero_observed_config_differences_is_rejected(tmp_path):
    root, contract_path = _workspace(tmp_path)
    shutil.copyfile(
        root / "configs" / "baseline.yaml",
        root / "configs" / "ablation_dropout_01.yaml",
    )

    with pytest.raises(AblationContractError, match="difference count"):
        validate_ablation_contract(contract_path, project_root=root)


def test_more_than_one_observed_config_difference_is_rejected(tmp_path):
    root, contract_path = _workspace(tmp_path)
    _mutate_yaml(
        root,
        "ablation_dropout_01.yaml",
        lambda document: document["training"].update({"weight_decay": 0.2}),
    )

    with pytest.raises(AblationContractError, match="difference count"):
        validate_ablation_contract(contract_path, project_root=root)


def test_control_dropout_drift_is_rejected(tmp_path):
    root, contract_path = _workspace(tmp_path)
    _mutate_yaml(
        root,
        "baseline.yaml",
        lambda document: document["model"].update({"dropout": 0.2}),
    )

    with pytest.raises(AblationContractError, match="control config value"):
        validate_ablation_contract(contract_path, project_root=root)


def test_treatment_dropout_drift_is_rejected(tmp_path):
    root, contract_path = _workspace(tmp_path)
    _mutate_yaml(
        root,
        "ablation_dropout_01.yaml",
        lambda document: document["model"].update({"dropout": 0.2}),
    )

    with pytest.raises(AblationContractError, match="treatment config value"):
        validate_ablation_contract(contract_path, project_root=root)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    (
        ("project", "seed", 1338),
        ("training", "target_tokens", 300_000_001),
        ("training", "weight_decay", 0.2),
        ("training", "micro_batch_size", 8),
        ("training", "gradient_accumulation_steps", 4),
        ("model", "n_layer", 9),
    ),
)
def test_held_constant_drift_is_rejected(
    tmp_path,
    section,
    field,
    value,
):
    root, contract_path = _workspace(tmp_path)
    _mutate_yaml(
        root,
        "ablation_dropout_01.yaml",
        lambda document: document[section].update({field: value}),
    )

    with pytest.raises(AblationContractError):
        validate_ablation_contract(contract_path, project_root=root)


def test_parameter_count_change_is_rejected_even_when_shared(tmp_path):
    root, contract_path = _workspace(tmp_path)
    for filename in ("baseline.yaml", "ablation_dropout_01.yaml"):
        _mutate_yaml(
            root,
            filename,
            lambda document: document["model"].update({"n_layer": 7}),
        )

    with pytest.raises(AblationContractError, match="parameter count"):
        validate_ablation_contract(contract_path, project_root=root)


def test_weight_tying_change_is_rejected(tmp_path):
    root, contract_path = _workspace(tmp_path)
    _mutate_yaml(
        root,
        "ablation_dropout_01.yaml",
        lambda document: document["model"].update({"tie_embeddings": False}),
    )

    with pytest.raises(AblationContractError, match="treatment config is invalid"):
        validate_ablation_contract(contract_path, project_root=root)


def test_generation_protocol_fingerprint_drift_is_rejected(tmp_path):
    root, contract_path = _workspace(tmp_path)
    _mutate_contract(
        contract_path,
        lambda document: document.update(
            {"generation_protocol_fingerprint": "0" * 64}
        ),
    )

    with pytest.raises(AblationContractError, match="generation protocol fingerprint"):
        validate_ablation_contract(contract_path, project_root=root)


def test_generation_protocol_raw_bytes_drift_is_rejected(tmp_path):
    root, contract_path = _workspace(tmp_path)
    protocol_path = root / "configs" / "day11_generation_protocol.json"
    protocol_path.write_bytes(protocol_path.read_bytes() + b"\n")

    with pytest.raises(AblationContractError, match="raw SHA-256"):
        validate_ablation_contract(contract_path, project_root=root)


def test_treatment_train_mode_enables_all_configured_dropout_modules():
    model = _tiny_model(0.1)
    model.train()
    dropout_modules = [
        module for module in model.modules() if isinstance(module, nn.Dropout)
    ]

    assert dropout_modules
    assert all(module.training for module in dropout_modules)
    assert all(module.p == pytest.approx(0.1) for module in dropout_modules)

    values = torch.ones(4_096)
    torch.manual_seed(1337)
    dropped = dropout_modules[0](values)

    assert torch.count_nonzero(dropped == 0).item() > 0
    assert not torch.equal(dropped, values)


def test_treatment_eval_mode_disables_all_dropout_modules():
    model = _tiny_model(0.1)
    model.eval()
    dropout_modules = [
        module for module in model.modules() if isinstance(module, nn.Dropout)
    ]
    values = torch.randn(256)

    assert dropout_modules
    assert all(not module.training for module in dropout_modules)
    assert all(torch.equal(module(values), values) for module in dropout_modules)


def test_control_zero_dropout_remains_identity_in_train_mode():
    model = _tiny_model(0.0)
    model.train()
    dropout_modules = [
        module for module in model.modules() if isinstance(module, nn.Dropout)
    ]
    values = torch.randn(256)

    assert dropout_modules
    assert all(module.training for module in dropout_modules)
    assert all(module.p == 0.0 for module in dropout_modules)
    assert all(torch.equal(module(values), values) for module in dropout_modules)
