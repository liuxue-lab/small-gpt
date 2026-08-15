"""Validate the frozen Day 12 single-variable ablation contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.check_config import validate_config  # noqa: E402


CONTRACT_SCHEMA_VERSION = 1
PROTOCOL_ID = "day12-dropout-01-ablation-v1"
EXPECTED_HYPOTHESIS = (
    "Dropout 0.1 may improve generalization or generation repetition control, "
    "but may also slow optimization under a fixed 300M-token budget."
)
CONTROL_RUN_ID = "baseline-full-300m-20260813-232952"
CONTROL_CHECKPOINT_SHA256 = (
    "a39f8378ebe4012afb992be451d355e814b856ffb5e690ac011758f9db614b51"
)
CONTROL_TRAINING_SOURCE_COMMIT = "07c22a42a696e4d2bab7e6396fcb4c417dc5f63e"
SOURCE_AUDIT_HEAD = "0c1f0040d5bae891e4445b4039cf842990755e7c"
CONTROL_CONFIG_FINGERPRINT = (
    "25d6b96b40abfdffba694a339759d157c2bfa651904913079cd70559a4f5d1e7"
)
CONTROL_MODEL_CONFIG_FINGERPRINT = (
    "ba82957a47a92cb0021f6e56b103f12dda2b4eb8cda3927c195c96d45d5c052e"
)
TREATMENT_CONFIG_FINGERPRINT = (
    "789707df0f04db2c348d4bd971956d69d2e511008319ad954dd3896a2dcacec2"
)
TREATMENT_MODEL_CONFIG_FINGERPRINT = (
    "7e1f052d8c5720a35f3ff3e4057aa142bebff87759c3b6ba835e071e4d5dbbf6"
)
TOKENIZER_SHA256 = (
    "b26835e02eebf777a257c4732abdd6f9732a115967d2ad839f3a1a00e45ee8c5"
)
TOKENIZED_MANIFEST_SHA256 = (
    "ce7cd91075c7c666c427e1aaa286096a7f386643f3a76de3c26ef770d6cce67e"
)
DATASET_FINGERPRINT = (
    "39dab5bacdf8719bbc849e85ddcd7422cba5777fc044b437d050a49b87ab174f"
)
GENERATION_PROTOCOL_ID = "day11-baseline-generation-v1"
GENERATION_PROTOCOL_RAW_SHA256 = (
    "bb6fd24c2d277d4369fcd21d551ff7023484b62d7bccfd7103beca6c71a8ce4a"
)
GENERATION_PROTOCOL_FINGERPRINT = (
    "e60f3fb381b3efd8f00bd3f3fc3071c11645c78977dc7c6c40e0fd124b6d1ed0"
)
EXPECTED_PARAMETER_COUNT = 33_833_984
EXPECTED_SEED = 1337
EXPECTED_TARGET_TOKENS = 300_000_000
EXPECTED_TOKENS_PER_UPDATE = 65_536
EXPECTED_TOTAL_UPDATES = 4_578
EXPECTED_PLANNED_TOKENS = 300_023_808
EXPECTED_CHANGED_FIELD = "model.dropout"
EXPECTED_CONTROL_DROPOUT = 0.0
EXPECTED_TREATMENT_DROPOUT = 0.1

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_CONTRACT_FIELDS = frozenset(
    {
        "schema_version",
        "protocol_id",
        "status",
        "hypothesis",
        "statistical_scope",
        "control_run_id",
        "control_checkpoint_sha256",
        "control_training_source_commit",
        "source_audit_head",
        "control_config_path",
        "control_config_fingerprint",
        "control_model_config_fingerprint",
        "treatment_config_path",
        "treatment_config_fingerprint",
        "treatment_model_config_fingerprint",
        "changed_field",
        "control_value",
        "treatment_value",
        "allowed_experimental_diff_count",
        "allowed_identity_diff_fields",
        "held_constant_fields",
        "training_seed",
        "tokens_per_update",
        "total_updates",
        "planned_tokens",
        "tokenizer_sha256",
        "tokenized_manifest_sha256",
        "dataset_fingerprint",
        "validation_protocol",
        "generation_protocol_path",
        "generation_protocol_id",
        "generation_protocol_raw_sha256",
        "generation_protocol_fingerprint",
        "source_mode",
        "created_at",
    }
)
_EXPECTED_IDENTITY_DIFF_FIELDS = (
    "checkpoint_path",
    "checkpoint_sha256",
    "created_at",
    "output_directory",
    "run_id",
    "source_commit",
)
_EXPECTED_HELD_CONSTANT_FIELDS = (
    "project.name",
    "project.seed",
    "data",
    "tokenizer",
    "model.architecture",
    "model.n_layer",
    "model.n_head",
    "model.n_embd",
    "model.ffn_hidden",
    "model.context_length",
    "model.vocab_size",
    "model.tie_embeddings",
    "model.normalization",
    "model.norm_position",
    "model.layer_norm_eps",
    "model.activation",
    "model.gelu_approximate",
    "model.position_encoding",
    "model.linear_bias",
    "model.lm_head_bias",
    "model.layer_norm_affine",
    "model.init_std",
    "model.scale_residual_projections",
    "training",
)
_EXPECTED_VALIDATION_PROTOCOL = {
    "format_name": "small_gpt_frozen_split_evaluation",
    "schema_version": 1,
    "split_order": ["validation", "test"],
    "max_batches": None,
    "window_mode": "sequential_non_overlapping",
    "verify_hashes": True,
    "precision": "bf16",
}


class AblationContractError(ValueError):
    """Raised when the Day 12 ablation contract is invalid or has drifted."""


@dataclass(frozen=True, slots=True)
class _ConfigBundle:
    document: dict[str, Any]
    model_config: Any
    training_config: Any
    plan: Any


@dataclass(frozen=True, slots=True)
class AblationValidationReport:
    protocol_id: str
    contract_fingerprint: str
    control_config_valid: bool
    treatment_config_valid: bool
    observed_experimental_diff_count: int
    observed_experimental_diff_paths: tuple[str, ...]
    control_dropout: float
    treatment_dropout: float
    control_parameters: int
    treatment_parameters: int
    parameter_delta: int
    control_tokens_per_update: int
    treatment_tokens_per_update: int
    control_total_updates: int
    treatment_total_updates: int
    control_planned_tokens: int
    treatment_planned_tokens: int
    held_constants_match: bool
    generation_protocol_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["observed_experimental_diff_paths"] = list(
            self.observed_experimental_diff_paths
        )
        return payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AblationContractError(message)


def _is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _strict_equal(left: object, right: object) -> bool:
    return type(left) is type(right) and left == right


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AblationContractError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_non_finite_constant(value: str) -> None:
    raise AblationContractError(
        f"contract JSON contains non-finite numeric constant {value!r}"
    )


def load_contract(path: str | Path) -> dict[str, Any]:
    contract_path = Path(path).resolve()
    try:
        raw_bytes = contract_path.read_bytes()
    except OSError as error:
        raise AblationContractError(
            f"could not read ablation contract {contract_path}: {error}"
        ) from error
    try:
        document = json.loads(
            raw_bytes.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite_constant,
        )
    except AblationContractError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AblationContractError(
            f"ablation contract must be valid UTF-8 JSON: {error}"
        ) from error
    if not isinstance(document, dict):
        raise AblationContractError("ablation contract root must be a mapping")
    return document


def canonical_sha256(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise AblationContractError(
            f"value cannot be represented as canonical strict JSON: {error}"
        ) from error
    return hashlib.sha256(encoded).hexdigest()


def strict_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                dict(value),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise AblationContractError(
            f"validation report is not strict JSON: {error}"
        ) from error


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise AblationContractError(
            f"{field} must be a lowercase 64-character SHA-256"
        )
    return value


def _require_non_empty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AblationContractError(f"{field} must be a non-empty string")
    return value


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str] | set[str],
    *,
    field: str,
) -> None:
    provided = set(value)
    missing = expected - provided
    unknown = provided - expected
    if missing:
        raise AblationContractError(
            f"{field} is missing fields: {sorted(missing)}"
        )
    if unknown:
        raise AblationContractError(
            f"{field} has unknown fields: {sorted(unknown)}"
        )


def _require_string_sequence(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise AblationContractError(f"{field} must be a non-empty JSON list")
    result: list[str] = []
    for index, item in enumerate(value):
        result.append(
            _require_non_empty_string(item, field=f"{field}[{index}]")
        )
    if len(set(result)) != len(result):
        raise AblationContractError(f"{field} must not contain duplicates")
    return tuple(result)


def _validate_created_at(value: object) -> None:
    text = _require_non_empty_string(value, field="created_at")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise AblationContractError(
            "created_at must be an ISO-8601 timestamp"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise AblationContractError("created_at must include a timezone")


def _validate_contract_shape(contract: Mapping[str, Any]) -> None:
    _require_exact_fields(contract, _CONTRACT_FIELDS, field="contract")
    _require(
        _strict_equal(contract["schema_version"], CONTRACT_SCHEMA_VERSION),
        f"schema_version must equal {CONTRACT_SCHEMA_VERSION}",
    )
    for field in (
        "protocol_id",
        "status",
        "hypothesis",
        "statistical_scope",
        "control_run_id",
        "control_config_path",
        "treatment_config_path",
        "changed_field",
        "generation_protocol_path",
        "generation_protocol_id",
        "source_mode",
    ):
        _require_non_empty_string(contract[field], field=field)
    for field in (
        "control_checkpoint_sha256",
        "control_config_fingerprint",
        "control_model_config_fingerprint",
        "treatment_config_fingerprint",
        "treatment_model_config_fingerprint",
        "tokenizer_sha256",
        "tokenized_manifest_sha256",
        "dataset_fingerprint",
        "generation_protocol_raw_sha256",
        "generation_protocol_fingerprint",
    ):
        _require_sha256(contract[field], field=field)
    for field in ("control_training_source_commit", "source_audit_head"):
        value = contract[field]
        if not isinstance(value, str) or _GIT_COMMIT_PATTERN.fullmatch(value) is None:
            raise AblationContractError(
                f"{field} must be a full lowercase 40-character Git commit"
            )
    for field in (
        "allowed_experimental_diff_count",
        "training_seed",
        "tokens_per_update",
        "total_updates",
        "planned_tokens",
    ):
        value = contract[field]
        if not _is_plain_int(value) or value <= 0:
            raise AblationContractError(f"{field} must be a positive integer")
    for field in ("control_value", "treatment_value"):
        if not isinstance(contract[field], float):
            raise AblationContractError(f"{field} must be a JSON float")
    identity_fields = _require_string_sequence(
        contract["allowed_identity_diff_fields"],
        field="allowed_identity_diff_fields",
    )
    held_fields = _require_string_sequence(
        contract["held_constant_fields"],
        field="held_constant_fields",
    )
    _require(
        identity_fields == _EXPECTED_IDENTITY_DIFF_FIELDS,
        "allowed_identity_diff_fields do not match the frozen contract",
    )
    _require(
        held_fields == _EXPECTED_HELD_CONSTANT_FIELDS,
        "held_constant_fields do not match the frozen contract",
    )
    _require(
        isinstance(contract["validation_protocol"], Mapping),
        "validation_protocol must be a mapping",
    )
    _validate_created_at(contract["created_at"])


def _resolve_project_file(
    project_root: Path,
    raw_path: object,
    *,
    field: str,
) -> Path:
    value = _require_non_empty_string(raw_path, field=field)
    relative = Path(value)
    if relative.is_absolute():
        raise AblationContractError(f"{field} must be project-relative")
    root = project_root.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise AblationContractError(
            f"{field} must not escape the project root"
        ) from error
    if not resolved.is_file():
        raise AblationContractError(f"{field} does not exist: {resolved}")
    return resolved


def _load_yaml_document(path: Path, *, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            document = yaml.safe_load(handle)
    except OSError as error:
        raise AblationContractError(f"could not read {label}: {error}") from error
    except yaml.YAMLError as error:
        raise AblationContractError(f"could not parse {label}: {error}") from error
    if not isinstance(document, dict):
        raise AblationContractError(f"{label} must contain a mapping")
    return document


def _load_config_bundle(path: Path, *, label: str) -> _ConfigBundle:
    from model import GPTConfig
    from train import TrainingConfig

    document = _load_yaml_document(path, label=label)
    try:
        validate_config(path.name, document)
        model_config = GPTConfig.from_yaml(path)
        training_config = TrainingConfig.from_yaml(path)
        plan = training_config.resolve()
    except (KeyError, TypeError, ValueError) as error:
        raise AblationContractError(f"{label} is invalid: {error}") from error
    return _ConfigBundle(
        document=document,
        model_config=model_config,
        training_config=training_config,
        plan=plan,
    )


def _difference_paths(
    control: object,
    treatment: object,
    *,
    prefix: str = "",
) -> list[str]:
    if isinstance(control, Mapping) and isinstance(treatment, Mapping):
        differences: list[str] = []
        for key in sorted(set(control) | set(treatment), key=str):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in control or key not in treatment:
                differences.append(path)
                continue
            differences.extend(
                _difference_paths(control[key], treatment[key], prefix=path)
            )
        return differences
    if _strict_equal(control, treatment):
        return []
    return [prefix or "<root>"]


def _value_at_path(document: Mapping[str, Any], path: str) -> object:
    current: object = document
    for component in path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            raise AblationContractError(
                f"configuration path {path!r} does not exist"
            )
        current = current[component]
    return current


def _validate_experimental_difference(
    contract: Mapping[str, Any],
    control: Mapping[str, Any],
    treatment: Mapping[str, Any],
) -> tuple[str, ...]:
    differences = tuple(_difference_paths(control, treatment))
    allowed_count = contract["allowed_experimental_diff_count"]
    _require(
        len(differences) == allowed_count,
        "observed experimental difference count does not match the contract: "
        f"observed={len(differences)}, allowed={allowed_count}, "
        f"paths={list(differences)}",
    )
    changed_field = contract["changed_field"]
    _require(
        differences == (changed_field,),
        "observed experimental difference paths do not match changed_field: "
        f"observed={list(differences)}, changed_field={changed_field!r}",
    )
    control_value = _value_at_path(control, changed_field)
    treatment_value = _value_at_path(treatment, changed_field)
    _require(
        _strict_equal(control_value, contract["control_value"]),
        "control config value does not match control_value",
    )
    _require(
        _strict_equal(treatment_value, contract["treatment_value"]),
        "treatment config value does not match treatment_value",
    )
    return differences


def _validate_held_constants(
    contract: Mapping[str, Any],
    control: Mapping[str, Any],
    treatment: Mapping[str, Any],
) -> None:
    for field in contract["held_constant_fields"]:
        control_value = _value_at_path(control, field)
        treatment_value = _value_at_path(treatment, field)
        if not _strict_equal(control_value, treatment_value):
            raise AblationContractError(
                f"held constant field drifted: {field}"
            )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_contract_value(
    contract: Mapping[str, Any],
    field: str,
    expected: object,
) -> None:
    actual = contract[field]
    _require(
        _strict_equal(actual, expected),
        f"{field} must equal {expected!r}, got {actual!r}",
    )


def validate_ablation_contract(
    contract_path: str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> AblationValidationReport:
    """Validate contract, both configs, derived plans, and frozen protocols."""

    root = Path(project_root).resolve()
    contract = load_contract(contract_path)
    _validate_contract_shape(contract)

    control_path = _resolve_project_file(
        root,
        contract["control_config_path"],
        field="control_config_path",
    )
    treatment_path = _resolve_project_file(
        root,
        contract["treatment_config_path"],
        field="treatment_config_path",
    )
    generation_protocol_path = _resolve_project_file(
        root,
        contract["generation_protocol_path"],
        field="generation_protocol_path",
    )

    control = _load_config_bundle(control_path, label="control config")
    treatment = _load_config_bundle(treatment_path, label="treatment config")
    differences = _validate_experimental_difference(
        contract,
        control.document,
        treatment.document,
    )
    _validate_held_constants(
        contract,
        control.document,
        treatment.document,
    )

    for field, expected in (
        ("protocol_id", PROTOCOL_ID),
        ("status", "FROZEN"),
        ("hypothesis", EXPECTED_HYPOTHESIS),
        ("control_run_id", CONTROL_RUN_ID),
        ("control_checkpoint_sha256", CONTROL_CHECKPOINT_SHA256),
        ("control_training_source_commit", CONTROL_TRAINING_SOURCE_COMMIT),
        ("source_audit_head", SOURCE_AUDIT_HEAD),
        ("control_config_path", "configs/baseline.yaml"),
        ("treatment_config_path", "configs/ablation_dropout_01.yaml"),
        ("changed_field", EXPECTED_CHANGED_FIELD),
        ("control_value", EXPECTED_CONTROL_DROPOUT),
        ("treatment_value", EXPECTED_TREATMENT_DROPOUT),
        ("allowed_experimental_diff_count", 1),
        ("training_seed", EXPECTED_SEED),
        ("tokens_per_update", EXPECTED_TOKENS_PER_UPDATE),
        ("total_updates", EXPECTED_TOTAL_UPDATES),
        ("planned_tokens", EXPECTED_PLANNED_TOKENS),
        ("tokenizer_sha256", TOKENIZER_SHA256),
        ("tokenized_manifest_sha256", TOKENIZED_MANIFEST_SHA256),
        ("dataset_fingerprint", DATASET_FINGERPRINT),
        ("generation_protocol_path", "configs/day11_generation_protocol.json"),
        ("generation_protocol_id", GENERATION_PROTOCOL_ID),
        ("source_mode", "CURRENT_MAIN"),
        (
            "statistical_scope",
            "single_seed_descriptive_engineering_ablation",
        ),
    ):
        _require_contract_value(contract, field, expected)

    control_dropout = float(control.model_config.dropout)
    treatment_dropout = float(treatment.model_config.dropout)
    _require(
        control_dropout == EXPECTED_CONTROL_DROPOUT,
        "control GPTConfig dropout must equal 0.0",
    )
    _require(
        treatment_dropout == EXPECTED_TREATMENT_DROPOUT,
        "treatment GPTConfig dropout must equal 0.1",
    )

    control_parameters = int(control.model_config.parameter_count)
    treatment_parameters = int(treatment.model_config.parameter_count)
    _require(
        control_parameters == EXPECTED_PARAMETER_COUNT,
        "control parameter count does not match the frozen baseline",
    )
    _require(
        treatment_parameters == EXPECTED_PARAMETER_COUNT,
        "treatment parameter count does not match the frozen baseline",
    )

    for label, bundle in (("control", control), ("treatment", treatment)):
        _require(
            bundle.training_config.seed == EXPECTED_SEED,
            f"{label} training seed drifted",
        )
        _require(
            bundle.plan.target_tokens == EXPECTED_TARGET_TOKENS,
            f"{label} target token budget drifted",
        )
        _require(
            bundle.plan.tokens_per_update == EXPECTED_TOKENS_PER_UPDATE,
            f"{label} tokens_per_update drifted",
        )
        _require(
            bundle.plan.total_updates == EXPECTED_TOTAL_UPDATES,
            f"{label} total_updates drifted",
        )
        _require(
            bundle.plan.planned_tokens == EXPECTED_PLANNED_TOKENS,
            f"{label} planned_tokens drifted",
        )

    control_config_fingerprint = canonical_sha256(control.document)
    treatment_config_fingerprint = canonical_sha256(treatment.document)
    control_model_fingerprint = canonical_sha256(control.document["model"])
    treatment_model_fingerprint = canonical_sha256(treatment.document["model"])
    for field, computed, frozen in (
        (
            "control_config_fingerprint",
            control_config_fingerprint,
            CONTROL_CONFIG_FINGERPRINT,
        ),
        (
            "treatment_config_fingerprint",
            treatment_config_fingerprint,
            TREATMENT_CONFIG_FINGERPRINT,
        ),
        (
            "control_model_config_fingerprint",
            control_model_fingerprint,
            CONTROL_MODEL_CONFIG_FINGERPRINT,
        ),
        (
            "treatment_model_config_fingerprint",
            treatment_model_fingerprint,
            TREATMENT_MODEL_CONFIG_FINGERPRINT,
        ),
    ):
        _require(
            computed == contract[field],
            f"{field} does not match the referenced config",
        )
        _require(
            computed == frozen,
            f"{field} does not match the frozen Day 12 identity",
        )

    _require(
        contract["validation_protocol"] == _EXPECTED_VALIDATION_PROTOCOL,
        "validation_protocol does not match the frozen full-split protocol",
    )

    from eval import generation_protocol_fingerprint, load_generation_protocol

    try:
        generation_protocol = load_generation_protocol(generation_protocol_path)
        generation_fingerprint = generation_protocol_fingerprint(
            generation_protocol
        )
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        raise AblationContractError(
            f"generation protocol is invalid: {error}"
        ) from error
    _require(
        generation_protocol.protocol_id == contract["generation_protocol_id"],
        "generation protocol ID does not match the contract",
    )
    _require(
        generation_protocol.protocol_id == GENERATION_PROTOCOL_ID,
        "generation protocol ID does not match the frozen Day 11 protocol",
    )
    raw_generation_sha256 = _sha256_file(generation_protocol_path)
    _require(
        raw_generation_sha256 == contract["generation_protocol_raw_sha256"],
        "generation protocol raw SHA-256 does not match the contract",
    )
    _require(
        raw_generation_sha256 == GENERATION_PROTOCOL_RAW_SHA256,
        "generation protocol raw SHA-256 drifted from Day 11",
    )
    _require(
        generation_fingerprint == contract["generation_protocol_fingerprint"],
        "generation protocol fingerprint does not match the contract",
    )
    _require(
        generation_fingerprint == GENERATION_PROTOCOL_FINGERPRINT,
        "generation protocol fingerprint drifted from Day 11",
    )

    report = AblationValidationReport(
        protocol_id=PROTOCOL_ID,
        contract_fingerprint=canonical_sha256(contract),
        control_config_valid=True,
        treatment_config_valid=True,
        observed_experimental_diff_count=len(differences),
        observed_experimental_diff_paths=differences,
        control_dropout=control_dropout,
        treatment_dropout=treatment_dropout,
        control_parameters=control_parameters,
        treatment_parameters=treatment_parameters,
        parameter_delta=treatment_parameters - control_parameters,
        control_tokens_per_update=control.plan.tokens_per_update,
        treatment_tokens_per_update=treatment.plan.tokens_per_update,
        control_total_updates=control.plan.total_updates,
        treatment_total_updates=treatment.plan.total_updates,
        control_planned_tokens=control.plan.planned_tokens,
        treatment_planned_tokens=treatment.plan.planned_tokens,
        held_constants_match=True,
        generation_protocol_fingerprint=generation_fingerprint,
    )
    strict_json_bytes(report.to_dict())
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the frozen Day 12 dropout ablation contract."
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=PROJECT_ROOT / "configs" / "day12_ablation_contract.json",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _print_report(report: AblationValidationReport) -> None:
    print(f"ProtocolID={report.protocol_id}")
    print(f"ContractFingerprint={report.contract_fingerprint}")
    print(f"ControlConfigValid={report.control_config_valid}")
    print(f"TreatmentConfigValid={report.treatment_config_valid}")
    print(
        "ObservedExperimentalDiffCount="
        f"{report.observed_experimental_diff_count}"
    )
    print(
        "ObservedExperimentalDiffPaths="
        + ",".join(report.observed_experimental_diff_paths)
    )
    print(f"ControlDropout={report.control_dropout}")
    print(f"TreatmentDropout={report.treatment_dropout}")
    print(f"ControlParameters={report.control_parameters}")
    print(f"TreatmentParameters={report.treatment_parameters}")
    print(f"ParameterDelta={report.parameter_delta}")
    print(f"ControlTokensPerUpdate={report.control_tokens_per_update}")
    print(f"TreatmentTokensPerUpdate={report.treatment_tokens_per_update}")
    print(f"ControlTotalUpdates={report.control_total_updates}")
    print(f"TreatmentTotalUpdates={report.treatment_total_updates}")
    print(f"ControlPlannedTokens={report.control_planned_tokens}")
    print(f"TreatmentPlannedTokens={report.treatment_planned_tokens}")
    print(f"HeldConstantsMatch={report.held_constants_match}")
    print(
        "GenerationProtocolFingerprint="
        f"{report.generation_protocol_fingerprint}"
    )
    print("Day12AblationContract=PASS")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = validate_ablation_contract(args.contract)
    except AblationContractError as error:
        print("Day12AblationContract=FAIL", file=sys.stderr)
        print(f"Error={error}", file=sys.stderr)
        return 1
    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
