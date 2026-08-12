from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from train import (
    TrainingConfig,
    TrainingConfigError,
    training_field_names,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEBUG_PATH = PROJECT_ROOT / "configs" / "debug.yaml"
BASELINE_PATH = PROJECT_ROOT / "configs" / "baseline.yaml"


def load_document(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        document = yaml.safe_load(file)
    assert isinstance(document, dict)
    return document


def write_document(tmp_path: Path, document: dict) -> Path:
    path = tmp_path / "config.yaml"
    with path.open("w", encoding="utf-8", newline="\n") as file:
        yaml.safe_dump(document, file, sort_keys=False)
    return path


def mutated_debug(tmp_path: Path, mutate) -> Path:
    document = deepcopy(load_document(DEBUG_PATH))
    mutate(document)
    return write_document(tmp_path, document)


@pytest.mark.parametrize("path", (DEBUG_PATH, BASELINE_PATH))
def test_project_training_configs_parse(path):
    config = TrainingConfig.from_yaml(path)

    assert config.project_name
    assert config.seed == 1337
    assert config.vocab_size == 16_384


def test_debug_training_contract():
    config = TrainingConfig.from_yaml(DEBUG_PATH)

    assert config.device == "auto"
    assert config.precision == "fp32"
    assert config.context_length == 128
    assert config.micro_batch_size == 4
    assert config.gradient_accumulation_steps == 1
    assert config.max_steps == 200
    assert config.target_tokens is None
    assert config.learning_rate == pytest.approx(3.0e-4)
    assert config.min_learning_rate == pytest.approx(3.0e-5)
    assert config.warmup_steps == 20
    assert config.warmup_ratio is None
    assert config.num_workers == 0
    assert config.pin_memory is False
    assert config.deterministic is True
    assert config.allow_tf32 is False
    assert config.is_execution_ready is True


def test_debug_resolved_plan_is_exact():
    plan = TrainingConfig.from_yaml(DEBUG_PATH).resolve()

    assert plan.tokens_per_micro_step == 4 * 128
    assert plan.tokens_per_update == 4 * 128 * 1
    assert plan.total_updates == 200
    assert plan.warmup_updates == 20
    assert plan.planned_tokens == 102_400
    assert plan.target_tokens is None
    assert plan.token_overshoot == 0


def test_baseline_contract_uses_frozen_rtx5090_resources():
    config = TrainingConfig.from_yaml(BASELINE_PATH)

    assert config.device == "cuda"
    assert config.precision == "bf16"
    assert config.context_length == 512
    assert config.max_steps is None
    assert config.target_tokens == 300_000_000
    assert config.micro_batch_size == 16
    assert config.gradient_accumulation_steps == 8
    assert config.warmup_steps is None
    assert config.warmup_ratio == pytest.approx(0.02)
    assert config.num_workers == 4
    assert config.pin_memory is False
    assert config.unresolved_fields == ()
    assert config.is_execution_ready is True


def test_baseline_resolved_plan_matches_day8_rtx5090_freeze():
    plan = TrainingConfig.from_yaml(BASELINE_PATH).resolve()

    assert plan.tokens_per_micro_step == 8_192
    assert plan.tokens_per_update == 65_536
    assert plan.total_updates == 4_578
    assert plan.warmup_updates == 92
    assert plan.planned_tokens == 300_023_808
    assert plan.target_tokens == 300_000_000
    assert plan.token_overshoot == 23_808


def test_resolved_target_token_plan_uses_ceil_and_records_overshoot():
    document = deepcopy(load_document(BASELINE_PATH))
    training = document["training"]
    training["micro_batch_size"] = 2
    training["gradient_accumulation_steps"] = 4
    training["num_workers"] = 0
    training["pin_memory"] = False
    config = TrainingConfig.from_mapping(
        project=document["project"],
        model=document["model"],
        training=training,
    )

    plan = config.resolve()

    assert plan.tokens_per_micro_step == 1_024
    assert plan.tokens_per_update == 4_096
    assert plan.total_updates == 73_243
    assert plan.warmup_updates == 1_465
    assert plan.planned_tokens == 300_003_328
    assert plan.target_tokens == 300_000_000
    assert plan.token_overshoot == 3_328


def test_yaml_training_fields_match_frozen_dataclass_contract():
    expected = training_field_names()

    assert set(load_document(DEBUG_PATH)["training"]) == expected
    assert set(load_document(BASELINE_PATH)["training"]) == expected


def test_to_dict_is_stable_and_contains_derived_project_fields():
    config = TrainingConfig.from_yaml(DEBUG_PATH)
    payload = config.to_dict()

    assert payload["project_name"] == "small-gpt-debug"
    assert payload["seed"] == 1337
    assert payload["context_length"] == 128
    assert payload["vocab_size"] == 16_384
    assert payload["micro_batch_size"] == 4


def test_resolved_plan_to_dict_is_stable():
    payload = TrainingConfig.from_yaml(DEBUG_PATH).resolve().to_dict()

    assert payload["tokens_per_update"] == 512
    assert payload["total_updates"] == 200
    assert payload["planned_tokens"] == 102_400


def test_missing_training_field_is_rejected(tmp_path):
    path = mutated_debug(
        tmp_path,
        lambda document: document["training"].pop("adam_eps"),
    )

    with pytest.raises(TrainingConfigError, match="missing fields"):
        TrainingConfig.from_yaml(path)


def test_unknown_training_field_is_rejected(tmp_path):
    path = mutated_debug(
        tmp_path,
        lambda document: document["training"].update({"lerning_rate": 1.0}),
    )

    with pytest.raises(TrainingConfigError, match="unknown fields"):
        TrainingConfig.from_yaml(path)


@pytest.mark.parametrize(
    "field",
    (
        "micro_batch_size",
        "gradient_accumulation_steps",
        "max_steps",
        "target_tokens",
        "log_interval",
        "eval_interval",
        "eval_batches",
        "save_interval",
        "num_workers",
    ),
)
def test_bool_never_counts_as_integer(tmp_path, field):
    def mutate(document):
        document["training"][field] = True
        if field == "target_tokens":
            document["training"]["max_steps"] = None

    path = mutated_debug(tmp_path, mutate)

    with pytest.raises(TrainingConfigError, match=field):
        TrainingConfig.from_yaml(path)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("micro_batch_size", 0),
        ("gradient_accumulation_steps", -1),
        ("max_steps", 0),
        ("log_interval", 0),
        ("eval_interval", -1),
        ("eval_batches", 0),
        ("save_interval", 0),
        ("num_workers", -1),
    ),
)
def test_invalid_integer_bounds_are_rejected(tmp_path, field, value):
    path = mutated_debug(
        tmp_path,
        lambda document: document["training"].update({field: value}),
    )

    with pytest.raises(TrainingConfigError, match=field):
        TrainingConfig.from_yaml(path)


@pytest.mark.parametrize(
    ("max_steps", "target_tokens"),
    ((200, 1_000), (None, None)),
)
def test_exactly_one_training_budget_is_required(
    tmp_path,
    max_steps,
    target_tokens,
):
    def mutate(document):
        document["training"]["max_steps"] = max_steps
        document["training"]["target_tokens"] = target_tokens

    path = mutated_debug(tmp_path, mutate)

    with pytest.raises(TrainingConfigError, match="exactly one.*max_steps"):
        TrainingConfig.from_yaml(path)


@pytest.mark.parametrize(
    ("warmup_steps", "warmup_ratio"),
    ((20, 0.1), (None, None)),
)
def test_exactly_one_warmup_form_is_required(
    tmp_path,
    warmup_steps,
    warmup_ratio,
):
    def mutate(document):
        document["training"]["warmup_steps"] = warmup_steps
        document["training"]["warmup_ratio"] = warmup_ratio

    path = mutated_debug(tmp_path, mutate)

    with pytest.raises(TrainingConfigError, match="exactly one.*warmup_steps"):
        TrainingConfig.from_yaml(path)


@pytest.mark.parametrize("ratio", (0.0, 1.0, -0.1, float("inf")))
def test_invalid_warmup_ratio_is_rejected(tmp_path, ratio):
    def mutate(document):
        document["training"]["warmup_steps"] = None
        document["training"]["warmup_ratio"] = ratio

    path = mutated_debug(tmp_path, mutate)

    with pytest.raises(TrainingConfigError, match="warmup_ratio"):
        TrainingConfig.from_yaml(path)


def test_resolved_warmup_cannot_consume_entire_run(tmp_path):
    def mutate(document):
        document["training"]["max_steps"] = 20
        document["training"]["warmup_steps"] = 20

    config = TrainingConfig.from_yaml(mutated_debug(tmp_path, mutate))

    with pytest.raises(TrainingConfigError, match="warmup updates"):
        config.resolve()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("learning_rate", 0.0),
        ("learning_rate", float("nan")),
        ("min_learning_rate", -1.0),
        ("weight_decay", -0.1),
        ("beta1", 1.0),
        ("beta2", -0.1),
        ("adam_eps", 0.0),
        ("gradient_clip", 0.0),
    ),
)
def test_invalid_numeric_hyperparameters_are_rejected(tmp_path, field, value):
    path = mutated_debug(
        tmp_path,
        lambda document: document["training"].update({field: value}),
    )

    with pytest.raises(TrainingConfigError, match=field):
        TrainingConfig.from_yaml(path)


def test_min_learning_rate_cannot_exceed_peak(tmp_path):
    path = mutated_debug(
        tmp_path,
        lambda document: document["training"].update(
            {"min_learning_rate": 0.001}
        ),
    )

    with pytest.raises(TrainingConfigError, match="min_learning_rate"):
        TrainingConfig.from_yaml(path)


@pytest.mark.parametrize("field", ("pin_memory", "deterministic", "allow_tf32"))
def test_boolean_fields_reject_integer_values(tmp_path, field):
    path = mutated_debug(
        tmp_path,
        lambda document: document["training"].update({field: 1}),
    )

    with pytest.raises(TrainingConfigError, match=field):
        TrainingConfig.from_yaml(path)


@pytest.mark.parametrize("field", ("run_dir", "checkpoint_dir"))
def test_output_directories_must_be_non_empty(tmp_path, field):
    path = mutated_debug(
        tmp_path,
        lambda document: document["training"].update({field: ""}),
    )

    with pytest.raises(TrainingConfigError, match=field):
        TrainingConfig.from_yaml(path)


def test_cpu_bf16_is_rejected_by_day7_contract(tmp_path):
    def mutate(document):
        document["training"]["device"] = "cpu"
        document["training"]["precision"] = "bf16"

    path = mutated_debug(tmp_path, mutate)

    with pytest.raises(TrainingConfigError, match="requires CUDA"):
        TrainingConfig.from_yaml(path)


@pytest.mark.parametrize(
    ("field", "value"),
    (("device", "mps"), ("precision", "fp16")),
)
def test_unsupported_device_or_precision_is_rejected(tmp_path, field, value):
    path = mutated_debug(
        tmp_path,
        lambda document: document["training"].update({field: value}),
    )

    with pytest.raises(TrainingConfigError, match=field):
        TrainingConfig.from_yaml(path)


def test_missing_project_section_is_rejected(tmp_path):
    path = mutated_debug(tmp_path, lambda document: document.pop("project"))

    with pytest.raises(TrainingConfigError, match="project section"):
        TrainingConfig.from_yaml(path)


def test_invalid_project_seed_is_rejected(tmp_path):
    path = mutated_debug(
        tmp_path,
        lambda document: document["project"].update({"seed": True}),
    )

    with pytest.raises(TrainingConfigError, match="project.seed"):
        TrainingConfig.from_yaml(path)


def test_invalid_model_context_length_is_rejected(tmp_path):
    path = mutated_debug(
        tmp_path,
        lambda document: document["model"].update({"context_length": 0}),
    )

    with pytest.raises(TrainingConfigError, match="model.context_length"):
        TrainingConfig.from_yaml(path)


def test_non_mapping_document_is_rejected(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(TrainingConfigError, match="top-level"):
        TrainingConfig.from_yaml(path)


def test_missing_file_has_contextual_error(tmp_path):
    with pytest.raises(TrainingConfigError, match="could not read"):
        TrainingConfig.from_yaml(tmp_path / "missing.yaml")
