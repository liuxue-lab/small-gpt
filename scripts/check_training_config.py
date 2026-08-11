from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train import TrainingConfig  # noqa: E402


CONFIG_PATHS = (
    PROJECT_ROOT / "configs" / "debug.yaml",
    PROJECT_ROOT / "configs" / "baseline.yaml",
)


def print_training_summary(config_path: Path, config: TrainingConfig) -> None:
    print(f"Training config     : {config_path.name}")
    print(f"Project             : {config.project_name}")
    print(f"Seed                : {config.seed}")
    print(f"Requested device    : {config.device}")
    print(f"Precision           : {config.precision}")
    print(f"Context length      : {config.context_length}")
    print(f"Micro batch size    : {config.micro_batch_size}")
    print(f"Accumulation steps  : {config.gradient_accumulation_steps}")
    print(f"Max steps           : {config.max_steps}")
    print(f"Target tokens       : {config.target_tokens}")
    print(f"Peak learning rate  : {config.learning_rate}")
    print(f"Minimum learning rate: {config.min_learning_rate}")
    print(f"Warmup steps        : {config.warmup_steps}")
    print(f"Warmup ratio        : {config.warmup_ratio}")
    print(f"Execution ready     : {config.is_execution_ready}")

    if config.is_execution_ready:
        plan = config.resolve()
        print(f"Tokens/micro-step   : {plan.tokens_per_micro_step:,}")
        print(f"Tokens/update       : {plan.tokens_per_update:,}")
        print(f"Total updates       : {plan.total_updates:,}")
        print(f"Resolved warmup     : {plan.warmup_updates:,}")
        print(f"Planned tokens      : {plan.planned_tokens:,}")
        print(f"Token overshoot     : {plan.token_overshoot:,}")
    else:
        print(f"Unresolved fields   : {list(config.unresolved_fields)}")
    print("-" * 60)


def main() -> int:
    for config_path in CONFIG_PATHS:
        config = TrainingConfig.from_yaml(config_path)
        print_training_summary(config_path, config)

    print("All training configuration checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
