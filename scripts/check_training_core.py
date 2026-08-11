from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import GPT, GPTConfig  # noqa: E402
from train import (  # noqa: E402
    TrainerState,
    TrainingConfig,
    build_optimizer,
    build_scheduler,
    partition_parameters,
)


DEBUG_PATH = PROJECT_ROOT / "configs" / "debug.yaml"


def main() -> int:
    model_config = GPTConfig.from_yaml(DEBUG_PATH)
    training_config = TrainingConfig.from_yaml(DEBUG_PATH)
    plan = training_config.resolve()
    model = GPT(model_config)

    groups = partition_parameters(model)
    optimizer = build_optimizer(model, training_config)
    scheduler = build_scheduler(optimizer, training_config, plan)
    state = TrainerState(run_id="day07-stage-c-inspection")
    state.validate_for_plan(plan)

    tied_aliases = groups.aliases_for("token_embedding.weight")
    assert tied_aliases == ("token_embedding.weight", "lm_head.weight")
    assert groups.unique_parameter_tensors == 20
    assert groups.decay_numel == 2_506_752
    assert groups.no_decay_numel == 1_280
    assert groups.total_numel == model_config.parameter_count == 2_508_032
    assert len(optimizer.param_groups) == 2

    print(f"Training config      : {DEBUG_PATH.name}")
    print(f"Exact parameters     : {groups.total_numel:,}")
    print(f"Unique tensors       : {groups.unique_parameter_tensors}")
    print(
        "Decay group         : "
        f"{len(groups.decay_parameters)} tensors / {groups.decay_numel:,} params"
    )
    print(
        "No-decay group      : "
        f"{len(groups.no_decay_parameters)} tensors / "
        f"{groups.no_decay_numel:,} params"
    )
    print(f"Tied aliases         : {list(tied_aliases)}")
    print(f"Total updates        : {plan.total_updates}")
    print(f"Warmup updates       : {plan.warmup_updates}")
    for update_index in (0, 19, 20, 199, 200):
        print(
            f"LR at update {update_index:>3}: "
            f"{scheduler.lr_for_update(update_index):.12g}"
        )
    print("Trainer state        : valid at update boundary 0")
    print("All training-core checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
