from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from model import GPT, GPTConfig  # noqa: E402
from train import (  # noqa: E402
    PrecisionPolicy,
    Trainer,
    TrainerState,
    TrainingConfig,
    build_optimizer,
    build_scheduler,
)


DEBUG_PATH = PROJECT_ROOT / "configs" / "debug.yaml"


def main() -> int:
    model_config = GPTConfig.from_yaml(DEBUG_PATH)
    training_config = replace(
        TrainingConfig.from_yaml(DEBUG_PATH),
        device="cpu",
        precision="fp32",
    )
    plan = training_config.resolve()
    precision = PrecisionPolicy.from_config(training_config)

    torch.manual_seed(training_config.seed)
    model = GPT(model_config).to(precision.device)
    optimizer = build_optimizer(model, training_config)
    scheduler = build_scheduler(optimizer, training_config, plan)
    state = TrainerState(run_id="day07-stage-d-synthetic-update")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        state=state,
        config=training_config,
        plan=plan,
        precision=precision,
    )

    generator = torch.Generator(device="cpu").manual_seed(
        training_config.seed + 1
    )
    tokens = torch.randint(
        0,
        model_config.vocab_size,
        (
            plan.micro_batch_size,
            plan.context_length + 1,
        ),
        generator=generator,
        dtype=torch.long,
    )
    input_ids = tokens[:, :-1]
    targets = tokens[:, 1:]
    before = model.blocks[0].attention.qkv_proj.weight.detach().clone()

    metrics = trainer.run_update([(input_ids, targets)])
    parameter_changed = not torch.equal(
        model.blocks[0].attention.qkv_proj.weight,
        before,
    )

    assert parameter_changed
    assert metrics.completed_global_step == 1
    assert metrics.tokens == plan.tokens_per_update == 512
    assert metrics.samples == plan.micro_batch_size == 4
    assert scheduler.next_update_index == state.global_step == 1
    state.validate_for_plan(plan)

    print(f"Training config       : {DEBUG_PATH.name}")
    print(f"Resolved device       : {precision.device}")
    print(f"Precision             : {precision.precision}")
    print(f"Accumulation steps    : {plan.gradient_accumulation_steps}")
    print(f"Update index          : {metrics.update_index}")
    print(f"Completed global step : {metrics.completed_global_step}")
    print(f"Raw weighted loss     : {metrics.raw_token_weighted_loss:.6f}")
    print(f"Learning rate         : {metrics.learning_rate:.12g}")
    print(f"Grad norm before clip : {metrics.grad_norm_before_clip:.6f}")
    print(f"Samples               : {metrics.samples}")
    print(f"Tokens                : {metrics.tokens}")
    print(f"Parameter changed     : {parameter_changed}")
    print(f"Scheduler next update : {scheduler.next_update_index}")
    print("All synthetic training-update checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
