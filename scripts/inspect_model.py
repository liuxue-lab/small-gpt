from __future__ import annotations

# ruff: noqa: I001

import argparse
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import GPT, GPTConfig


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "debug.yaml"
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "data" / "tokenized" / "fineweb_edu_pilot" / "manifest.json"
)


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def non_negative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not torch.isfinite(torch.tensor(parsed)) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect the Day 6 GPT with a small synthetic or tokenized Pilot batch."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="model YAML path (default: configs/debug.yaml)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="execution device (default: auto)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--batch-source",
        choices=("synthetic", "pilot"),
        default="synthetic",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="tokenized Pilot manifest used with --batch-source pilot",
    )
    parser.add_argument("--batch-size", type=positive_integer, default=4)
    parser.add_argument(
        "--sequence-length",
        type=positive_integer,
        default=None,
        help="sequence length (default: model context length)",
    )
    parser.add_argument("--num-workers", type=non_negative_integer, default=0)
    parser.add_argument(
        "--backward",
        action="store_true",
        help="run one loss.backward() and summarize gradients",
    )
    parser.add_argument(
        "--overfit-steps",
        type=non_negative_integer,
        default=0,
        help="diagnostically optimize one fixed batch for N steps",
    )
    parser.add_argument(
        "--learning-rate",
        type=positive_float,
        default=3.0e-3,
        help="diagnostic overfit learning rate",
    )
    return parser


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return torch.device(requested)


def build_synthetic_batch(
    config: GPTConfig,
    *,
    batch_size: int,
    sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.randint(
        0,
        config.vocab_size,
        (batch_size, sequence_length + 1),
        dtype=torch.long,
    )
    return tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()


def build_pilot_batch(
    manifest: Path,
    *,
    batch_size: int,
    sequence_length: int,
    num_workers: int,
    seed: int,
    pin_memory: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    from data_pipeline import (
        CausalWindowDataset,
        EpochRandomWindowSampler,
        SplitTokenStore,
        build_dataloader,
    )

    store = SplitTokenStore(manifest, "train")
    try:
        dataset = CausalWindowDataset(
            store,
            context_length=sequence_length,
            mode="all_starts",
        )
        sampler = EpochRandomWindowSampler(
            dataset,
            samples_per_epoch=batch_size,
            base_seed=seed,
            epoch=0,
        )
        loader = build_dataloader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )
        inputs, targets = next(iter(loader))
    finally:
        store.close()

    return inputs, targets


def gradient_summary(model: GPT) -> tuple[int, int, bool]:
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    finite = all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    nonzero = sum(bool(torch.count_nonzero(gradient)) for gradient in gradients)
    return len(gradients), nonzero, finite


def run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    config_path = args.config.resolve()
    config = GPTConfig.from_yaml(config_path)
    sequence_length = args.sequence_length or config.context_length
    if sequence_length > config.context_length:
        raise ValueError(
            "sequence length exceeds model context length: "
            f"{sequence_length} > {config.context_length}"
        )

    model = GPT(config).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    if args.batch_source == "synthetic":
        inputs, targets = build_synthetic_batch(
            config,
            batch_size=args.batch_size,
            sequence_length=sequence_length,
        )
    else:
        inputs, targets = build_pilot_batch(
            args.manifest.resolve(),
            batch_size=args.batch_size,
            sequence_length=sequence_length,
            num_workers=args.num_workers,
            seed=args.seed,
            pin_memory=device.type == "cuda",
        )

    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    if args.backward and args.overfit_steps == 0:
        model.train()
        output = model(inputs, targets)
    else:
        model.train(args.overfit_steps > 0)
        with torch.no_grad():
            output = model(inputs, targets)
    if output.loss is None:
        raise RuntimeError("model did not return a loss for a batch with targets")

    print(f"Configuration       : {config_path}")
    print(f"Device              : {device}")
    print(f"Seed                : {args.seed}")
    print(f"Batch source        : {args.batch_source}")
    print(f"Parameters          : {parameter_count:,}")
    print(f"Trainable parameters: {trainable_count:,}")
    print(
        f"Weight tying        : {model.lm_head.weight is model.token_embedding.weight}"
    )
    print(f"Input shape/dtype   : {tuple(inputs.shape)} / {inputs.dtype}")
    print(f"Target shape/dtype  : {tuple(targets.shape)} / {targets.dtype}")
    print(f"Causal x/y shift    : {bool(torch.equal(targets[:, :-1], inputs[:, 1:]))}")
    print(
        "Input token range   : "
        f"[{int(inputs.min().item())}, {int(inputs.max().item())}]"
    )
    print(f"Logits shape        : {tuple(output.logits.shape)}")
    print(f"Loss                : {output.loss.item():.6f}")
    print(f"Logits finite       : {bool(torch.isfinite(output.logits).all())}")

    if args.overfit_steps > 0:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=0.0,
        )
        initial_loss = output.loss.item()
        for _ in range(args.overfit_steps):
            optimizer.zero_grad(set_to_none=True)
            step_output = model(inputs, targets)
            if step_output.loss is None:
                raise RuntimeError("model did not return a diagnostic loss")
            step_output.loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            final_output = model(inputs, targets)
        if final_output.loss is None:
            raise RuntimeError("model did not return a final diagnostic loss")
        gradient_count, nonzero_count, gradients_finite = gradient_summary(model)
        print(f"Gradient tensors    : {gradient_count}")
        print(f"Nonzero gradients   : {nonzero_count}")
        print(f"Gradients finite    : {gradients_finite}")
        print(f"Overfit steps       : {args.overfit_steps}")
        print(f"Overfit initial loss: {initial_loss:.6f}")
        print(f"Overfit final loss  : {final_output.loss.item():.6f}")
    elif args.backward:
        output.loss.backward()
        gradient_count, nonzero_count, gradients_finite = gradient_summary(model)
        print(f"Gradient tensors    : {gradient_count}")
        print(f"Nonzero gradients   : {nonzero_count}")
        print(f"Gradients finite    : {gradients_finite}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"Model inspection failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
