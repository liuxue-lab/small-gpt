"""Read-only validation and Dataset/DataLoader inspection for Day 5 output."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline import (  # noqa: E402
    CausalWindowDataset,
    EpochRandomWindowSampler,
    SplitTokenStore,
    TokenizedDataError,
    build_dataloader,
    validate_completed_corpus,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a complete tokenized corpus and inspect causal T+1 "
            "Dataset/DataLoader samples without modifying the data."
        )
    )
    parser.add_argument(
        "--manifest",
        default="data/tokenized/fineweb_edu_pilot/manifest.json",
        help="Tokenized corpus manifest relative to the project root.",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=512,
        help="Causal model context length T; each item reads T+1 tokens.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of sample windows to display for each split.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Build one DataLoader batch per split when greater than zero.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers used with --batch-size (validate 0, then 2).",
    )
    args = parser.parse_args(argv)
    if args.context_length <= 0:
        parser.error("--context-length must be greater than zero")
    if args.samples <= 0:
        parser.error("--samples must be greater than zero")
    if args.batch_size < 0:
        parser.error("--batch-size must be zero or greater")
    if args.num_workers < 0:
        parser.error("--num-workers must be zero or greater")
    if args.num_workers and args.batch_size == 0:
        parser.error("--num-workers requires --batch-size greater than zero")
    return args


def _project_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _window_summary(
    dataset: CausalWindowDataset,
    index: int,
) -> str:
    start = dataset.start_for_index(index)
    x, y = dataset[index]
    shift_ok = torch.equal(y[:-1], x[1:])
    minimum = min(int(x.min()), int(y.min()))
    maximum = max(int(x.max()), int(y.max()))
    return (
        f"start={start}, x={tuple(x.shape)}, y={tuple(y.shape)}, "
        f"dtype={x.dtype}, ids=[{minimum}, {maximum}], shift={shift_ok}"
    )


def _inspect_split(
    manifest_path: Path,
    manifest: dict,
    split: str,
    args: argparse.Namespace,
) -> None:
    store = SplitTokenStore(manifest_path, split)
    try:
        mode = "all_starts" if split == "train" else "sequential"
        dataset = CausalWindowDataset(
            store,
            args.context_length,
            mode=mode,
        )
        print(
            f"{split}: tokens={len(store)}, documents={store.document_count}, "
            f"mode={mode}, windows={len(dataset)}, "
            f"remainder={dataset.evaluation_remainder if mode == 'sequential' else 'n/a'}"
        )

        if split == "train":
            sample_sampler = EpochRandomWindowSampler(
                dataset,
                samples_per_epoch=min(args.samples, len(dataset)),
                base_seed=int(manifest["dataset_contract"]["train_base_seed"]),
                epoch=0,
            )
            sample_indices = list(sample_sampler)
        else:
            sample_indices = list(range(min(args.samples, len(dataset))))
        for sample_number, index in enumerate(sample_indices):
            print(f"  sample {sample_number}: {_window_summary(dataset, index)}")

        if args.batch_size > 0:
            sampler = None
            drop_last = False
            if split == "train":
                sampler = EpochRandomWindowSampler(
                    dataset,
                    samples_per_epoch=args.batch_size * 2,
                    base_seed=42,
                    epoch=0,
                )
                drop_last = True
            loader = build_dataloader(
                dataset,
                batch_size=args.batch_size,
                sampler=sampler,
                num_workers=args.num_workers,
                pin_memory=False,
                drop_last=drop_last,
                persistent_workers=False,
            )
            loader_iterator = iter(loader)
            batch_x, batch_y = next(loader_iterator)
            shift_ok = torch.equal(batch_y[:, :-1], batch_x[:, 1:])
            print(
                f"  batch: x={tuple(batch_x.shape)}, y={tuple(batch_y.shape)}, "
                f"dtype={batch_x.dtype}, workers={args.num_workers}, shift={shift_ok}"
            )
            del loader_iterator, loader
    finally:
        store.close()


def run(args: argparse.Namespace) -> int:
    manifest_path = _project_path(args.manifest)
    manifest = validate_completed_corpus(
        manifest_path,
        project_root=PROJECT_ROOT,
        verify_identities=True,
        scan_payload=True,
    )
    print("Tokenized corpus validation complete")
    print(f"  manifest: {manifest_path}")
    print(f"  fingerprint: {manifest['config_fingerprint']}")
    print(f"  records: {manifest['totals']['records']}")
    print(f"  model tokens: {manifest['totals']['model_tokens']}")
    print(f"  payload bytes: {manifest['totals']['token_payload_bytes']}")
    for split in ("train", "validation", "test"):
        _inspect_split(manifest_path, manifest, split, args)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        print("Inspection interrupted; no data were modified.", file=sys.stderr)
        return 130
    except (TokenizedDataError, OSError, ValueError, IndexError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
