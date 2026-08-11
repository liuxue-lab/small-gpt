from __future__ import annotations

from collections.abc import Iterator
from itertools import islice
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Sampler

from data_pipeline import (
    CausalWindowDataset,
    EpochRandomWindowSampler,
    SplitTokenStore,
    build_dataloader,
)

from .config import ResolvedTrainingPlan, TrainingConfig
from .state import TrainerState


class DataStreamError(RuntimeError):
    """Raised when the deterministic train/evaluation stream is inconsistent."""


class OffsetSampler(Sampler[int]):
    """Skip sampler indexes without reading the corresponding dataset items."""

    def __init__(self, sampler: Sampler[int], *, offset: int) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError(
                f"sampler must be a torch Sampler, got {type(sampler)!r}"
            )
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise DataStreamError(
                f"offset must be a non-negative integer, got {offset!r}"
            )
        try:
            total_samples = len(sampler)
        except TypeError as error:
            raise DataStreamError(
                "offset sampler requires a base sampler with a finite length"
            ) from error
        if offset > total_samples:
            raise DataStreamError(
                f"offset {offset} exceeds sampler length {total_samples}"
            )
        self.sampler = sampler
        self.offset = offset
        self.total_samples = total_samples

    def __len__(self) -> int:
        return self.total_samples - self.offset

    def __iter__(self) -> Iterator[int]:
        yield from islice(iter(self.sampler), self.offset, None)


def _shutdown_loader_iterator(iterator: object | None) -> None:
    if iterator is None:
        return
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()


class TrainingDataStream(
    Iterator[tuple[torch.Tensor, torch.Tensor]],
):
    """Single deterministic random-window stream with checkpointable offset."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        config: TrainingConfig,
        plan: ResolvedTrainingPlan,
        state: TrainerState,
        verify_hashes: bool = False,
    ) -> None:
        if not isinstance(config, TrainingConfig):
            raise TypeError(
                f"config must be a TrainingConfig, got {type(config)!r}"
            )
        if not isinstance(plan, ResolvedTrainingPlan):
            raise TypeError(
                "plan must be a ResolvedTrainingPlan, "
                f"got {type(plan)!r}"
            )
        if not isinstance(state, TrainerState):
            raise TypeError(
                f"state must be a TrainerState, got {type(state)!r}"
            )
        if not isinstance(verify_hashes, bool):
            raise TypeError("verify_hashes must be a boolean")
        if config.resolve() != plan:
            raise DataStreamError(
                "training config and resolved plan do not describe the same "
                "execution"
            )
        state.validate_for_plan(plan)
        if state.data_epoch != 0:
            raise DataStreamError(
                "Day 7 single-stream training requires data_epoch=0"
            )
        if state.batches_consumed_in_epoch != state.micro_steps_seen:
            raise DataStreamError(
                "batches_consumed_in_epoch must equal micro_steps_seen in the "
                "Day 7 single-stream contract"
            )

        samples_per_update = (
            plan.micro_batch_size * plan.gradient_accumulation_steps
        )
        total_samples = plan.total_updates * samples_per_update
        if state.samples_consumed > total_samples:
            raise DataStreamError(
                "trainer state consumed more samples than the resolved plan"
            )
        if state.samples_consumed % plan.micro_batch_size != 0:
            raise DataStreamError(
                "samples_consumed must end on a complete micro-batch boundary"
            )
        remaining_samples = total_samples - state.samples_consumed
        if remaining_samples <= 0:
            raise DataStreamError("training data stream has no samples remaining")
        if remaining_samples % plan.micro_batch_size != 0:
            raise DataStreamError(
                "remaining samples do not form complete micro-batches"
            )

        self.manifest_path = Path(manifest_path).resolve()
        self.config = config
        self.plan = plan
        self.initial_sample_offset = state.samples_consumed
        self.total_samples = total_samples
        self.remaining_samples = remaining_samples
        self.total_micro_batches = remaining_samples // plan.micro_batch_size
        self._yielded_micro_batches = 0
        self._closed = False
        self._iterator: Iterator[Any] | None = None

        store: SplitTokenStore | None = None
        try:
            store = SplitTokenStore(
                self.manifest_path,
                "train",
                verify_hashes=verify_hashes,
            )
            dataset = CausalWindowDataset(
                store,
                plan.context_length,
                mode="all_starts",
            )
            base_sampler = EpochRandomWindowSampler(
                dataset,
                samples_per_epoch=total_samples,
                base_seed=config.seed,
                epoch=0,
            )
            sampler = OffsetSampler(
                base_sampler,
                offset=state.samples_consumed,
            )
            loader = build_dataloader(
                dataset,
                batch_size=plan.micro_batch_size,
                sampler=sampler,
                num_workers=plan.num_workers,
                pin_memory=plan.pin_memory,
                drop_last=True,
                persistent_workers=False,
            )
            iterator = iter(loader)
        except Exception:
            if store is not None:
                store.close()
            raise

        self.store = store
        self.dataset = dataset
        self.base_sampler = base_sampler
        self.sampler = sampler
        self.loader: DataLoader[Any] = loader
        self._iterator = iterator

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def yielded_micro_batches(self) -> int:
        return self._yielded_micro_batches

    @property
    def micro_batches_remaining(self) -> int:
        return self.total_micro_batches - self._yielded_micro_batches

    def __len__(self) -> int:
        return self.total_micro_batches

    def __iter__(self) -> TrainingDataStream:
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._closed or self._iterator is None:
            raise DataStreamError("training data stream is closed")
        batch = next(self._iterator)
        self._yielded_micro_batches += 1
        input_ids, targets = batch
        return input_ids, targets

    def close(self) -> None:
        if self._closed:
            return
        _shutdown_loader_iterator(self._iterator)
        self._iterator = None
        self.store.close()
        self._closed = True

    def __enter__(self) -> TrainingDataStream:
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()


class ValidationDataStream:
    """Repeatable non-overlapping validation windows in fixed split order."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        plan: ResolvedTrainingPlan,
        verify_hashes: bool = False,
    ) -> None:
        if not isinstance(plan, ResolvedTrainingPlan):
            raise TypeError(
                "plan must be a ResolvedTrainingPlan, "
                f"got {type(plan)!r}"
            )
        if not isinstance(verify_hashes, bool):
            raise TypeError("verify_hashes must be a boolean")

        self.manifest_path = Path(manifest_path).resolve()
        self.plan = plan
        self._closed = False
        store: SplitTokenStore | None = None
        try:
            store = SplitTokenStore(
                self.manifest_path,
                "validation",
                verify_hashes=verify_hashes,
            )
            dataset = CausalWindowDataset(
                store,
                plan.context_length,
                mode="sequential",
            )
            loader = build_dataloader(
                dataset,
                batch_size=plan.micro_batch_size,
                num_workers=plan.num_workers,
                pin_memory=plan.pin_memory,
                drop_last=False,
                persistent_workers=False,
            )
        except Exception:
            if store is not None:
                store.close()
            raise

        self.store = store
        self.dataset = dataset
        self.loader: DataLoader[Any] = loader

    @property
    def is_closed(self) -> bool:
        return self._closed

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        if self._closed:
            raise DataStreamError("validation data stream is closed")
        return iter(self.loader)

    def close(self) -> None:
        if self._closed:
            return
        self.store.close()
        self._closed = True

    def __enter__(self) -> ValidationDataStream:
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()
