"""Memory-mapped logical token streams and causal next-token datasets."""

from __future__ import annotations

import bisect
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .binary_format import (
    TOKEN_DTYPE,
    DatasetContractError,
    TokenShardHeader,
    close_memmap,
    map_token_payload,
    sha256_file,
    validate_token_shard,
)


ALLOWED_SPLITS = ("train", "validation", "test")
DATASET_MODES = ("all_starts", "sequential")


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetContractError(f"cannot read valid manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DatasetContractError(f"manifest must be a JSON object: {path}")
    if payload.get("schema_version") != 1:
        raise DatasetContractError("manifest schema_version must be 1")
    if payload.get("format_name") != "small_gpt_tokenized_corpus":
        raise DatasetContractError("manifest format_name is invalid")
    if payload.get("status") != "complete":
        raise DatasetContractError("manifest must have status='complete'")
    return payload


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DatasetContractError(f"{field} must be a mapping")
    return value


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DatasetContractError(f"{field} must be a positive integer")
    return value


def _safe_shard_path(corpus_root: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise DatasetContractError(f"{field} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute():
        raise DatasetContractError(f"{field} must be relative")
    resolved = (corpus_root / relative).resolve()
    try:
        resolved.relative_to(corpus_root.resolve())
    except ValueError as exc:
        raise DatasetContractError(f"{field} escapes the corpus root") from exc
    return resolved


class MemmapTokenShard:
    """One validated token shard with a lazily opened payload memmap."""

    def __init__(
        self,
        path: str | Path,
        *,
        split: str,
        expected_token_count: int | None = None,
        expected_vocab_size: int | None = None,
        expected_eos_token_id: int | None = None,
        expected_sha256: str | None = None,
        verify_hash: bool = False,
    ) -> None:
        self.path = Path(path).resolve()
        self.header: TokenShardHeader = validate_token_shard(
            self.path,
            expected_split=split,
            expected_vocab_size=expected_vocab_size,
            expected_eos_token_id=expected_eos_token_id,
            scan_payload=False,
        )
        if (
            expected_token_count is not None
            and self.header.token_count != expected_token_count
        ):
            raise DatasetContractError(
                f"token count mismatch for {self.path}: "
                f"header={self.header.token_count}, expected={expected_token_count}"
            )
        self.expected_sha256 = expected_sha256
        if verify_hash:
            if expected_sha256 is None:
                raise DatasetContractError("verify_hash requires expected_sha256")
            actual = sha256_file(self.path)
            if actual != expected_sha256:
                raise DatasetContractError(
                    f"token shard SHA-256 mismatch for {self.path}: "
                    f"found {actual}, expected {expected_sha256}"
                )
        self._mapped: np.memmap | None = None

    def __len__(self) -> int:
        return self.header.token_count

    @property
    def is_open(self) -> bool:
        return self._mapped is not None

    def _ensure_open(self) -> np.memmap:
        if self._mapped is None:
            self._mapped = map_token_payload(self.path, self.header)
        return self._mapped

    def read(self, start: int, length: int) -> np.ndarray:
        if isinstance(start, bool) or not isinstance(start, int) or start < 0:
            raise DatasetContractError("shard-local start must be a non-negative integer")
        if isinstance(length, bool) or not isinstance(length, int) or length < 0:
            raise DatasetContractError("slice length must be a non-negative integer")
        if start + length > len(self):
            raise DatasetContractError(
                f"slice [{start}, {start + length}) exceeds shard length {len(self)}"
            )
        if length == 0:
            return np.empty(0, dtype=TOKEN_DTYPE)
        return self._ensure_open()[start : start + length]

    def close(self) -> None:
        close_memmap(self._mapped)
        self._mapped = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_mapped"] = None
        return state

    def __del__(self) -> None:  # pragma: no cover - timing depends on GC
        try:
            self.close()
        except Exception:
            pass


class SplitTokenStore:
    """Expose all physical shards in one split as one contiguous uint16 stream."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        *,
        verify_hashes: bool = False,
    ) -> None:
        if split not in ALLOWED_SPLITS:
            raise DatasetContractError(
                f"split must be one of {ALLOWED_SPLITS}; found {split!r}"
            )
        self.manifest_path = Path(manifest_path).resolve()
        self.split = split
        manifest = _load_manifest(self.manifest_path)
        source = _mapping(manifest.get("source"), "manifest.source")
        if source.get("split_order") != list(ALLOWED_SPLITS):
            raise DatasetContractError(
                f"manifest source.split_order must be {list(ALLOWED_SPLITS)}"
            )
        splits = _mapping(manifest.get("splits"), "manifest.splits")
        if set(splits) != set(ALLOWED_SPLITS):
            raise DatasetContractError(
                f"manifest split keys must be {list(ALLOWED_SPLITS)}"
            )
        split_payload = _mapping(splits[split], f"manifest.splits.{split}")
        shards = split_payload.get("shards")
        if not isinstance(shards, list) or not shards:
            raise DatasetContractError(f"manifest {split} shards must be non-empty")

        tokenizer = _mapping(manifest.get("tokenizer"), "manifest.tokenizer")
        vocab_size = _positive_int(tokenizer.get("vocab_size"), "manifest.tokenizer.vocab_size")
        special_ids = _mapping(
            tokenizer.get("special_token_ids"),
            "manifest.tokenizer.special_token_ids",
        )
        eos_token_id = int(special_ids.get("eos", -1))
        if not 0 <= eos_token_id < vocab_size:
            raise DatasetContractError("manifest EOS token ID is invalid")

        corpus_root = self.manifest_path.parent
        self.shards: list[MemmapTokenShard] = []
        self._starts: list[int] = []
        expected_token_start = 0
        expected_document_start = 0
        for shard_index, raw_shard in enumerate(shards):
            shard = _mapping(raw_shard, f"manifest {split} shard {shard_index}")
            if shard.get("shard_index") != shard_index:
                raise DatasetContractError(f"{split} shard indexes are not contiguous")
            if shard.get("global_token_start") != expected_token_start:
                raise DatasetContractError(f"{split} token prefixes are not contiguous")
            if shard.get("global_document_start") != expected_document_start:
                raise DatasetContractError(f"{split} document prefixes are not contiguous")
            binary = _mapping(shard.get("binary"), f"{split} shard {shard_index}.binary")
            binary_path = _safe_shard_path(
                corpus_root,
                binary.get("path"),
                f"{split} shard {shard_index}.binary.path",
            )
            if binary_path.parent != corpus_root / split:
                raise DatasetContractError(
                    f"{split} token shard must be a direct child of its split directory"
                )
            token_count = _positive_int(
                shard.get("token_count"),
                f"{split} shard {shard_index}.token_count",
            )
            document_count = _positive_int(
                shard.get("document_count"),
                f"{split} shard {shard_index}.document_count",
            )
            expected_hash = binary.get("sha256")
            if not isinstance(expected_hash, str) or len(expected_hash) != 64:
                raise DatasetContractError(f"invalid binary hash for {binary_path}")
            self._starts.append(expected_token_start)
            self.shards.append(
                MemmapTokenShard(
                    binary_path,
                    split=split,
                    expected_token_count=token_count,
                    expected_vocab_size=vocab_size,
                    expected_eos_token_id=eos_token_id,
                    expected_sha256=expected_hash,
                    verify_hash=verify_hashes,
                )
            )
            expected_token_start += token_count
            expected_document_start += document_count

        self.token_count = expected_token_start
        self.document_count = expected_document_start
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        declared_tokens = _positive_int(
            split_payload.get("model_tokens"),
            f"manifest.splits.{split}.model_tokens",
        )
        declared_documents = _positive_int(
            split_payload.get("records"),
            f"manifest.splits.{split}.records",
        )
        if self.token_count != declared_tokens:
            raise DatasetContractError(
                f"{split} shard tokens sum to {self.token_count}; "
                f"manifest declares {declared_tokens}"
            )
        if self.document_count != declared_documents:
            raise DatasetContractError(
                f"{split} shard documents sum to {self.document_count}; "
                f"manifest declares {declared_documents}"
            )

    def __len__(self) -> int:
        return self.token_count

    def read(self, start: int, length: int) -> np.ndarray:
        """Read a logical split slice, joining physical shards only when needed."""

        if isinstance(start, bool) or not isinstance(start, int) or start < 0:
            raise DatasetContractError("logical start must be a non-negative integer")
        if isinstance(length, bool) or not isinstance(length, int) or length < 0:
            raise DatasetContractError("logical length must be a non-negative integer")
        end = start + length
        if end > self.token_count:
            raise DatasetContractError(
                f"logical slice [{start}, {end}) exceeds {self.split} length "
                f"{self.token_count}"
            )
        if length == 0:
            return np.empty(0, dtype=TOKEN_DTYPE)

        shard_index = bisect.bisect_right(self._starts, start) - 1
        shard_start = self._starts[shard_index]
        shard = self.shards[shard_index]
        local_start = start - shard_start
        if end <= shard_start + len(shard):
            return shard.read(local_start, length)

        result = np.empty(length, dtype=TOKEN_DTYPE)
        output_offset = 0
        logical_position = start
        while logical_position < end:
            shard_index = bisect.bisect_right(self._starts, logical_position) - 1
            shard_start = self._starts[shard_index]
            shard = self.shards[shard_index]
            local_start = logical_position - shard_start
            take = min(end - logical_position, len(shard) - local_start)
            result[output_offset : output_offset + take] = shard.read(local_start, take)
            logical_position += take
            output_offset += take
        return result

    def close(self) -> None:
        for shard in self.shards:
            shard.close()

    def __enter__(self) -> "SplitTokenStore":
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()


class CausalWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Create `(x, y)` windows from `context_length + 1` logical tokens."""

    def __init__(
        self,
        store: SplitTokenStore,
        context_length: int,
        *,
        mode: str,
    ) -> None:
        if (
            isinstance(context_length, bool)
            or not isinstance(context_length, int)
            or context_length <= 0
        ):
            raise DatasetContractError("context_length must be a positive integer")
        if mode not in DATASET_MODES:
            raise DatasetContractError(
                f"mode must be one of {DATASET_MODES}; found {mode!r}"
            )
        if len(store) < context_length + 1:
            raise DatasetContractError(
                f"{store.split} has {len(store)} tokens; at least "
                f"{context_length + 1} are required"
            )
        self.store = store
        self.context_length = context_length
        self.mode = mode
        self.possible_starts = len(store) - context_length
        self.evaluation_remainder = (len(store) - 1) % context_length

    def __len__(self) -> int:
        if self.mode == "all_starts":
            return self.possible_starts
        return (len(self.store) - 1) // self.context_length

    def start_for_index(self, index: int) -> int:
        if isinstance(index, bool) or not isinstance(index, int):
            raise DatasetContractError("dataset index must be an integer")
        if index < 0 or index >= len(self):
            raise IndexError(f"dataset index {index} is outside [0, {len(self)})")
        return index if self.mode == "all_starts" else index * self.context_length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.start_for_index(index)
        chunk = self.store.read(start, self.context_length + 1)
        tokens = torch.from_numpy(np.array(chunk, dtype=np.int64, copy=True))
        return tokens[:-1], tokens[1:]


class EpochRandomWindowSampler(Sampler[int]):
    """Token-uniform random starts with replacement and deterministic epochs."""

    def __init__(
        self,
        dataset: CausalWindowDataset,
        *,
        samples_per_epoch: int,
        base_seed: int,
        epoch: int = 0,
        chunk_size: int = 65_536,
    ) -> None:
        if dataset.mode != "all_starts":
            raise DatasetContractError(
                "EpochRandomWindowSampler requires an all_starts dataset"
            )
        for value, field, allow_zero in (
            (samples_per_epoch, "samples_per_epoch", False),
            (base_seed, "base_seed", True),
            (epoch, "epoch", True),
            (chunk_size, "chunk_size", False),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or (not allow_zero and value == 0)
            ):
                qualifier = "non-negative" if allow_zero else "positive"
                raise DatasetContractError(f"{field} must be a {qualifier} integer")
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch
        self.base_seed = base_seed
        self.epoch = epoch
        self.chunk_size = chunk_size

    def __len__(self) -> int:
        return self.samples_per_epoch

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise DatasetContractError("epoch must be a non-negative integer")
        self.epoch = epoch

    def _epoch_seed(self) -> int:
        digest = hashlib.sha256(
            f"small-gpt-window-sampler:{self.base_seed}:{self.epoch}".encode("ascii")
        ).digest()
        return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self._epoch_seed())
        remaining = self.samples_per_epoch
        high = len(self.dataset)
        while remaining:
            count = min(remaining, self.chunk_size)
            starts = torch.randint(
                low=0,
                high=high,
                size=(count,),
                generator=generator,
                dtype=torch.int64,
            )
            yield from starts.tolist()
            remaining -= count


def build_dataloader(
    dataset: CausalWindowDataset,
    *,
    batch_size: int,
    sampler: Sampler[int] | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """Build a Windows-safe DataLoader with explicit, non-shuffle sampling."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise DatasetContractError("batch_size must be a positive integer")
    if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 0:
        raise DatasetContractError("num_workers must be a non-negative integer")
    if not isinstance(pin_memory, bool) or not isinstance(drop_last, bool):
        raise DatasetContractError("pin_memory and drop_last must be boolean")
    if not isinstance(persistent_workers, bool):
        raise DatasetContractError("persistent_workers must be boolean")
    if persistent_workers and num_workers == 0:
        raise DatasetContractError("persistent_workers requires num_workers > 0")
    if (
        isinstance(prefetch_factor, bool)
        or not isinstance(prefetch_factor, int)
        or prefetch_factor <= 0
    ):
        raise DatasetContractError("prefetch_factor must be a positive integer")

    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
        "persistent_workers": persistent_workers,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs)
