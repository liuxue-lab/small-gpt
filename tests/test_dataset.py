import gc
import hashlib
import json
import pickle
from pathlib import Path

import pytest
import torch

from data_pipeline import (
    CausalWindowDataset,
    DatasetContractError,
    EpochRandomWindowSampler,
    SplitTokenStore,
    TokenShardWriter,
    build_dataloader,
)


SPECIAL_IDS = {"bos": 0, "eos": 1, "pad": 2, "unk": 3}


def _document_hash(split: str, shard_index: int, document_index: int) -> bytes:
    label = f"{split}:{shard_index}:{document_index}"
    return hashlib.sha256(label.encode("utf-8")).digest()


def _write_dataset_fixture(tmp_path: Path):
    root = tmp_path / "tokenized"
    layout = {
        "train": [
            [(10, 11, 1)],
            [(20, 21, 22, 1)],
            [(30, 31, 32, 33, 1)],
        ],
        "validation": [[(40, 41, 42, 43, 44, 45, 46, 47, 48, 1)]],
        "test": [[(50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 1)]],
    }
    streams = {}
    splits = {}
    for split, shard_documents in layout.items():
        global_token_start = 0
        global_document_start = 0
        shard_metadata = []
        stream = []
        for shard_index, documents in enumerate(shard_documents):
            writer = TokenShardWriter(
                staging_root=root,
                split=split,
                shard_index=shard_index,
                global_token_start=global_token_start,
                global_document_start=global_document_start,
                vocab_size=16_384,
                special_token_ids=SPECIAL_IDS,
            )
            for document_index, token_ids in enumerate(documents):
                writer.append_document(
                    token_ids,
                    text_sha256=_document_hash(
                        split,
                        shard_index,
                        document_index,
                    ),
                    provided_tokens=len(token_ids) - 1,
                )
                stream.extend(token_ids)
            metadata = writer.finalize()
            shard_metadata.append(metadata)
            global_token_start += metadata["token_count"]
            global_document_start += metadata["document_count"]
        streams[split] = stream
        splits[split] = {
            "records": global_document_start,
            "model_tokens": global_token_start,
            "storage_shards": len(shard_metadata),
            "shards": shard_metadata,
        }

    manifest = {
        "schema_version": 1,
        "format_name": "small_gpt_tokenized_corpus",
        "status": "complete",
        "source": {"split_order": ["train", "validation", "test"]},
        "tokenizer": {
            "vocab_size": 16_384,
            "special_token_ids": SPECIAL_IDS,
        },
        "splits": splits,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path, streams


def test_split_store_is_lazy_and_reads_single_exact_and_cross_shard_slices(
    tmp_path,
):
    manifest_path, streams = _write_dataset_fixture(tmp_path)
    store = SplitTokenStore(manifest_path, "train", verify_hashes=True)
    try:
        assert len(store) == 12
        assert store.document_count == 3
        assert not any(shard.is_open for shard in store.shards)

        assert store.read(0, 2).tolist() == streams["train"][0:2]
        assert [shard.is_open for shard in store.shards] == [True, False, False]

        assert store.read(3, 4).tolist() == streams["train"][3:7]
        assert [shard.is_open for shard in store.shards] == [True, True, False]

        assert store.read(2, 9).tolist() == streams["train"][2:11]
        assert all(shard.is_open for shard in store.shards)

        with pytest.raises(DatasetContractError):
            store.read(-1, 1)
        with pytest.raises(DatasetContractError):
            store.read(11, 2)
    finally:
        store.close()

    validation = SplitTokenStore(manifest_path, "validation")
    try:
        assert validation.read(0, len(validation)).tolist() == streams["validation"]
        assert validation.read(0, 1).tolist() != streams["train"][0:1]
        with pytest.raises(DatasetContractError):
            validation.read(len(validation), 1)
    finally:
        validation.close()


def test_store_pickle_drops_open_memmaps_and_reopens_lazily(tmp_path):
    manifest_path, streams = _write_dataset_fixture(tmp_path)
    store = SplitTokenStore(manifest_path, "train")
    assert store.read(1, 8).tolist() == streams["train"][1:9]
    assert any(shard.is_open for shard in store.shards)

    restored = pickle.loads(pickle.dumps(store))
    try:
        assert not any(shard.is_open for shard in restored.shards)
        assert restored.read(1, 8).tolist() == streams["train"][1:9]
        assert any(shard.is_open for shard in restored.shards)
    finally:
        store.close()
        restored.close()


def test_causal_dataset_reads_t_plus_one_and_preserves_next_token_shift(tmp_path):
    manifest_path, streams = _write_dataset_fixture(tmp_path)
    store = SplitTokenStore(manifest_path, "train")
    try:
        dataset = CausalWindowDataset(store, 4, mode="all_starts")
        assert len(dataset) == 12 - 4
        assert dataset.possible_starts == 8

        first_x, first_y = dataset[0]
        assert first_x.tolist() == streams["train"][0:4]
        assert first_y.tolist() == streams["train"][1:5]
        assert first_x.shape == first_y.shape == (4,)
        assert first_x.dtype == first_y.dtype == torch.long
        assert torch.equal(first_y[:-1], first_x[1:])

        cross_x, cross_y = dataset[2]
        assert cross_x.tolist() == streams["train"][2:6]
        assert cross_y.tolist() == streams["train"][3:7]
        assert torch.equal(cross_y[:-1], cross_x[1:])

        last_x, last_y = dataset[len(dataset) - 1]
        assert last_x.tolist() == streams["train"][7:11]
        assert last_y.tolist() == streams["train"][8:12]

        with pytest.raises(IndexError):
            _ = dataset[len(dataset)]
        with pytest.raises(IndexError):
            _ = dataset[-1]
        with pytest.raises(DatasetContractError, match="at least 13"):
            CausalWindowDataset(store, 12, mode="all_starts")
    finally:
        store.close()


def test_sequential_dataset_uses_split_total_not_per_shard_rounding(tmp_path):
    manifest_path, streams = _write_dataset_fixture(tmp_path)
    store = SplitTokenStore(manifest_path, "train")
    try:
        dataset = CausalWindowDataset(store, 4, mode="sequential")
        assert len(dataset) == (len(streams["train"]) - 1) // 4 == 2
        assert dataset.evaluation_remainder == 3
        assert dataset.start_for_index(0) == 0
        assert dataset.start_for_index(1) == 4

        first_x, _first_y = dataset[0]
        second_x, _second_y = dataset[1]
        assert first_x.tolist() == streams["train"][0:4]
        assert second_x.tolist() == streams["train"][4:8]
    finally:
        store.close()


def test_epoch_sampler_is_reproducible_bounded_and_changes_by_epoch(tmp_path):
    manifest_path, _streams = _write_dataset_fixture(tmp_path)
    store = SplitTokenStore(manifest_path, "train")
    try:
        dataset = CausalWindowDataset(store, 4, mode="all_starts")
        first = EpochRandomWindowSampler(
            dataset,
            samples_per_epoch=200,
            base_seed=42,
            epoch=3,
            chunk_size=17,
        )
        second = EpochRandomWindowSampler(
            dataset,
            samples_per_epoch=200,
            base_seed=42,
            epoch=3,
            chunk_size=17,
        )
        first_sequence = list(first)
        second_sequence = list(second)
        assert len(first) == len(first_sequence) == 200
        assert first_sequence == second_sequence
        assert all(0 <= start < len(dataset) for start in first_sequence)

        second.set_epoch(4)
        assert list(second) != first_sequence
        with pytest.raises(DatasetContractError):
            EpochRandomWindowSampler(
                CausalWindowDataset(store, 4, mode="sequential"),
                samples_per_epoch=10,
                base_seed=42,
            )
    finally:
        store.close()


@pytest.mark.parametrize("num_workers", [0, 2])
def test_dataloader_batch_shape_shift_and_worker_file_release(
    tmp_path,
    num_workers,
):
    manifest_path, _streams = _write_dataset_fixture(tmp_path)
    store = SplitTokenStore(manifest_path, "train")
    dataset = CausalWindowDataset(store, 4, mode="all_starts")
    sampler = EpochRandomWindowSampler(
        dataset,
        samples_per_epoch=8,
        base_seed=42,
        epoch=0,
    )
    loader = build_dataloader(
        dataset,
        batch_size=4,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
        persistent_workers=False,
    )
    iterator = iter(loader)
    batch_x, batch_y = next(iterator)
    assert batch_x.shape == batch_y.shape == (4, 4)
    assert batch_x.dtype == batch_y.dtype == torch.long
    assert torch.equal(batch_y[:, :-1], batch_x[:, 1:])

    shutdown = getattr(iterator, "_shutdown_workers", None)
    if shutdown is not None:
        shutdown()
    del iterator, loader
    store.close()
    gc.collect()

    binary_path = store.shards[0].path
    probe_path = binary_path.with_suffix(".rename-probe")
    binary_path.rename(probe_path)
    probe_path.rename(binary_path)


def test_evaluation_dataloader_keeps_complete_dataset_tail_batch(tmp_path):
    manifest_path, _streams = _write_dataset_fixture(tmp_path)
    store = SplitTokenStore(manifest_path, "train")
    try:
        dataset = CausalWindowDataset(store, 4, mode="sequential")
        loader = build_dataloader(
            dataset,
            batch_size=4,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )
        batch_x, batch_y = next(iter(loader))
        assert batch_x.shape == batch_y.shape == (2, 4)
        assert torch.equal(batch_y[:, :-1], batch_x[:, 1:])
    finally:
        store.close()

