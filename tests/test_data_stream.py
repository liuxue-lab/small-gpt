from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from data_pipeline import TokenShardWriter
from torch import nn
from torch.utils.data import Dataset, SequentialSampler

from train import (
    DataStreamError,
    EvaluationDataStream,
    OffsetSampler,
    PrecisionPolicy,
    TrainerState,
    TrainingConfig,
    TrainingDataStream,
    ValidationDataStream,
    evaluate_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEBUG_PATH = PROJECT_ROOT / "configs" / "debug.yaml"
SPECIAL_IDS = {"bos": 0, "eos": 1, "pad": 2, "unk": 3}


def _document_hash(split: str, shard_index: int, document_index: int) -> bytes:
    label = f"stage-e:{split}:{shard_index}:{document_index}"
    return hashlib.sha256(label.encode("utf-8")).digest()


def write_tokenized_fixture(tmp_path: Path):
    root = tmp_path / "tokenized"
    layout = {
        "train": [
            [(10, 11, 12, 13, 14, 15, 16, 17, 18, 1)],
            [(20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 1)],
        ],
        "validation": [
            [
                (
                    40,
                    41,
                    42,
                    43,
                    44,
                    45,
                    46,
                    47,
                    48,
                    49,
                    50,
                    51,
                    52,
                    53,
                    54,
                    55,
                    56,
                    57,
                    58,
                    59,
                    1,
                )
            ]
        ],
        "test": [[(60, 61, 62, 63, 64, 65, 66, 67, 1)]],
    }
    streams: dict[str, list[int]] = {}
    splits: dict[str, dict] = {}
    for split, shard_documents in layout.items():
        global_token_start = 0
        global_document_start = 0
        shard_metadata = []
        stream: list[int] = []
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


def stream_config(*, seed: int = 1337) -> TrainingConfig:
    return replace(
        TrainingConfig.from_yaml(DEBUG_PATH),
        seed=seed,
        context_length=4,
        device="cpu",
        precision="fp32",
        micro_batch_size=2,
        gradient_accumulation_steps=1,
        max_steps=4,
        target_tokens=None,
        warmup_steps=1,
        warmup_ratio=None,
        num_workers=0,
        pin_memory=False,
    )


class RecordingDataset(Dataset[int]):
    def __init__(self, length: int):
        self.length = length
        self.requested: list[int] = []

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        self.requested.append(index)
        return index


class CursorIsolationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids, targets=None):
        assert targets is not None
        loss = self.anchor * 0.0 + targets.float().mean()
        return SimpleNamespace(logits=None, loss=loss)


def test_offset_sampler_skips_indexes_before_dataset_reads():
    dataset = RecordingDataset(10)
    sampler = OffsetSampler(SequentialSampler(dataset), offset=3)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        sampler=sampler,
    )

    values = torch.cat(list(loader)).tolist()

    assert len(sampler) == 7
    assert values == list(range(3, 10))
    assert dataset.requested == list(range(3, 10))


@pytest.mark.parametrize("offset", (-1, True, 1.5, 11))
def test_offset_sampler_rejects_invalid_offset(offset):
    dataset = RecordingDataset(10)

    with pytest.raises(DataStreamError, match="offset"):
        OffsetSampler(SequentialSampler(dataset), offset=offset)


def test_training_stream_is_deterministic_and_preserves_causal_shift(tmp_path):
    manifest_path, _ = write_tokenized_fixture(tmp_path)
    config = stream_config()
    plan = config.resolve()
    first_state = TrainerState(run_id="first")
    second_state = TrainerState(run_id="second")

    with TrainingDataStream(
        manifest_path,
        config=config,
        plan=plan,
        state=first_state,
    ) as first, TrainingDataStream(
        manifest_path,
        config=config,
        plan=plan,
        state=second_state,
    ) as second:
        first_batches = [next(first), next(first)]
        second_batches = [next(second), next(second)]

        assert len(first) == len(second) == 4
        assert first.yielded_micro_batches == 2
        assert first.micro_batches_remaining == 2
        for (first_x, first_y), (second_x, second_y) in zip(
            first_batches,
            second_batches,
            strict=True,
        ):
            assert torch.equal(first_x, second_x)
            assert torch.equal(first_y, second_y)
            assert first_x.shape == first_y.shape == (2, 4)
            assert first_x.dtype == first_y.dtype == torch.long
            assert torch.equal(first_y[:, :-1], first_x[:, 1:])

    assert first.is_closed is True
    assert second.is_closed is True
    assert not any(shard.is_open for shard in first.store.shards)
    assert not any(shard.is_open for shard in second.store.shards)


def test_resumed_stream_next_batch_matches_continuous_stream(tmp_path):
    manifest_path, _ = write_tokenized_fixture(tmp_path)
    config = stream_config()
    plan = config.resolve()

    with TrainingDataStream(
        manifest_path,
        config=config,
        plan=plan,
        state=TrainerState(run_id="continuous"),
    ) as continuous:
        _first_batch = next(continuous)
        expected_next = next(continuous)

    resumed_state = TrainerState(run_id="resumed")
    resumed_state.record_update(micro_steps=1, tokens=8, samples=2)
    with TrainingDataStream(
        manifest_path,
        config=config,
        plan=plan,
        state=resumed_state,
    ) as resumed:
        actual_next = next(resumed)

        assert resumed.initial_sample_offset == 2
        assert len(resumed.sampler) == 6
        assert torch.equal(actual_next[0], expected_next[0])
        assert torch.equal(actual_next[1], expected_next[1])


def test_different_seed_changes_random_training_windows(tmp_path):
    manifest_path, _ = write_tokenized_fixture(tmp_path)
    first_config = stream_config(seed=1)
    second_config = stream_config(seed=2)

    with TrainingDataStream(
        manifest_path,
        config=first_config,
        plan=first_config.resolve(),
        state=TrainerState(run_id="seed-1"),
    ) as first, TrainingDataStream(
        manifest_path,
        config=second_config,
        plan=second_config.resolve(),
        state=TrainerState(run_id="seed-2"),
    ) as second:
        first_batches = [next(first) for _ in range(4)]
        second_batches = [next(second) for _ in range(4)]

    assert any(
        not torch.equal(first_batch[0], second_batch[0])
        or not torch.equal(first_batch[1], second_batch[1])
        for first_batch, second_batch in zip(
            first_batches,
            second_batches,
            strict=True,
        )
    )


def test_validation_stream_is_repeatable_sequential_and_keeps_tail_batch(tmp_path):
    manifest_path, streams = write_tokenized_fixture(tmp_path)
    plan = stream_config().resolve()

    with ValidationDataStream(manifest_path, plan=plan) as validation:
        first_pass = list(validation)
        second_pass = list(validation)

        assert validation.dataset.mode == "sequential"
        assert [
            validation.dataset.start_for_index(index)
            for index in range(len(validation.dataset))
        ] == [0, 4, 8, 12, 16]
        assert len(validation) == 3
        assert first_pass[-1][0].shape == (1, 4)
        assert first_pass[0][0][0].tolist() == streams["validation"][0:4]
        for first_batch, second_batch in zip(
            first_pass,
            second_pass,
            strict=True,
        ):
            assert torch.equal(first_batch[0], second_batch[0])
            assert torch.equal(first_batch[1], second_batch[1])

    assert validation.is_closed is True
    assert not any(shard.is_open for shard in validation.store.shards)


@pytest.mark.parametrize(
    ("split", "expected_starts", "expected_batches", "last_batch_size"),
    (
        ("validation", [0, 4, 8, 12, 16], 3, 1),
        ("test", [0, 4], 1, 2),
    ),
)
def test_evaluation_stream_is_explicit_repeatable_and_reports_full_coverage(
    tmp_path,
    split,
    expected_starts,
    expected_batches,
    last_batch_size,
):
    manifest_path, streams = write_tokenized_fixture(tmp_path)
    plan = stream_config().resolve()

    with EvaluationDataStream(
        manifest_path,
        split=split,
        plan=plan,
    ) as evaluation:
        first_pass = list(evaluation)
        second_pass = list(evaluation)

        assert evaluation.split == split
        assert evaluation.store.split == split
        assert evaluation.dataset.mode == "sequential"
        assert [
            evaluation.dataset.start_for_index(index)
            for index in range(len(evaluation.dataset))
        ] == expected_starts
        assert evaluation.total_windows == len(expected_starts)
        assert evaluation.total_evaluation_tokens == len(expected_starts) * 4
        assert evaluation.discarded_tokens == (
            len(streams[split]) - 1
        ) % plan.context_length
        assert (
            evaluation.total_evaluation_tokens + evaluation.discarded_tokens
            == len(streams[split]) - 1
        )
        assert len(evaluation) == expected_batches
        assert first_pass[-1][0].shape == (last_batch_size, 4)
        assert first_pass[0][0][0].tolist() == streams[split][0:4]
        for first_batch, second_batch in zip(
            first_pass,
            second_pass,
            strict=True,
        ):
            assert torch.equal(first_batch[0], second_batch[0])
            assert torch.equal(first_batch[1], second_batch[1])

    assert evaluation.is_closed is True
    assert not any(shard.is_open for shard in evaluation.store.shards)


@pytest.mark.parametrize("split", ("train", "Validation", "", None, True))
def test_evaluation_stream_rejects_non_frozen_split(tmp_path, split):
    manifest_path, _ = write_tokenized_fixture(tmp_path)

    with pytest.raises(DataStreamError, match="evaluation split"):
        EvaluationDataStream(
            manifest_path,
            split=split,
            plan=stream_config().resolve(),
        )


def test_closed_test_evaluation_stream_cannot_be_reused(tmp_path):
    manifest_path, _ = write_tokenized_fixture(tmp_path)
    evaluation = EvaluationDataStream(
        manifest_path,
        split="test",
        plan=stream_config().resolve(),
    )

    evaluation.close()

    with pytest.raises(DataStreamError, match="test evaluation data stream is closed"):
        iter(evaluation)


def test_evaluation_does_not_change_the_next_training_batch(tmp_path):
    manifest_path, _ = write_tokenized_fixture(tmp_path)
    config = stream_config()
    plan = config.resolve()
    policy = PrecisionPolicy.from_config(config)

    with TrainingDataStream(
        manifest_path,
        config=config,
        plan=plan,
        state=TrainerState(run_id="continuous"),
    ) as continuous:
        _ = next(continuous)
        expected_next = next(continuous)

    with TrainingDataStream(
        manifest_path,
        config=config,
        plan=plan,
        state=TrainerState(run_id="with-eval"),
    ) as with_eval, ValidationDataStream(
        manifest_path,
        plan=plan,
    ) as validation:
        _ = next(with_eval)
        evaluate_model(
            CursorIsolationModel(),
            validation,
            precision=policy,
            global_step=0,
            max_batches=1,
        )
        actual_next = next(with_eval)

    assert torch.equal(actual_next[0], expected_next[0])
    assert torch.equal(actual_next[1], expected_next[1])


def test_train_and_validation_loaders_do_not_advance_global_torch_rng(tmp_path):
    manifest_path, _ = write_tokenized_fixture(tmp_path)
    config = stream_config()
    plan = config.resolve()
    torch.manual_seed(9876)
    before = torch.get_rng_state().clone()

    with TrainingDataStream(
        manifest_path,
        config=config,
        plan=plan,
        state=TrainerState(run_id="rng-isolation"),
    ) as training, ValidationDataStream(
        manifest_path,
        plan=plan,
    ) as validation:
        _training_batch = next(training)
        _validation_batch = next(iter(validation))

    assert torch.equal(torch.get_rng_state(), before)


def test_training_stream_closes_store_on_exception(tmp_path):
    manifest_path, _ = write_tokenized_fixture(tmp_path)
    config = stream_config()
    stream = None

    with pytest.raises(RuntimeError, match="controlled failure"):
        with TrainingDataStream(
            manifest_path,
            config=config,
            plan=config.resolve(),
            state=TrainerState(run_id="close-test"),
        ) as active_stream:
            stream = active_stream
            _ = next(active_stream)
            raise RuntimeError("controlled failure")

    assert stream is not None
    assert stream.is_closed is True
    assert not any(shard.is_open for shard in stream.store.shards)


def test_training_stream_rejects_inconsistent_batch_cursor(tmp_path):
    manifest_path, _ = write_tokenized_fixture(tmp_path)
    config = stream_config()
    state = TrainerState(
        run_id="bad-cursor",
        batches_consumed_in_epoch=1,
    )

    with pytest.raises(DataStreamError, match="batches_consumed_in_epoch"):
        TrainingDataStream(
            manifest_path,
            config=config,
            plan=config.resolve(),
            state=state,
        )


def test_training_stream_rejects_completed_sample_budget(tmp_path):
    manifest_path, _ = write_tokenized_fixture(tmp_path)
    config = stream_config()
    plan = config.resolve()
    state = TrainerState(
        run_id="complete",
        global_step=plan.total_updates,
        micro_steps_seen=plan.total_updates,
        tokens_seen=plan.planned_tokens,
        samples_consumed=plan.total_updates * plan.micro_batch_size,
        batches_consumed_in_epoch=plan.total_updates,
    )

    with pytest.raises(DataStreamError, match="no samples remaining"):
        TrainingDataStream(
            manifest_path,
            config=config,
            plan=plan,
            state=state,
        )
