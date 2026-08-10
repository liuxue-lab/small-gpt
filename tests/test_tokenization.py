import copy
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from data_pipeline import (
    BuildContext,
    ControlledInterruption,
    ResumeStateError,
    TokenizationBuildError,
    build_tokenized_corpus,
    config_fingerprint,
    load_tokenized_data_config,
    map_token_payload,
    read_token_header,
    sha256_file,
    validate_completed_corpus,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG_PATH = PROJECT_ROOT / "configs" / "tokenized_data.yaml"
SPECIAL_IDS = {"bos": 0, "eos": 1, "pad": 2, "unk": 3}


class _Encoding:
    def __init__(self, token_ids):
        self.ids = token_ids


class DummyTokenizer:
    """Small deterministic tokenizer used only by offline temporary fixtures."""

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return _Encoding([100 + (ord(character) % 97) for character in text])


@dataclass(frozen=True)
class SyntheticBuild:
    root: Path
    config: dict
    context: BuildContext
    records: dict[str, list[dict]]

    def context_for(self, output_name: str) -> BuildContext:
        output = self.root / "data" / "tokenized" / output_name
        return replace(
            self.context,
            output_dir=output,
            staging_dir=output.with_name(f".{output.name}.inprogress"),
        )


def _json_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_source_record(path: Path, record: dict | None) -> None:
    if record is None:
        path.write_text("", encoding="utf-8")
    else:
        path.write_text(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def make_synthetic_build(tmp_path: Path) -> SyntheticBuild:
    root = tmp_path.resolve()
    corpus_dir = root / "data" / "processed" / "synthetic"
    shards_dir = corpus_dir / "shards"
    shards_dir.mkdir(parents=True)

    source_records = {
        "train": [
            {
                "id": "train-a",
                "text": "a" * 400,
                "provided_token_count": 10,
                "text_sha256": _json_sha256("a" * 400),
            },
            {
                "id": "train-b",
                "text": "b" * 400,
                "provided_token_count": 11,
                "text_sha256": _json_sha256("b" * 400),
            },
        ],
        "validation": [
            {
                "id": "validation-c",
                "text": "c" * 600,
                "provided_token_count": 12,
                "text_sha256": _json_sha256("c" * 600),
            }
        ],
        "test": [
            {
                "id": "test-d",
                "text": "d" * 700,
                "provided_token_count": 13,
                "text_sha256": _json_sha256("d" * 700),
            }
        ],
    }
    source_files: dict[str, list[Path]] = {
        split: [] for split in source_records
    }
    source_manifest_shards = []
    for shard_index in range(4):
        shard_dir = shards_dir / f"shard-{shard_index:05d}"
        shard_dir.mkdir()
        files = {}
        for split, records in source_records.items():
            path = shard_dir / f"{split}.jsonl"
            source_files[split].append(path)
            record = records[shard_index] if shard_index < len(records) else None
            _write_source_record(path, record)
            files[split] = {
                "path": path.relative_to(corpus_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "records": int(record is not None),
                "provided_tokens": (
                    int(record["provided_token_count"])
                    if record is not None
                    else 0
                ),
            }
        source_manifest_shards.append({"files": files})

    source_manifest = {
        "schema_version": 1,
        "status": "complete",
        "dataset": {
            "name": "synthetic",
            "configuration": "offline",
            "revision": "fixed",
        },
        "shards": source_manifest_shards,
    }
    source_manifest_path = corpus_dir / "manifest.json"
    source_manifest_path.write_text(
        json.dumps(source_manifest, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    artifacts_dir = root / "tokenizer" / "artifacts"
    artifacts_dir.mkdir(parents=True)
    tokenizer_path = artifacts_dir / "tokenizer.json"
    metadata_path = artifacts_dir / "tokenizer_config.json"
    tokenizer_path.write_bytes(b"synthetic-tokenizer-v1")
    metadata_path.write_bytes(b"synthetic-tokenizer-metadata-v1")

    config = copy.deepcopy(load_tokenized_data_config(BASE_CONFIG_PATH))
    config["source"].update(
        {
            "dataset": "synthetic",
            "configuration": "offline",
            "revision": "fixed",
            "corpus_dir": "data/processed/synthetic",
            "manifest": "data/processed/synthetic/manifest.json",
            "manifest_sha256": sha256_file(source_manifest_path),
        }
    )
    config["tokenizer"].update(
        {
            "file": "tokenizer/artifacts/tokenizer.json",
            "sha256": sha256_file(tokenizer_path),
            "metadata": "tokenizer/artifacts/tokenizer_config.json",
            "metadata_sha256": sha256_file(metadata_path),
        }
    )
    config["profiles"]["pilot"].update(
        {
            "output_dir": "data/tokenized/run-a",
            "staging_dir": "data/tokenized/.run-a.inprogress",
            "target_model_tokens_per_shard": 514,
        }
    )
    expected = {
        "train": {
            "records": 2,
            "provided_tokens": 21,
            "raw_bpe_tokens": 800,
            "appended_eos_tokens": 2,
            "model_tokens": 802,
            "unknown_tokens": 0,
        },
        "validation": {
            "records": 1,
            "provided_tokens": 12,
            "raw_bpe_tokens": 600,
            "appended_eos_tokens": 1,
            "model_tokens": 601,
            "unknown_tokens": 0,
        },
        "test": {
            "records": 1,
            "provided_tokens": 13,
            "raw_bpe_tokens": 700,
            "appended_eos_tokens": 1,
            "model_tokens": 701,
            "unknown_tokens": 0,
        },
    }
    total_fields = tuple(next(iter(expected.values())))
    totals = {
        field: sum(expected[split][field] for split in expected)
        for field in total_fields
    }
    totals["token_payload_bytes"] = totals["model_tokens"] * 2
    config["expected"] = {**expected, "totals": totals}

    output_dir = root / "data" / "tokenized" / "run-a"
    context = BuildContext(
        config=config,
        config_path=root / "configs" / "tokenized_data.yaml",
        project_root=root,
        profile="pilot",
        fingerprint=config_fingerprint(config),
        output_dir=output_dir,
        staging_dir=output_dir.with_name(".run-a.inprogress"),
        corpus_dir=corpus_dir,
        source_manifest_path=source_manifest_path,
        source_manifest=source_manifest,
        source_files=source_files,
        tokenizer_path=tokenizer_path,
        tokenizer_metadata_path=metadata_path,
        tokenizer=DummyTokenizer(),
        special_token_ids=SPECIAL_IDS,
    )
    return SyntheticBuild(root, config, context, source_records)


def _snapshot(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def test_config_fingerprint_ignores_order_and_runtime_paths(tmp_path):
    fixture = make_synthetic_build(tmp_path)
    first = config_fingerprint(fixture.config)

    reordered = json.loads(json.dumps(fixture.config, sort_keys=True))
    assert config_fingerprint(reordered) == first

    relocated = copy.deepcopy(fixture.config)
    relocated["profiles"]["pilot"]["output_dir"] = "data/tokenized/elsewhere"
    relocated["profiles"]["pilot"]["staging_dir"] = (
        "data/tokenized/.elsewhere.inprogress"
    )
    assert config_fingerprint(relocated) == first

    semantic_change = copy.deepcopy(fixture.config)
    semantic_change["profiles"]["pilot"]["target_model_tokens_per_shard"] = 515
    assert config_fingerprint(semantic_change) != first


def test_build_conserves_splits_and_keeps_oversize_document_atomic(tmp_path):
    fixture = make_synthetic_build(tmp_path)
    messages = []
    result = build_tokenized_corpus(fixture.context, progress=messages.append)

    assert result.already_complete is False
    assert result.manifest["totals"]["records"] == 4
    assert result.manifest["totals"]["raw_bpe_tokens"] == 2_100
    assert result.manifest["totals"]["appended_eos_tokens"] == 4
    assert result.manifest["totals"]["model_tokens"] == 2_104
    assert result.manifest["totals"]["token_payload_bytes"] == 4_208
    assert result.manifest["totals"]["unknown_tokens"] == 0
    assert result.manifest["totals"]["writer_inserted_bos_tokens"] == 0
    assert result.manifest["totals"]["writer_inserted_pad_tokens"] == 0
    assert result.manifest["totals"]["storage_dropped_tokens"] == 0
    assert [
        len(result.manifest["splits"][split]["shards"])
        for split in ("train", "validation", "test")
    ] == [2, 1, 1]
    assert result.manifest["splits"]["validation"]["shards"][0][
        "token_count"
    ] == 601
    assert result.manifest["splits"]["test"]["shards"][0][
        "token_count"
    ] == 701
    assert any("published:" in message for message in messages)
    assert not fixture.context.staging_dir.exists()
    assert not list(result.output_dir.rglob("*.part"))

    manifest_path = result.output_dir / "manifest.json"
    validate_completed_corpus(
        manifest_path,
        project_root=fixture.root,
        expected_fingerprint=fixture.context.fingerprint,
        verify_identities=True,
        scan_payload=True,
    )
    validation_binary = result.output_dir / result.manifest["splits"][
        "validation"
    ]["shards"][0]["binary"]["path"]
    validation_header = read_token_header(validation_binary)
    validation_tokens = map_token_payload(validation_binary, validation_header)
    try:
        assert validation_tokens.shape == (601,)
        assert np_unique(validation_tokens[:-1]) == {102}
        assert int(validation_tokens[-1]) == 1
    finally:
        validation_tokens._mmap.close()


def np_unique(values) -> set[int]:
    return {int(value) for value in values}


def test_interrupted_build_resumes_without_rewriting_checkpoint(tmp_path):
    fixture = make_synthetic_build(tmp_path)
    context = fixture.context

    with pytest.raises(ControlledInterruption):
        build_tokenized_corpus(context, interrupt_after_completed_shards=1)
    assert not context.output_dir.exists()
    assert context.staging_dir.is_dir()
    assert not (context.staging_dir / "manifest.json").exists()

    first_binary = context.staging_dir / "train" / "shard-00000.bin"
    checkpoint_hash = sha256_file(first_binary)
    (context.staging_dir / "train" / "shard-99999.bin.part").write_bytes(
        b"orphan part"
    )
    (context.staging_dir / "train" / "shard-00001.bin").write_bytes(
        b"orphan final"
    )

    with pytest.raises(ResumeStateError, match="--resume"):
        build_tokenized_corpus(context, resume=False)
    resumed = build_tokenized_corpus(context, resume=True)
    assert sha256_file(
        resumed.output_dir / "train" / "shard-00000.bin"
    ) == checkpoint_hash
    assert not list(resumed.output_dir.rglob("*.part"))

    one_shot_context = fixture.context_for("run-b")
    one_shot = build_tokenized_corpus(one_shot_context)
    assert resumed.manifest == one_shot.manifest
    assert _snapshot(resumed.output_dir) == _snapshot(one_shot.output_dir)


def test_resume_rejects_corrupted_completed_checkpoint(tmp_path):
    fixture = make_synthetic_build(tmp_path)
    context = fixture.context
    with pytest.raises(ControlledInterruption):
        build_tokenized_corpus(context, interrupt_after_completed_shards=1)

    completed_binary = context.staging_dir / "train" / "shard-00000.bin"
    with completed_binary.open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ResumeStateError, match="size mismatch"):
        build_tokenized_corpus(context, resume=True)
    assert not context.output_dir.exists()


def test_complete_output_is_verified_noop_and_rejects_other_identity(tmp_path):
    fixture = make_synthetic_build(tmp_path)
    first = build_tokenized_corpus(fixture.context)
    before = _snapshot(first.output_dir)

    second = build_tokenized_corpus(fixture.context)
    after = _snapshot(second.output_dir)
    assert second.already_complete is True
    assert second.manifest == first.manifest
    assert after == before

    different_identity = replace(fixture.context, fingerprint="0" * 64)
    with pytest.raises(
        TokenizationBuildError,
        match="different semantic configuration",
    ):
        build_tokenized_corpus(different_identity)


def test_source_text_hash_mismatch_reports_file_and_line(tmp_path):
    fixture = make_synthetic_build(tmp_path)
    source_path = fixture.context.source_files["train"][0]
    record = json.loads(source_path.read_text(encoding="utf-8"))
    record["text_sha256"] = "0" * 64
    _write_source_record(source_path, record)

    with pytest.raises(TokenizationBuildError) as captured:
        build_tokenized_corpus(fixture.context)
    message = str(captured.value)
    assert str(source_path) in message
    assert ":1" in message
    assert "text_sha256 mismatch" in message

