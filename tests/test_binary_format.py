import hashlib
import struct
from pathlib import Path

import numpy as np
import pytest

from data_pipeline import (
    INDEX_ENTRY_BYTES,
    INDEX_ENTRY_STRUCT,
    INDEX_HEADER_BYTES,
    INDEX_HEADER_STRUCT,
    TOKEN_HEADER_BYTES,
    TOKEN_HEADER_STRUCT,
    BinaryFormatError,
    DocumentIndexEntry,
    IndexFormatError,
    IndexShardHeader,
    TokenShardHeader,
    TokenShardWriter,
    iter_index_entries,
    map_token_payload,
    pack_index_entry,
    pack_index_header,
    pack_token_header,
    read_index_header,
    read_token_header,
    sha256_file,
    unpack_index_entry,
    unpack_index_header,
    unpack_token_header,
    validate_index_shard,
    validate_token_shard,
)


SPECIAL_IDS = {"bos": 0, "eos": 1, "pad": 2, "unk": 3}


def _digest(label: str) -> bytes:
    return hashlib.sha256(label.encode("utf-8")).digest()


def _write_pair(
    tmp_path: Path,
    *,
    split: str = "train",
    documents: tuple[tuple[int, ...], ...] = ((10, 11, 1), (12, 1)),
):
    root = tmp_path / "tokenized"
    writer = TokenShardWriter(
        staging_root=root,
        split=split,
        shard_index=0,
        global_token_start=0,
        global_document_start=0,
        vocab_size=16_384,
        special_token_ids=SPECIAL_IDS,
    )
    for index, token_ids in enumerate(documents):
        writer.append_document(
            token_ids,
            text_sha256=_digest(f"document-{index}"),
            provided_tokens=index + 5,
        )
    metadata = writer.finalize()
    return root, metadata, writer.binary_path, writer.index_path


def _copy_and_patch(
    source: Path,
    destination: Path,
    *,
    offset: int,
    replacement: bytes,
) -> Path:
    destination.write_bytes(source.read_bytes())
    with destination.open("r+b") as handle:
        handle.seek(offset)
        handle.write(replacement)
    return destination


def test_struct_sizes_and_pack_unpack_round_trip():
    assert TOKEN_HEADER_STRUCT.size == TOKEN_HEADER_BYTES == 64
    assert INDEX_HEADER_STRUCT.size == INDEX_HEADER_BYTES == 128
    assert INDEX_ENTRY_STRUCT.size == INDEX_ENTRY_BYTES == 48

    token_header = TokenShardHeader(
        schema_version=1,
        header_bytes=64,
        dtype_code=1,
        endian_code=1,
        flags=1,
        vocab_size=16_384,
        split_code=0,
        token_count=5,
        document_count=2,
        payload_bytes=10,
        eos_token_id=1,
        minimum_token_id=1,
        maximum_token_id=12,
    )
    packed_token = pack_token_header(token_header)
    assert len(packed_token) == 64
    assert packed_token[21:24] == b"\x00" * 3
    assert packed_token[60:64] == b"\x00" * 4
    assert unpack_token_header(packed_token) == token_header

    binary_hash = hashlib.sha256(b"binary").digest()
    index_header = IndexShardHeader(
        schema_version=1,
        header_bytes=128,
        record_bytes=48,
        offset_semantics=1,
        length_semantics=1,
        token_count=5,
        document_count=2,
        global_token_start=100,
        global_document_start=20,
        binary_sha256=binary_hash,
        entries_bytes=96,
        eos_token_id=1,
    )
    packed_index = pack_index_header(index_header)
    assert len(packed_index) == 128
    assert packed_index[92:128] == b"\x00" * 36
    assert unpack_index_header(packed_index) == index_header

    entry = DocumentIndexEntry(3, 7, _digest("entry"))
    assert unpack_index_entry(pack_index_entry(entry)) == entry


def test_writer_produces_valid_little_endian_payload_and_gapless_index(tmp_path):
    _root, metadata, binary_path, index_path = _write_pair(tmp_path)

    token_header = validate_token_shard(
        binary_path,
        expected_split="train",
        expected_vocab_size=16_384,
        expected_eos_token_id=1,
        scan_payload=True,
    )
    index_header = validate_index_shard(
        index_path,
        binary_path,
        expected_global_token_start=0,
        expected_global_document_start=0,
        expected_binary_sha256=metadata["binary"]["sha256"],
    )
    assert read_token_header(binary_path) == token_header
    assert read_index_header(index_path) == index_header
    assert binary_path.stat().st_size == 64 + 5 * 2
    assert index_path.stat().st_size == 128 + 2 * 48
    assert token_header.token_count == 5
    assert token_header.document_count == 2
    assert token_header.minimum_token_id == 1
    assert token_header.maximum_token_id == 12
    assert index_header.binary_sha256.hex() == sha256_file(binary_path)

    mapped = map_token_payload(binary_path, token_header)
    try:
        assert mapped.dtype == np.dtype("<u2")
        assert mapped.tolist() == [10, 11, 1, 12, 1]
    finally:
        mapped._mmap.close()

    entries = list(iter_index_entries(index_path, index_header))
    assert [(entry.start_token, entry.length_tokens) for entry in entries] == [
        (0, 3),
        (3, 2),
    ]
    assert entries[0].text_sha256 == _digest("document-0")
    assert entries[1].text_sha256 == _digest("document-1")


@pytest.mark.parametrize(
    ("name", "offset", "replacement"),
    [
        ("magic", 0, b"BROKEN!!"),
        ("schema", 8, struct.pack("<H", 2)),
        ("dtype", 12, b"\x02"),
        ("endian", 13, b"\x02"),
        ("flags", 14, struct.pack("<H", 3)),
        ("split", 20, b"\x09"),
        ("reserved-middle", 21, b"X"),
        ("reserved-tail", 60, b"X"),
    ],
)
def test_token_header_rejects_invalid_identity_fields(
    tmp_path,
    name,
    offset,
    replacement,
):
    _root, _metadata, binary_path, _index_path = _write_pair(tmp_path)
    corrupted = _copy_and_patch(
        binary_path,
        tmp_path / f"{name}.bin",
        offset=offset,
        replacement=replacement,
    )
    with pytest.raises(BinaryFormatError):
        validate_token_shard(corrupted)


def test_token_shard_rejects_truncation_odd_payload_and_range_corruption(tmp_path):
    _root, _metadata, binary_path, _index_path = _write_pair(tmp_path)

    truncated = tmp_path / "truncated.bin"
    truncated.write_bytes(binary_path.read_bytes()[:-2])
    with pytest.raises(BinaryFormatError, match="size mismatch"):
        validate_token_shard(truncated)

    odd_payload = tmp_path / "odd.bin"
    odd_payload.write_bytes(binary_path.read_bytes()[:-1])
    with pytest.raises(BinaryFormatError, match="odd payload byte count"):
        validate_token_shard(odd_payload)

    wrong_count = _copy_and_patch(
        binary_path,
        tmp_path / "wrong-count.bin",
        offset=24,
        replacement=struct.pack("<Q", 6),
    )
    with pytest.raises(BinaryFormatError):
        validate_token_shard(wrong_count)

    out_of_range = _copy_and_patch(
        binary_path,
        tmp_path / "out-of-range.bin",
        offset=64,
        replacement=struct.pack("<H", 16_384),
    )
    with pytest.raises(BinaryFormatError):
        validate_token_shard(out_of_range, scan_payload=True)

    with pytest.raises(BinaryFormatError, match="split mismatch"):
        validate_token_shard(binary_path, expected_split="validation")


def test_index_rejects_gap_overlap_zero_length_and_hash_corruption(tmp_path):
    _root, _metadata, binary_path, index_path = _write_pair(tmp_path)

    gap = _copy_and_patch(
        index_path,
        tmp_path / "gap.idx",
        offset=128,
        replacement=struct.pack("<Q", 1),
    )
    with pytest.raises(IndexFormatError, match="gap"):
        validate_index_shard(gap, binary_path)

    overlap = _copy_and_patch(
        index_path,
        tmp_path / "overlap.idx",
        offset=128 + 48,
        replacement=struct.pack("<Q", 2),
    )
    with pytest.raises(IndexFormatError, match="overlap"):
        validate_index_shard(overlap, binary_path)

    zero_length = _copy_and_patch(
        index_path,
        tmp_path / "zero-length.idx",
        offset=128 + 8,
        replacement=struct.pack("<Q", 0),
    )
    with pytest.raises(IndexFormatError, match="positive"):
        validate_index_shard(zero_length, binary_path)

    wrong_hash = _copy_and_patch(
        index_path,
        tmp_path / "wrong-hash.idx",
        offset=48,
        replacement=b"\x00" * 32,
    )
    with pytest.raises(IndexFormatError, match="binding mismatch"):
        validate_index_shard(wrong_hash, binary_path)

    truncated = tmp_path / "truncated.idx"
    truncated.write_bytes(index_path.read_bytes()[:-1])
    with pytest.raises(IndexFormatError, match="size mismatch"):
        validate_index_shard(truncated, binary_path)


def test_index_rejects_document_that_does_not_end_in_eos(tmp_path):
    _root, _metadata, binary_path, index_path = _write_pair(tmp_path)
    corrupted_binary = tmp_path / "bad-eos.bin"
    corrupted_binary.write_bytes(binary_path.read_bytes())
    with corrupted_binary.open("r+b") as handle:
        handle.seek(64 + 4 * 2)
        handle.write(struct.pack("<H", 9))

    rebound_index = tmp_path / "bad-eos.idx"
    rebound_index.write_bytes(index_path.read_bytes())
    with rebound_index.open("r+b") as handle:
        handle.seek(48)
        handle.write(bytes.fromhex(sha256_file(corrupted_binary)))

    with pytest.raises(IndexFormatError, match="does not end in EOS"):
        validate_index_shard(rebound_index, corrupted_binary)

