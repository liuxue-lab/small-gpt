"""Binary token-shard and document-index formats for small-gpt.

The module is intentionally side-effect free: importing it defines the fixed
width structs and validation helpers, but never opens project data files.
"""

from __future__ import annotations

import hashlib
import os
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

import numpy as np


TOKEN_MAGIC = b"SGPTTOK1"
INDEX_MAGIC = b"SGPTIDX1"
SCHEMA_VERSION = 1

TOKEN_HEADER_STRUCT = struct.Struct("<8sHHBBHIB3xQQQIII4x")
INDEX_HEADER_STRUCT = struct.Struct("<8sHHHBBQQQQ32sQI36x")
INDEX_ENTRY_STRUCT = struct.Struct("<QQ32s")

TOKEN_HEADER_BYTES = TOKEN_HEADER_STRUCT.size
INDEX_HEADER_BYTES = INDEX_HEADER_STRUCT.size
INDEX_ENTRY_BYTES = INDEX_ENTRY_STRUCT.size

TOKEN_DTYPE = np.dtype("<u2")
TOKEN_DTYPE_CODE = 1
LITTLE_ENDIAN_CODE = 1
INDEX_OFFSET_SEMANTICS = 1
INDEX_LENGTH_SEMANTICS = 1

APPEND_EOS_FLAG = 1 << 0
ADD_BOS_FLAG = 1 << 1
ADD_PAD_FLAG = 1 << 2
KNOWN_TOKEN_FLAGS = APPEND_EOS_FLAG | ADD_BOS_FLAG | ADD_PAD_FLAG

SPLIT_CODES = {"train": 0, "validation": 1, "test": 2}
SPLIT_NAMES = {code: name for name, code in SPLIT_CODES.items()}


class TokenizedDataError(RuntimeError):
    """Base class for expected tokenized-data failures."""


class TokenizedDataConfigError(TokenizedDataError):
    """Raised when the tokenized-data configuration is invalid."""


class BinaryFormatError(TokenizedDataError):
    """Raised when a token shard violates the binary format contract."""


class IndexFormatError(TokenizedDataError):
    """Raised when a document index violates the index contract."""


class TokenizationBuildError(TokenizedDataError):
    """Raised when a tokenized corpus cannot be built safely."""


class ResumeStateError(TokenizedDataError):
    """Raised when a staged build cannot be resumed safely."""


class DatasetContractError(TokenizedDataError):
    """Raised when a Dataset request violates the logical-stream contract."""


@dataclass(frozen=True)
class TokenShardHeader:
    """Decoded 64-byte token-shard header."""

    schema_version: int
    header_bytes: int
    dtype_code: int
    endian_code: int
    flags: int
    vocab_size: int
    split_code: int
    token_count: int
    document_count: int
    payload_bytes: int
    eos_token_id: int
    minimum_token_id: int
    maximum_token_id: int

    @property
    def split(self) -> str:
        return SPLIT_NAMES[self.split_code]

    def to_dict(self) -> dict[str, int | str]:
        payload = asdict(self)
        payload["split"] = self.split
        return payload


@dataclass(frozen=True)
class IndexShardHeader:
    """Decoded 128-byte document-index header."""

    schema_version: int
    header_bytes: int
    record_bytes: int
    offset_semantics: int
    length_semantics: int
    token_count: int
    document_count: int
    global_token_start: int
    global_document_start: int
    binary_sha256: bytes
    entries_bytes: int
    eos_token_id: int

    def to_dict(self) -> dict[str, int | str]:
        payload: dict[str, int | str] = asdict(self)
        payload["binary_sha256"] = self.binary_sha256.hex()
        return payload


@dataclass(frozen=True)
class DocumentIndexEntry:
    """One fixed-width document entry in a shard-local index."""

    start_token: int
    length_tokens: int
    text_sha256: bytes

    @property
    def end_token(self) -> int:
        return self.start_token + self.length_tokens

    def to_dict(self) -> dict[str, int | str]:
        return {
            "start_token": self.start_token,
            "length_tokens": self.length_tokens,
            "text_sha256": self.text_sha256.hex(),
        }


def _require_nonnegative_int(value: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _require_positive_int(value: int, field: str) -> int:
    if _require_nonnegative_int(value, field) == 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _validate_token_header_values(header: TokenShardHeader) -> None:
    if header.schema_version != SCHEMA_VERSION:
        raise BinaryFormatError(
            f"token schema is {header.schema_version}; expected {SCHEMA_VERSION}"
        )
    if header.header_bytes != TOKEN_HEADER_BYTES:
        raise BinaryFormatError(
            f"token header is {header.header_bytes} bytes; "
            f"expected {TOKEN_HEADER_BYTES}"
        )
    if header.dtype_code != TOKEN_DTYPE_CODE:
        raise BinaryFormatError(
            f"token dtype code is {header.dtype_code}; expected {TOKEN_DTYPE_CODE}"
        )
    if header.endian_code != LITTLE_ENDIAN_CODE:
        raise BinaryFormatError(
            f"token endian code is {header.endian_code}; "
            f"expected {LITTLE_ENDIAN_CODE} (little-endian)"
        )
    if header.flags & ~KNOWN_TOKEN_FLAGS:
        raise BinaryFormatError(f"token header contains unknown flags: {header.flags}")
    if header.flags != APPEND_EOS_FLAG:
        raise BinaryFormatError(
            "token flags must encode append_eos=true, add_bos=false, add_pad=false"
        )
    if header.vocab_size <= 0 or header.vocab_size > 65_536:
        raise BinaryFormatError(f"invalid vocabulary size: {header.vocab_size}")
    if header.split_code not in SPLIT_NAMES:
        raise BinaryFormatError(f"unknown split code: {header.split_code}")
    if header.token_count <= 0:
        raise BinaryFormatError("token shard must contain at least one token")
    if header.document_count <= 0:
        raise BinaryFormatError("token shard must contain at least one document")
    expected_payload_bytes = header.token_count * TOKEN_DTYPE.itemsize
    if header.payload_bytes != expected_payload_bytes:
        raise BinaryFormatError(
            f"payload_bytes is {header.payload_bytes}; expected {expected_payload_bytes}"
        )
    if not 0 <= header.eos_token_id < header.vocab_size:
        raise BinaryFormatError(f"EOS token ID is out of range: {header.eos_token_id}")
    if not (
        0
        <= header.minimum_token_id
        <= header.maximum_token_id
        < header.vocab_size
    ):
        raise BinaryFormatError(
            "token min/max do not satisfy "
            f"0 <= min <= max < {header.vocab_size}: "
            f"{header.minimum_token_id}, {header.maximum_token_id}"
        )


def pack_token_header(header: TokenShardHeader) -> bytes:
    """Pack a validated token-shard header."""

    _validate_token_header_values(header)
    return TOKEN_HEADER_STRUCT.pack(
        TOKEN_MAGIC,
        header.schema_version,
        header.header_bytes,
        header.dtype_code,
        header.endian_code,
        header.flags,
        header.vocab_size,
        header.split_code,
        header.token_count,
        header.document_count,
        header.payload_bytes,
        header.eos_token_id,
        header.minimum_token_id,
        header.maximum_token_id,
    )


def unpack_token_header(raw: bytes, *, source: str = "token header") -> TokenShardHeader:
    """Decode and validate exactly one token-shard header."""

    if len(raw) != TOKEN_HEADER_BYTES:
        raise BinaryFormatError(
            f"{source} has {len(raw)} header bytes; expected {TOKEN_HEADER_BYTES}"
        )
    if raw[:8] != TOKEN_MAGIC:
        raise BinaryFormatError(
            f"invalid token magic in {source}: {raw[:8]!r}; expected {TOKEN_MAGIC!r}"
        )
    if raw[21:24] != b"\x00" * 3 or raw[60:64] != b"\x00" * 4:
        raise BinaryFormatError(f"reserved token-header bytes are non-zero in {source}")

    unpacked = TOKEN_HEADER_STRUCT.unpack(raw)
    header = TokenShardHeader(*unpacked[1:])
    _validate_token_header_values(header)
    return header


def read_token_header(path: str | Path) -> TokenShardHeader:
    shard_path = Path(path)
    try:
        with shard_path.open("rb") as handle:
            raw = handle.read(TOKEN_HEADER_BYTES)
    except OSError as exc:
        raise BinaryFormatError(f"cannot read token shard {shard_path}: {exc}") from exc
    return unpack_token_header(raw, source=str(shard_path))


def _validate_index_header_values(header: IndexShardHeader) -> None:
    if header.schema_version != SCHEMA_VERSION:
        raise IndexFormatError(
            f"index schema is {header.schema_version}; expected {SCHEMA_VERSION}"
        )
    if header.header_bytes != INDEX_HEADER_BYTES:
        raise IndexFormatError(
            f"index header is {header.header_bytes} bytes; "
            f"expected {INDEX_HEADER_BYTES}"
        )
    if header.record_bytes != INDEX_ENTRY_BYTES:
        raise IndexFormatError(
            f"index record is {header.record_bytes} bytes; "
            f"expected {INDEX_ENTRY_BYTES}"
        )
    if header.offset_semantics != INDEX_OFFSET_SEMANTICS:
        raise IndexFormatError(
            f"unsupported index offset semantics: {header.offset_semantics}"
        )
    if header.length_semantics != INDEX_LENGTH_SEMANTICS:
        raise IndexFormatError(
            f"unsupported index length semantics: {header.length_semantics}"
        )
    if header.token_count <= 0:
        raise IndexFormatError("index token_count must be positive")
    if header.document_count <= 0:
        raise IndexFormatError("index document_count must be positive")
    if len(header.binary_sha256) != 32:
        raise IndexFormatError("binary_sha256 must contain exactly 32 bytes")
    if header.entries_bytes != header.document_count * INDEX_ENTRY_BYTES:
        raise IndexFormatError(
            f"entries_bytes is {header.entries_bytes}; expected "
            f"{header.document_count * INDEX_ENTRY_BYTES}"
        )
    if header.eos_token_id < 0:
        raise IndexFormatError("EOS token ID must be non-negative")


def pack_index_header(header: IndexShardHeader) -> bytes:
    """Pack a validated document-index header."""

    _validate_index_header_values(header)
    return INDEX_HEADER_STRUCT.pack(
        INDEX_MAGIC,
        header.schema_version,
        header.header_bytes,
        header.record_bytes,
        header.offset_semantics,
        header.length_semantics,
        header.token_count,
        header.document_count,
        header.global_token_start,
        header.global_document_start,
        header.binary_sha256,
        header.entries_bytes,
        header.eos_token_id,
    )


def unpack_index_header(raw: bytes, *, source: str = "index header") -> IndexShardHeader:
    """Decode and validate exactly one document-index header."""

    if len(raw) != INDEX_HEADER_BYTES:
        raise IndexFormatError(
            f"{source} has {len(raw)} header bytes; expected {INDEX_HEADER_BYTES}"
        )
    if raw[:8] != INDEX_MAGIC:
        raise IndexFormatError(
            f"invalid index magic in {source}: {raw[:8]!r}; expected {INDEX_MAGIC!r}"
        )
    if raw[92:128] != b"\x00" * 36:
        raise IndexFormatError(f"reserved index-header bytes are non-zero in {source}")

    unpacked = INDEX_HEADER_STRUCT.unpack(raw)
    header = IndexShardHeader(*unpacked[1:])
    _validate_index_header_values(header)
    return header


def read_index_header(path: str | Path) -> IndexShardHeader:
    index_path = Path(path)
    try:
        with index_path.open("rb") as handle:
            raw = handle.read(INDEX_HEADER_BYTES)
    except OSError as exc:
        raise IndexFormatError(f"cannot read index shard {index_path}: {exc}") from exc
    return unpack_index_header(raw, source=str(index_path))


def pack_index_entry(entry: DocumentIndexEntry) -> bytes:
    _require_nonnegative_int(entry.start_token, "start_token")
    _require_positive_int(entry.length_tokens, "length_tokens")
    if len(entry.text_sha256) != 32:
        raise ValueError("text_sha256 must contain exactly 32 raw bytes")
    return INDEX_ENTRY_STRUCT.pack(
        entry.start_token,
        entry.length_tokens,
        entry.text_sha256,
    )


def unpack_index_entry(raw: bytes, *, source: str = "index entry") -> DocumentIndexEntry:
    if len(raw) != INDEX_ENTRY_BYTES:
        raise IndexFormatError(
            f"{source} has {len(raw)} bytes; expected {INDEX_ENTRY_BYTES}"
        )
    entry = DocumentIndexEntry(*INDEX_ENTRY_STRUCT.unpack(raw))
    if entry.length_tokens <= 0:
        raise IndexFormatError(f"document length must be positive in {source}")
    return entry


def iter_index_entries(
    path: str | Path,
    header: IndexShardHeader | None = None,
) -> Iterator[DocumentIndexEntry]:
    """Stream index entries without retaining the whole index in memory."""

    index_path = Path(path)
    decoded_header = header or read_index_header(index_path)
    try:
        with index_path.open("rb") as handle:
            handle.seek(INDEX_HEADER_BYTES)
            for entry_index in range(decoded_header.document_count):
                raw = handle.read(INDEX_ENTRY_BYTES)
                yield unpack_index_entry(
                    raw,
                    source=f"{index_path} entry {entry_index}",
                )
            if handle.read(1):
                raise IndexFormatError(f"unexpected trailing bytes in {index_path}")
    except OSError as exc:
        raise IndexFormatError(f"cannot read index entries from {index_path}: {exc}") from exc


def sha256_file(path: str | Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    """Return the lowercase SHA-256 hex digest of a file using bounded memory."""

    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    digest = hashlib.sha256()
    file_path = Path(path)
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flush_and_fsync(handle: BinaryIO) -> None:
    """Flush Python and operating-system buffers for an open binary file."""

    handle.flush()
    os.fsync(handle.fileno())


def map_token_payload(
    path: str | Path,
    header: TokenShardHeader | None = None,
) -> np.memmap:
    """Memory-map only the little-endian uint16 payload of a token shard."""

    shard_path = Path(path)
    decoded_header = header or read_token_header(shard_path)
    expected_size = TOKEN_HEADER_BYTES + decoded_header.payload_bytes
    try:
        actual_size = shard_path.stat().st_size
    except OSError as exc:
        raise BinaryFormatError(f"cannot stat token shard {shard_path}: {exc}") from exc
    if actual_size != expected_size:
        raise BinaryFormatError(
            f"token shard size mismatch for {shard_path}: "
            f"found {actual_size}, expected {expected_size}"
        )
    return np.memmap(
        shard_path,
        mode="r",
        dtype=TOKEN_DTYPE,
        offset=TOKEN_HEADER_BYTES,
        shape=(decoded_header.token_count,),
    )


def close_memmap(mapped: np.memmap | None) -> None:
    """Release a memmap promptly so Windows can rename/delete its file."""

    if mapped is None:
        return
    mmap_handle = getattr(mapped, "_mmap", None)
    if mmap_handle is not None:
        mmap_handle.close()


def validate_token_shard(
    path: str | Path,
    *,
    expected_split: str | None = None,
    expected_vocab_size: int | None = None,
    expected_eos_token_id: int | None = None,
    scan_payload: bool = True,
    scan_chunk_tokens: int = 1024 * 1024,
) -> TokenShardHeader:
    """Validate header, size, and optionally the complete token-ID range."""

    shard_path = Path(path)
    header = read_token_header(shard_path)
    expected_size = TOKEN_HEADER_BYTES + header.payload_bytes
    actual_size = shard_path.stat().st_size
    if actual_size != expected_size:
        payload_size = max(0, actual_size - TOKEN_HEADER_BYTES)
        suffix = " (odd payload byte count)" if payload_size % 2 else ""
        raise BinaryFormatError(
            f"token shard size mismatch for {shard_path}: found {actual_size}, "
            f"expected {expected_size}{suffix}"
        )
    if expected_split is not None:
        if expected_split not in SPLIT_CODES:
            raise ValueError(f"unknown expected split: {expected_split}")
        if header.split_code != SPLIT_CODES[expected_split]:
            raise BinaryFormatError(
                f"split mismatch for {shard_path}: header={header.split}, "
                f"expected={expected_split}"
            )
    if expected_vocab_size is not None and header.vocab_size != expected_vocab_size:
        raise BinaryFormatError(
            f"vocabulary mismatch for {shard_path}: header={header.vocab_size}, "
            f"expected={expected_vocab_size}"
        )
    if (
        expected_eos_token_id is not None
        and header.eos_token_id != expected_eos_token_id
    ):
        raise BinaryFormatError(
            f"EOS mismatch for {shard_path}: header={header.eos_token_id}, "
            f"expected={expected_eos_token_id}"
        )

    if scan_payload:
        if scan_chunk_tokens <= 0:
            raise ValueError("scan_chunk_tokens must be positive")
        mapped = map_token_payload(shard_path, header)
        try:
            observed_min: int | None = None
            observed_max: int | None = None
            for start in range(0, header.token_count, scan_chunk_tokens):
                chunk = mapped[start : start + scan_chunk_tokens]
                local_min = int(chunk.min())
                local_max = int(chunk.max())
                observed_min = (
                    local_min if observed_min is None else min(observed_min, local_min)
                )
                observed_max = (
                    local_max if observed_max is None else max(observed_max, local_max)
                )
            if observed_min != header.minimum_token_id:
                raise BinaryFormatError(
                    f"minimum token ID mismatch for {shard_path}: "
                    f"header={header.minimum_token_id}, observed={observed_min}"
                )
            if observed_max != header.maximum_token_id:
                raise BinaryFormatError(
                    f"maximum token ID mismatch for {shard_path}: "
                    f"header={header.maximum_token_id}, observed={observed_max}"
                )
            if observed_max is not None and observed_max >= header.vocab_size:
                raise BinaryFormatError(
                    f"out-of-range token ID {observed_max} in {shard_path}"
                )
        finally:
            close_memmap(mapped)
    return header


def validate_index_shard(
    index_path: str | Path,
    binary_path: str | Path,
    *,
    expected_global_token_start: int | None = None,
    expected_global_document_start: int | None = None,
    expected_binary_sha256: str | None = None,
    validate_document_eos: bool = True,
) -> IndexShardHeader:
    """Validate an index and its cryptographic/structural binding to a token shard."""

    idx_path = Path(index_path)
    bin_path = Path(binary_path)
    token_header = read_token_header(bin_path)
    index_header = read_index_header(idx_path)

    expected_size = INDEX_HEADER_BYTES + index_header.entries_bytes
    actual_size = idx_path.stat().st_size
    if actual_size != expected_size:
        raise IndexFormatError(
            f"index size mismatch for {idx_path}: found {actual_size}, "
            f"expected {expected_size}"
        )
    if index_header.token_count != token_header.token_count:
        raise IndexFormatError(
            f"token_count mismatch between {idx_path} and {bin_path}"
        )
    if index_header.document_count != token_header.document_count:
        raise IndexFormatError(
            f"document_count mismatch between {idx_path} and {bin_path}"
        )
    if index_header.eos_token_id != token_header.eos_token_id:
        raise IndexFormatError(f"EOS token ID mismatch between {idx_path} and {bin_path}")

    actual_binary_sha256 = sha256_file(bin_path)
    if index_header.binary_sha256.hex() != actual_binary_sha256:
        raise IndexFormatError(
            f"binary SHA-256 binding mismatch in {idx_path}: "
            f"header={index_header.binary_sha256.hex()}, actual={actual_binary_sha256}"
        )
    if expected_binary_sha256 is not None:
        normalized_expected = expected_binary_sha256.lower()
        if actual_binary_sha256 != normalized_expected:
            raise IndexFormatError(
                f"binary SHA-256 mismatch for {bin_path}: "
                f"found {actual_binary_sha256}, expected {normalized_expected}"
            )
    if (
        expected_global_token_start is not None
        and index_header.global_token_start != expected_global_token_start
    ):
        raise IndexFormatError(
            f"global token start mismatch for {idx_path}: "
            f"header={index_header.global_token_start}, "
            f"expected={expected_global_token_start}"
        )
    if (
        expected_global_document_start is not None
        and index_header.global_document_start != expected_global_document_start
    ):
        raise IndexFormatError(
            f"global document start mismatch for {idx_path}: "
            f"header={index_header.global_document_start}, "
            f"expected={expected_global_document_start}"
        )

    mapped = map_token_payload(bin_path, token_header) if validate_document_eos else None
    try:
        expected_start = 0
        entries_seen = 0
        for entries_seen, entry in enumerate(
            iter_index_entries(idx_path, index_header),
            start=1,
        ):
            if entry.start_token != expected_start:
                relation = "overlap" if entry.start_token < expected_start else "gap"
                raise IndexFormatError(
                    f"index {relation} in {idx_path} at entry {entries_seen - 1}: "
                    f"found start {entry.start_token}, expected {expected_start}"
                )
            if entry.end_token > index_header.token_count:
                raise IndexFormatError(
                    f"document entry {entries_seen - 1} exceeds token shard in {idx_path}"
                )
            if (
                mapped is not None
                and int(mapped[entry.end_token - 1]) != index_header.eos_token_id
            ):
                raise IndexFormatError(
                    f"document entry {entries_seen - 1} does not end in EOS in {idx_path}"
                )
            expected_start = entry.end_token
        if entries_seen != index_header.document_count:
            raise IndexFormatError(
                f"index entry count mismatch for {idx_path}: "
                f"read {entries_seen}, expected {index_header.document_count}"
            )
        if expected_start != index_header.token_count:
            raise IndexFormatError(
                f"index does not cover token shard {idx_path}: "
                f"covered {expected_start}, expected {index_header.token_count}"
            )
    finally:
        close_memmap(mapped)
    return index_header
