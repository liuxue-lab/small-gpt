"""Deterministic, resumable corpus tokenization for small-gpt."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import tokenizers as tokenizers_library
import yaml
from tokenizers import Tokenizer

from tokenizer.bpe import (
    CorpusFormatError,
    atomic_write_json,
    discover_split_files,
    encode_document,
    iter_jsonl_records,
    load_source_manifest,
    load_tokenizer,
    project_relative_path,
)

from .binary_format import (
    APPEND_EOS_FLAG,
    INDEX_ENTRY_BYTES,
    INDEX_HEADER_BYTES,
    INDEX_LENGTH_SEMANTICS,
    INDEX_MAGIC,
    INDEX_OFFSET_SEMANTICS,
    LITTLE_ENDIAN_CODE,
    SCHEMA_VERSION,
    SPLIT_CODES,
    TOKEN_DTYPE,
    TOKEN_DTYPE_CODE,
    TOKEN_HEADER_BYTES,
    TOKEN_MAGIC,
    DocumentIndexEntry,
    IndexShardHeader,
    ResumeStateError,
    TokenizationBuildError,
    TokenizedDataConfigError,
    TokenShardHeader,
    flush_and_fsync,
    pack_index_entry,
    pack_index_header,
    pack_token_header,
    sha256_file,
    validate_index_shard,
    validate_token_shard,
)


FORMAT_NAME = "small_gpt_tokenized_corpus"
ALLOWED_SPLITS = ("train", "validation", "test")
ALLOWED_PROFILE = "pilot"
ALLOWED_PROFILES = ("pilot", "full")
FROZEN_SOURCE_IDENTITY = "frozen_manifest"
CAPTURED_SOURCE_IDENTITY = "capture_complete_manifest"
OBSERVED_STATISTICS_MODE = "source_manifest_observed"
DAY4_STATS_PATH = "reports/day-04-tokenizer-stats.json"
SHARD_FILE_RE = re.compile(r"shard-(\d{5})\.(bin|idx)(\.part)?")


class ControlledInterruption(TokenizationBuildError):
    """Test hook used to exercise shard-level recovery deterministically."""


@dataclass(frozen=True)
class SourceRecord:
    split: str
    path: Path
    line_number: int
    record_number: int
    record: dict[str, Any]
    text_sha256: bytes


@dataclass(frozen=True)
class BuildContext:
    config: dict[str, Any]
    config_path: Path
    project_root: Path
    profile: str
    fingerprint: str
    output_dir: Path
    staging_dir: Path
    corpus_dir: Path
    source_manifest_path: Path
    source_manifest: dict[str, Any]
    source_files: dict[str, list[Path]]
    tokenizer_path: Path
    tokenizer_metadata_path: Path
    tokenizer: Tokenizer
    special_token_ids: dict[str, int]


@dataclass(frozen=True)
class BuildResult:
    manifest: dict[str, Any]
    output_dir: Path
    already_complete: bool


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TokenizedDataConfigError(f"{field} must be a mapping")
    return value


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TokenizedDataConfigError(f"{field} must be a non-empty string")
    return value


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TokenizedDataConfigError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TokenizedDataConfigError(f"{field} must be a non-negative integer")
    return value


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise TokenizedDataConfigError(f"{field} must be boolean")
    return value


def _sha256_string(value: Any, field: str) -> str:
    digest = _nonempty_string(value, field).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise TokenizedDataConfigError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def _source_file_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise TokenizedDataConfigError(f"{field} must be a non-empty list")
    normalized: list[str] = []
    for index, raw_path in enumerate(value):
        source_file = _nonempty_string(raw_path, f"{field}[{index}]")
        path = PurePosixPath(source_file)
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != source_file
            or path.suffix != ".parquet"
        ):
            raise TokenizedDataConfigError(
                f"{field}[{index}] must be a normalized relative Parquet path"
            )
        normalized.append(source_file)
    if len(normalized) != len(set(normalized)):
        raise TokenizedDataConfigError(f"{field} must not contain duplicates")
    return normalized


def _require_equal(actual: Any, expected: Any, field: str) -> None:
    if actual != expected:
        raise TokenizedDataConfigError(
            f"{field} must be {expected!r}; found {actual!r}"
        )


def load_tokenized_data_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a side-effect-free tokenized-data YAML contract."""

    config_path = Path(path)
    if not config_path.is_file():
        raise TokenizedDataConfigError(
            f"tokenized-data config does not exist: {config_path}"
        )
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise TokenizedDataConfigError(
            f"cannot read tokenized-data config {config_path}: {exc}"
        ) from exc
    config = dict(_mapping(loaded, "config"))
    validate_tokenized_data_config(config)
    return config


def validate_tokenized_data_config(config: Mapping[str, Any]) -> None:
    """Validate the frozen Pilot or manifest-bound Full data contract."""

    _require_equal(config.get("schema_version"), SCHEMA_VERSION, "schema_version")
    _require_equal(config.get("format_name"), FORMAT_NAME, "format_name")

    profiles = _mapping(config.get("profiles"), "profiles")
    if len(profiles) != 1 or not set(profiles).issubset(ALLOWED_PROFILES):
        raise TokenizedDataConfigError(
            "profiles must contain exactly one executable profile: pilot or full"
        )
    configured_profile = next(iter(profiles))

    source = _mapping(config.get("source"), "source")
    for field in (
        "dataset",
        "configuration",
        "revision",
        "corpus_dir",
        "manifest",
        "file_pattern",
        "deterministic_order",
    ):
        _nonempty_string(source.get(field), f"source.{field}")
    identity_mode = source.get("identity_mode", FROZEN_SOURCE_IDENTITY)
    if configured_profile == ALLOWED_PROFILE:
        _require_equal(
            identity_mode,
            FROZEN_SOURCE_IDENTITY,
            "source.identity_mode",
        )
        _sha256_string(source.get("manifest_sha256"), "source.manifest_sha256")
        _positive_int(
            source.get("expected_source_shards"),
            "source.expected_source_shards",
        )
    else:
        _require_equal(
            identity_mode,
            CAPTURED_SOURCE_IDENTITY,
            "source.identity_mode",
        )
        has_manifest_sha = "manifest_sha256" in source
        has_shard_count = "expected_source_shards" in source
        if has_manifest_sha != has_shard_count:
            raise TokenizedDataConfigError(
                "Full source.manifest_sha256 and expected_source_shards must "
                "either both be absent or both be resolved"
            )
        if has_manifest_sha:
            _sha256_string(source.get("manifest_sha256"), "source.manifest_sha256")
            _positive_int(
                source.get("expected_source_shards"),
                "source.expected_source_shards",
            )
        _sha256_string(
            source.get("expected_config_fingerprint"),
            "source.expected_config_fingerprint",
        )
        _source_file_list(
            source.get("expected_source_files"),
            "source.expected_source_files",
        )
        expected_profile = _mapping(
            source.get("expected_profile"),
            "source.expected_profile",
        )
        _require_equal(
            expected_profile.get("name"),
            "full",
            "source.expected_profile.name",
        )
        target_provided_tokens = _positive_int(
            expected_profile.get("target_provided_tokens"),
            "source.expected_profile.target_provided_tokens",
        )
        shard_target_provided_tokens = _positive_int(
            expected_profile.get("shard_target_provided_tokens"),
            "source.expected_profile.shard_target_provided_tokens",
        )
        if shard_target_provided_tokens > target_provided_tokens:
            raise TokenizedDataConfigError(
                "source.expected_profile.shard_target_provided_tokens must not "
                "exceed target_provided_tokens"
            )
        estimated_shards = _positive_int(
            expected_profile.get("estimated_shards"),
            "source.expected_profile.estimated_shards",
        )
        calculated_shards = (
            target_provided_tokens + shard_target_provided_tokens - 1
        ) // shard_target_provided_tokens
        if estimated_shards != calculated_shards:
            raise TokenizedDataConfigError(
                "source.expected_profile.estimated_shards does not match "
                "the Full budgets"
            )
    _require_equal(
        source.get("split_order"),
        list(ALLOWED_SPLITS),
        "source.split_order",
    )
    _require_equal(
        source.get("file_pattern"),
        "shards/shard-*/{split}.jsonl",
        "source.file_pattern",
    )
    _require_equal(
        source.get("deterministic_order"),
        "source_shard_then_line",
        "source.deterministic_order",
    )
    _require_equal(source.get("validate_text_sha256"), True, "source.validate_text_sha256")

    tokenizer = _mapping(config.get("tokenizer"), "tokenizer")
    for field in ("file", "metadata", "normalizer"):
        _nonempty_string(tokenizer.get(field), f"tokenizer.{field}")
    _sha256_string(tokenizer.get("sha256"), "tokenizer.sha256")
    _sha256_string(tokenizer.get("metadata_sha256"), "tokenizer.metadata_sha256")
    library = _mapping(tokenizer.get("library"), "tokenizer.library")
    _require_equal(library.get("name"), "tokenizers", "tokenizer.library.name")
    _nonempty_string(library.get("version"), "tokenizer.library.version")
    vocab_size = _positive_int(tokenizer.get("vocab_size"), "tokenizer.vocab_size")
    if vocab_size != 16_384:
        raise TokenizedDataConfigError("tokenizer.vocab_size must be 16384")
    _require_equal(tokenizer.get("normalizer"), "nfc", "tokenizer.normalizer")
    special_tokens = _mapping(tokenizer.get("special_tokens"), "tokenizer.special_tokens")
    expected_special = {
        "bos": {"token": "<bos>", "id": 0},
        "eos": {"token": "<eos>", "id": 1},
        "pad": {"token": "<pad>", "id": 2},
        "unk": {"token": "<unk>", "id": 3},
    }
    if dict(special_tokens) != expected_special:
        raise TokenizedDataConfigError(
            "tokenizer.special_tokens must freeze <bos>=0, <eos>=1, "
            "<pad>=2, <unk>=3"
        )

    encoding = _mapping(config.get("encoding"), "encoding")
    expected_encoding = {
        "append_eos_per_document": True,
        "add_bos": False,
        "add_pad": False,
        "reject_empty_document": True,
        "allow_cross_document_windows": True,
        "allow_cross_storage_shard_windows": True,
        "allow_cross_split_windows": False,
    }
    for field, expected in expected_encoding.items():
        _require_equal(encoding.get(field), expected, f"encoding.{field}")

    binary = _mapping(config.get("binary_format"), "binary_format")
    expected_binary = {
        "schema_version": SCHEMA_VERSION,
        "magic": TOKEN_MAGIC.decode("ascii"),
        "header_bytes": TOKEN_HEADER_BYTES,
        "struct_format": "<8sHHBBHIB3xQQQIII4x",
        "dtype": "<u2",
        "dtype_code": TOKEN_DTYPE_CODE,
        "endian": "little",
        "endian_code": LITTLE_ENDIAN_CODE,
    }
    for field, expected in expected_binary.items():
        _require_equal(binary.get(field), expected, f"binary_format.{field}")
    flags = _mapping(binary.get("flags"), "binary_format.flags")
    _require_equal(
        dict(flags),
        {"append_eos_bit": 0, "add_bos_bit": 1, "add_pad_bit": 2},
        "binary_format.flags",
    )
    split_codes = _mapping(binary.get("split_codes"), "binary_format.split_codes")
    _require_equal(dict(split_codes), SPLIT_CODES, "binary_format.split_codes")

    index = _mapping(config.get("index_format"), "index_format")
    expected_index = {
        "schema_version": SCHEMA_VERSION,
        "magic": INDEX_MAGIC.decode("ascii"),
        "header_bytes": INDEX_HEADER_BYTES,
        "header_struct_format": "<8sHHHBBQQQQ32sQI36x",
        "record_bytes": INDEX_ENTRY_BYTES,
        "record_struct_format": "<QQ32s",
        "offset_dtype": "<u8",
        "offset_semantics": "shard_local_token_offset",
        "length_dtype": "<u8",
        "length_includes_eos": True,
        "document_identity": "text_sha256_raw_32_bytes",
    }
    for field, expected in expected_index.items():
        _require_equal(index.get(field), expected, f"index_format.{field}")

    publication = _mapping(config.get("publication"), "publication")
    expected_publication = {
        "manifest_file": "manifest.json",
        "state_file": "state.json",
        "part_suffix": ".part",
        "completed_output_behavior": "identity_match_noop",
        "overwrite_complete_output": False,
        "atomic_publish": True,
        "fsync_files": True,
    }
    for field, expected in expected_publication.items():
        _require_equal(publication.get(field), expected, f"publication.{field}")

    selected_profile = _mapping(
        profiles[configured_profile],
        f"profiles.{configured_profile}",
    )
    _nonempty_string(
        selected_profile.get("output_dir"),
        f"profiles.{configured_profile}.output_dir",
    )
    _nonempty_string(
        selected_profile.get("staging_dir"),
        f"profiles.{configured_profile}.staging_dir",
    )
    shard_target = _positive_int(
        selected_profile.get("target_model_tokens_per_shard"),
        f"profiles.{configured_profile}.target_model_tokens_per_shard",
    )
    if shard_target <= 513:
        raise TokenizedDataConfigError(
            f"profiles.{configured_profile}.target_model_tokens_per_shard "
            "must exceed 513"
        )
    _require_equal(
        selected_profile.get("document_atomic"),
        True,
        f"profiles.{configured_profile}.document_atomic",
    )
    _require_equal(
        selected_profile.get("resume_enabled"),
        True,
        f"profiles.{configured_profile}.resume_enabled",
    )

    if configured_profile == ALLOWED_PROFILE:
        expected = _mapping(config.get("expected"), "expected")
        expected_fields = (
            "records",
            "provided_tokens",
            "raw_bpe_tokens",
            "appended_eos_tokens",
            "model_tokens",
            "unknown_tokens",
        )
        summed = Counter()
        for split in ALLOWED_SPLITS:
            split_expected = _mapping(expected.get(split), f"expected.{split}")
            for field in expected_fields:
                value = _nonnegative_int(
                    split_expected.get(field),
                    f"expected.{split}.{field}",
                )
                summed[field] += value
            if split_expected["records"] <= 0:
                raise TokenizedDataConfigError(
                    f"expected.{split}.records must be positive"
                )
            if split_expected["records"] != split_expected["appended_eos_tokens"]:
                raise TokenizedDataConfigError(
                    f"expected.{split}: records must equal appended_eos_tokens"
                )
            if (
                split_expected["raw_bpe_tokens"]
                + split_expected["appended_eos_tokens"]
                != split_expected["model_tokens"]
            ):
                raise TokenizedDataConfigError(
                    f"expected.{split}: raw BPE + EOS must equal model tokens"
                )
        totals = _mapping(expected.get("totals"), "expected.totals")
        for field in expected_fields:
            total_value = _nonnegative_int(
                totals.get(field),
                f"expected.totals.{field}",
            )
            if total_value != summed[field]:
                raise TokenizedDataConfigError(
                    f"expected.totals.{field} is {total_value}; "
                    f"split sum is {summed[field]}"
                )
        payload_bytes = _nonnegative_int(
            totals.get("token_payload_bytes"),
            "expected.totals.token_payload_bytes",
        )
        if payload_bytes != totals["model_tokens"] * TOKEN_DTYPE.itemsize:
            raise TokenizedDataConfigError(
                "expected.totals.token_payload_bytes must equal model_tokens * 2"
            )
        if "statistics" in config:
            raise TokenizedDataConfigError(
                "Pilot config must use frozen expected statistics"
            )
    else:
        if "expected" in config:
            raise TokenizedDataConfigError(
                "Full config must not freeze token-output statistics"
            )
        statistics = _mapping(config.get("statistics"), "statistics")
        _require_equal(
            statistics.get("mode"),
            OBSERVED_STATISTICS_MODE,
            "statistics.mode",
        )
        _require_equal(
            statistics.get("require_zero_unknown_tokens"),
            True,
            "statistics.require_zero_unknown_tokens",
        )

    dataset = _mapping(config.get("dataset"), "dataset")
    _require_equal(dataset.get("storage_view"), "split_logical_stream", "dataset.storage_view")
    _require_equal(dataset.get("lazy_memmap"), True, "dataset.lazy_memmap")
    _require_equal(dataset.get("output_torch_dtype"), "long", "dataset.output_torch_dtype")
    train = _mapping(dataset.get("train"), "dataset.train")
    _require_equal(train.get("dataset_mode"), "all_starts", "dataset.train.dataset_mode")
    _require_equal(
        train.get("sampler"),
        "epoch_random_window_with_replacement",
        "dataset.train.sampler",
    )
    _nonnegative_int(train.get("base_seed"), "dataset.train.base_seed")
    _require_equal(train.get("shuffle"), False, "dataset.train.shuffle")
    _require_equal(train.get("drop_last_batch"), True, "dataset.train.drop_last_batch")
    evaluation = _mapping(dataset.get("evaluation"), "dataset.evaluation")
    _require_equal(
        evaluation.get("dataset_mode"),
        "sequential_non_overlapping",
        "dataset.evaluation.dataset_mode",
    )
    _require_equal(evaluation.get("start_token"), 0, "dataset.evaluation.start_token")
    _require_equal(evaluation.get("shuffle"), False, "dataset.evaluation.shuffle")
    _require_equal(
        evaluation.get("drop_last_batch"),
        False,
        "dataset.evaluation.drop_last_batch",
    )
    dataloader = _mapping(dataset.get("dataloader"), "dataset.dataloader")
    _nonnegative_int(
        dataloader.get("default_num_workers"),
        "dataset.dataloader.default_num_workers",
    )
    _require_equal(
        dataloader.get("windows_validation_num_workers"),
        [0, 2],
        "dataset.dataloader.windows_validation_num_workers",
    )
    _require_equal(
        dataloader.get("default_pin_memory"),
        False,
        "dataset.dataloader.default_pin_memory",
    )

    fingerprint = _mapping(config.get("fingerprint"), "fingerprint")
    _require_equal(fingerprint.get("algorithm"), "sha256", "fingerprint.algorithm")
    _require_equal(
        fingerprint.get("canonical_json_sort_keys"),
        True,
        "fingerprint.canonical_json_sort_keys",
    )
    expected_exclusions = [
        f"profiles.{configured_profile}.output_dir",
        f"profiles.{configured_profile}.staging_dir",
        "cli.resume",
        "cli.no_progress",
    ]
    _require_equal(
        fingerprint.get("exclude_runtime_fields"),
        expected_exclusions,
        "fingerprint.exclude_runtime_fields",
    )


def _delete_nested_field(payload: dict[str, Any], dotted_path: str) -> None:
    parts = dotted_path.split(".")
    current: Any = payload
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            return
        current = current[part]
    if isinstance(current, dict):
        current.pop(parts[-1], None)


def config_fingerprint(config: Mapping[str, Any]) -> str:
    """Hash only fields that can change data content or read semantics."""

    validate_tokenized_data_config(config)
    source = config["source"]
    if (
        source.get("identity_mode") == CAPTURED_SOURCE_IDENTITY
        and "manifest_sha256" not in source
    ):
        raise TokenizedDataConfigError(
            "captured Full source identity must be resolved before fingerprinting"
        )
    semantic = copy.deepcopy(dict(config))
    excluded = semantic["fingerprint"]["exclude_runtime_fields"]
    for dotted_path in excluded:
        _delete_nested_field(semantic, str(dotted_path))
    canonical = json.dumps(
        semantic,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _resolve_project_path(project_root: Path, configured_path: str | Path) -> Path:
    path = Path(configured_path)
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TokenizationBuildError(f"cannot read valid {label} from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TokenizationBuildError(f"{label} must be a JSON object: {path}")
    return payload


def _verify_file_identity(path: Path, expected_sha256: str, label: str) -> None:
    if not path.is_file():
        raise TokenizationBuildError(f"{label} does not exist: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise TokenizationBuildError(
            f"{label} SHA-256 mismatch for {path}: found {actual}, expected {expected_sha256}"
        )


def _verify_tokenizer_metadata_identity(path: Path, expected_sha256: str) -> None:
    """Verify metadata across Git's platform-dependent text line endings.

    The frozen Day 4 metadata digest was recorded from a CRLF working tree. Git
    materializes the same tracked JSON with LF on Linux, so first preserve strict
    raw-byte verification and then allow only an LF/CRLF normalization fallback.
    All non-newline content remains covered by SHA-256.
    """

    if not path.is_file():
        raise TokenizationBuildError(f"tokenizer metadata does not exist: {path}")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise TokenizationBuildError(
            f"cannot read tokenizer metadata from {path}: {exc}"
        ) from exc

    raw_sha256 = hashlib.sha256(payload).hexdigest()
    if raw_sha256 == expected_sha256:
        return

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TokenizationBuildError(
            f"tokenizer metadata is not valid UTF-8: {path}: {exc}"
        ) from exc
    normalized_lf = text.replace("\r\n", "\n").replace("\r", "\n")
    canonical_lf = normalized_lf.encode("utf-8")
    canonical_crlf = normalized_lf.replace("\n", "\r\n").encode("utf-8")
    canonical_lf_sha256 = hashlib.sha256(canonical_lf).hexdigest()
    canonical_crlf_sha256 = hashlib.sha256(canonical_crlf).hexdigest()
    if expected_sha256 not in (canonical_lf_sha256, canonical_crlf_sha256):
        raise TokenizationBuildError(
            f"tokenizer metadata SHA-256 mismatch for {path}: found raw "
            f"{raw_sha256}, canonical LF {canonical_lf_sha256}, and canonical "
            f"CRLF {canonical_crlf_sha256}; expected {expected_sha256}"
        )


def _validate_output_paths(
    project_root: Path,
    output_dir: Path,
    staging_dir: Path,
    corpus_dir: Path,
    tokenizer_path: Path,
) -> None:
    tokenized_root = (project_root / "data" / "tokenized").resolve()
    for path, label in ((output_dir, "output"), (staging_dir, "staging")):
        if path == tokenized_root or not _is_relative_to(path, tokenized_root):
            raise TokenizedDataConfigError(
                f"{label} directory must be a child of {tokenized_root}: {path}"
            )
    if output_dir == staging_dir:
        raise TokenizedDataConfigError("output and staging directories must differ")
    protected = (corpus_dir.resolve(), tokenizer_path.parent.resolve())
    for protected_path in protected:
        if _is_relative_to(output_dir, protected_path) or _is_relative_to(
            protected_path, output_dir
        ):
            raise TokenizedDataConfigError(
                f"output directory overlaps protected input path: {protected_path}"
            )
        if _is_relative_to(staging_dir, protected_path) or _is_relative_to(
            protected_path, staging_dir
        ):
            raise TokenizedDataConfigError(
                f"staging directory overlaps protected input path: {protected_path}"
            )


def _validate_source_files(
    *,
    corpus_dir: Path,
    manifest: Mapping[str, Any],
    source_files: Mapping[str, Sequence[Path]],
) -> None:
    manifest_files: dict[str, Mapping[str, Any]] = {}
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        raise TokenizationBuildError("source manifest shards must be a list")
    for shard_index, raw_shard in enumerate(shards):
        shard = _mapping(raw_shard, f"source manifest shards[{shard_index}]")
        files = _mapping(shard.get("files"), f"source manifest shards[{shard_index}].files")
        for split in ALLOWED_SPLITS:
            info = _mapping(files.get(split), f"source shard {shard_index} {split}")
            relative = _nonempty_string(
                info.get("path"),
                f"source shard {shard_index} {split}.path",
            )
            if relative in manifest_files:
                raise TokenizationBuildError(f"duplicate source file in manifest: {relative}")
            manifest_files[relative] = info

    discovered_relative: set[str] = set()
    for split in ALLOWED_SPLITS:
        for path in source_files[split]:
            relative = path.resolve().relative_to(corpus_dir.resolve()).as_posix()
            discovered_relative.add(relative)
            if relative not in manifest_files:
                raise TokenizationBuildError(
                    f"source file is missing from manifest: {relative}"
                )
            info = manifest_files[relative]
            expected_size = _nonnegative_int(info.get("bytes"), f"{relative}.bytes")
            if path.stat().st_size != expected_size:
                raise TokenizationBuildError(
                    f"source file size mismatch for {path}: found {path.stat().st_size}, "
                    f"expected {expected_size}"
                )
            expected_hash = _sha256_string(info.get("sha256"), f"{relative}.sha256")
            actual_hash = sha256_file(path)
            if actual_hash != expected_hash:
                raise TokenizationBuildError(
                    f"source file SHA-256 mismatch for {path}: "
                    f"found {actual_hash}, expected {expected_hash}"
                )
    if discovered_relative != set(manifest_files):
        missing = sorted(set(manifest_files) - discovered_relative)
        raise TokenizationBuildError(
            f"source manifest references undiscovered files: {missing}"
        )


def _validate_tokenizer_metadata(
    config: Mapping[str, Any],
    metadata: Mapping[str, Any],
    tokenizer: Tokenizer,
) -> dict[str, int]:
    configured = config["tokenizer"]
    if metadata.get("schema_version") != 1:
        raise TokenizationBuildError("tokenizer metadata schema_version must be 1")
    if metadata.get("library") != configured["library"]:
        raise TokenizationBuildError("tokenizer metadata library identity mismatch")
    token_meta = _mapping(metadata.get("tokenizer"), "tokenizer metadata.tokenizer")
    if token_meta.get("vocab_size") != configured["vocab_size"]:
        raise TokenizationBuildError("tokenizer metadata vocabulary size mismatch")
    if token_meta.get("normalizer") != configured["normalizer"]:
        raise TokenizationBuildError("tokenizer metadata normalizer mismatch")
    boundaries = _mapping(
        metadata.get("document_boundaries"),
        "tokenizer metadata.document_boundaries",
    )
    if (
        boundaries.get("append_eos_per_document") is not True
        or boundaries.get("add_bos") is not False
    ):
        raise TokenizationBuildError("tokenizer document-boundary metadata mismatch")

    actual_vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    if actual_vocab_size != configured["vocab_size"]:
        raise TokenizationBuildError(
            f"tokenizer vocabulary is {actual_vocab_size}; expected {configured['vocab_size']}"
        )
    actual_ids: dict[str, int] = {}
    for name, item in configured["special_tokens"].items():
        actual_id = tokenizer.token_to_id(item["token"])
        if actual_id != item["id"]:
            raise TokenizationBuildError(
                f"special token {item['token']} has ID {actual_id}; expected {item['id']}"
            )
        actual_ids[name] = int(actual_id)
    expected_metadata_special = [
        {"name": name, "token": item["token"], "id": item["id"]}
        for name, item in configured["special_tokens"].items()
    ]
    if token_meta.get("special_tokens") != expected_metadata_special:
        raise TokenizationBuildError("tokenizer metadata special-token ordering mismatch")
    return actual_ids


def _validate_day4_statistics(config: Mapping[str, Any], project_root: Path) -> None:
    stats_path = project_root / DAY4_STATS_PATH
    stats = _load_json_object(stats_path, "Day 4 tokenizer statistics")
    source = stats.get("source")
    tokenizer = stats.get("tokenizer")
    if not isinstance(source, Mapping) or not isinstance(tokenizer, Mapping):
        raise TokenizationBuildError("Day 4 statistics are missing source/tokenizer identity")
    if source.get("manifest_sha256") != config["source"]["manifest_sha256"]:
        raise TokenizationBuildError("Day 4 source manifest identity does not match Day 5 config")
    if tokenizer.get("sha256") != config["tokenizer"]["sha256"]:
        raise TokenizationBuildError("Day 4 tokenizer identity does not match Day 5 config")
    for split in ALLOWED_SPLITS:
        actual = _mapping(stats.get("splits", {}).get(split), f"Day 4 stats.{split}")
        expected = config["expected"][split]
        mapping = {
            "records": "records",
            "provided_tokens": "provided_tokens",
            "raw_bpe_tokens": "bpe_tokens_without_eos",
            "appended_eos_tokens": "eos_tokens",
            "model_tokens": "total_model_tokens",
            "unknown_tokens": "unknown_tokens",
        }
        for expected_key, stats_key in mapping.items():
            if int(actual.get(stats_key, -1)) != int(expected[expected_key]):
                raise TokenizationBuildError(
                    f"Day 4 {split}.{stats_key} does not match Day 5 expected.{expected_key}"
                )


def _source_split_counts(
    statistics: Mapping[str, Any],
    field: str,
) -> dict[str, int]:
    values = _mapping(statistics.get(field), f"source manifest statistics.{field}")
    if set(values) != set(ALLOWED_SPLITS):
        raise TokenizationBuildError(
            f"source manifest statistics.{field} must contain "
            f"{list(ALLOWED_SPLITS)}"
        )
    return {
        split: _nonnegative_int(
            values[split],
            f"source manifest statistics.{field}.{split}",
        )
        for split in ALLOWED_SPLITS
    }


def _validate_source_manifest_identity(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    profile: str,
) -> None:
    """Bind a source manifest to its dataset and, for Full, its build contract."""

    dataset = _mapping(manifest.get("dataset"), "source manifest.dataset")
    identity_mapping = {
        "name": config["source"]["dataset"],
        "configuration": config["source"]["configuration"],
        "revision": config["source"]["revision"],
    }
    for field, expected in identity_mapping.items():
        if dataset.get(field) != expected:
            raise TokenizationBuildError(
                f"source manifest dataset.{field} is {dataset.get(field)!r}; "
                f"expected {expected!r}"
            )

    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise TokenizationBuildError("source manifest shards must be a non-empty list")
    if len(shards) != int(config["source"]["expected_source_shards"]):
        raise TokenizationBuildError(
            f"source manifest has {len(shards)} shards; expected "
            f"{config['source']['expected_source_shards']}"
        )

    if profile == ALLOWED_PROFILE:
        return

    expected_source_files = list(config["source"]["expected_source_files"])
    if dataset.get("source_files") != expected_source_files:
        raise TokenizationBuildError(
            "Full source manifest file list does not match the frozen source files"
        )
    expected_fingerprint = config["source"]["expected_config_fingerprint"]
    if manifest.get("config_fingerprint") != expected_fingerprint:
        raise TokenizationBuildError(
            "Full source manifest configuration fingerprint mismatch"
        )
    manifest_profile = _mapping(manifest.get("profile"), "source manifest.profile")
    expected_profile = config["source"]["expected_profile"]
    if dict(manifest_profile) != dict(expected_profile):
        raise TokenizationBuildError(
            "Full source manifest profile does not match the frozen Full budget"
        )

    aggregate_records = Counter({split: 0 for split in ALLOWED_SPLITS})
    aggregate_tokens = Counter({split: 0 for split in ALLOWED_SPLITS})
    aggregate_removals: Counter[str] = Counter()
    previous_source_end = 0
    for shard_index, raw_shard in enumerate(shards):
        shard = _mapping(raw_shard, f"source manifest.shards[{shard_index}]")
        if shard.get("schema_version") != SCHEMA_VERSION:
            raise TokenizationBuildError(
                f"source shard {shard_index} schema version mismatch"
            )
        if shard.get("config_fingerprint") != expected_fingerprint:
            raise TokenizationBuildError(
                f"source shard {shard_index} configuration fingerprint mismatch"
            )
        if shard.get("shard_index") != shard_index:
            raise TokenizationBuildError("source shard indexes are not contiguous")

        source_start = _nonnegative_int(
            shard.get("source_records_start_exclusive"),
            f"source shard {shard_index}.source_records_start_exclusive",
        )
        source_end = _positive_int(
            shard.get("source_records_end_inclusive"),
            f"source shard {shard_index}.source_records_end_inclusive",
        )
        if source_start != previous_source_end or source_end <= source_start:
            raise TokenizationBuildError(
                f"source shard {shard_index} record range is not contiguous"
            )
        input_records = _positive_int(
            shard.get("input_records"),
            f"source shard {shard_index}.input_records",
        )
        if input_records != source_end - source_start:
            raise TokenizationBuildError(
                f"source shard {shard_index} input-record count does not conserve"
            )

        files = _mapping(shard.get("files"), f"source shard {shard_index}.files")
        split_records = _mapping(
            shard.get("split_records"),
            f"source shard {shard_index}.split_records",
        )
        split_tokens = _mapping(
            shard.get("split_provided_tokens"),
            f"source shard {shard_index}.split_provided_tokens",
        )
        if (
            set(files) != set(ALLOWED_SPLITS)
            or set(split_records) != set(ALLOWED_SPLITS)
            or set(split_tokens) != set(ALLOWED_SPLITS)
        ):
            raise TokenizationBuildError(
                f"source shard {shard_index} split set is invalid"
            )

        shard_records = 0
        shard_tokens = 0
        for split in ALLOWED_SPLITS:
            info = _mapping(
                files[split],
                f"source shard {shard_index}.files.{split}",
            )
            expected_path = (
                f"shards/shard-{shard_index:05d}/{split}.jsonl"
            )
            if info.get("path") != expected_path:
                raise TokenizationBuildError(
                    f"source shard {shard_index} {split} path is invalid"
                )
            file_records = _nonnegative_int(
                info.get("records"),
                f"source shard {shard_index}.files.{split}.records",
            )
            file_tokens = _nonnegative_int(
                info.get("provided_tokens"),
                f"source shard {shard_index}.files.{split}.provided_tokens",
            )
            declared_records = _nonnegative_int(
                split_records[split],
                f"source shard {shard_index}.split_records.{split}",
            )
            declared_tokens = _nonnegative_int(
                split_tokens[split],
                f"source shard {shard_index}.split_provided_tokens.{split}",
            )
            if file_records != declared_records or file_tokens != declared_tokens:
                raise TokenizationBuildError(
                    f"source shard {shard_index} {split} file statistics mismatch"
                )
            shard_records += file_records
            shard_tokens += file_tokens
            aggregate_records[split] += file_records
            aggregate_tokens[split] += file_tokens

        if shard_records != _nonnegative_int(
            shard.get("kept_records"),
            f"source shard {shard_index}.kept_records",
        ):
            raise TokenizationBuildError(
                f"source shard {shard_index} kept-record count does not conserve"
            )
        if shard_tokens != _nonnegative_int(
            shard.get("kept_provided_tokens"),
            f"source shard {shard_index}.kept_provided_tokens",
        ):
            raise TokenizationBuildError(
                f"source shard {shard_index} provided-token count does not conserve"
            )
        removals = _mapping(
            shard.get("removal_counts"),
            f"source shard {shard_index}.removal_counts",
        )
        removal_total = 0
        for reason, raw_count in removals.items():
            count = _nonnegative_int(
                raw_count,
                f"source shard {shard_index}.removal_counts.{reason}",
            )
            removal_total += count
            aggregate_removals[str(reason)] += count
        if shard_records + removal_total != input_records:
            raise TokenizationBuildError(
                f"source shard {shard_index} input statistics do not conserve"
            )
        previous_source_end = source_end

    statistics = _mapping(manifest.get("statistics"), "source manifest.statistics")
    manifest_records = _source_split_counts(statistics, "split_records")
    manifest_tokens = _source_split_counts(statistics, "split_provided_tokens")
    if manifest_records != dict(aggregate_records):
        raise TokenizationBuildError(
            "source manifest split record totals do not match its shards"
        )
    if manifest_tokens != dict(aggregate_tokens):
        raise TokenizationBuildError(
            "source manifest split provided-token totals do not match its shards"
        )
    kept_records = sum(manifest_records.values())
    kept_tokens = sum(manifest_tokens.values())
    expected_totals = {
        "input_records": previous_source_end,
        "kept_records": kept_records,
        "kept_provided_tokens": kept_tokens,
    }
    for field, expected in expected_totals.items():
        if _nonnegative_int(
            statistics.get(field),
            f"source manifest.statistics.{field}",
        ) != expected:
            raise TokenizationBuildError(
                f"source manifest statistics.{field} does not conserve"
            )
    manifest_removals = _mapping(
        statistics.get("removal_counts"),
        "source manifest.statistics.removal_counts",
    )
    normalized_removals = {
        str(reason): _nonnegative_int(
            count,
            f"source manifest.statistics.removal_counts.{reason}",
        )
        for reason, count in manifest_removals.items()
    }
    if normalized_removals != dict(aggregate_removals):
        raise TokenizationBuildError(
            "source manifest removal totals do not match its shards"
        )
    if kept_records + sum(normalized_removals.values()) != previous_source_end:
        raise TokenizationBuildError("source manifest input statistics do not conserve")
    if kept_tokens < int(expected_profile["target_provided_tokens"]):
        raise TokenizationBuildError(
            "Full source manifest did not reach its provided-token target"
        )
    for split in ALLOWED_SPLITS:
        if manifest_records[split] <= 0 or manifest_tokens[split] <= 0:
            raise TokenizationBuildError(
                f"Full source manifest {split} split must be non-empty"
            )


def prepare_build_context(
    config_path: str | Path,
    *,
    project_root: str | Path,
    profile: str = ALLOWED_PROFILE,
    output_dir_override: str | Path | None = None,
) -> BuildContext:
    """Perform all read-only identity checks before creating staging data."""

    root = Path(project_root).resolve()
    resolved_config_path = _resolve_project_path(root, config_path)
    config = load_tokenized_data_config(resolved_config_path)
    configured_profile = next(iter(config["profiles"]))
    if profile not in ALLOWED_PROFILES:
        raise TokenizedDataConfigError(
            f"profile must be one of {ALLOWED_PROFILES}; found {profile!r}"
        )
    if profile != configured_profile:
        raise TokenizedDataConfigError(
            f"config contains profile {configured_profile!r}, but "
            f"{profile!r} was requested"
        )

    corpus_dir = _resolve_project_path(root, config["source"]["corpus_dir"])
    source_manifest_path = _resolve_project_path(root, config["source"]["manifest"])
    tokenizer_path = _resolve_project_path(root, config["tokenizer"]["file"])
    tokenizer_metadata_path = _resolve_project_path(root, config["tokenizer"]["metadata"])

    if config["source"].get("identity_mode") == CAPTURED_SOURCE_IDENTITY:
        if not source_manifest_path.is_file():
            raise TokenizationBuildError(
                f"Full source manifest does not exist: {source_manifest_path}"
            )
        source_manifest_preview = _load_json_object(
            source_manifest_path,
            "Full source manifest",
        )
        source_shards = source_manifest_preview.get("shards")
        if not isinstance(source_shards, list) or not source_shards:
            raise TokenizationBuildError(
                "Full source manifest shards must be a non-empty list"
            )
        config = copy.deepcopy(config)
        config["source"]["manifest_sha256"] = sha256_file(source_manifest_path)
        config["source"]["expected_source_shards"] = len(source_shards)
        validate_tokenized_data_config(config)

    profile_config = config["profiles"][profile]
    if output_dir_override is None:
        output_dir = _resolve_project_path(root, profile_config["output_dir"])
        staging_dir = _resolve_project_path(root, profile_config["staging_dir"])
    else:
        output_dir = _resolve_project_path(root, output_dir_override)
        staging_dir = output_dir.with_name(f".{output_dir.name}.inprogress")
    _validate_output_paths(root, output_dir, staging_dir, corpus_dir, tokenizer_path)

    _verify_file_identity(
        source_manifest_path,
        config["source"]["manifest_sha256"],
        "source manifest",
    )
    source_manifest = load_source_manifest(
        source_manifest_path,
        int(config["source"]["expected_source_shards"]),
    )
    _validate_source_manifest_identity(config, source_manifest, profile=profile)

    source_files = {
        split: discover_split_files(
            corpus_dir,
            split,
            expected_shards=int(config["source"]["expected_source_shards"]),
        )
        for split in ALLOWED_SPLITS
    }
    _validate_source_files(
        corpus_dir=corpus_dir,
        manifest=source_manifest,
        source_files=source_files,
    )

    _verify_file_identity(tokenizer_path, config["tokenizer"]["sha256"], "tokenizer")
    _verify_tokenizer_metadata_identity(
        tokenizer_metadata_path,
        config["tokenizer"]["metadata_sha256"],
    )
    expected_library_version = config["tokenizer"]["library"]["version"]
    if tokenizers_library.__version__ != expected_library_version:
        raise TokenizationBuildError(
            f"tokenizers version is {tokenizers_library.__version__}; "
            f"expected {expected_library_version}"
        )
    tokenizer_metadata = _load_json_object(tokenizer_metadata_path, "tokenizer metadata")
    tokenizer = load_tokenizer(tokenizer_path)
    special_ids = _validate_tokenizer_metadata(config, tokenizer_metadata, tokenizer)
    if profile == ALLOWED_PROFILE:
        _validate_day4_statistics(config, root)

    return BuildContext(
        config=config,
        config_path=resolved_config_path,
        project_root=root,
        profile=profile,
        fingerprint=config_fingerprint(config),
        output_dir=output_dir,
        staging_dir=staging_dir,
        corpus_dir=corpus_dir,
        source_manifest_path=source_manifest_path,
        source_manifest=source_manifest,
        source_files=source_files,
        tokenizer_path=tokenizer_path,
        tokenizer_metadata_path=tokenizer_metadata_path,
        tokenizer=tokenizer,
        special_token_ids=special_ids,
    )


def iter_source_records_with_cursor(
    paths: Sequence[Path],
    *,
    split: str,
) -> Iterator[SourceRecord]:
    """Stream strict source records while preserving file/line diagnostics."""

    if split not in ALLOWED_SPLITS:
        raise ValueError(f"unsupported split: {split}")
    record_number = 0
    for path in paths:
        try:
            records = iter_jsonl_records([path], require_provided_tokens=True)
            for line_number, record in enumerate(records, start=1):
                record_number += 1
                document_id = record.get("id")
                if not isinstance(document_id, str) or not document_id:
                    raise TokenizationBuildError(
                        f"id must be a non-empty string at {path}:{line_number}"
                    )
                provided_tokens = record.get("provided_token_count")
                if (
                    isinstance(provided_tokens, bool)
                    or not isinstance(provided_tokens, int)
                    or provided_tokens <= 0
                ):
                    raise TokenizationBuildError(
                        "provided_token_count must be a positive integer at "
                        f"{path}:{line_number}"
                    )
                digest = record.get("text_sha256")
                if not isinstance(digest, str) or not re.fullmatch(
                    r"[0-9a-f]{64}", digest
                ):
                    raise TokenizationBuildError(
                        f"text_sha256 must be 64 lowercase hex characters at "
                        f"{path}:{line_number}"
                    )
                actual_digest = hashlib.sha256(
                    record["text"].encode("utf-8")
                ).hexdigest()
                if digest != actual_digest:
                    raise TokenizationBuildError(
                        f"text_sha256 mismatch at {path}:{line_number}: "
                        f"found {digest}, recomputed {actual_digest}"
                    )
                yield SourceRecord(
                    split=split,
                    path=path,
                    line_number=line_number,
                    record_number=record_number,
                    record=record,
                    text_sha256=bytes.fromhex(digest),
                )
        except CorpusFormatError as exc:
            raise TokenizationBuildError(str(exc)) from exc


def encode_and_validate_document(
    tokenizer: Tokenizer,
    text: str,
    *,
    eos_token_id: int,
    unk_token_id: int,
    vocab_size: int,
) -> list[int]:
    """Encode one complete document and enforce the frozen ID contract."""

    try:
        token_ids = encode_document(tokenizer, text, eos_token_id)
    except (CorpusFormatError, TypeError) as exc:
        raise TokenizationBuildError(f"cannot encode document: {exc}") from exc
    if len(token_ids) < 2:
        raise TokenizationBuildError("document produced no raw BPE tokens before EOS")
    if token_ids[-1] != eos_token_id:
        raise TokenizationBuildError("encoded document does not end in the configured EOS")
    local_min = min(token_ids)
    local_max = max(token_ids)
    if local_min < 0 or local_max >= vocab_size:
        raise TokenizationBuildError(
            f"document produced token IDs outside [0, {vocab_size}): "
            f"min={local_min}, max={local_max}"
        )
    if token_ids[:-1].count(unk_token_id):
        raise TokenizationBuildError("document produced an unknown token; expected zero")
    return token_ids


def _empty_statistics() -> dict[str, Any]:
    return {
        "records": 0,
        "provided_tokens": 0,
        "raw_bpe_tokens": 0,
        "appended_eos_tokens": 0,
        "model_tokens": 0,
        "unknown_tokens": 0,
        "writer_inserted_bos_tokens": 0,
        "writer_inserted_pad_tokens": 0,
        "special_token_occurrences": {
            "bos": 0,
            "eos": 0,
            "pad": 0,
            "unk": 0,
        },
        "minimum_token_id": None,
        "maximum_token_id": None,
        "storage_dropped_tokens": 0,
    }


class TokenShardWriter:
    """Write one document-atomic token/index shard through `.part` files."""

    def __init__(
        self,
        *,
        staging_root: Path,
        split: str,
        shard_index: int,
        global_token_start: int,
        global_document_start: int,
        vocab_size: int,
        special_token_ids: Mapping[str, int],
    ) -> None:
        if split not in SPLIT_CODES:
            raise ValueError(f"unsupported split: {split}")
        if shard_index < 0 or global_token_start < 0 or global_document_start < 0:
            raise ValueError("shard/global indexes must be non-negative")
        self.staging_root = Path(staging_root)
        self.split = split
        self.shard_index = shard_index
        self.global_token_start = global_token_start
        self.global_document_start = global_document_start
        self.vocab_size = vocab_size
        self.special_token_ids = {
            name: int(token_id) for name, token_id in special_token_ids.items()
        }
        self.eos_token_id = self.special_token_ids["eos"]
        self.statistics = _empty_statistics()
        self._closed = False
        self._finalized = False

        shard_name = f"shard-{shard_index:05d}"
        split_dir = self.staging_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        self.binary_path = split_dir / f"{shard_name}.bin"
        self.index_path = split_dir / f"{shard_name}.idx"
        self.binary_part_path = split_dir / f"{shard_name}.bin.part"
        self.index_part_path = split_dir / f"{shard_name}.idx.part"
        for path in (
            self.binary_path,
            self.index_path,
            self.binary_part_path,
            self.index_part_path,
        ):
            if path.exists():
                raise TokenizationBuildError(f"refusing to overwrite shard file: {path}")

        self._binary_handle = self.binary_part_path.open("x+b")
        self._index_handle = self.index_part_path.open("x+b")
        self._binary_handle.write(b"\x00" * TOKEN_HEADER_BYTES)
        self._index_handle.write(b"\x00" * INDEX_HEADER_BYTES)

    @property
    def token_count(self) -> int:
        return int(self.statistics["model_tokens"])

    @property
    def document_count(self) -> int:
        return int(self.statistics["records"])

    def append_document(
        self,
        token_ids: Sequence[int],
        *,
        text_sha256: bytes,
        provided_tokens: int,
    ) -> None:
        if self._closed:
            raise TokenizationBuildError("cannot append to a closed token shard")
        if not token_ids or token_ids[-1] != self.eos_token_id:
            raise TokenizationBuildError("document must end in the configured EOS")
        if len(text_sha256) != 32:
            raise TokenizationBuildError("text_sha256 must contain 32 raw bytes")
        if provided_tokens <= 0:
            raise TokenizationBuildError("provided_tokens must be positive")

        local_min = min(token_ids)
        local_max = max(token_ids)
        if local_min < 0 or local_max >= self.vocab_size:
            raise TokenizationBuildError("document token ID is outside the vocabulary")

        local_start = self.token_count
        payload = np.asarray(token_ids, dtype=TOKEN_DTYPE)
        self._binary_handle.write(payload.tobytes(order="C"))
        self._index_handle.write(
            pack_index_entry(
                DocumentIndexEntry(
                    start_token=local_start,
                    length_tokens=len(token_ids),
                    text_sha256=text_sha256,
                )
            )
        )

        self.statistics["records"] += 1
        self.statistics["provided_tokens"] += provided_tokens
        self.statistics["raw_bpe_tokens"] += len(token_ids) - 1
        self.statistics["appended_eos_tokens"] += 1
        self.statistics["model_tokens"] += len(token_ids)
        counts = Counter(token_ids)
        for name, token_id in self.special_token_ids.items():
            self.statistics["special_token_occurrences"][name] += counts[token_id]
        self.statistics["unknown_tokens"] += counts[self.special_token_ids["unk"]]
        current_min = self.statistics["minimum_token_id"]
        current_max = self.statistics["maximum_token_id"]
        self.statistics["minimum_token_id"] = (
            local_min if current_min is None else min(current_min, local_min)
        )
        self.statistics["maximum_token_id"] = (
            local_max if current_max is None else max(current_max, local_max)
        )

    def _close_handles(self) -> None:
        if self._closed:
            return
        first_error: Exception | None = None
        for handle in (self._binary_handle, self._index_handle):
            try:
                if not handle.closed:
                    handle.close()
            except Exception as exc:  # pragma: no cover - filesystem failure
                if first_error is None:
                    first_error = exc
        self._closed = True
        if first_error is not None:
            raise first_error

    def abort(self) -> None:
        """Close handles while retaining `.part` files for explicit recovery."""

        self._close_handles()

    def finalize(self) -> dict[str, Any]:
        if self._finalized:
            raise TokenizationBuildError("token shard has already been finalized")
        if self.document_count <= 0 or self.token_count <= 0:
            raise TokenizationBuildError("refusing to finalize an empty token shard")

        binary_sha256: str | None = None
        try:
            flush_and_fsync(self._binary_handle)
            flush_and_fsync(self._index_handle)

            token_header = TokenShardHeader(
                schema_version=SCHEMA_VERSION,
                header_bytes=TOKEN_HEADER_BYTES,
                dtype_code=TOKEN_DTYPE_CODE,
                endian_code=LITTLE_ENDIAN_CODE,
                flags=APPEND_EOS_FLAG,
                vocab_size=self.vocab_size,
                split_code=SPLIT_CODES[self.split],
                token_count=self.token_count,
                document_count=self.document_count,
                payload_bytes=self.token_count * TOKEN_DTYPE.itemsize,
                eos_token_id=self.eos_token_id,
                minimum_token_id=int(self.statistics["minimum_token_id"]),
                maximum_token_id=int(self.statistics["maximum_token_id"]),
            )
            self._binary_handle.seek(0)
            self._binary_handle.write(pack_token_header(token_header))
            flush_and_fsync(self._binary_handle)
            self._binary_handle.close()

            validate_token_shard(
                self.binary_part_path,
                expected_split=self.split,
                expected_vocab_size=self.vocab_size,
                expected_eos_token_id=self.eos_token_id,
                scan_payload=True,
            )
            binary_sha256 = sha256_file(self.binary_part_path)

            index_header = IndexShardHeader(
                schema_version=SCHEMA_VERSION,
                header_bytes=INDEX_HEADER_BYTES,
                record_bytes=INDEX_ENTRY_BYTES,
                offset_semantics=INDEX_OFFSET_SEMANTICS,
                length_semantics=INDEX_LENGTH_SEMANTICS,
                token_count=self.token_count,
                document_count=self.document_count,
                global_token_start=self.global_token_start,
                global_document_start=self.global_document_start,
                binary_sha256=bytes.fromhex(binary_sha256),
                entries_bytes=self.document_count * INDEX_ENTRY_BYTES,
                eos_token_id=self.eos_token_id,
            )
            self._index_handle.seek(0)
            self._index_handle.write(pack_index_header(index_header))
            flush_and_fsync(self._index_handle)
            self._index_handle.close()
            self._closed = True

            validate_index_shard(
                self.index_part_path,
                self.binary_part_path,
                expected_global_token_start=self.global_token_start,
                expected_global_document_start=self.global_document_start,
                expected_binary_sha256=binary_sha256,
                validate_document_eos=True,
            )
            index_sha256 = sha256_file(self.index_part_path)

            if self.binary_path.exists() or self.index_path.exists():
                raise TokenizationBuildError(
                    f"refusing to replace a completed shard: {self.binary_path}"
                )
            self.binary_part_path.rename(self.binary_path)
            self.index_part_path.rename(self.index_path)
            self._finalized = True

            return {
                "shard_index": self.shard_index,
                "global_token_start": self.global_token_start,
                "global_document_start": self.global_document_start,
                "token_count": self.token_count,
                "document_count": self.document_count,
                "statistics": copy.deepcopy(self.statistics),
                "binary": {
                    "path": self.binary_path.relative_to(self.staging_root).as_posix(),
                    "bytes": self.binary_path.stat().st_size,
                    "payload_bytes": self.token_count * TOKEN_DTYPE.itemsize,
                    "sha256": binary_sha256,
                },
                "index": {
                    "path": self.index_path.relative_to(self.staging_root).as_posix(),
                    "bytes": self.index_path.stat().st_size,
                    "entries_bytes": self.document_count * INDEX_ENTRY_BYTES,
                    "sha256": index_sha256,
                },
            }
        except Exception:
            self._close_handles()
            raise


def _empty_state(context: BuildContext) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "profile": context.profile,
        "config_fingerprint": context.fingerprint,
        "source_manifest_sha256": context.config["source"]["manifest_sha256"],
        "tokenizer_sha256": context.config["tokenizer"]["sha256"],
        "splits": {
            split: {
                "complete": False,
                "statistics": _empty_statistics(),
                "shards": [],
            }
            for split in ALLOWED_SPLITS
        },
    }


def _state_path(context: BuildContext) -> Path:
    return context.staging_dir / context.config["publication"]["state_file"]


def _write_state(context: BuildContext, state: Mapping[str, Any]) -> None:
    atomic_write_json(_state_path(context), state)


def _combine_statistics(
    target: dict[str, Any],
    addition: Mapping[str, Any],
) -> None:
    for field in (
        "records",
        "provided_tokens",
        "raw_bpe_tokens",
        "appended_eos_tokens",
        "model_tokens",
        "unknown_tokens",
        "writer_inserted_bos_tokens",
        "writer_inserted_pad_tokens",
        "storage_dropped_tokens",
    ):
        target[field] += int(addition[field])
    for name in target["special_token_occurrences"]:
        target["special_token_occurrences"][name] += int(
            addition["special_token_occurrences"][name]
        )
    addition_min = addition["minimum_token_id"]
    addition_max = addition["maximum_token_id"]
    if addition_min is not None:
        target["minimum_token_id"] = (
            int(addition_min)
            if target["minimum_token_id"] is None
            else min(int(target["minimum_token_id"]), int(addition_min))
        )
    if addition_max is not None:
        target["maximum_token_id"] = (
            int(addition_max)
            if target["maximum_token_id"] is None
            else max(int(target["maximum_token_id"]), int(addition_max))
        )


def _aggregate_shards(shards: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    aggregate = _empty_statistics()
    for shard in shards:
        _combine_statistics(aggregate, _mapping(shard.get("statistics"), "shard.statistics"))
    return aggregate


def _apply_completed_shard(
    state: dict[str, Any],
    split: str,
    shard: dict[str, Any],
) -> None:
    split_state = state["splits"][split]
    expected_index = len(split_state["shards"])
    if shard["shard_index"] != expected_index:
        raise ResumeStateError(
            f"{split} shard index is {shard['shard_index']}; expected {expected_index}"
        )
    expected_token_start = split_state["statistics"]["model_tokens"]
    expected_document_start = split_state["statistics"]["records"]
    if shard["global_token_start"] != expected_token_start:
        raise ResumeStateError(f"{split} shard global token prefix is not contiguous")
    if shard["global_document_start"] != expected_document_start:
        raise ResumeStateError(f"{split} shard global document prefix is not contiguous")
    split_state["shards"].append(shard)
    _combine_statistics(split_state["statistics"], shard["statistics"])


def _registered_paths(state: Mapping[str, Any]) -> set[str]:
    registered: set[str] = set()
    for split in ALLOWED_SPLITS:
        for shard in state["splits"][split]["shards"]:
            registered.add(str(shard["binary"]["path"]))
            registered.add(str(shard["index"]["path"]))
    return registered


def _validate_state_identity(context: BuildContext, state: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": SCHEMA_VERSION,
        "profile": context.profile,
        "config_fingerprint": context.fingerprint,
        "source_manifest_sha256": context.config["source"]["manifest_sha256"],
        "tokenizer_sha256": context.config["tokenizer"]["sha256"],
    }
    for field, expected_value in expected.items():
        if state.get(field) != expected_value:
            raise ResumeStateError(
                f"resume state {field} mismatch: "
                f"found {state.get(field)!r}, expected {expected_value!r}"
            )
    splits = state.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != set(ALLOWED_SPLITS):
        raise ResumeStateError("resume state split set is invalid")


def _reconcile_staging(context: BuildContext, state: dict[str, Any]) -> None:
    """Validate checkpoints and discard only precisely identified orphans."""

    registered = _registered_paths(state)
    allowed_root_files = {context.config["publication"]["state_file"]}
    stale_manifest = context.staging_dir / context.config["publication"]["manifest_file"]
    if stale_manifest.is_file():
        stale_manifest.unlink()
    for child in context.staging_dir.iterdir():
        if child.name in ALLOWED_SPLITS and child.is_dir():
            continue
        if child.is_file() and child.name in allowed_root_files:
            continue
        raise ResumeStateError(f"unexpected item in staging directory: {child}")

    for split in ALLOWED_SPLITS:
        split_dir = context.staging_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for path in split_dir.iterdir():
            match = SHARD_FILE_RE.fullmatch(path.name)
            if not path.is_file() or match is None:
                raise ResumeStateError(f"unexpected item in staging split: {path}")
            relative = path.relative_to(context.staging_dir).as_posix()
            if match.group(3) == ".part" or relative not in registered:
                path.unlink()

    for split in ALLOWED_SPLITS:
        split_state = state["splits"][split]
        if not isinstance(split_state, Mapping):
            raise ResumeStateError(f"resume state for {split} must be a mapping")
        shards = split_state.get("shards")
        statistics = split_state.get("statistics")
        if not isinstance(shards, list) or not isinstance(statistics, Mapping):
            raise ResumeStateError(f"resume state for {split} is incomplete")
        expected_token_start = 0
        expected_document_start = 0
        for shard_index, shard in enumerate(shards):
            if shard.get("shard_index") != shard_index:
                raise ResumeStateError(f"{split} checkpoint shard indexes are not contiguous")
            if shard.get("global_token_start") != expected_token_start:
                raise ResumeStateError(f"{split} checkpoint token prefixes are not contiguous")
            if shard.get("global_document_start") != expected_document_start:
                raise ResumeStateError(f"{split} checkpoint document prefixes are not contiguous")
            binary_path = context.staging_dir / shard["binary"]["path"]
            index_path = context.staging_dir / shard["index"]["path"]
            if not binary_path.is_file() or not index_path.is_file():
                raise ResumeStateError(
                    "registered checkpoint files are missing for "
                    f"{split} shard {shard_index}"
                )
            if binary_path.stat().st_size != int(shard["binary"]["bytes"]):
                raise ResumeStateError(f"checkpoint binary size mismatch: {binary_path}")
            if index_path.stat().st_size != int(shard["index"]["bytes"]):
                raise ResumeStateError(f"checkpoint index size mismatch: {index_path}")
            if sha256_file(binary_path) != shard["binary"]["sha256"]:
                raise ResumeStateError(f"checkpoint binary hash mismatch: {binary_path}")
            if sha256_file(index_path) != shard["index"]["sha256"]:
                raise ResumeStateError(f"checkpoint index hash mismatch: {index_path}")
            token_header = validate_token_shard(
                binary_path,
                expected_split=split,
                expected_vocab_size=context.config["tokenizer"]["vocab_size"],
                expected_eos_token_id=context.special_token_ids["eos"],
                scan_payload=False,
            )
            _validate_shard_statistics(
                shard,
                token_header,
                label=f"checkpoint {split} shard {shard_index}",
            )
            validate_index_shard(
                index_path,
                binary_path,
                expected_global_token_start=expected_token_start,
                expected_global_document_start=expected_document_start,
                expected_binary_sha256=shard["binary"]["sha256"],
            )
            expected_token_start += int(shard["token_count"])
            expected_document_start += int(shard["document_count"])
        if _aggregate_shards(shards) != dict(statistics):
            raise ResumeStateError(f"checkpoint aggregate statistics mismatch for {split}")


def _load_or_initialize_state(
    context: BuildContext,
    *,
    resume: bool,
) -> dict[str, Any]:
    if context.staging_dir.exists():
        if not resume:
            raise ResumeStateError(
                f"staging directory already exists: {context.staging_dir}; "
                "rerun with --resume after reviewing it"
            )
        state_path = _state_path(context)
        if not state_path.is_file():
            raise ResumeStateError(f"resume state does not exist: {state_path}")
        state = _load_json_object(state_path, "resume state")
        _validate_state_identity(context, state)
        _reconcile_staging(context, state)
        return state

    context.staging_dir.parent.mkdir(parents=True, exist_ok=True)
    context.staging_dir.mkdir()
    for split in ALLOWED_SPLITS:
        (context.staging_dir / split).mkdir()
    state = _empty_state(context)
    _write_state(context, state)
    return state


def _core_expected_statistics(config: Mapping[str, Any], split: str) -> dict[str, int]:
    return {
        field: int(config["expected"][split][field])
        for field in (
            "records",
            "provided_tokens",
            "raw_bpe_tokens",
            "appended_eos_tokens",
            "model_tokens",
            "unknown_tokens",
        )
    }


def _expected_source_counts(context: BuildContext, split: str) -> dict[str, int]:
    if context.profile == ALLOWED_PROFILE:
        return {
            "records": int(context.config["expected"][split]["records"]),
            "provided_tokens": int(
                context.config["expected"][split]["provided_tokens"]
            ),
        }
    statistics = _mapping(
        context.source_manifest.get("statistics"),
        "source manifest.statistics",
    )
    records = _source_split_counts(statistics, "split_records")
    provided_tokens = _source_split_counts(
        statistics,
        "split_provided_tokens",
    )
    return {
        "records": records[split],
        "provided_tokens": provided_tokens[split],
    }


def _validate_split_against_expected(
    context: BuildContext,
    split: str,
    statistics: Mapping[str, Any],
) -> None:
    if context.profile == ALLOWED_PROFILE:
        expected = _core_expected_statistics(context.config, split)
        for field, expected_value in expected.items():
            actual = int(statistics[field])
            if actual != expected_value:
                raise TokenizationBuildError(
                    f"{split}.{field} is {actual}; Day 4 expected {expected_value}"
                )
    else:
        expected_source = _expected_source_counts(context, split)
        for field, expected_value in expected_source.items():
            actual = int(statistics[field])
            if actual != expected_value:
                raise TokenizationBuildError(
                    f"{split}.{field} is {actual}; Full source manifest "
                    f"declares {expected_value}"
                )
        _validate_statistics_conservation(statistics, label=split)
    if int(statistics["writer_inserted_bos_tokens"]) != 0:
        raise TokenizationBuildError(f"{split} writer inserted BOS tokens")
    if int(statistics["writer_inserted_pad_tokens"]) != 0:
        raise TokenizationBuildError(f"{split} writer inserted PAD tokens")
    if int(statistics["storage_dropped_tokens"]) != 0:
        raise TokenizationBuildError(f"{split} storage dropped tokens")


def _completed_shard_count(state: Mapping[str, Any]) -> int:
    return sum(len(state["splits"][split]["shards"]) for split in ALLOWED_SPLITS)


def _commit_writer(
    context: BuildContext,
    state: dict[str, Any],
    writer: TokenShardWriter,
    *,
    progress: Callable[[str], None],
    interrupt_after_completed_shards: int | None,
) -> None:
    shard = writer.finalize()
    _apply_completed_shard(state, writer.split, shard)
    _write_state(context, state)
    progress(
        f"{writer.split} shard-{writer.shard_index:05d}: "
        f"documents={writer.document_count}, tokens={writer.token_count}"
    )
    if (
        interrupt_after_completed_shards is not None
        and _completed_shard_count(state) >= interrupt_after_completed_shards
    ):
        raise ControlledInterruption(
            "controlled interruption after "
            f"{_completed_shard_count(state)} completed storage shard(s)"
        )


def _build_split(
    context: BuildContext,
    state: dict[str, Any],
    split: str,
    *,
    progress: Callable[[str], None],
    interrupt_after_completed_shards: int | None,
) -> None:
    split_state = state["splits"][split]
    if split_state["complete"]:
        _validate_split_against_expected(context, split, split_state["statistics"])
        return

    committed_records = int(split_state["statistics"]["records"])
    shard_target = int(
        context.config["profiles"][context.profile]["target_model_tokens_per_shard"]
    )
    vocab_size = int(context.config["tokenizer"]["vocab_size"])
    writer: TokenShardWriter | None = None
    records_seen = 0

    try:
        for source_record in iter_source_records_with_cursor(
            context.source_files[split],
            split=split,
        ):
            records_seen = source_record.record_number
            if source_record.record_number <= committed_records:
                continue

            token_ids = encode_and_validate_document(
                context.tokenizer,
                source_record.record["text"],
                eos_token_id=context.special_token_ids["eos"],
                unk_token_id=context.special_token_ids["unk"],
                vocab_size=vocab_size,
            )
            if (
                writer is not None
                and writer.document_count > 0
                and writer.token_count + len(token_ids) > shard_target
            ):
                _commit_writer(
                    context,
                    state,
                    writer,
                    progress=progress,
                    interrupt_after_completed_shards=interrupt_after_completed_shards,
                )
                writer = None

            if writer is None:
                writer = TokenShardWriter(
                    staging_root=context.staging_dir,
                    split=split,
                    shard_index=len(split_state["shards"]),
                    global_token_start=int(split_state["statistics"]["model_tokens"]),
                    global_document_start=int(split_state["statistics"]["records"]),
                    vocab_size=vocab_size,
                    special_token_ids=context.special_token_ids,
                )
            writer.append_document(
                token_ids,
                text_sha256=source_record.text_sha256,
                provided_tokens=int(source_record.record["provided_token_count"]),
            )

        if writer is not None and writer.document_count:
            _commit_writer(
                context,
                state,
                writer,
                progress=progress,
                interrupt_after_completed_shards=interrupt_after_completed_shards,
            )
            writer = None
    finally:
        if writer is not None:
            writer.abort()

    if committed_records > records_seen:
        raise ResumeStateError(
            f"{split} source has {records_seen} records, fewer than "
            f"the {committed_records} committed records"
        )
    expected_records = _expected_source_counts(context, split)["records"]
    if records_seen != expected_records:
        raise TokenizationBuildError(
            f"{split} source contains {records_seen} records; expected "
            f"{expected_records}"
        )
    _validate_split_against_expected(context, split, split_state["statistics"])
    split_state["complete"] = True
    _write_state(context, state)
    progress(
        f"{split} complete: documents={split_state['statistics']['records']}, "
        f"tokens={split_state['statistics']['model_tokens']}"
    )


def _manifest_split_payload(
    context: BuildContext,
    state: Mapping[str, Any],
    split: str,
) -> dict[str, Any]:
    split_state = state["splits"][split]
    payload = copy.deepcopy(split_state["statistics"])
    payload["source_files"] = [
        project_relative_path(path, context.project_root)
        for path in context.source_files[split]
    ]
    payload["storage_shards"] = len(split_state["shards"])
    payload["shards"] = copy.deepcopy(split_state["shards"])
    return payload


def _manifest_source_payload(context: BuildContext) -> dict[str, Any]:
    payload = {
        "dataset": context.config["source"]["dataset"],
        "configuration": context.config["source"]["configuration"],
        "revision": context.config["source"]["revision"],
        "manifest": project_relative_path(
            context.source_manifest_path,
            context.project_root,
        ),
        "manifest_sha256": context.config["source"]["manifest_sha256"],
        "expected_source_shards": context.config["source"][
            "expected_source_shards"
        ],
        "split_order": list(ALLOWED_SPLITS),
        "deterministic_order": context.config["source"]["deterministic_order"],
    }
    if context.profile == "full":
        source_statistics = _mapping(
            context.source_manifest.get("statistics"),
            "source manifest.statistics",
        )
        payload["config_fingerprint"] = context.source_manifest[
            "config_fingerprint"
        ]
        payload["profile"] = copy.deepcopy(context.source_manifest["profile"])
        payload["statistics"] = {
            "split_records": copy.deepcopy(source_statistics["split_records"]),
            "split_provided_tokens": copy.deepcopy(
                source_statistics["split_provided_tokens"]
            ),
        }
    return payload


def build_manifest(
    context: BuildContext,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the deterministic, location-independent corpus authority file."""

    if state.get("status") != "complete":
        raise TokenizationBuildError("cannot build a manifest from an incomplete state")
    for split in ALLOWED_SPLITS:
        if state["splits"][split].get("complete") is not True:
            raise TokenizationBuildError(f"cannot publish incomplete split: {split}")

    totals = _empty_statistics()
    for split in ALLOWED_SPLITS:
        _combine_statistics(totals, state["splits"][split]["statistics"])
    totals["token_payload_bytes"] = totals["model_tokens"] * TOKEN_DTYPE.itemsize
    totals["storage_shards"] = sum(
        len(state["splits"][split]["shards"]) for split in ALLOWED_SPLITS
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "format_name": FORMAT_NAME,
        "status": "complete",
        "profile": context.profile,
        "config_fingerprint": context.fingerprint,
        "source": _manifest_source_payload(context),
        "tokenizer": {
            "file": project_relative_path(context.tokenizer_path, context.project_root),
            "sha256": context.config["tokenizer"]["sha256"],
            "metadata": project_relative_path(
                context.tokenizer_metadata_path,
                context.project_root,
            ),
            "metadata_sha256": context.config["tokenizer"]["metadata_sha256"],
            "library": copy.deepcopy(context.config["tokenizer"]["library"]),
            "vocab_size": context.config["tokenizer"]["vocab_size"],
            "normalizer": context.config["tokenizer"]["normalizer"],
            "special_token_ids": copy.deepcopy(context.special_token_ids),
        },
        "encoding": copy.deepcopy(context.config["encoding"]),
        "binary_format": copy.deepcopy(context.config["binary_format"]),
        "index_format": copy.deepcopy(context.config["index_format"]),
        "sharding": {
            "target_model_tokens_per_shard": context.config["profiles"][
                context.profile
            ]["target_model_tokens_per_shard"],
            "document_atomic": True,
            "storage_dropped_tokens": 0,
        },
        "splits": {
            split: _manifest_split_payload(context, state, split)
            for split in ALLOWED_SPLITS
        },
        "totals": totals,
        "dataset_contract": {
            "storage_view": "split_logical_stream",
            "token_dtype_on_disk": "<u2",
            "output_torch_dtype": "long",
            "next_token_read_length": "context_length + 1",
            "allow_cross_document_windows": True,
            "allow_cross_storage_shard_windows": True,
            "allow_cross_split_windows": False,
            "train_mode": "all_starts",
            "train_sampler": "epoch_random_window_with_replacement",
            "train_base_seed": context.config["dataset"]["train"]["base_seed"],
            "evaluation_mode": "sequential_non_overlapping",
            "default_pin_memory": False,
        },
    }


def _safe_manifest_path(root: Path, value: Any, field: str) -> Path:
    relative = Path(_nonempty_string(value, field))
    if relative.is_absolute():
        raise TokenizationBuildError(f"{field} must be relative to the tokenized corpus")
    resolved = (root / relative).resolve()
    if not _is_relative_to(resolved, root.resolve()):
        raise TokenizationBuildError(f"{field} escapes the tokenized corpus: {value}")
    return resolved


def _safe_project_identity_path(
    project_root: Path,
    value: Any,
    field: str,
) -> Path:
    relative = Path(_nonempty_string(value, field))
    if relative.is_absolute():
        raise TokenizationBuildError(f"{field} must be project-relative")
    resolved = (project_root / relative).resolve()
    if not _is_relative_to(resolved, project_root.resolve()):
        raise TokenizationBuildError(f"{field} escapes the project root")
    return resolved


def _statistics_view(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = _empty_statistics()
    for name in result:
        if name not in payload:
            raise TokenizationBuildError(f"{field}.{name} is missing")
        if name == "special_token_occurrences":
            special = _mapping(payload[name], f"{field}.{name}")
            if set(special) != {"bos", "eos", "pad", "unk"}:
                raise TokenizationBuildError(f"{field}.{name} has invalid keys")
            result[name] = {token: int(special[token]) for token in special}
        elif name in ("minimum_token_id", "maximum_token_id"):
            result[name] = None if payload[name] is None else int(payload[name])
        else:
            result[name] = int(payload[name])
    return result


def _validate_statistics_conservation(
    statistics: Mapping[str, Any],
    *,
    label: str,
) -> None:
    if int(statistics["records"]) <= 0:
        raise TokenizationBuildError(f"{label} must contain at least one document")
    if int(statistics["records"]) != int(statistics["appended_eos_tokens"]):
        raise TokenizationBuildError(f"{label}: records must equal appended EOS tokens")
    if (
        int(statistics["raw_bpe_tokens"])
        + int(statistics["appended_eos_tokens"])
        != int(statistics["model_tokens"])
    ):
        raise TokenizationBuildError(f"{label}: raw BPE + EOS != model tokens")
    for field in (
        "unknown_tokens",
        "writer_inserted_bos_tokens",
        "writer_inserted_pad_tokens",
        "storage_dropped_tokens",
    ):
        if int(statistics[field]) != 0:
            raise TokenizationBuildError(f"{label}.{field} must be zero")
    minimum = statistics["minimum_token_id"]
    maximum = statistics["maximum_token_id"]
    if minimum is None or maximum is None or int(minimum) > int(maximum):
        raise TokenizationBuildError(f"{label} token min/max are invalid")


def _validate_shard_statistics(
    shard: Mapping[str, Any],
    header: TokenShardHeader,
    *,
    label: str,
) -> dict[str, Any]:
    statistics = _statistics_view(
        _mapping(shard.get("statistics"), f"{label}.statistics"),
        f"{label}.statistics",
    )
    _validate_statistics_conservation(statistics, label=label)
    expected = {
        "records": header.document_count,
        "model_tokens": header.token_count,
        "minimum_token_id": header.minimum_token_id,
        "maximum_token_id": header.maximum_token_id,
    }
    for field, expected_value in expected.items():
        if int(statistics[field]) != expected_value:
            raise TokenizationBuildError(
                f"{label}.{field} is {statistics[field]}; header declares {expected_value}"
            )
    if int(shard.get("token_count", -1)) != header.token_count:
        raise TokenizationBuildError(f"{label}.token_count does not match its header")
    if int(shard.get("document_count", -1)) != header.document_count:
        raise TokenizationBuildError(f"{label}.document_count does not match its header")
    return statistics


def _validate_manifest_contract(manifest: Mapping[str, Any]) -> None:
    encoding = _mapping(manifest.get("encoding"), "manifest.encoding")
    expected_encoding = {
        "append_eos_per_document": True,
        "add_bos": False,
        "add_pad": False,
        "reject_empty_document": True,
        "allow_cross_document_windows": True,
        "allow_cross_storage_shard_windows": True,
        "allow_cross_split_windows": False,
    }
    for field, expected in expected_encoding.items():
        if encoding.get(field) != expected:
            raise TokenizationBuildError(
                f"manifest.encoding.{field} must be {expected!r}"
            )

    binary = _mapping(manifest.get("binary_format"), "manifest.binary_format")
    binary_expected = {
        "schema_version": SCHEMA_VERSION,
        "magic": TOKEN_MAGIC.decode("ascii"),
        "header_bytes": TOKEN_HEADER_BYTES,
        "struct_format": "<8sHHBBHIB3xQQQIII4x",
        "dtype": "<u2",
        "dtype_code": TOKEN_DTYPE_CODE,
        "endian": "little",
        "endian_code": LITTLE_ENDIAN_CODE,
    }
    for field, expected in binary_expected.items():
        if binary.get(field) != expected:
            raise TokenizationBuildError(
                f"manifest.binary_format.{field} must be {expected!r}"
            )

    index = _mapping(manifest.get("index_format"), "manifest.index_format")
    index_expected = {
        "schema_version": SCHEMA_VERSION,
        "magic": INDEX_MAGIC.decode("ascii"),
        "header_bytes": INDEX_HEADER_BYTES,
        "header_struct_format": "<8sHHHBBQQQQ32sQI36x",
        "record_bytes": INDEX_ENTRY_BYTES,
        "record_struct_format": "<QQ32s",
        "offset_semantics": "shard_local_token_offset",
        "length_includes_eos": True,
    }
    for field, expected in index_expected.items():
        if index.get(field) != expected:
            raise TokenizationBuildError(
                f"manifest.index_format.{field} must be {expected!r}"
            )

    contract = _mapping(
        manifest.get("dataset_contract"),
        "manifest.dataset_contract",
    )
    if contract.get("storage_view") != "split_logical_stream":
        raise TokenizationBuildError("manifest Dataset storage view is invalid")
    if contract.get("next_token_read_length") != "context_length + 1":
        raise TokenizationBuildError("manifest Dataset T+1 contract is invalid")
    if contract.get("allow_cross_split_windows") is not False:
        raise TokenizationBuildError("manifest must forbid cross-split windows")


def _verify_manifest_identities(
    manifest: Mapping[str, Any],
    project_root: Path,
) -> None:
    source = _mapping(manifest.get("source"), "manifest.source")
    source_manifest_path = _safe_project_identity_path(
        project_root,
        source.get("manifest"),
        "manifest.source.manifest",
    )
    expected_source_sha = _sha256_string(
        source.get("manifest_sha256"),
        "manifest.source.manifest_sha256",
    )
    _verify_file_identity(source_manifest_path, expected_source_sha, "source manifest")
    source_manifest = _load_json_object(source_manifest_path, "source manifest")
    if source_manifest.get("status") != "complete":
        raise TokenizationBuildError("source manifest is no longer complete")
    source_dataset = _mapping(
        source_manifest.get("dataset"),
        "source manifest.dataset",
    )
    for manifest_field, source_field in (
        ("dataset", "name"),
        ("configuration", "configuration"),
        ("revision", "revision"),
    ):
        if source_dataset.get(source_field) != source.get(manifest_field):
            raise TokenizationBuildError(
                f"source manifest dataset.{source_field} identity mismatch"
            )
    source_shards = source_manifest.get("shards")
    expected_source_shards = _positive_int(
        source.get("expected_source_shards"),
        "manifest.source.expected_source_shards",
    )
    if not isinstance(source_shards, list) or len(source_shards) != expected_source_shards:
        raise TokenizationBuildError("source manifest shard-count identity mismatch")

    has_full_identity = "config_fingerprint" in source
    full_identity_fields = {"config_fingerprint", "profile", "statistics"}
    present_full_fields = full_identity_fields.intersection(source)
    if present_full_fields and present_full_fields != full_identity_fields:
        raise TokenizationBuildError("manifest Full source identity is incomplete")
    if has_full_identity:
        source_fingerprint = _sha256_string(
            source.get("config_fingerprint"),
            "manifest.source.config_fingerprint",
        )
        if source_manifest.get("config_fingerprint") != source_fingerprint:
            raise TokenizationBuildError(
                "source manifest configuration fingerprint identity mismatch"
            )
        if source_manifest.get("profile") != source.get("profile"):
            raise TokenizationBuildError("source manifest Full profile identity mismatch")
        source_statistics = _mapping(
            source.get("statistics"),
            "manifest.source.statistics",
        )
        current_statistics = _mapping(
            source_manifest.get("statistics"),
            "source manifest.statistics",
        )
        for field in ("split_records", "split_provided_tokens"):
            if source_statistics.get(field) != current_statistics.get(field):
                raise TokenizationBuildError(
                    f"source manifest {field} identity mismatch"
                )

    tokenizer = _mapping(manifest.get("tokenizer"), "manifest.tokenizer")
    tokenizer_path = _safe_project_identity_path(
        project_root,
        tokenizer.get("file"),
        "manifest.tokenizer.file",
    )
    metadata_path = _safe_project_identity_path(
        project_root,
        tokenizer.get("metadata"),
        "manifest.tokenizer.metadata",
    )
    _verify_file_identity(
        tokenizer_path,
        _sha256_string(tokenizer.get("sha256"), "manifest.tokenizer.sha256"),
        "tokenizer",
    )
    _verify_tokenizer_metadata_identity(
        metadata_path,
        _sha256_string(
            tokenizer.get("metadata_sha256"),
            "manifest.tokenizer.metadata_sha256",
        ),
    )


def validate_completed_corpus(
    manifest_path: str | Path,
    *,
    project_root: str | Path | None = None,
    expected_fingerprint: str | None = None,
    verify_identities: bool = True,
    scan_payload: bool = True,
) -> dict[str, Any]:
    """Validate a complete manifest and every registered token/index shard."""

    path = Path(manifest_path).resolve()
    manifest = _load_json_object(path, "tokenized corpus manifest")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise TokenizationBuildError("tokenized corpus manifest schema mismatch")
    if manifest.get("format_name") != FORMAT_NAME:
        raise TokenizationBuildError("tokenized corpus manifest format mismatch")
    if manifest.get("status") != "complete":
        raise TokenizationBuildError("tokenized corpus manifest is not complete")
    profile = manifest.get("profile")
    if profile not in ALLOWED_PROFILES:
        raise TokenizationBuildError(
            f"tokenized corpus manifest profile must be one of {ALLOWED_PROFILES}"
        )
    fingerprint = manifest.get("config_fingerprint")
    if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise TokenizationBuildError("manifest config_fingerprint is invalid")
    if expected_fingerprint is not None and fingerprint != expected_fingerprint:
        raise TokenizationBuildError(
            "completed output belongs to a different semantic configuration"
        )
    _validate_manifest_contract(manifest)
    if verify_identities:
        if project_root is None:
            raise ValueError("project_root is required when verify_identities=True")
        _verify_manifest_identities(manifest, Path(project_root).resolve())

    source_info = _mapping(manifest.get("source"), "manifest.source")
    if source_info.get("split_order") != list(ALLOWED_SPLITS):
        raise TokenizationBuildError(
            f"manifest source.split_order must be {list(ALLOWED_SPLITS)}"
        )
    if profile == "full" and not {
        "config_fingerprint",
        "profile",
        "statistics",
    }.issubset(source_info):
        raise TokenizationBuildError("Full manifest source identity is incomplete")
    full_source_records: dict[str, int] | None = None
    full_source_tokens: dict[str, int] | None = None
    if profile == "full":
        full_source_statistics = _mapping(
            source_info.get("statistics"),
            "manifest.source.statistics",
        )
        full_source_records = _source_split_counts(
            full_source_statistics,
            "split_records",
        )
        full_source_tokens = _source_split_counts(
            full_source_statistics,
            "split_provided_tokens",
        )

    tokenizer_info = _mapping(manifest.get("tokenizer"), "manifest.tokenizer")
    vocab_size = _positive_int(tokenizer_info.get("vocab_size"), "manifest.tokenizer.vocab_size")
    special_ids = _mapping(
        tokenizer_info.get("special_token_ids"),
        "manifest.tokenizer.special_token_ids",
    )
    if dict(special_ids) != {"bos": 0, "eos": 1, "pad": 2, "unk": 3}:
        raise TokenizationBuildError(
            "manifest special IDs must be bos=0, eos=1, pad=2, unk=3"
        )
    eos_token_id = int(special_ids.get("eos", -1))
    if eos_token_id < 0 or eos_token_id >= vocab_size:
        raise TokenizationBuildError("manifest EOS token ID is invalid")

    splits = _mapping(manifest.get("splits"), "manifest.splits")
    if set(splits) != set(ALLOWED_SPLITS):
        raise TokenizationBuildError(
            f"manifest split keys must be {list(ALLOWED_SPLITS)}"
        )
    corpus_root = path.parent
    overall = _empty_statistics()

    for split in ALLOWED_SPLITS:
        split_payload = _mapping(splits[split], f"manifest.splits.{split}")
        split_statistics = _statistics_view(
            split_payload,
            f"manifest.splits.{split}",
        )
        _validate_statistics_conservation(split_statistics, label=split)
        if full_source_records is not None and full_source_tokens is not None:
            if int(split_statistics["records"]) != full_source_records[split]:
                raise TokenizationBuildError(
                    f"manifest {split} records do not match the Full source"
                )
            if int(split_statistics["provided_tokens"]) != full_source_tokens[split]:
                raise TokenizationBuildError(
                    f"manifest {split} provided tokens do not match the Full source"
                )
        shards = split_payload.get("shards")
        if not isinstance(shards, list) or not shards:
            raise TokenizationBuildError(f"manifest {split} shards must be non-empty")
        if int(split_payload.get("storage_shards", -1)) != len(shards):
            raise TokenizationBuildError(f"manifest {split} storage shard count mismatch")

        expected_token_start = 0
        expected_document_start = 0
        registered_files: set[Path] = set()
        for shard_index, raw_shard in enumerate(shards):
            shard = _mapping(raw_shard, f"manifest {split} shard {shard_index}")
            if shard.get("shard_index") != shard_index:
                raise TokenizationBuildError(f"{split} shard indexes are not contiguous")
            if shard.get("global_token_start") != expected_token_start:
                raise TokenizationBuildError(f"{split} global token prefixes are not contiguous")
            if shard.get("global_document_start") != expected_document_start:
                raise TokenizationBuildError(
                    f"{split} global document prefixes are not contiguous"
                )

            binary = _mapping(shard.get("binary"), f"{split} shard {shard_index}.binary")
            index = _mapping(shard.get("index"), f"{split} shard {shard_index}.index")
            binary_path = _safe_manifest_path(
                corpus_root,
                binary.get("path"),
                f"{split} shard {shard_index}.binary.path",
            )
            index_path = _safe_manifest_path(
                corpus_root,
                index.get("path"),
                f"{split} shard {shard_index}.index.path",
            )
            if (
                binary_path.parent != corpus_root / split
                or index_path.parent != corpus_root / split
            ):
                raise TokenizationBuildError(
                    f"{split} shard files must be direct children of the split directory"
                )
            registered_files.update((binary_path, index_path))
            if not binary_path.is_file() or not index_path.is_file():
                raise TokenizationBuildError(
                    f"manifest shard files are missing for {split} {shard_index}"
                )
            if binary_path.stat().st_size != int(binary.get("bytes", -1)):
                raise TokenizationBuildError(f"binary size mismatch: {binary_path}")
            if index_path.stat().st_size != int(index.get("bytes", -1)):
                raise TokenizationBuildError(f"index size mismatch: {index_path}")
            binary_sha256 = sha256_file(binary_path)
            if binary_sha256 != binary.get("sha256"):
                raise TokenizationBuildError(f"binary SHA-256 mismatch: {binary_path}")
            if sha256_file(index_path) != index.get("sha256"):
                raise TokenizationBuildError(f"index SHA-256 mismatch: {index_path}")

            token_header = validate_token_shard(
                binary_path,
                expected_split=split,
                expected_vocab_size=vocab_size,
                expected_eos_token_id=eos_token_id,
                scan_payload=scan_payload,
            )
            _validate_shard_statistics(
                shard,
                token_header,
                label=f"{split} shard {shard_index}",
            )
            validate_index_shard(
                index_path,
                binary_path,
                expected_global_token_start=expected_token_start,
                expected_global_document_start=expected_document_start,
                expected_binary_sha256=binary_sha256,
            )
            if int(binary.get("payload_bytes", -1)) != token_header.payload_bytes:
                raise TokenizationBuildError(f"{split} shard payload byte count mismatch")
            if int(index.get("entries_bytes", -1)) != (
                token_header.document_count * INDEX_ENTRY_BYTES
            ):
                raise TokenizationBuildError(f"{split} shard index byte count mismatch")

            expected_token_start += token_header.token_count
            expected_document_start += token_header.document_count

        actual_split_dir_files = {
            child.resolve()
            for child in (corpus_root / split).iterdir()
            if child.is_file()
        }
        if actual_split_dir_files != registered_files:
            extras = sorted(str(item) for item in actual_split_dir_files - registered_files)
            missing = sorted(str(item) for item in registered_files - actual_split_dir_files)
            raise TokenizationBuildError(
                f"{split} registered file set mismatch; extras={extras}, missing={missing}"
            )

        aggregate = _aggregate_shards(shards)
        if aggregate != split_statistics:
            raise TokenizationBuildError(f"{split} shard statistics do not match split totals")
        if expected_token_start != int(split_statistics["model_tokens"]):
            raise TokenizationBuildError(f"{split} token prefix does not match split total")
        if expected_document_start != int(split_statistics["records"]):
            raise TokenizationBuildError(f"{split} document prefix does not match split total")
        _combine_statistics(overall, split_statistics)

    totals_payload = _mapping(manifest.get("totals"), "manifest.totals")
    totals_statistics = _statistics_view(totals_payload, "manifest.totals")
    if totals_statistics != overall:
        raise TokenizationBuildError("manifest totals do not equal the split sum")
    _validate_statistics_conservation(totals_statistics, label="totals")
    if int(totals_payload.get("token_payload_bytes", -1)) != int(
        overall["model_tokens"]
    ) * TOKEN_DTYPE.itemsize:
        raise TokenizationBuildError("manifest total token payload bytes do not conserve")
    expected_storage_shards = sum(
        len(splits[split]["shards"]) for split in ALLOWED_SPLITS
    )
    if int(totals_payload.get("storage_shards", -1)) != expected_storage_shards:
        raise TokenizationBuildError("manifest total storage shard count mismatch")
    return manifest


def preflight_summary(context: BuildContext, *, resume: bool) -> list[str]:
    """Return a compact, user-facing summary after all read-only checks pass."""

    statistics_contract = (
        "frozen Day 4 exact statistics"
        if context.profile == ALLOWED_PROFILE
        else "source-manifest counts plus observed token conservation"
    )
    return [
        "Tokenized corpus preflight",
        f"  profile: {context.profile}",
        f"  source corpus: {project_relative_path(context.corpus_dir, context.project_root)}",
        "  source manifest SHA-256: "
        f"{context.config['source']['manifest_sha256'][:12]}...",
        f"  tokenizer SHA-256: {context.config['tokenizer']['sha256'][:12]}...",
        f"  vocabulary: {context.config['tokenizer']['vocab_size']}",
        "  special IDs: "
        + ", ".join(
            f"{name}={token_id}" for name, token_id in context.special_token_ids.items()
        ),
        f"  split order: {' -> '.join(ALLOWED_SPLITS)}",
        "  shard target: "
        f"{context.config['profiles'][context.profile]['target_model_tokens_per_shard']}",
        "  storage: little-endian uint16 (<u2)",
        f"  output: {project_relative_path(context.output_dir, context.project_root)}",
        f"  staging: {project_relative_path(context.staging_dir, context.project_root)}",
        f"  resume: {resume}",
        f"  statistics contract: {statistics_contract}",
    ]


def build_tokenized_corpus(
    context: BuildContext,
    *,
    resume: bool = False,
    progress: Callable[[str], None] | None = None,
    interrupt_after_completed_shards: int | None = None,
) -> BuildResult:
    """Build, validate, and atomically publish the configured token corpus."""

    progress_callback = progress or (lambda _message: None)
    manifest_name = context.config["publication"]["manifest_file"]
    completed_manifest_path = context.output_dir / manifest_name

    if context.output_dir.exists():
        if not completed_manifest_path.is_file():
            raise TokenizationBuildError(
                f"output directory exists without a complete manifest: {context.output_dir}"
            )
        manifest = validate_completed_corpus(
            completed_manifest_path,
            project_root=context.project_root,
            expected_fingerprint=context.fingerprint,
            verify_identities=True,
            scan_payload=True,
        )
        progress_callback("already complete: identity and all shard checks passed")
        return BuildResult(
            manifest=manifest,
            output_dir=context.output_dir,
            already_complete=True,
        )

    state = _load_or_initialize_state(context, resume=resume)
    for split in ALLOWED_SPLITS:
        _build_split(
            context,
            state,
            split,
            progress=progress_callback,
            interrupt_after_completed_shards=interrupt_after_completed_shards,
        )

    state["status"] = "complete"
    _write_state(context, state)
    manifest = build_manifest(context, state)
    staged_manifest_path = context.staging_dir / manifest_name
    atomic_write_json(staged_manifest_path, manifest)
    validate_completed_corpus(
        staged_manifest_path,
        project_root=context.project_root,
        expected_fingerprint=context.fingerprint,
        verify_identities=True,
        scan_payload=True,
    )
    if context.output_dir.exists():
        raise TokenizationBuildError(
            f"output appeared during validation; refusing to replace it: {context.output_dir}"
        )
    context.staging_dir.rename(context.output_dir)
    progress_callback(
        f"published: {project_relative_path(context.output_dir, context.project_root)}"
    )
    return BuildResult(
        manifest=manifest,
        output_dir=context.output_dir,
        already_complete=False,
    )


def tokenize_corpus(
    config_path: str | Path,
    *,
    project_root: str | Path,
    profile: str = ALLOWED_PROFILE,
    output_dir_override: str | Path | None = None,
    resume: bool = False,
    progress: Callable[[str], None] | None = None,
    interrupt_after_completed_shards: int | None = None,
) -> BuildResult:
    """High-level API used by the CLI and offline integration tests."""

    context = prepare_build_context(
        config_path,
        project_root=project_root,
        profile=profile,
        output_dir_override=output_dir_override,
    )
    return build_tokenized_corpus(
        context,
        resume=resume,
        progress=progress,
        interrupt_after_completed_shards=interrupt_after_completed_shards,
    )
