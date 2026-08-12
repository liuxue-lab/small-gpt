from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, TextIO
from urllib.parse import quote, urlsplit

import yaml

try:
    from scripts.prepare_data import (
        CleaningConfig,
        SPLIT_NAMES,
        assign_split,
        clean_record,
        file_sha256,
        text_sha256,
    )
except ModuleNotFoundError:  # Support direct execution from the scripts directory.
    from prepare_data import (  # type: ignore[no-redef]
        CleaningConfig,
        SPLIT_NAMES,
        assign_split,
        clean_record,
        file_sha256,
        text_sha256,
    )


SCHEMA_VERSION = 1
SHARD_DIRECTORY_RE = re.compile(r"^shard-(\d{5})$")
TEMPORARY_SHARD_RE = re.compile(r"^\.shard-(\d{5})\.tmp$")


class CorpusIntegrityError(RuntimeError):
    """Raised when completed corpus files do not match their metadata."""


class ControlledInterruption(RuntimeError):
    """Test-only interruption used to verify deterministic recovery."""


class SourceExhaustedError(RuntimeError):
    """Raised when the source ends before the requested token budget is met."""


@dataclass(frozen=True)
class CorpusRunConfig:
    dataset_name: str
    dataset_configuration: str
    dataset_split: str
    dataset_revision: str
    streaming: bool
    cleaning: CleaningConfig
    profile: str
    target_provided_tokens: int
    shard_target_provided_tokens: int
    estimated_shards: int
    output_dir: Path
    manifest_filename: str
    state_filename: str
    output_format: str
    encoding: str
    save_raw_records: bool
    source_files: tuple[str, ...] = ()

    def validate(self) -> None:
        self.cleaning.validate()

        if not self.dataset_name:
            raise ValueError("dataset name must not be empty")
        if not self.dataset_configuration:
            raise ValueError("dataset configuration must not be empty")
        if not self.dataset_split:
            raise ValueError("dataset split must not be empty")
        if not self.dataset_revision:
            raise ValueError("dataset revision must not be empty")
        if self.streaming is not True:
            raise ValueError("the formal corpus pipeline requires streaming=True")
        if self.target_provided_tokens <= 0:
            raise ValueError("target_provided_tokens must be positive")
        if self.shard_target_provided_tokens <= 0:
            raise ValueError("shard_target_provided_tokens must be positive")
        if self.shard_target_provided_tokens > self.target_provided_tokens:
            raise ValueError(
                "shard_target_provided_tokens must not exceed the run target"
            )
        if self.output_format != "jsonl":
            raise ValueError("only JSONL output is supported")
        if self.encoding.lower().replace("_", "-") != "utf-8":
            raise ValueError("only UTF-8 output is supported")
        if self.save_raw_records:
            raise ValueError("formal collection must not save raw source records")
        if not self.manifest_filename.endswith(".json"):
            raise ValueError("manifest_filename must be a JSON filename")
        if not self.state_filename.endswith(".json"):
            raise ValueError("state_filename must be a JSON filename")
        if self.profile == "full" and not self.source_files:
            raise ValueError("the Full profile requires explicit source_files")
        if len(self.source_files) != len(set(self.source_files)):
            raise ValueError("source_files must not contain duplicates")
        for index, source_file in enumerate(self.source_files):
            path = PurePosixPath(source_file)
            if (
                not source_file
                or path.is_absolute()
                or ".." in path.parts
                or path.as_posix() != source_file
                or path.suffix != ".parquet"
            ):
                raise ValueError(
                    f"source_files[{index}] must be a normalized relative "
                    f"Parquet path: {source_file!r}"
                )


@dataclass
class RecoveryState:
    completed_shards: list[dict[str, Any]]
    seen_text_hashes: set[str]
    source_records_seen: int
    kept_records: int
    kept_provided_tokens: int
    split_records: Counter[str]
    split_provided_tokens: Counter[str]
    removal_counts: Counter[str]


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping")
    return value


def load_run_config(
    config_path: Path,
    profile: str,
    *,
    output_dir_override: Path | None = None,
    target_provided_tokens_override: int | None = None,
    shard_target_provided_tokens_override: int | None = None,
) -> CorpusRunConfig:
    with config_path.open("r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file)

    root = _require_mapping(raw_config, "configuration")
    dataset = _require_mapping(root.get("dataset"), "dataset")
    cleaning = _require_mapping(root.get("cleaning"), "cleaning")
    splits = _require_mapping(root.get("splits"), "splits")
    output = _require_mapping(root.get("output"), "output")
    profiles = _require_mapping(root.get("profiles"), "profiles")

    if profile not in profiles:
        raise ValueError(
            f"unknown profile {profile!r}; expected one of {sorted(profiles)}"
        )
    selected_profile = _require_mapping(profiles[profile], f"profiles.{profile}")

    raw_source_files = selected_profile.get("source_files", [])
    if not isinstance(raw_source_files, list) or any(
        not isinstance(item, str) for item in raw_source_files
    ):
        raise TypeError(f"profiles.{profile}.source_files must be a list of strings")
    source_files = tuple(raw_source_files)

    cleaning_config = CleaningConfig(
        min_characters=int(cleaning["min_characters"]),
        min_language_score=float(cleaning["min_language_score"]),
        min_quality_score=int(cleaning["min_quality_score"]),
        seed=int(splits["seed"]),
        train_ratio=float(splits["train_ratio"]),
        validation_ratio=float(splits["validation_ratio"]),
    )

    configured_test_ratio = float(splits["test_ratio"])
    if not math.isclose(
        cleaning_config.test_ratio,
        configured_test_ratio,
        abs_tol=1e-9,
    ):
        raise ValueError("train, validation, and test ratios must sum to one")

    target = (
        int(target_provided_tokens_override)
        if target_provided_tokens_override is not None
        else int(selected_profile["target_provided_tokens"])
    )
    shard_target = (
        int(shard_target_provided_tokens_override)
        if shard_target_provided_tokens_override is not None
        else int(selected_profile["shard_target_provided_tokens"])
    )

    run_config = CorpusRunConfig(
        dataset_name=str(dataset["name"]),
        dataset_configuration=str(dataset["configuration"]),
        dataset_split=str(dataset["split"]),
        dataset_revision=str(dataset["revision"]),
        streaming=bool(dataset["streaming"]),
        cleaning=cleaning_config,
        profile=profile,
        target_provided_tokens=target,
        shard_target_provided_tokens=shard_target,
        estimated_shards=math.ceil(target / shard_target),
        output_dir=(
            output_dir_override
            if output_dir_override is not None
            else Path(str(selected_profile.get("output_dir", output["directory"])))
        ),
        manifest_filename=str(output["manifest_filename"]),
        state_filename=str(output["state_filename"]),
        output_format=str(output["format"]),
        encoding=str(output["encoding"]),
        save_raw_records=bool(output["save_raw_records"]),
        source_files=source_files,
    )
    run_config.validate()

    if (
        target_provided_tokens_override is None
        and shard_target_provided_tokens_override is None
        and int(selected_profile["estimated_shards"])
        != run_config.estimated_shards
    ):
        raise ValueError(
            f"profiles.{profile}.estimated_shards does not match the budgets"
        )

    return run_config


def config_fingerprint(config: CorpusRunConfig) -> str:
    dataset_identity: dict[str, Any] = {
        "name": config.dataset_name,
        "configuration": config.dataset_configuration,
        "split": config.dataset_split,
        "revision": config.dataset_revision,
        "streaming": config.streaming,
    }
    if config.source_files:
        dataset_identity["source_files"] = list(config.source_files)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset_identity,
        "cleaning": asdict(config.cleaning),
        "profile": config.profile,
        "target_provided_tokens": config.target_provided_tokens,
        "shard_target_provided_tokens": config.shard_target_provided_tokens,
        "output_format": config.output_format,
        "encoding": config.encoding,
        "save_raw_records": config.save_raw_records,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"

    if path.is_file() and path.read_text(encoding="utf-8") == serialized:
        return

    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as file:
            file.write(serialized)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CorpusIntegrityError(f"cannot read valid JSON from {path}") from error
    return _require_mapping(payload, str(path))


def _empty_recovery_state() -> RecoveryState:
    return RecoveryState(
        completed_shards=[],
        seen_text_hashes=set(),
        source_records_seen=0,
        kept_records=0,
        kept_provided_tokens=0,
        split_records=Counter(),
        split_provided_tokens=Counter(),
        removal_counts=Counter(),
    )


def _cleanup_temporary_shards(shards_dir: Path) -> None:
    if not shards_dir.exists():
        return
    for path in shards_dir.iterdir():
        if path.is_dir() and TEMPORARY_SHARD_RE.fullmatch(path.name):
            shutil.rmtree(path)


def _verify_split_file(
    path: Path,
    expected: dict[str, Any],
    split_name: str,
    config: CorpusRunConfig,
    seen_hashes: set[str],
) -> tuple[int, int]:
    if not path.is_file():
        raise CorpusIntegrityError(f"missing completed shard file: {path}")
    if path.stat().st_size != int(expected["bytes"]):
        raise CorpusIntegrityError(f"file size mismatch: {path}")
    if file_sha256(path) != str(expected["sha256"]):
        raise CorpusIntegrityError(f"SHA-256 mismatch: {path}")

    records = 0
    provided_tokens = 0
    with path.open("r", encoding=config.encoding) as file:
        for line_number, line in enumerate(file, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise CorpusIntegrityError(
                    f"invalid JSON in {path} on line {line_number}"
                ) from error
            if not isinstance(record, dict):
                raise CorpusIntegrityError(
                    f"non-object JSON in {path} on line {line_number}"
                )

            text = record.get("text")
            digest = record.get("text_sha256")
            token_count = record.get("provided_token_count")
            if not isinstance(text, str) or not isinstance(digest, str):
                raise CorpusIntegrityError(f"missing text identity in {path}")
            if digest != text_sha256(text):
                raise CorpusIntegrityError(f"text hash mismatch in {path}")
            if digest in seen_hashes:
                raise CorpusIntegrityError(
                    f"duplicate text hash across completed shards: {digest}"
                )
            if assign_split(digest, config.cleaning) != split_name:
                raise CorpusIntegrityError(f"split mismatch in {path}")
            if not isinstance(token_count, int) or token_count <= 0:
                raise CorpusIntegrityError(f"invalid provided token count in {path}")

            seen_hashes.add(digest)
            records += 1
            provided_tokens += token_count

    if records != int(expected["records"]):
        raise CorpusIntegrityError(f"record count mismatch: {path}")
    if provided_tokens != int(expected["provided_tokens"]):
        raise CorpusIntegrityError(f"provided-token count mismatch: {path}")
    return records, provided_tokens


def recover_completed_shards(config: CorpusRunConfig) -> RecoveryState:
    output_dir = config.output_dir
    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_temporary_shards(shards_dir)

    indexed_directories: list[tuple[int, Path]] = []
    for path in shards_dir.iterdir():
        if not path.is_dir():
            continue
        match = SHARD_DIRECTORY_RE.fullmatch(path.name)
        if match:
            indexed_directories.append((int(match.group(1)), path))
    indexed_directories.sort()

    actual_indexes = [index for index, _ in indexed_directories]
    expected_indexes = list(range(len(indexed_directories)))
    if actual_indexes != expected_indexes:
        raise CorpusIntegrityError(
            "completed shard indexes must be contiguous and start at zero"
        )

    recovery = _empty_recovery_state()
    expected_fingerprint = config_fingerprint(config)
    previous_source_end = 0

    for shard_index, shard_dir in indexed_directories:
        metadata = _read_json(shard_dir / "metadata.json")
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise CorpusIntegrityError(f"schema mismatch in {shard_dir}")
        if metadata.get("config_fingerprint") != expected_fingerprint:
            raise CorpusIntegrityError(
                f"configuration fingerprint mismatch in {shard_dir}"
            )
        if metadata.get("shard_index") != shard_index:
            raise CorpusIntegrityError(f"shard index mismatch in {shard_dir}")

        source_start = int(metadata["source_records_start_exclusive"])
        source_end = int(metadata["source_records_end_inclusive"])
        if source_start != previous_source_end or source_end <= source_start:
            raise CorpusIntegrityError(
                f"source record range is not contiguous in {shard_dir}"
            )

        files = _require_mapping(metadata.get("files"), "metadata.files")
        if set(files) != set(SPLIT_NAMES):
            raise CorpusIntegrityError(f"split file set mismatch in {shard_dir}")

        shard_records = 0
        shard_tokens = 0
        for split_name in SPLIT_NAMES:
            file_metadata = _require_mapping(
                files[split_name],
                f"metadata.files.{split_name}",
            )
            actual_records, actual_tokens = _verify_split_file(
                shard_dir / f"{split_name}.jsonl",
                file_metadata,
                split_name,
                config,
                recovery.seen_text_hashes,
            )
            shard_records += actual_records
            shard_tokens += actual_tokens
            recovery.split_records[split_name] += actual_records
            recovery.split_provided_tokens[split_name] += actual_tokens

        if shard_records != int(metadata["kept_records"]):
            raise CorpusIntegrityError(f"kept-record mismatch in {shard_dir}")
        if shard_tokens != int(metadata["kept_provided_tokens"]):
            raise CorpusIntegrityError(f"kept-token mismatch in {shard_dir}")

        removals = _require_mapping(
            metadata.get("removal_counts", {}),
            "metadata.removal_counts",
        )
        removal_total = sum(int(value) for value in removals.values())
        input_records = source_end - source_start
        if shard_records + removal_total != input_records:
            raise CorpusIntegrityError(f"statistics do not conserve in {shard_dir}")

        recovery.removal_counts.update(
            {str(key): int(value) for key, value in removals.items()}
        )
        recovery.kept_records += shard_records
        recovery.kept_provided_tokens += shard_tokens
        recovery.source_records_seen = source_end
        recovery.completed_shards.append(metadata)
        previous_source_end = source_end

    return recovery


def _statistics_payload(recovery: RecoveryState) -> dict[str, Any]:
    input_records = recovery.source_records_seen
    retention_rate = (
        round(recovery.kept_records / input_records, 8)
        if input_records
        else 0.0
    )
    return {
        "input_records": input_records,
        "kept_records": recovery.kept_records,
        "kept_provided_tokens": recovery.kept_provided_tokens,
        "removal_counts": dict(sorted(recovery.removal_counts.items())),
        "split_records": {
            name: recovery.split_records[name] for name in SPLIT_NAMES
        },
        "split_provided_tokens": {
            name: recovery.split_provided_tokens[name] for name in SPLIT_NAMES
        },
        "retention_rate": retention_rate,
    }


def _manifest_payload(
    config: CorpusRunConfig,
    recovery: RecoveryState,
    status: str,
) -> dict[str, Any]:
    dataset_identity: dict[str, Any] = {
        "name": config.dataset_name,
        "configuration": config.dataset_configuration,
        "split": config.dataset_split,
        "revision": config.dataset_revision,
        "streaming": config.streaming,
    }
    if config.source_files:
        dataset_identity["source_files"] = list(config.source_files)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "config_fingerprint": config_fingerprint(config),
        "dataset": dataset_identity,
        "profile": {
            "name": config.profile,
            "target_provided_tokens": config.target_provided_tokens,
            "shard_target_provided_tokens": (
                config.shard_target_provided_tokens
            ),
            "estimated_shards": config.estimated_shards,
        },
        "output": {
            "format": config.output_format,
            "encoding": config.encoding,
            "save_raw_records": config.save_raw_records,
        },
        "statistics": _statistics_payload(recovery),
        "shards": recovery.completed_shards,
    }


def _state_payload(
    config: CorpusRunConfig,
    recovery: RecoveryState,
    status: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "config_fingerprint": config_fingerprint(config),
        "next_shard_index": len(recovery.completed_shards),
        "source_records_seen": recovery.source_records_seen,
        "kept_records": recovery.kept_records,
        "kept_provided_tokens": recovery.kept_provided_tokens,
    }


def _write_run_documents(
    config: CorpusRunConfig,
    recovery: RecoveryState,
    status: str,
) -> dict[str, Any]:
    manifest = _manifest_payload(config, recovery, status)
    atomic_write_json(config.output_dir / config.manifest_filename, manifest)
    atomic_write_json(
        config.output_dir / config.state_filename,
        _state_payload(config, recovery, status),
    )
    return manifest


def _flush_and_close_writers(writers: dict[str, TextIO]) -> None:
    first_error: Exception | None = None
    for writer in writers.values():
        try:
            writer.flush()
            os.fsync(writer.fileno())
            writer.close()
        except Exception as error:  # pragma: no cover - rare filesystem failure.
            if first_error is None:
                first_error = error
    if first_error is not None:
        raise first_error


def _finalize_shard(
    config: CorpusRunConfig,
    shard_index: int,
    temporary_dir: Path,
    source_start: int,
    source_end: int,
    split_records: Counter[str],
    split_tokens: Counter[str],
    removal_counts: Counter[str],
) -> dict[str, Any]:
    shard_name = f"shard-{shard_index:05d}"
    final_dir = temporary_dir.parent / shard_name
    if final_dir.exists():
        raise CorpusIntegrityError(f"refusing to overwrite {final_dir}")

    files: dict[str, dict[str, Any]] = {}
    for split_name in SPLIT_NAMES:
        path = temporary_dir / f"{split_name}.jsonl"
        files[split_name] = {
            "path": f"shards/{shard_name}/{split_name}.jsonl",
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
            "records": split_records[split_name],
            "provided_tokens": split_tokens[split_name],
        }

    kept_records = sum(split_records.values())
    kept_tokens = sum(split_tokens.values())
    input_records = source_end - source_start
    if kept_records + sum(removal_counts.values()) != input_records:
        raise RuntimeError("shard statistics do not conserve")

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "config_fingerprint": config_fingerprint(config),
        "shard_index": shard_index,
        "source_records_start_exclusive": source_start,
        "source_records_end_inclusive": source_end,
        "input_records": input_records,
        "kept_records": kept_records,
        "kept_provided_tokens": kept_tokens,
        "split_records": {
            name: split_records[name] for name in SPLIT_NAMES
        },
        "split_provided_tokens": {
            name: split_tokens[name] for name in SPLIT_NAMES
        },
        "removal_counts": dict(sorted(removal_counts.items())),
        "files": files,
    }
    atomic_write_json(temporary_dir / "metadata.json", metadata)
    temporary_dir.rename(final_dir)
    return metadata


def _apply_completed_shard(
    recovery: RecoveryState,
    metadata: dict[str, Any],
) -> None:
    recovery.completed_shards.append(metadata)
    recovery.source_records_seen = int(metadata["source_records_end_inclusive"])
    recovery.kept_records += int(metadata["kept_records"])
    recovery.kept_provided_tokens += int(metadata["kept_provided_tokens"])
    recovery.removal_counts.update(metadata["removal_counts"])
    recovery.split_records.update(metadata["split_records"])
    recovery.split_provided_tokens.update(metadata["split_provided_tokens"])


def _skip_committed_records(
    records: Iterable[dict[str, Any]],
    count: int,
) -> Any:
    iterator = iter(records)
    for index in range(count):
        try:
            next(iterator)
        except StopIteration as error:
            raise SourceExhaustedError(
                "source ended while seeking to committed record "
                f"{index + 1} of {count}"
            ) from error
    return iterator


def build_corpus(
    records: Iterable[dict[str, Any]],
    config: CorpusRunConfig,
    *,
    interrupt_after_source_records: int | None = None,
) -> dict[str, Any]:
    config.validate()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = config.output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    recovery = recover_completed_shards(config)
    initial_status = (
        "complete"
        if recovery.kept_provided_tokens >= config.target_provided_tokens
        else "running"
    )
    manifest = _write_run_documents(config, recovery, initial_status)
    if initial_status == "complete":
        return manifest

    iterator = _skip_committed_records(records, recovery.source_records_seen)
    source_index = recovery.source_records_seen

    while recovery.kept_provided_tokens < config.target_provided_tokens:
        shard_index = len(recovery.completed_shards)
        shard_name = f"shard-{shard_index:05d}"
        temporary_dir = shards_dir / f".{shard_name}.tmp"
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        temporary_dir.mkdir(parents=False)

        writers = {
            name: (temporary_dir / f"{name}.jsonl").open(
                "w",
                encoding=config.encoding,
                newline="\n",
            )
            for name in SPLIT_NAMES
        }
        source_start = source_index
        split_records: Counter[str] = Counter()
        split_tokens: Counter[str] = Counter()
        removal_counts: Counter[str] = Counter()
        current_hashes: set[str] = set()
        shard_tokens = 0
        source_exhausted = False

        try:
            while (
                shard_tokens < config.shard_target_provided_tokens
                and recovery.kept_provided_tokens + shard_tokens
                < config.target_provided_tokens
            ):
                try:
                    record = next(iterator)
                except StopIteration:
                    source_exhausted = True
                    break

                source_index += 1
                cleaned, removal_reason = clean_record(record, config.cleaning)
                if cleaned is None:
                    removal_counts[str(removal_reason)] += 1
                else:
                    provided_tokens = cleaned.get("provided_token_count")
                    if (
                        not isinstance(provided_tokens, int)
                        or provided_tokens <= 0
                    ):
                        removal_counts[
                            "missing_or_invalid_provided_token_count"
                        ] += 1
                    else:
                        digest = str(cleaned["text_sha256"])
                        if (
                            digest in recovery.seen_text_hashes
                            or digest in current_hashes
                        ):
                            removal_counts["exact_duplicate"] += 1
                        else:
                            split_name = assign_split(digest, config.cleaning)
                            writers[split_name].write(
                                json.dumps(cleaned, ensure_ascii=False) + "\n"
                            )
                            current_hashes.add(digest)
                            split_records[split_name] += 1
                            split_tokens[split_name] += provided_tokens
                            shard_tokens += provided_tokens

                if (
                    interrupt_after_source_records is not None
                    and source_index >= interrupt_after_source_records
                ):
                    raise ControlledInterruption(
                        f"controlled interruption after source record {source_index}"
                    )
        finally:
            _flush_and_close_writers(writers)

        input_records = source_index - source_start
        if input_records == 0:
            shutil.rmtree(temporary_dir)
            manifest = _write_run_documents(config, recovery, "source_exhausted")
            raise SourceExhaustedError(
                "source ended before the requested token budget was reached"
            )

        metadata = _finalize_shard(
            config,
            shard_index,
            temporary_dir,
            source_start,
            source_index,
            split_records,
            split_tokens,
            removal_counts,
        )
        recovery.seen_text_hashes.update(current_hashes)
        _apply_completed_shard(recovery, metadata)

        status = (
            "complete"
            if recovery.kept_provided_tokens >= config.target_provided_tokens
            else "source_exhausted"
            if source_exhausted
            else "running"
        )
        manifest = _write_run_documents(config, recovery, status)

        if source_exhausted and status != "complete":
            raise SourceExhaustedError(
                "source ended before the requested token budget was reached"
            )

    return manifest


def _explicit_source_urls(config: CorpusRunConfig) -> list[str]:
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    parsed_endpoint = urlsplit(endpoint)
    if parsed_endpoint.scheme not in {"http", "https"} or not parsed_endpoint.netloc:
        raise ValueError(f"HF_ENDPOINT is not a valid HTTP(S) endpoint: {endpoint!r}")
    dataset = quote(config.dataset_name, safe="/")
    revision = quote(config.dataset_revision, safe="")
    return [
        f"{endpoint}/datasets/{dataset}/resolve/{revision}/"
        f"{quote(source_file, safe='/')}"
        for source_file in config.source_files
    ]


def open_fineweb_edu_stream(config: CorpusRunConfig) -> Iterable[dict[str, Any]]:
    from datasets import load_dataset

    if config.source_files:
        return load_dataset(
            "parquet",
            data_files={config.dataset_split: _explicit_source_urls(config)},
            split=config.dataset_split,
            streaming=config.streaming,
        )
    return load_dataset(
        config.dataset_name,
        name=config.dataset_configuration,
        split=config.dataset_split,
        revision=config.dataset_revision,
        streaming=config.streaming,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream, clean, deduplicate, split, and shard a bounded "
            "FineWeb-Edu corpus."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data_fineweb_edu.yaml"),
    )
    parser.add_argument("--profile", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--target-provided-tokens", type=int)
    parser.add_argument("--shard-target-provided-tokens", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_run_config(
        args.config,
        args.profile,
        output_dir_override=args.output_dir,
        target_provided_tokens_override=args.target_provided_tokens,
        shard_target_provided_tokens_override=(
            args.shard_target_provided_tokens
        ),
    )
    print(
        f"Dataset revision : {config.dataset_revision}\n"
        f"Profile          : {config.profile}\n"
        f"Target tokens    : {config.target_provided_tokens}\n"
        f"Shard target     : {config.shard_target_provided_tokens}\n"
        f"Explicit files   : {len(config.source_files)}\n"
        f"Output directory : {config.output_dir}"
    )
    manifest = build_corpus(open_fineweb_edu_stream(config), config)
    print(json.dumps(manifest["statistics"], ensure_ascii=False, indent=2))
    print(f"Manifest saved to: {config.output_dir / config.manifest_filename}")


if __name__ == "__main__":
    main()
