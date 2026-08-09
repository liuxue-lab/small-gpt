"""Reusable ByteLevel BPE tokenizer utilities for the small-gpt project."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import unicodedata
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import tokenizers
import yaml
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers


EXPECTED_SPECIAL_TOKENS = (
    ("bos", "<bos>", 0),
    ("eos", "<eos>", 1),
    ("pad", "<pad>", 2),
    ("unk", "<unk>", 3),
)
ALLOWED_SPLITS = ("train", "validation", "test")


class TokenizerPipelineError(RuntimeError):
    """Base error for expected tokenizer pipeline failures."""


class TokenizerConfigError(TokenizerPipelineError):
    """Raised when the tokenizer configuration is invalid."""


class CorpusFormatError(TokenizerPipelineError):
    """Raised when an input JSONL record violates the corpus contract."""


class ArtifactError(TokenizerPipelineError):
    """Raised when tokenizer artifacts cannot be validated or published."""


@dataclass
class CorpusStats:
    records: int = 0
    provided_tokens: int = 0
    characters: int = 0
    utf8_bytes: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class ArtifactFile:
    filename: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class ArtifactBundle:
    output_dir: Path
    tokenizer: ArtifactFile
    vocab: ArtifactFile
    merges: ArtifactFile
    config: ArtifactFile

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": self.output_dir.as_posix(),
            "tokenizer": asdict(self.tokenizer),
            "vocab": asdict(self.vocab),
            "merges": asdict(self.merges),
            "config": asdict(self.config),
        }


def _as_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TokenizerConfigError(f"{field} must be a mapping")
    return value


def _as_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TokenizerConfigError(f"{field} must be a non-empty string")
    return value


def _as_positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TokenizerConfigError(f"{field} must be a positive integer")
    return value


def _as_nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TokenizerConfigError(f"{field} must be a non-negative integer")
    return value


def load_tokenizer_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise TokenizerConfigError(f"tokenizer config does not exist: {config_path}")

    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise TokenizerConfigError(f"cannot read tokenizer config {config_path}: {exc}") from exc

    config = dict(_as_mapping(loaded, "config"))
    validate_tokenizer_config(config)
    return config


def validate_tokenizer_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise TokenizerConfigError("schema_version must be 1")

    library = _as_mapping(config.get("library"), "library")
    if library.get("name") != "tokenizers":
        raise TokenizerConfigError("library.name must be 'tokenizers'")
    _as_nonempty_string(library.get("version"), "library.version")

    model = _as_mapping(config.get("model"), "model")
    if model.get("type") != "byte_level_bpe":
        raise TokenizerConfigError("model.type must be 'byte_level_bpe'")
    if _as_positive_int(model.get("vocab_size"), "model.vocab_size") != 16384:
        raise TokenizerConfigError("model.vocab_size must be 16384")
    _as_positive_int(model.get("min_frequency"), "model.min_frequency")
    _as_positive_int(model.get("max_token_length"), "model.max_token_length")
    if model.get("dropout") is not None:
        raise TokenizerConfigError("model.dropout must be null for deterministic encoding")
    if model.get("unk_token") != "<unk>":
        raise TokenizerConfigError("model.unk_token must be '<unk>'")
    if model.get("byte_fallback") is not False:
        raise TokenizerConfigError("model.byte_fallback must be false for ByteLevel BPE")

    normalizer = _as_mapping(config.get("normalizer"), "normalizer")
    if normalizer.get("type") != "nfc":
        raise TokenizerConfigError("normalizer.type must be 'nfc'")

    pre_tokenizer = _as_mapping(config.get("pre_tokenizer"), "pre_tokenizer")
    expected_pre_tokenizer = {
        "type": "byte_level",
        "add_prefix_space": False,
        "use_regex": True,
        "initial_alphabet": "byte_level",
    }
    for key, expected in expected_pre_tokenizer.items():
        if pre_tokenizer.get(key) != expected:
            raise TokenizerConfigError(f"pre_tokenizer.{key} must be {expected!r}")

    decoder = _as_mapping(config.get("decoder"), "decoder")
    if decoder.get("type") != "byte_level":
        raise TokenizerConfigError("decoder.type must be 'byte_level'")

    special_tokens = config.get("special_tokens")
    if not isinstance(special_tokens, list) or len(special_tokens) != 4:
        raise TokenizerConfigError("special_tokens must contain exactly four entries")

    parsed_special_tokens: list[tuple[str, str, int]] = []
    for index, raw_token in enumerate(special_tokens):
        token = _as_mapping(raw_token, f"special_tokens[{index}]")
        name = _as_nonempty_string(token.get("name"), f"special_tokens[{index}].name")
        value = _as_nonempty_string(token.get("token"), f"special_tokens[{index}].token")
        token_id = _as_nonnegative_int(token.get("id"), f"special_tokens[{index}].id")
        parsed_special_tokens.append((name, value, token_id))

    if tuple(parsed_special_tokens) != EXPECTED_SPECIAL_TOKENS:
        raise TokenizerConfigError(
            "special_tokens must be ordered as "
            "<bos>=0, <eos>=1, <pad>=2, <unk>=3"
        )

    source = _as_mapping(config.get("source"), "source")
    for field in (
        "dataset",
        "configuration",
        "revision",
        "corpus_dir",
        "manifest",
        "train_file_pattern",
        "deterministic_order",
    ):
        _as_nonempty_string(source.get(field), f"source.{field}")
    if source.get("split") != "train":
        raise TokenizerConfigError("source.split must be 'train'")
    if source.get("train_file_pattern") != "shards/shard-*/train.jsonl":
        raise TokenizerConfigError(
            "source.train_file_pattern must be 'shards/shard-*/train.jsonl'"
        )
    _as_positive_int(source.get("expected_shards"), "source.expected_shards")
    _as_positive_int(source.get("expected_records"), "source.expected_records")
    _as_positive_int(
        source.get("expected_provided_tokens"),
        "source.expected_provided_tokens",
    )
    if source.get("deterministic_order") != "shard_then_line":
        raise TokenizerConfigError(
            "source.deterministic_order must be 'shard_then_line'"
        )

    training = _as_mapping(config.get("training"), "training")
    if not isinstance(training.get("show_progress"), bool):
        raise TokenizerConfigError("training.show_progress must be boolean")
    if not isinstance(training.get("overwrite_existing"), bool):
        raise TokenizerConfigError("training.overwrite_existing must be boolean")

    artifacts = _as_mapping(config.get("artifacts"), "artifacts")
    _as_nonempty_string(artifacts.get("output_dir"), "artifacts.output_dir")
    for field in ("tokenizer_file", "config_file", "vocab_file", "merges_file"):
        filename = _as_nonempty_string(artifacts.get(field), f"artifacts.{field}")
        if Path(filename).name != filename:
            raise TokenizerConfigError(f"artifacts.{field} must be a filename")

    configured_names = [
        artifacts["tokenizer_file"],
        artifacts["config_file"],
        artifacts["vocab_file"],
        artifacts["merges_file"],
    ]
    if len(set(configured_names)) != len(configured_names):
        raise TokenizerConfigError("artifact filenames must be unique")

    evaluation = _as_mapping(config.get("evaluation"), "evaluation")
    if evaluation.get("append_eos_per_document") is not True:
        raise TokenizerConfigError("evaluation.append_eos_per_document must be true")
    if evaluation.get("add_bos") is not False:
        raise TokenizerConfigError("evaluation.add_bos must be false")
    _as_nonempty_string(evaluation.get("stats_file"), "evaluation.stats_file")

    validation = _as_mapping(config.get("validation"), "validation")
    model_config_paths = validation.get("model_config_paths")
    if not isinstance(model_config_paths, list) or not model_config_paths:
        raise TokenizerConfigError("validation.model_config_paths must be a non-empty list")
    for index, path in enumerate(model_config_paths):
        _as_nonempty_string(path, f"validation.model_config_paths[{index}]")
    if validation.get("require_zero_unknown_tokens") is not True:
        raise TokenizerConfigError(
            "validation.require_zero_unknown_tokens must be true"
        )


def validate_installed_tokenizers_version(config: Mapping[str, Any]) -> None:
    expected = str(_as_mapping(config["library"], "library")["version"])
    actual = tokenizers.__version__
    if actual != expected:
        raise TokenizerConfigError(
            f"tokenizers version mismatch: expected {expected}, found {actual}"
        )


def config_fingerprint(config: Mapping[str, Any]) -> str:
    serialized = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_project_path(project_root: str | Path, configured_path: str | Path) -> Path:
    path = Path(configured_path)
    if path.is_absolute():
        return path.resolve()
    return (Path(project_root).resolve() / path).resolve()


def project_relative_path(path: str | Path, project_root: str | Path) -> str:
    resolved_path = Path(path).resolve()
    resolved_root = Path(project_root).resolve()
    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError:
        return resolved_path.as_posix()


def ordered_special_tokens(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    special_tokens = config["special_tokens"]
    return [dict(item) for item in special_tokens]


def special_token_ids(config: Mapping[str, Any]) -> dict[str, int]:
    return {item["name"]: int(item["id"]) for item in ordered_special_tokens(config)}


def validate_model_configs(
    config: Mapping[str, Any],
    project_root: str | Path,
) -> dict[str, Any]:
    expected_vocab_size = int(config["model"]["vocab_size"])
    expected_special_tokens = [item["token"] for item in ordered_special_tokens(config)]
    results: list[dict[str, Any]] = []

    paths = config["validation"]["model_config_paths"]
    for configured_path in paths:
        path = resolve_project_path(project_root, configured_path)
        if not path.is_file():
            raise TokenizerConfigError(f"model config does not exist: {path}")
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise TokenizerConfigError(f"cannot read model config {path}: {exc}") from exc

        model_config = _as_mapping(loaded, str(path))
        tokenizer_section = _as_mapping(model_config.get("tokenizer"), f"{path}.tokenizer")
        model_section = _as_mapping(model_config.get("model"), f"{path}.model")

        tokenizer_vocab = tokenizer_section.get("vocab_size")
        model_vocab = model_section.get("vocab_size")
        if tokenizer_vocab != expected_vocab_size or model_vocab != expected_vocab_size:
            raise TokenizerConfigError(
                f"vocab size mismatch in {path}: tokenizer={tokenizer_vocab}, "
                f"model={model_vocab}, expected={expected_vocab_size}"
            )

        configured_special_tokens = tokenizer_section.get("special_tokens")
        if configured_special_tokens is not None:
            if configured_special_tokens != expected_special_tokens:
                raise TokenizerConfigError(
                    f"special token mismatch in {path}: {configured_special_tokens!r}"
                )

        context_length = model_section.get("context_length")
        _as_positive_int(context_length, f"{path}.model.context_length")
        results.append(
            {
                "path": project_relative_path(path, project_root),
                "vocab_size": expected_vocab_size,
                "context_length": context_length,
            }
        )

    return {
        "configs": results,
        "max_context_length": max(item["context_length"] for item in results),
    }


def load_source_manifest(
    path: str | Path,
    expected_shards: int,
) -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise TokenizerConfigError(f"source manifest does not exist: {manifest_path}")
    try:
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TokenizerConfigError(f"cannot read source manifest {manifest_path}: {exc}") from exc

    manifest = dict(_as_mapping(loaded, "source manifest"))
    if manifest.get("status") != "complete":
        raise TokenizerConfigError(
            f"source manifest status must be 'complete', found {manifest.get('status')!r}"
        )
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        raise TokenizerConfigError("source manifest shards must be a list")
    if len(shards) != expected_shards:
        raise TokenizerConfigError(
            f"source manifest has {len(shards)} shards; expected {expected_shards}"
        )
    return manifest


def discover_split_files(
    corpus_dir: str | Path,
    split: str,
    expected_shards: int | None = None,
) -> list[Path]:
    if split not in ALLOWED_SPLITS:
        raise TokenizerConfigError(
            f"unsupported split {split!r}; expected one of {ALLOWED_SPLITS}"
        )

    root = Path(corpus_dir)
    shards_dir = root / "shards"
    if not shards_dir.is_dir():
        raise TokenizerConfigError(f"shards directory does not exist: {shards_dir}")

    paths = sorted(
        shards_dir.glob(f"shard-*/{split}.jsonl"),
        key=lambda path: (path.parent.name, path.name),
    )
    if not paths:
        raise TokenizerConfigError(f"no {split} JSONL files found under {shards_dir}")
    if expected_shards is not None and len(paths) != expected_shards:
        raise TokenizerConfigError(
            f"found {len(paths)} {split} files; expected {expected_shards}"
        )

    shard_names = [path.parent.name for path in paths]
    if len(shard_names) != len(set(shard_names)):
        raise TokenizerConfigError(f"duplicate shard directories for split {split}")
    return paths


def iter_jsonl_records(
    paths: Sequence[Path],
    *,
    require_provided_tokens: bool = True,
) -> Iterator[dict[str, Any]]:
    for path in paths:
        try:
            handle = path.open("r", encoding="utf-8")
        except OSError as exc:
            raise CorpusFormatError(f"cannot open corpus file {path}: {exc}") from exc

        with handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    raise CorpusFormatError(f"blank JSONL line at {path}:{line_number}")
                try:
                    raw_record = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise CorpusFormatError(
                        f"invalid JSON at {path}:{line_number}: {exc.msg}"
                    ) from exc
                if not isinstance(raw_record, dict):
                    raise CorpusFormatError(
                        f"record must be an object at {path}:{line_number}"
                    )

                text = raw_record.get("text")
                if not isinstance(text, str) or not text.strip():
                    raise CorpusFormatError(
                        f"text must be a non-empty string at {path}:{line_number}"
                    )

                if require_provided_tokens:
                    provided_tokens = raw_record.get("provided_token_count")
                    if (
                        isinstance(provided_tokens, bool)
                        or not isinstance(provided_tokens, int)
                        or provided_tokens < 0
                    ):
                        raise CorpusFormatError(
                            "provided_token_count must be a non-negative integer "
                            f"at {path}:{line_number}"
                        )

                yield raw_record


def iter_jsonl_texts(
    paths: Sequence[Path],
    stats: CorpusStats,
) -> Iterator[str]:
    for record in iter_jsonl_records(paths, require_provided_tokens=True):
        text = record["text"]
        stats.records += 1
        stats.provided_tokens += int(record["provided_token_count"])
        stats.characters += len(text)
        stats.utf8_bytes += len(text.encode("utf-8"))
        yield text


def build_tokenizer(config: Mapping[str, Any]) -> Tokenizer:
    model_config = config["model"]
    tokenizer = Tokenizer(
        models.BPE(
            dropout=model_config["dropout"],
            unk_token=model_config["unk_token"],
            fuse_unk=False,
            byte_fallback=model_config["byte_fallback"],
        )
    )
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=config["pre_tokenizer"]["add_prefix_space"],
        use_regex=config["pre_tokenizer"]["use_regex"],
    )
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def build_bpe_trainer(
    config: Mapping[str, Any],
    *,
    show_progress: bool | None = None,
) -> trainers.BpeTrainer:
    model_config = config["model"]
    if show_progress is None:
        show_progress = bool(config["training"]["show_progress"])
    return trainers.BpeTrainer(
        vocab_size=int(model_config["vocab_size"]),
        min_frequency=int(model_config["min_frequency"]),
        show_progress=show_progress,
        special_tokens=[item["token"] for item in ordered_special_tokens(config)],
        initial_alphabet=sorted(pre_tokenizers.ByteLevel.alphabet()),
        max_token_length=int(model_config["max_token_length"]),
    )


def train_tokenizer(
    config: Mapping[str, Any],
    texts: Iterable[str],
    *,
    length: int | None = None,
    show_progress: bool | None = None,
) -> Tokenizer:
    tokenizer = build_tokenizer(config)
    trainer = build_bpe_trainer(config, show_progress=show_progress)
    tokenizer.train_from_iterator(texts, trainer=trainer, length=length)
    validate_trained_tokenizer(tokenizer, config)
    return tokenizer


def validate_trained_tokenizer(
    tokenizer: Tokenizer,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    expected_vocab_size = int(config["model"]["vocab_size"])
    actual_vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    if actual_vocab_size != expected_vocab_size:
        raise ArtifactError(
            f"trained vocab size is {actual_vocab_size}; expected {expected_vocab_size}"
        )

    vocab = tokenizer.get_vocab(with_added_tokens=True)
    vocab_ids = sorted(vocab.values())
    if vocab_ids != list(range(expected_vocab_size)):
        raise ArtifactError("trained vocabulary IDs are not contiguous")

    actual_special_ids: dict[str, int] = {}
    for item in ordered_special_tokens(config):
        actual_id = tokenizer.token_to_id(item["token"])
        expected_id = int(item["id"])
        if actual_id != expected_id:
            raise ArtifactError(
                f"special token {item['token']} has ID {actual_id}; expected {expected_id}"
            )
        actual_special_ids[item["name"]] = actual_id

    sample_ids = encode_text(tokenizer, "Tokenizer validation: café 🤖")
    if not sample_ids:
        raise ArtifactError("trained tokenizer produced no IDs for validation text")
    if min(sample_ids) < 0 or max(sample_ids) >= expected_vocab_size:
        raise ArtifactError("trained tokenizer produced an out-of-range ID")

    return {
        "vocab_size": actual_vocab_size,
        "special_token_ids": actual_special_ids,
    }


def encode_text(tokenizer: Tokenizer, text: str) -> list[int]:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return list(tokenizer.encode(text, add_special_tokens=False).ids)


def encode_document(tokenizer: Tokenizer, text: str, eos_id: int) -> list[int]:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text.strip():
        raise CorpusFormatError("cannot encode an empty document")
    ids = encode_text(tokenizer, text)
    ids.append(eos_id)
    return ids


def normalized_round_trip(tokenizer: Tokenizer, text: str) -> str:
    ids = encode_text(tokenizer, text)
    return tokenizer.decode(ids, skip_special_tokens=True)


def load_tokenizer(path: str | Path) -> Tokenizer:
    tokenizer_path = Path(path)
    if not tokenizer_path.is_file():
        raise ArtifactError(f"tokenizer artifact does not exist: {tokenizer_path}")
    try:
        return Tokenizer.from_file(str(tokenizer_path))
    except Exception as exc:  # tokenizers exposes multiple native exception types
        raise ArtifactError(f"cannot load tokenizer artifact {tokenizer_path}: {exc}") from exc


def _artifact_file(path: Path) -> ArtifactFile:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ArtifactError(f"artifact is missing or empty: {path}")
    return ArtifactFile(
        filename=path.name,
        size_bytes=path.stat().st_size,
        sha256=sha256_file(path),
    )


def _rename_model_artifact(generated: Sequence[str], suffix: str, target: Path) -> None:
    candidates = [Path(path) for path in generated if Path(path).name.endswith(suffix)]
    if len(candidates) != 1:
        raise ArtifactError(
            f"expected one generated *{suffix} artifact, found {len(candidates)}"
        )
    source = candidates[0]
    if source.resolve() != target.resolve():
        source.replace(target)


def _publish_directory(temp_dir: Path, output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and not overwrite:
        raise ArtifactError(
            f"artifact directory already exists: {output_dir}; "
            "use --overwrite only after reviewing the existing artifacts"
        )

    backup_dir: Path | None = None
    if output_dir.exists():
        backup_dir = output_dir.with_name(
            f".{output_dir.name}.backup-{uuid.uuid4().hex}"
        )
        output_dir.rename(backup_dir)

    try:
        temp_dir.rename(output_dir)
    except Exception:
        if backup_dir is not None and backup_dir.exists() and not output_dir.exists():
            backup_dir.rename(output_dir)
        raise
    else:
        if backup_dir is not None and backup_dir.exists():
            shutil.rmtree(backup_dir)


def save_tokenizer_artifacts(
    tokenizer: Tokenizer,
    config: Mapping[str, Any],
    output_dir: str | Path,
    *,
    project_root: str | Path,
    training_metadata: Mapping[str, Any],
    validation_samples: Sequence[str],
    overwrite: bool = False,
) -> ArtifactBundle:
    validate_trained_tokenizer(tokenizer, config)
    output = Path(output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise ArtifactError(
            f"artifact directory already exists: {output}; refusing to overwrite"
        )

    artifacts_config = config["artifacts"]
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=str(output.parent))
    )
    try:
        tokenizer_path = temp_dir / artifacts_config["tokenizer_file"]
        config_path = temp_dir / artifacts_config["config_file"]
        vocab_path = temp_dir / artifacts_config["vocab_file"]
        merges_path = temp_dir / artifacts_config["merges_file"]

        tokenizer.save(str(tokenizer_path), pretty=True)
        generated = tokenizer.model.save(str(temp_dir))
        _rename_model_artifact(generated, "vocab.json", vocab_path)
        _rename_model_artifact(generated, "merges.txt", merges_path)

        tokenizer_file = _artifact_file(tokenizer_path)
        vocab_file = _artifact_file(vocab_path)
        merges_file = _artifact_file(merges_path)

        metadata = {
            "schema_version": 1,
            "library": dict(config["library"]),
            "tokenizer": {
                "type": config["model"]["type"],
                "vocab_size": tokenizer.get_vocab_size(with_added_tokens=True),
                "normalizer": config["normalizer"]["type"],
                "pre_tokenizer": dict(config["pre_tokenizer"]),
                "decoder": dict(config["decoder"]),
                "special_tokens": ordered_special_tokens(config),
                "byte_fallback": config["model"]["byte_fallback"],
            },
            "document_boundaries": {
                "append_eos_per_document": config["evaluation"][
                    "append_eos_per_document"
                ],
                "add_bos": config["evaluation"]["add_bos"],
            },
            "config_fingerprint": config_fingerprint(config),
            "training": dict(training_metadata),
            "artifacts": {
                "tokenizer": asdict(tokenizer_file),
                "vocab": asdict(vocab_file),
                "merges": asdict(merges_file),
            },
        }
        config_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        config_file = _artifact_file(config_path)

        reloaded = load_tokenizer(tokenizer_path)
        validate_trained_tokenizer(reloaded, config)
        for sample in validation_samples:
            original_ids = encode_text(tokenizer, sample)
            reloaded_ids = encode_text(reloaded, sample)
            if original_ids != reloaded_ids:
                raise ArtifactError(
                    f"save/reload IDs differ for validation sample {sample!r}"
                )
            expected_text = unicodedata.normalize("NFC", sample)
            decoded = reloaded.decode(reloaded_ids, skip_special_tokens=True)
            if decoded != expected_text:
                raise ArtifactError(
                    "NFC round-trip mismatch for validation sample "
                    f"{sample!r}: decoded {decoded!r}"
                )

        _publish_directory(temp_dir, output, overwrite)
        return ArtifactBundle(
            output_dir=output,
            tokenizer=tokenizer_file,
            vocab=vocab_file,
            merges=merges_file,
            config=config_file,
        )
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def percentile(values: Sequence[int], probability: float) -> float | None:
    if not values:
        return None
    if probability < 0.0 or probability > 1.0:
        raise ValueError("probability must be between 0 and 1")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def evaluate_split(
    tokenizer: Tokenizer,
    paths: Sequence[Path],
    *,
    split: str,
    config: Mapping[str, Any],
    context_length: int,
) -> dict[str, Any]:
    vocab_size = int(config["model"]["vocab_size"])
    ids_by_name = special_token_ids(config)
    eos_id = ids_by_name["eos"]
    unk_id = ids_by_name["unk"]

    records = 0
    characters = 0
    utf8_bytes = 0
    provided_tokens = 0
    bpe_tokens_without_eos = 0
    unknown_tokens = 0
    minimum_id: int | None = None
    maximum_id: int | None = None
    document_lengths: list[int] = []

    for record in iter_jsonl_records(paths, require_provided_tokens=True):
        text = record["text"]
        raw_ids = encode_text(tokenizer, text)
        if not raw_ids:
            raise ArtifactError(f"{split} document encoded to an empty ID sequence")
        if min(raw_ids) < 0 or max(raw_ids) >= vocab_size:
            raise ArtifactError(f"{split} document produced an out-of-range token ID")

        document_ids = raw_ids + [eos_id]
        records += 1
        characters += len(text)
        utf8_bytes += len(text.encode("utf-8"))
        provided_tokens += int(record["provided_token_count"])
        bpe_tokens_without_eos += len(raw_ids)
        unknown_tokens += raw_ids.count(unk_id)
        document_lengths.append(len(document_ids))

        local_min = min(document_ids)
        local_max = max(document_ids)
        minimum_id = local_min if minimum_id is None else min(minimum_id, local_min)
        maximum_id = local_max if maximum_id is None else max(maximum_id, local_max)

    eos_tokens = records
    total_model_tokens = bpe_tokens_without_eos + eos_tokens
    over_context = sum(length > context_length for length in document_lengths)

    return {
        "split": split,
        "files": len(paths),
        "records": records,
        "characters": characters,
        "utf8_bytes": utf8_bytes,
        "provided_tokens": provided_tokens,
        "bpe_tokens_without_eos": bpe_tokens_without_eos,
        "eos_tokens": eos_tokens,
        "total_model_tokens": total_model_tokens,
        "unknown_tokens": unknown_tokens,
        "minimum_token_id": minimum_id,
        "maximum_token_id": maximum_id,
        "document_tokens": {
            "minimum": min(document_lengths) if document_lengths else None,
            "maximum": max(document_lengths) if document_lengths else None,
            "mean": _ratio(sum(document_lengths), len(document_lengths)),
            "p50": percentile(document_lengths, 0.50),
            "p95": percentile(document_lengths, 0.95),
            "p99": percentile(document_lengths, 0.99),
        },
        "characters_per_bpe_token": _ratio(characters, bpe_tokens_without_eos),
        "bytes_per_bpe_token": _ratio(utf8_bytes, bpe_tokens_without_eos),
        "bpe_to_provided_token_ratio": _ratio(
            bpe_tokens_without_eos,
            provided_tokens,
        ),
        "model_to_provided_token_ratio": _ratio(
            total_model_tokens,
            provided_tokens,
        ),
        "context_length": context_length,
        "documents_over_context": over_context,
        "documents_over_context_ratio": _ratio(over_context, records),
    }


def combine_split_statistics(split_stats: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    additive_fields = (
        "files",
        "records",
        "characters",
        "utf8_bytes",
        "provided_tokens",
        "bpe_tokens_without_eos",
        "eos_tokens",
        "total_model_tokens",
        "unknown_tokens",
        "documents_over_context",
    )
    totals = {
        field: sum(int(stats[field]) for stats in split_stats.values())
        for field in additive_fields
    }
    totals["characters_per_bpe_token"] = _ratio(
        totals["characters"],
        totals["bpe_tokens_without_eos"],
    )
    totals["bytes_per_bpe_token"] = _ratio(
        totals["utf8_bytes"],
        totals["bpe_tokens_without_eos"],
    )
    totals["bpe_to_provided_token_ratio"] = _ratio(
        totals["bpe_tokens_without_eos"],
        totals["provided_tokens"],
    )
    totals["model_to_provided_token_ratio"] = _ratio(
        totals["total_model_tokens"],
        totals["provided_tokens"],
    )
    totals["documents_over_context_ratio"] = _ratio(
        totals["documents_over_context"],
        totals["records"],
    )
    return totals


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(
        f".{output_path.name}.tmp-{uuid.uuid4().hex}"
    )
    try:
        with temp_path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, output_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise
