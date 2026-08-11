from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


class GPTConfigError(ValueError):
    """Raised when a GPT model configuration violates the frozen contract."""


def _is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


@dataclass(frozen=True, slots=True)
class GPTConfig:
    architecture: str
    n_layer: int
    n_head: int
    n_embd: int
    ffn_hidden: int
    context_length: int
    vocab_size: int
    dropout: float
    tie_embeddings: bool
    normalization: str
    norm_position: str
    layer_norm_eps: float
    activation: str
    gelu_approximate: str
    position_encoding: str
    linear_bias: bool
    lm_head_bias: bool
    layer_norm_affine: bool
    init_std: float
    scale_residual_projections: bool

    def __post_init__(self) -> None:
        for field_name in (
            "n_layer",
            "n_head",
            "n_embd",
            "ffn_hidden",
            "context_length",
            "vocab_size",
        ):
            value = getattr(self, field_name)
            if not _is_plain_int(value) or value <= 0:
                raise GPTConfigError(
                    f"{field_name} must be a positive integer, got {value!r}"
                )

        if self.n_embd % self.n_head != 0:
            raise GPTConfigError(
                "n_embd must be divisible by n_head, "
                f"got n_embd={self.n_embd}, n_head={self.n_head}"
            )

        if self.ffn_hidden != 4 * self.n_embd:
            raise GPTConfigError(
                "ffn_hidden must equal 4 * n_embd, "
                f"got ffn_hidden={self.ffn_hidden}, "
                f"n_embd={self.n_embd}"
            )

        if not _is_finite_number(self.dropout) or not 0.0 <= float(self.dropout) < 1.0:
            raise GPTConfigError(
                f"dropout must be finite and in [0, 1), got {self.dropout!r}"
            )

        if (
            not _is_finite_number(self.layer_norm_eps)
            or float(self.layer_norm_eps) <= 0.0
        ):
            raise GPTConfigError(
                "layer_norm_eps must be finite and positive, "
                f"got {self.layer_norm_eps!r}"
            )

        if not _is_finite_number(self.init_std) or float(self.init_std) <= 0.0:
            raise GPTConfigError(
                f"init_std must be finite and positive, got {self.init_std!r}"
            )

        for field_name in (
            "tie_embeddings",
            "linear_bias",
            "lm_head_bias",
            "layer_norm_affine",
            "scale_residual_projections",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, bool):
                raise GPTConfigError(f"{field_name} must be a boolean, got {value!r}")

        frozen_values = {
            "architecture": "decoder_only_gpt",
            "normalization": "layernorm",
            "norm_position": "pre",
            "activation": "gelu",
            "gelu_approximate": "tanh",
            "position_encoding": "learned_absolute",
            "tie_embeddings": True,
            "linear_bias": False,
            "lm_head_bias": False,
            "layer_norm_affine": True,
            "scale_residual_projections": True,
        }
        for field_name, expected in frozen_values.items():
            actual = getattr(self, field_name)
            if actual != expected:
                raise GPTConfigError(
                    f"{field_name} must equal {expected!r}, got {actual!r}"
                )

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def residual_init_std(self) -> float:
        return self.init_std / math.sqrt(2 * self.n_layer)

    @property
    def parameter_count(self) -> int:
        token_embedding = self.vocab_size * self.n_embd
        position_embedding = self.context_length * self.n_embd

        qkv_projection = 3 * self.n_embd * self.n_embd
        attention_output_projection = self.n_embd * self.n_embd
        mlp_projections = 2 * self.n_embd * self.ffn_hidden

        if self.linear_bias:
            qkv_projection += 3 * self.n_embd
            attention_output_projection += self.n_embd
            mlp_projections += self.ffn_hidden + self.n_embd

        layer_norms_per_block = 4 * self.n_embd
        if not self.layer_norm_affine:
            layer_norms_per_block = 0

        transformer_blocks = self.n_layer * (
            qkv_projection
            + attention_output_projection
            + mlp_projections
            + layer_norms_per_block
        )

        final_norm = 2 * self.n_embd if self.layer_norm_affine else 0

        output_head = 0
        if not self.tie_embeddings:
            output_head += self.vocab_size * self.n_embd
        if self.lm_head_bias:
            output_head += self.vocab_size

        return (
            token_embedding
            + position_embedding
            + transformer_blocks
            + final_norm
            + output_head
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, model: Mapping[str, Any]) -> GPTConfig:
        if not isinstance(model, Mapping):
            raise GPTConfigError(
                f"model configuration must be a mapping, got {type(model)!r}"
            )

        expected_fields = {field.name for field in fields(cls)}
        provided_fields = set(model)
        missing_fields = expected_fields - provided_fields
        unknown_fields = provided_fields - expected_fields

        if missing_fields:
            raise GPTConfigError(
                f"model configuration is missing fields: {sorted(missing_fields)}"
            )
        if unknown_fields:
            raise GPTConfigError(
                f"model configuration has unknown fields: {sorted(unknown_fields)}"
            )

        try:
            return cls(**dict(model))
        except TypeError as error:
            raise GPTConfigError(f"could not construct GPTConfig: {error}") from error

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        require_project_contract: bool = True,
    ) -> GPTConfig:
        config_path = Path(path)

        try:
            with config_path.open("r", encoding="utf-8") as file:
                document = yaml.safe_load(file)
        except OSError as error:
            raise GPTConfigError(
                f"could not read model configuration {config_path}: {error}"
            ) from error
        except yaml.YAMLError as error:
            raise GPTConfigError(
                f"could not parse model configuration {config_path}: {error}"
            ) from error

        if not isinstance(document, dict):
            raise GPTConfigError(
                f"{config_path}: top-level configuration must be a mapping"
            )

        model = document.get("model")
        if not isinstance(model, dict):
            raise GPTConfigError(f"{config_path}: model section must be a mapping")

        try:
            config = cls.from_mapping(model)
        except GPTConfigError as error:
            raise GPTConfigError(f"{config_path}: {error}") from error

        if require_project_contract:
            tokenizer = document.get("tokenizer")
            if not isinstance(tokenizer, dict):
                raise GPTConfigError(
                    f"{config_path}: tokenizer section must be a mapping"
                )

            tokenizer_vocab_size = tokenizer.get("vocab_size")
            if tokenizer_vocab_size != config.vocab_size:
                raise GPTConfigError(
                    f"{config_path}: tokenizer vocab_size "
                    f"{tokenizer_vocab_size!r} does not match model "
                    f"vocab_size {config.vocab_size}"
                )
            if config.vocab_size != 16_384:
                raise GPTConfigError(
                    f"{config_path}: project vocab_size must equal 16384, "
                    f"got {config.vocab_size}"
                )

        return config
