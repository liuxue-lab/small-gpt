from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .attention import LayerKVCache
from .block import TransformerBlock
from .config import GPTConfig


PastKeyValues = tuple[LayerKVCache, ...]


@dataclass(frozen=True, slots=True)
class GPTOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None


@dataclass(frozen=True, slots=True)
class GPTCachedOutput:
    logits: torch.Tensor
    past_key_values: PastKeyValues


class GPT(nn.Module):
    """Decoder-only GPT language model with tied token/output embeddings."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.n_embd,
        )
        self.position_embedding = nn.Embedding(
            config.context_length,
            config.n_embd,
        )
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layer)
        )
        self.final_norm = nn.LayerNorm(
            config.n_embd,
            eps=config.layer_norm_eps,
            elementwise_affine=config.layer_norm_affine,
        )
        self.lm_head = nn.Linear(
            config.n_embd,
            config.vocab_size,
            bias=config.lm_head_bias,
        )

        self.apply(self._initialize_module)
        self._scale_residual_projections()
        self.lm_head.weight = self.token_embedding.weight

    def _initialize_module(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Embedding, nn.Linear)):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.init_std,
            )
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            if module.elementwise_affine:
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _scale_residual_projections(self) -> None:
        if not self.config.scale_residual_projections:
            return

        for block in self.blocks:
            nn.init.normal_(
                block.attention.out_proj.weight,
                mean=0.0,
                std=self.config.residual_init_std,
            )
            nn.init.normal_(
                block.mlp.fc_out.weight,
                mean=0.0,
                std=self.config.residual_init_std,
            )

    def _validate_input_ids(self, input_ids: torch.Tensor) -> None:
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError(
                f"input_ids must be a torch.Tensor, got {type(input_ids)!r}"
            )
        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape [batch, sequence], "
                f"got {tuple(input_ids.shape)}"
            )
        if input_ids.dtype != torch.long:
            raise TypeError(
                f"input_ids must have dtype torch.long, got {input_ids.dtype}"
            )

        batch_size, sequence_length = input_ids.shape
        if batch_size <= 0:
            raise ValueError("input_ids batch dimension must be positive")
        if sequence_length <= 0:
            raise ValueError("input_ids sequence dimension must be positive")
        if sequence_length > self.config.context_length:
            raise ValueError(
                "input sequence length exceeds configured context length: "
                f"{sequence_length} > {self.config.context_length}"
            )

        minimum_id, maximum_id = torch.aminmax(input_ids)
        minimum_id_value = int(minimum_id.item())
        maximum_id_value = int(maximum_id.item())
        if minimum_id_value < 0 or maximum_id_value >= self.config.vocab_size:
            raise ValueError(
                "input_ids contain token IDs outside the valid range "
                f"[0, {self.config.vocab_size}): "
                f"min={minimum_id_value}, max={maximum_id_value}"
            )

    def _validate_targets(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        if not isinstance(targets, torch.Tensor):
            raise TypeError(f"targets must be a torch.Tensor, got {type(targets)!r}")
        if targets.shape != input_ids.shape:
            raise ValueError(
                "targets shape must match input_ids shape, "
                f"got targets={tuple(targets.shape)} and "
                f"input_ids={tuple(input_ids.shape)}"
            )
        if targets.dtype != torch.long:
            raise TypeError(f"targets must have dtype torch.long, got {targets.dtype}")
        if targets.device != input_ids.device:
            raise ValueError(
                "targets and input_ids must be on the same device, "
                f"got targets={targets.device} and input_ids={input_ids.device}"
            )

        minimum_id, maximum_id = torch.aminmax(targets)
        minimum_id_value = int(minimum_id.item())
        maximum_id_value = int(maximum_id.item())
        if minimum_id_value < 0 or maximum_id_value >= self.config.vocab_size:
            raise ValueError(
                "targets contain token IDs outside the valid range "
                f"[0, {self.config.vocab_size}): "
                f"min={minimum_id_value}, max={maximum_id_value}"
            )

    def _validate_past_key_values(
        self,
        input_ids: torch.Tensor,
        past_key_values: PastKeyValues | None,
    ) -> int:
        model_device = self.token_embedding.weight.device
        if input_ids.device != model_device:
            raise ValueError(
                "input_ids and model parameters must be on the same device, "
                f"got input_ids={input_ids.device} and model={model_device}"
            )
        if past_key_values is None:
            return 0
        if not isinstance(past_key_values, tuple):
            raise TypeError(
                "past_key_values must be a tuple with one key/value pair per layer"
            )
        if len(past_key_values) != self.config.n_layer:
            raise ValueError(
                "past_key_values layer count must equal n_layer, "
                f"got {len(past_key_values)} and n_layer={self.config.n_layer}"
            )
        if input_ids.shape[1] != 1:
            raise ValueError(
                "cached decode requires input_ids sequence length 1 when "
                "past_key_values is provided"
            )

        expected_batch = input_ids.shape[0]
        past_length: int | None = None
        cache_dtype: torch.dtype | None = None
        for layer_index, layer_cache in enumerate(past_key_values):
            if not isinstance(layer_cache, tuple) or len(layer_cache) != 2:
                raise TypeError(
                    f"past_key_values[{layer_index}] must be a two-item tuple"
                )
            key, value = layer_cache
            for cache_name, tensor in (("key", key), ("value", value)):
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError(
                        f"past_key_values[{layer_index}].{cache_name} must be "
                        f"a torch.Tensor, got {type(tensor)!r}"
                    )
                if tensor.ndim != 4:
                    raise ValueError(
                        f"past_key_values[{layer_index}].{cache_name} must have "
                        "shape [batch, head, sequence, head_dim], "
                        f"got {tuple(tensor.shape)}"
                    )
                if not tensor.is_floating_point():
                    raise TypeError(
                        f"past_key_values[{layer_index}].{cache_name} must use "
                        f"a floating-point dtype, got {tensor.dtype}"
                    )
                if tensor.device != input_ids.device:
                    raise ValueError(
                        f"past_key_values[{layer_index}].{cache_name} and "
                        "input_ids must be on the same device"
                    )

            if key.shape != value.shape:
                raise ValueError(
                    f"past_key_values[{layer_index}] key/value shapes must match"
                )
            if key.shape[0] != expected_batch:
                raise ValueError(
                    f"past_key_values[{layer_index}] batch dimension must equal "
                    f"{expected_batch}, got {key.shape[0]}"
                )
            if key.shape[1] != self.config.n_head:
                raise ValueError(
                    f"past_key_values[{layer_index}] head dimension must equal "
                    f"{self.config.n_head}, got {key.shape[1]}"
                )
            if key.shape[3] != self.config.head_dim:
                raise ValueError(
                    f"past_key_values[{layer_index}] head_dim must equal "
                    f"{self.config.head_dim}, got {key.shape[3]}"
                )
            if key.shape[2] <= 0:
                raise ValueError(
                    f"past_key_values[{layer_index}] sequence length must be positive"
                )
            if key.dtype != value.dtype:
                raise TypeError(
                    f"past_key_values[{layer_index}] key/value dtypes must match"
                )

            layer_past_length = key.shape[2]
            if past_length is None:
                past_length = layer_past_length
                cache_dtype = key.dtype
            elif layer_past_length != past_length:
                raise ValueError("all cache layers must have the same sequence length")
            elif key.dtype != cache_dtype:
                raise TypeError("all cache layers must use the same dtype")

        if past_length is None:
            raise ValueError("past_key_values must not be empty")
        total_length = past_length + input_ids.shape[1]
        if total_length > self.config.context_length:
            raise ValueError(
                "cached sequence length exceeds configured context length: "
                f"{total_length} > {self.config.context_length}"
            )
        return past_length

    def forward_cached(
        self,
        input_ids: torch.Tensor,
        past_key_values: PastKeyValues | None = None,
    ) -> GPTCachedOutput:
        """Run inference with per-layer KV caches and absolute positions."""

        if self.training:
            raise RuntimeError("forward_cached requires eval mode")
        if not torch.is_inference_mode_enabled():
            raise RuntimeError("forward_cached requires torch.inference_mode()")

        self._validate_input_ids(input_ids)
        past_length = self._validate_past_key_values(input_ids, past_key_values)

        _, sequence_length = input_ids.shape
        position_ids = torch.arange(
            past_length,
            past_length + sequence_length,
            device=input_ids.device,
            dtype=torch.long,
        )
        hidden_states = self.token_embedding(input_ids)
        hidden_states = hidden_states + self.position_embedding(position_ids)
        hidden_states = self.embedding_dropout(hidden_states)

        present_key_values: list[LayerKVCache] = []
        for layer_index, block in enumerate(self.blocks):
            layer_past = (
                None
                if past_key_values is None
                else past_key_values[layer_index]
            )
            hidden_states, layer_present = block.forward_cached(
                hidden_states,
                layer_past,
            )
            present_key_values.append(layer_present)

        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return GPTCachedOutput(
            logits=logits,
            past_key_values=tuple(present_key_values),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> GPTOutput:
        self._validate_input_ids(input_ids)
        if targets is not None:
            self._validate_targets(input_ids, targets)

        _, sequence_length = input_ids.shape
        position_ids = torch.arange(
            sequence_length,
            device=input_ids.device,
            dtype=torch.long,
        )
        hidden_states = self.token_embedding(input_ids)
        hidden_states = hidden_states + self.position_embedding(position_ids)
        hidden_states = self.embedding_dropout(hidden_states)

        for block in self.blocks:
            hidden_states = block(hidden_states)

        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
            )

        return GPTOutput(logits=logits, loss=loss)