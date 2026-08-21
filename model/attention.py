from __future__ import annotations

import torch
from torch import nn

from .config import GPTConfig


LayerKVCache = tuple[torch.Tensor, torch.Tensor]


class CausalSelfAttention(nn.Module):
    """Handwritten scaled dot-product causal multi-head self-attention."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5

        self.qkv_proj = nn.Linear(
            config.n_embd,
            3 * config.n_embd,
            bias=config.linear_bias,
        )
        self.out_proj = nn.Linear(
            config.n_embd,
            config.n_embd,
            bias=config.linear_bias,
        )
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        causal_mask = torch.tril(
            torch.ones(
                config.context_length,
                config.context_length,
                dtype=torch.bool,
            )
        ).view(1, 1, config.context_length, config.context_length)
        self.register_buffer(
            "causal_mask",
            causal_mask,
            persistent=False,
        )

    def _split_heads(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        return hidden_states.view(
            batch_size,
            sequence_length,
            self.n_head,
            self.head_dim,
        ).transpose(1, 2)

    def _merge_heads(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, _, sequence_length, _ = hidden_states.shape
        return (
            hidden_states.transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                sequence_length,
                self.config.n_embd,
            )
        )

    def _attention_probabilities(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> torch.Tensor:
        sequence_length = query.shape[-2]
        attention_scores = torch.matmul(
            query,
            key.transpose(-2, -1),
        )
        attention_scores = attention_scores * self.scale

        causal_mask = self.causal_mask[
            :,
            :,
            :sequence_length,
            :sequence_length,
        ]
        attention_scores = attention_scores.masked_fill(
            ~causal_mask,
            torch.finfo(attention_scores.dtype).min,
        )
        probabilities = torch.softmax(
            attention_scores,
            dim=-1,
            dtype=torch.float32,
        ).to(query.dtype)
        return self.attn_dropout(probabilities)

    def _validate_cached_hidden_states(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[int, int]:
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError(
                f"hidden_states must be a torch.Tensor, got {type(hidden_states)!r}"
            )
        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape [batch, sequence, embedding], "
                f"got {tuple(hidden_states.shape)}"
            )
        if hidden_states.shape[-1] != self.config.n_embd:
            raise ValueError(
                "hidden_states last dimension must equal n_embd, "
                f"got {hidden_states.shape[-1]} and "
                f"n_embd={self.config.n_embd}"
            )
        if not hidden_states.is_floating_point():
            raise TypeError(
                "hidden_states must use a floating-point dtype, "
                f"got {hidden_states.dtype}"
            )

        batch_size, sequence_length, _ = hidden_states.shape
        if batch_size <= 0:
            raise ValueError("hidden_states batch dimension must be positive")
        if sequence_length <= 0:
            raise ValueError("hidden_states sequence dimension must be positive")
        if sequence_length > self.config.context_length:
            raise ValueError(
                "hidden_states sequence length exceeds configured context length: "
                f"{sequence_length} > {self.config.context_length}"
            )
        return batch_size, sequence_length

    def _validate_past_key_value(
        self,
        past_key_value: object,
        *,
        batch_size: int,
        current_key: torch.Tensor,
    ) -> LayerKVCache:
        if not isinstance(past_key_value, tuple) or len(past_key_value) != 2:
            raise TypeError(
                "past_key_value must be a two-item tuple of key/value tensors"
            )

        past_key, past_value = past_key_value
        for name, tensor in (("past_key", past_key), ("past_value", past_value)):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor)!r}")
            if tensor.ndim != 4:
                raise ValueError(
                    f"{name} must have shape [batch, head, sequence, head_dim], "
                    f"got {tuple(tensor.shape)}"
                )
            if not tensor.is_floating_point():
                raise TypeError(
                    f"{name} must use a floating-point dtype, got {tensor.dtype}"
                )

        if past_key.shape != past_value.shape:
            raise ValueError(
                "past key/value shapes must match, "
                f"got key={tuple(past_key.shape)} and "
                f"value={tuple(past_value.shape)}"
            )

        expected_prefix = (batch_size, self.n_head)
        if past_key.shape[:2] != expected_prefix:
            raise ValueError(
                "past key/value batch and head dimensions must equal "
                f"{expected_prefix}, got {tuple(past_key.shape[:2])}"
            )
        if past_key.shape[-1] != self.head_dim:
            raise ValueError(
                "past key/value head dimension must equal configured head_dim, "
                f"got {past_key.shape[-1]} and head_dim={self.head_dim}"
            )
        if past_key.shape[-2] <= 0:
            raise ValueError("past key/value sequence dimension must be positive")
        if (
            past_key.device != current_key.device
            or past_value.device != current_key.device
        ):
            raise ValueError(
                "past key/value tensors and current key must be on the same device"
            )
        if past_key.dtype != current_key.dtype or past_value.dtype != current_key.dtype:
            raise TypeError(
                "past key/value tensors and current key must use the same dtype"
            )
        return past_key, past_value

    def forward_cached(
        self,
        hidden_states: torch.Tensor,
        past_key_value: LayerKVCache | None = None,
    ) -> tuple[torch.Tensor, LayerKVCache]:
        """Run inference-only attention and return an immutable-style KV cache."""

        if self.training:
            raise RuntimeError("forward_cached requires eval mode")
        if not torch.is_inference_mode_enabled():
            raise RuntimeError("forward_cached requires torch.inference_mode()")

        batch_size, sequence_length = self._validate_cached_hidden_states(
            hidden_states
        )
        if past_key_value is not None and sequence_length != 1:
            raise ValueError(
                "cached decode requires exactly one current token when "
                "past_key_value is provided"
            )

        query_key_value = self.qkv_proj(hidden_states)
        query, key, value = query_key_value.split(
            self.config.n_embd,
            dim=-1,
        )
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        past_length = 0
        if past_key_value is not None:
            past_key, past_value = self._validate_past_key_value(
                past_key_value,
                batch_size=batch_size,
                current_key=key,
            )
            past_length = past_key.shape[-2]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        total_length = past_length + sequence_length
        if total_length > self.config.context_length:
            raise ValueError(
                "cached sequence length exceeds configured context length: "
                f"{total_length} > {self.config.context_length}"
            )

        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = attention_scores * self.scale
        causal_mask = self.causal_mask[
            :,
            :,
            past_length:total_length,
            :total_length,
        ]
        attention_scores = attention_scores.masked_fill(
            ~causal_mask,
            torch.finfo(attention_scores.dtype).min,
        )
        probabilities = torch.softmax(
            attention_scores,
            dim=-1,
            dtype=torch.float32,
        ).to(query.dtype)
        probabilities = self.attn_dropout(probabilities)
        attended_values = torch.matmul(probabilities, value)
        attended_values = self._merge_heads(attended_values)
        output = self.out_proj(attended_values)
        return self.resid_dropout(output), (key, value)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError(
                f"hidden_states must be a torch.Tensor, got {type(hidden_states)!r}"
            )
        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape [batch, sequence, embedding], "
                f"got {tuple(hidden_states.shape)}"
            )
        if hidden_states.shape[-1] != self.config.n_embd:
            raise ValueError(
                "hidden_states last dimension must equal n_embd, "
                f"got {hidden_states.shape[-1]} and "
                f"n_embd={self.config.n_embd}"
            )
        if not hidden_states.is_floating_point():
            raise TypeError(
                "hidden_states must use a floating-point dtype, "
                f"got {hidden_states.dtype}"
            )

        batch_size, sequence_length, _ = hidden_states.shape
        if batch_size <= 0:
            raise ValueError("hidden_states batch dimension must be positive")
        if sequence_length <= 0:
            raise ValueError("hidden_states sequence dimension must be positive")
        if sequence_length > self.config.context_length:
            raise ValueError(
                "hidden_states sequence length exceeds configured context length: "
                f"{sequence_length} > {self.config.context_length}"
            )

        query_key_value = self.qkv_proj(hidden_states)
        query, key, value = query_key_value.split(
            self.config.n_embd,
            dim=-1,
        )
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        probabilities = self._attention_probabilities(query, key)
        attended_values = torch.matmul(probabilities, value)
        attended_values = self._merge_heads(attended_values)
        output = self.out_proj(attended_values)
        return self.resid_dropout(output)