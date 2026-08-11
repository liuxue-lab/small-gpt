from __future__ import annotations

import torch
from torch import nn

from .config import GPTConfig


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
