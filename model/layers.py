from __future__ import annotations

import torch
from torch import nn

from .config import GPTConfig


class TokenPositionEmbedding(nn.Module):
    """Combine token and learned absolute position embeddings."""

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
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
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

        position_ids = torch.arange(
            sequence_length,
            device=input_ids.device,
            dtype=torch.long,
        )
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)

        return self.dropout(token_embeddings + position_embeddings)


class MLP(nn.Module):
    """Position-wise GELU feed-forward network."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.fc_in = nn.Linear(
            config.n_embd,
            config.ffn_hidden,
            bias=config.linear_bias,
        )
        self.activation = nn.GELU(
            approximate=config.gelu_approximate,
        )
        self.fc_out = nn.Linear(
            config.ffn_hidden,
            config.n_embd,
            bias=config.linear_bias,
        )
        self.dropout = nn.Dropout(config.dropout)

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

        hidden_states = self.fc_in(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        return self.dropout(hidden_states)
