from __future__ import annotations

import torch
from torch import nn

from .attention import CausalSelfAttention
from .config import GPTConfig
from .layers import MLP


class TransformerBlock(nn.Module):
    """Pre-LayerNorm decoder-only Transformer block."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.norm1 = nn.LayerNorm(
            config.n_embd,
            eps=config.layer_norm_eps,
            elementwise_affine=config.layer_norm_affine,
        )
        self.attention = CausalSelfAttention(config)
        self.norm2 = nn.LayerNorm(
            config.n_embd,
            eps=config.layer_norm_eps,
            elementwise_affine=config.layer_norm_affine,
        )
        self.mlp = MLP(config)

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

        hidden_states = hidden_states + self.attention(self.norm1(hidden_states))
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states
