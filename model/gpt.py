from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .block import TransformerBlock
from .config import GPTConfig


@dataclass(frozen=True, slots=True)
class GPTOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None


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
