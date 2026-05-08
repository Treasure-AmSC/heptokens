"""Shared transformer backbone for token sequences."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from heptokens.data.sequence import MASK_KEY, TOKENS_KEY, TYPE_IDS_KEY
from heptokens.models.transformer import Transformer


@dataclass
class SequenceBackboneConfig:
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    max_seq_length: int = 128
    vocab_size: int = 20000
    num_type_ids: int = 12
    mask_token_id: int = 3
    pad_token_id: int = 0
    mask_prob: float = 0.15
    hierarchical: bool = False
    n_groups: int = 3
    use_type_embedding: bool = True
    use_position_embedding: bool = True


class GroupAttentionLayer(nn.Module):
    """Optional per-token group attention used by the sequence backbone."""

    def __init__(
        self,
        hidden_dim: int,
        n_groups: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_groups = n_groups
        self.group_projections = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_groups)
        )
        self.group_type_emb = nn.Embedding(n_groups, hidden_dim)
        self.inner_attention = Transformer(
            input_dim=hidden_dim,
            output_dim=hidden_dim,
            d_model=hidden_dim,
            n_heads=num_heads,
            num_layers=1,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
        )
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.shape
        n_groups = self.n_groups
        sub_vectors = torch.stack([proj(x) for proj in self.group_projections], dim=2)
        group_ids = torch.arange(n_groups, device=x.device)
        sub_vectors = sub_vectors + self.group_type_emb(group_ids).view(
            1,
            1,
            n_groups,
            hidden_dim,
        )
        sub_flat = sub_vectors.reshape(batch_size * seq_len, n_groups, hidden_dim)

        if mask is None:
            position_valid = torch.ones(batch_size * seq_len, dtype=torch.bool, device=x.device)
        else:
            position_valid = mask.reshape(batch_size * seq_len).bool()

        out_flat = torch.zeros_like(sub_flat)
        if position_valid.any():
            group_mask = torch.ones(
                (int(position_valid.sum()), n_groups),
                dtype=torch.bool,
                device=x.device,
            )
            out_flat[position_valid] = self.inner_attention(
                sub_flat[position_valid],
                mask=group_mask,
            )

        pooled = self.output_proj(out_flat.mean(dim=1)).reshape(batch_size, seq_len, hidden_dim)
        return self.layer_norm(x + pooled)


class SequenceBackbone(nn.Module):
    """Token, type, position, and transformer encoder stack for sequence tokens."""

    def __init__(self, config: SequenceBackboneConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.position_embedding = (
            nn.Embedding(config.max_seq_length, config.hidden_dim)
            if config.use_position_embedding
            else None
        )
        self.type_embedding = (
            nn.Embedding(config.num_type_ids, config.hidden_dim)
            if config.use_type_embedding
            else None
        )
        self.group_attention = (
            GroupAttentionLayer(
                config.hidden_dim,
                config.n_groups,
                config.num_heads,
                config.dropout,
            )
            if config.hierarchical
            else None
        )
        self.transformer = Transformer(
            input_dim=config.hidden_dim,
            output_dim=config.hidden_dim,
            d_model=config.hidden_dim,
            n_heads=config.num_heads,
            num_layers=config.num_layers,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len = mask.shape
        x = self.token_embedding(tokens)
        if self.type_embedding is not None:
            if type_ids is None:
                type_ids = torch.zeros_like(tokens)
            x = x + self.type_embedding(type_ids)
        if self.position_embedding is not None:
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
            x = x + self.position_embedding(positions)
        x = self.dropout(x)
        if self.group_attention is not None:
            x = self.group_attention(x, mask)
        return self.layer_norm(self.transformer(x, mask=mask.bool()))

    def forward_batch(self, batch: dict) -> torch.Tensor:
        return self(
            batch[TOKENS_KEY],
            batch[MASK_KEY],
            batch.get(TYPE_IDS_KEY),
        )
