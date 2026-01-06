"""Transformer building block for sequence-aware encoding/decoding."""

from __future__ import annotations

import torch
import torch.nn as nn


class Transformer(nn.Module):
    """Transformer encoder stack with input/output projections.

    This module preserves the input sequence length and supports a padding mask.
    It is designed to be used inside Encoder/Decoder wrappers.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        d_model: int = 128,
        n_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass through the transformer.

        Args:
            x: Tensor of shape [batch, n_csts, input_dim].
            mask: Boolean tensor of shape [batch, n_csts] where True means valid.

        Returns:
            Tensor of shape [batch, n_csts, output_dim].
        """
        squeeze_batch = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_batch = True

        key_padding_mask = None
        empty_sequences = None
        if mask is not None:
            mask = mask.bool()
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            empty_sequences = ~mask.any(dim=1)
            x = x.clone()
            # Zero out padded positions to avoid NaNs in attention/FFN paths.
            x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
            if empty_sequences.any():
                mask_for_encoder = mask.clone()
                # Ensure at least one valid token so attention doesn't see all padding.
                mask_for_encoder[empty_sequences, 0] = True
                key_padding_mask = ~mask_for_encoder
                x[empty_sequences] = 0.0
            else:
                key_padding_mask = ~mask
        x = self.input_proj(x)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.output_proj(x)

        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
            if empty_sequences is not None and empty_sequences.any():
                x[empty_sequences] = 0.0

        if squeeze_batch:
            x = x.squeeze(0)

        return x
