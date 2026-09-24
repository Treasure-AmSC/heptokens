"""Standalone, non-learned position tokenizer (HEP4M-style).

Copied from heptokens-cocoa ``heptokens_cocoa/models/pos_tokenizer.py`` (code unchanged).
Auxiliary position features (e.g. eta, cos_phi, sin_phi) are binned onto a fixed,
uniform grid. This module holds no learnable parameters and stays fully outside the
VQ-VAE path: it is never an encoder input, never a reconstruction target, and never
contributes to the loss. Its tokens are stored alongside the VQ-VAE codebook indices.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

# Default matches the (eta, cos_phi, sin_phi) convention used by the
# HEP4M/COCOA "positions" tensors.
_DEFAULT_RANGES = [(-3.0, 3.0), (-1.0, 1.0), (-1.0, 1.0)]
_DEFAULT_N_BINS = 1024


class PositionTokenizer(nn.Module):
    """Fixed-bin quantizer for auxiliary position features.

    Args:
        ranges: List of ``(min, max)`` per position feature. Defaults to the
            (eta, cos_phi, sin_phi) ranges used elsewhere in this repo.
        n_bins: Number of bins per feature: a single int (applied to every
            feature) or a list with one value per feature.
    """

    def __init__(
        self,
        ranges: list[tuple[float, float]] | None = None,
        n_bins: int | list[int] = _DEFAULT_N_BINS,
    ) -> None:
        super().__init__()
        if ranges is None:
            ranges = list(_DEFAULT_RANGES)
        n_features = len(ranges)

        if isinstance(n_bins, int):
            n_bins = [n_bins] * n_features
        if len(n_bins) != n_features:
            raise ValueError(
                f"n_bins must be an int or a list of length {n_features}, got {n_bins}"
            )

        mins = torch.tensor([r[0] for r in ranges], dtype=torch.float32)
        maxs = torch.tensor([r[1] for r in ranges], dtype=torch.float32)
        n_bins_t = torch.tensor(n_bins, dtype=torch.float32)
        widths = (maxs - mins) / n_bins_t

        self.register_buffer("mins", mins)
        self.register_buffer("maxs", maxs)
        self.register_buffer("widths", widths)
        self.register_buffer("max_idx", n_bins_t.long() - 1)
        self.n_features = n_features
        self.n_bins = list(n_bins)

    @torch.no_grad()
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Bin ``positions`` into per-feature indices.

        Args:
            positions: Tensor of shape [..., n_features].

        Returns:
            LongTensor of shape [..., n_features] with per-feature bin
            indices, clamped to a valid range.
        """
        if positions.shape[-1] != self.n_features:
            raise ValueError(
                f"Expected last dim {self.n_features}, got {positions.shape[-1]}"
            )
        eps = 1e-6
        clamped = torch.clamp(positions, self.mins, self.maxs - eps)
        idx = ((clamped - self.mins) / self.widths).long()
        return torch.clamp(idx, min=torch.zeros_like(self.max_idx), max=self.max_idx)

    @torch.no_grad()
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Map bin indices back to bin-center position values.

        Args:
            indices: LongTensor of shape [..., n_features].

        Returns:
            Tensor of shape [..., n_features] with bin-center positions.
        """
        return self.mins + (indices.float() + 0.5) * self.widths
