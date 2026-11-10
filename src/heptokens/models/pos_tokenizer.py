"""Fixed-bin (non-learned) tokenizer for per-element positions.

Used for positions-out tokenization: positions are kept out of the VQ-VAE input
(``csts``), carried as a separate ``positions`` tensor, and binned onto a uniform
grid per feature. Holds no learnable parameters and never enters
encode/decode/loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_DEFAULT_N_BINS = 1024


class PositionTokenizer(nn.Module):
    """Uniform fixed-bin quantizer, one integer token per position feature.

    Args:
        ranges: ``(min, max)`` per position feature. Values outside are clamped
            into the first/last bin.
        n_bins: Bins per feature: an int for all features, or one per feature.
    """

    def __init__(
        self,
        ranges: list[tuple[float, float]],
        n_bins: int | list[int] = _DEFAULT_N_BINS,
    ) -> None:
        super().__init__()
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
        """Bin ``positions`` [..., n_features] into LongTensor indices of the same shape."""
        if positions.shape[-1] != self.n_features:
            raise ValueError(f"Expected last dim {self.n_features}, got {positions.shape[-1]}")
        # Buffers may sit on a different device than ``positions`` when the tokenizer
        # is held outside the LightningModule (e.g. by the prediction writer).
        device = positions.device
        mins, maxs = self.mins.to(device), self.maxs.to(device)
        widths, max_idx = self.widths.to(device), self.max_idx.to(device)
        clamped = torch.clamp(positions, mins, maxs)
        # The maximum maps to bin n_bins; the index clamp folds it into the last bin.
        idx = ((clamped - mins) / widths).long()
        return torch.clamp(idx, min=torch.zeros_like(max_idx), max=max_idx)

    @torch.no_grad()
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Map bin indices [..., n_features] back to bin-centre positions."""
        device = indices.device
        return self.mins.to(device) + (indices.float() + 0.5) * self.widths.to(device)
