"""LitVqVae bottlenecks: the :class:`Quantizer` interface and its implementations.

A quantizer maps encoder outputs ``z_e [..., dim]`` to ``num_quantizers`` integer tokens
per element, each taking one of ``codebook_size`` values. LitVqVae, token export and the
TorchScript inference modules only use the interface, so a new tokenization scheme is a
:class:`Quantizer` subclass plus a ``conf/model/quantizer/`` config.

Implementations:
    ResidualVQQuantizer: residual vector quantization with optional dead-code revival.
    LatentBins: uniform bins per latent dimension, one token per dimension.
"""

from abc import ABC, abstractmethod
from typing import List

import torch
import torch.nn as nn
from vector_quantize_pytorch import VectorQuantize

# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


class Quantizer(nn.Module, ABC):
    """Base class for LitVqVae bottlenecks.

    Subclasses take the latent dimension as a ``dim`` keyword argument (LitVqVae passes
    ``codebook_dim``) and set ``num_quantizers`` (tokens per element) and
    ``codebook_size`` (values per token).
    """

    num_quantizers: int
    codebook_size: int

    @abstractmethod
    def forward(
        self, z_e: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize ``z_e [..., dim]``.

        Args:
            mask: Optional bool tensor matching ``z_e.shape[:-1]`` (True = valid), for
                quantizers whose training updates should ignore padding.

        Returns:
            ``z_q [..., dim]``, integer ``indices [..., num_quantizers]`` and a scalar
            loss that LitVqVae adds to the reconstruction loss.
        """

    @abstractmethod
    def index_encoder(self) -> nn.Module:
        """A TorchScript-able module mapping ``z_e [..., dim]`` to int64 tokens
        ``[..., num_quantizers]``, holding a snapshot of any state it needs (e.g. codebooks).

        This is the quantizer's inference path: ``encode_indices`` uses it, and token
        export scripts it into ``index_encoder.pt``.
        """

    @torch.no_grad()
    def encode_indices(self, z_e: torch.Tensor) -> torch.Tensor:
        """Tokens ``[..., num_quantizers]`` for ``z_e [..., dim]``, with no state updates."""
        return self.index_encoder()(z_e)

    @abstractmethod
    def codebooks(self) -> torch.Tensor:
        """``[num_quantizers, codebook_size, dim]`` embeddings such that an element's
        quantized latent is ``sum over q of codebooks[q, indices[..., q]]``.

        This is what token export saves as ``codebooks``.
        """


# ---------------------------------------------------------------------------
# Residual vector quantization
# ---------------------------------------------------------------------------


# ``vector_quantize_pytorch`` 0.2.2 has no protection against codebook collapse: once a
# code's EMA cluster size decays to ~0 it is never revived. RevivingVectorQuantize
# reinitializes dead codes to a real encoder output from the current batch. With
# ``threshold_ema_dead_code <= 0`` it behaves exactly like VectorQuantize. See #26 for
# replacing this with upstream's own dead-code handling.


class RevivingVectorQuantize(VectorQuantize):
    """VectorQuantize with dead-code revival.

    Args:
        threshold_ema_dead_code: codebook entries whose EMA ``cluster_size``
            falls below this threshold are reinitialized (during training
            only) to a random encoder output vector from the current batch.
            Set to 0 to disable (identical behavior to the base class).
    """

    def __init__(self, *args, threshold_ema_dead_code: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold_ema_dead_code = threshold_ema_dead_code

    def forward(self, input: torch.Tensor, mask: torch.Tensor | None = None):
        """Quantize ``input`` [..., dim], then revive dead codes (training only).

        Args:
            mask: Optional boolean tensor matching ``input.shape[:-1]``, True for valid
                positions. Revived codes are only sampled from valid positions, so padding
                never becomes a codebook entry.
        """
        quantize, embed_ind, loss = super().forward(input)

        if self.training and self.threshold_ema_dead_code > 0:
            with torch.no_grad():
                dead_mask = self.cluster_size < self.threshold_ema_dead_code
                n_dead = int(dead_mask.sum().item())
                if n_dead > 0:
                    flatten = input.reshape(-1, self.dim)
                    if mask is not None:
                        flatten = flatten[mask.reshape(-1).bool()]
                    n_samples = flatten.shape[0]
                    if n_samples > 0:
                        rand_idx = torch.randint(0, n_samples, (n_dead,), device=flatten.device)
                        # Match the codebook buffers' dtype (inputs may be fp16/bf16 under AMP).
                        replacement = flatten[rand_idx].t().to(self.embed)  # [dim, n_dead]
                        self.embed.data[:, dead_mask] = replacement
                        # embed_avg is an EMA *sum* that gets divided by cluster_size, so
                        # scale it to keep embed_avg / cluster_size == replacement.
                        self.embed_avg.data[:, dead_mask] = (
                            replacement * self.threshold_ema_dead_code
                        )
                        # Reset cluster_size just above threshold so revived
                        # codes aren't immediately re-flagged as dead next step.
                        self.cluster_size.data[dead_mask] = self.threshold_ema_dead_code

        return quantize, embed_ind, loss


class ResidualVQQuantizer(Quantizer):
    """Residual vector quantization, optionally with dead-code revival.

    Each of the ``num_quantizers`` layers quantizes the residual left by the previous ones,
    so an element's latent is the sum of one code from each codebook.

    Args:
        dim: Latent dimension (LitVqVae passes ``codebook_dim``).
        codebook_size: Codes per codebook.
        num_quantizers: Number of residual layers (tokens per element).
        commitment: Weight of the commitment loss.
        threshold_ema_dead_code: Revive codes whose EMA cluster size falls below this
            (see :class:`RevivingVectorQuantize`); 0 disables revival.
        **kwargs: Passed to each ``VectorQuantize`` layer (e.g. ``decay``).
    """

    def __init__(
        self,
        *,
        dim: int,
        codebook_size: int,
        num_quantizers: int,
        commitment: float = 1.0,
        threshold_ema_dead_code: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.layers = torch.nn.ModuleList(
            [
                RevivingVectorQuantize(
                    dim=dim,
                    codebook_size=codebook_size,
                    commitment=commitment,
                    threshold_ema_dead_code=threshold_ema_dead_code,
                    **kwargs,
                )
                for _ in range(num_quantizers)
            ]
        )

    def forward(self, z_e: torch.Tensor, mask: torch.Tensor | None = None):
        z_q = torch.zeros_like(z_e)
        residual = z_e
        indices, losses = [], []
        for layer in self.layers:
            quantized, layer_indices, loss = layer(residual, mask=mask)
            residual = residual - quantized
            z_q = z_q + quantized
            indices.append(layer_indices)
            losses.append(loss)
        return z_q, torch.stack(indices, dim=-1), torch.stack(losses).mean()

    def codebooks(self) -> torch.Tensor:
        return torch.stack([layer.codebook for layer in self.layers])

    def index_encoder(self) -> "ResidualVQIndexEncoder":
        return ResidualVQIndexEncoder(self.codebooks())


class ResidualVQIndexEncoder(torch.nn.Module):
    """TorchScript-able residual-VQ argmin search: ``z_e [..., dim]`` -> ``[..., Q]`` tokens.

    Skips VectorQuantize's ``[N, codebook_size]`` one-hot, which dominates memory for large
    codebooks and is only needed for training's EMA updates.

    Args:
        codebooks: ``[Q, codebook_size, dim]``, as returned by ``ResidualVQQuantizer.codebooks``.
    """

    def __init__(self, codebooks: torch.Tensor):
        super().__init__()
        # [Q, dim, K]: the layout of VectorQuantize.embed, so distances match training's.
        self.register_buffer("embeds", codebooks.detach().transpose(1, 2).contiguous())

    def forward(self, z_e: torch.Tensor) -> torch.Tensor:
        residual = z_e.reshape(-1, z_e.shape[-1])
        indices: List[torch.Tensor] = []
        for q in range(self.embeds.shape[0]):
            embed = self.embeds[q]  # [dim, K]
            dist = (
                residual.pow(2).sum(1, keepdim=True)
                - 2 * residual @ embed
                + embed.pow(2).sum(0, keepdim=True)
            )
            idx = (-dist).max(1).indices
            residual = residual - torch.nn.functional.embedding(idx, embed.t())
            indices.append(idx)
        shape = list(z_e.shape[:-1])
        return torch.stack(indices, dim=-1).reshape(shape + [len(indices)])


# ---------------------------------------------------------------------------
# Latent binning
# ---------------------------------------------------------------------------


class LatentBins(Quantizer):
    """Uniform bins on [-bin_range, bin_range] for each latent dimension, one token per dimension.

    mu = bin_range * tanh(h / bin_range) keeps the latent inside the bins.
    Training: z = mu + noise_std_bins * bin width * N(0, 1).
    Evaluation: z = the centre of mu's bin, which is what the tokens encode.

    Args:
        dim: Latent dimensions (tokens per element).
        num_bins: Bins per dimension (values per token).
        bin_range: Half-width of the binned interval.
        noise_std_bins: Training noise std, in bin widths.
    """

    def __init__(self, dim: int, num_bins: int, bin_range: float, noise_std_bins: float):
        super().__init__()
        assert dim > 0 and num_bins >= 2 and bin_range > 0 and noise_std_bins >= 0
        self.dim = dim
        self.num_bins = num_bins
        self.num_quantizers = dim  # one token per latent dimension
        self.codebook_size = num_bins
        self.bin_range = float(bin_range)
        self.bin_width = 2 * self.bin_range / num_bins
        self.noise_std = noise_std_bins * self.bin_width

    def bins(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """h [..., dim] -> mu, bin indices [..., dim]."""
        mu = self.bin_range * torch.tanh(h / self.bin_range)
        return mu, self.index_encoder()(h.detach())

    def get_output_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Bin indices [..., dim] -> bin centres [..., dim]."""
        return -self.bin_range + (indices + 0.5) * self.bin_width

    def forward(
        self, h: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """h [..., dim] -> z [..., dim], indices [..., dim], zero loss (mask is unused)."""
        assert h.shape[-1] == self.dim, f"{h.shape[-1]} latent dimensions, expected {self.dim}"
        mu, indices = self.bins(h)
        if self.training:
            z = mu + self.noise_std * torch.randn_like(mu)
        else:
            z = self.get_output_from_indices(indices).to(mu.dtype)
        return z, indices, h.new_zeros(())

    def index_encoder(self) -> "BinIndexEncoder":
        return BinIndexEncoder(self.num_bins, self.bin_range)

    def codebooks(self) -> torch.Tensor:
        """Each latent dimension is one quantizer: ``codebooks[q, k]`` holds the centre of
        bin ``k`` on axis ``q`` and zeros elsewhere, so summing over quantizers gives the
        binned latent."""
        centres = self.get_output_from_indices(torch.arange(self.num_bins))  # [num_bins]
        axes = torch.eye(self.dim)  # [dim, dim]
        return axes[:, None, :] * centres[None, :, None]  # [dim, num_bins, dim]


class BinIndexEncoder(torch.nn.Module):
    """TorchScript-able latent binning: ``h [..., dim]`` -> bin indices ``[..., dim]``."""

    def __init__(self, num_bins: int, bin_range: float):
        super().__init__()
        self.num_bins = num_bins
        self.bin_range = float(bin_range)
        self.bin_width = 2 * self.bin_range / num_bins

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        mu = self.bin_range * torch.tanh(h / self.bin_range)
        indices = torch.floor((mu + self.bin_range) / self.bin_width).long()
        return indices.clamp(0, self.num_bins - 1)
