"""PyTorch Lightning wrapper for a VQ-VAE with a pluggable quantizer."""

import logging
from collections.abc import Callable, Mapping
from typing import Dict, Tuple

import torch
from lightning import LightningModule

from heptokens.models.coders import Decoder, Encoder
from heptokens.models.quantizers import Quantizer
from heptokens.models.utils import ScheduledOptimiserMixin, compute_codebook_utilization

log = logging.getLogger(__name__)


class LitVqVae(ScheduledOptimiserMixin, LightningModule):
    """Lightning Module for a VQ-VAE: encoder -> quantizer -> decoder.

    Args:
        encoder: Partial Encoder, called with input_dim, output_dim=codebook_dim, data_sample
        decoder: Partial Decoder, called with input_dim=codebook_dim, output_dim, data_sample
        quantizer: Partial :class:`~heptokens.models.quantizers.Quantizer`, called with
            ``dim=codebook_dim``, e.g. ``partial(ResidualVQQuantizer, codebook_size=512,
            num_quantizers=3)`` or ``partial(LatentBins, num_bins=4096, ...)``. In Hydra, a
            ``quantizer:`` block (see ``conf/model/quantizer/``).
        codebook_dim: Latent dimension: encoder output, quantizer and decoder input
        learning_rate: Learning rate for optimizer
        reconstruction_weight: Weight for reconstruction loss
        data_sample: Sample data for initialization (optional, for compatibility)
        **kwargs: Extra hyperparameters (e.g. ``scheduler_interval``)
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        quantizer: Callable[..., Quantizer],
        codebook_dim: int = 256,
        learning_rate: float = 1e-3,
        reconstruction_weight: float = 1.0,
        optimizer=None,
        scheduler=None,
        input_key: str = "csts",
        mask_key: str | None = "mask",
        data_sample: torch.Tensor | dict = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.reconstruction_weight = reconstruction_weight
        self.mask_key = mask_key

        # Infer input dimension from data_sample if provided.
        if isinstance(data_sample, Mapping):
            input_dim = data_sample[input_key].shape[-1]
        elif data_sample is not None:
            input_dim = data_sample.shape[-1]
        else:
            input_dim = 3  # Default for now

        # Declare encoder
        self.encoder = encoder(
            input_dim=input_dim, output_dim=codebook_dim, data_sample=data_sample
        )
        # Declare decoder
        self.decoder = decoder(
            input_dim=codebook_dim, output_dim=input_dim, data_sample=data_sample
        )
        self.vector_quantization = quantizer(dim=codebook_dim)

    @property
    def codebook_size(self) -> int:
        """Values each token can take."""
        return self.vector_quantization.codebook_size

    def _mask_indices(self, indices: torch.Tensor, batch: Dict[str, torch.Tensor]):
        """Set tokens at padded positions to -1. The mask is optional; some modalities have
        no padding, and mask_key may be None to disable masking entirely."""
        mask = batch.get(self.mask_key)
        if mask is not None:
            indices = indices.masked_fill(~mask.unsqueeze(-1), -1)
        return indices

    def encode(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode input data to quantized embeddings.

        Args:
            batch: Dictionary containing 'csts' tensor of shape [batch_size, n_csts, d_vector]
                and 'mask' tensor of shape [batch_size, n_csts]

        Returns:
            z_q: Quantized embeddings of shape [batch_size, n_csts, codebook_dim]
            indices: Tokens of shape [batch_size, n_csts, num_quantizers], -1 where masked
            commit_loss: Scalar quantizer loss
        """
        z_e = self.encoder(batch)  # [batch_size, n_csts, codebook_dim]
        # The quantizer gets the mask so training updates (e.g. dead-code revival) can
        # ignore padding.
        z_q, indices, commit_loss = self.vector_quantization(z_e, mask=batch.get(self.mask_key))
        return z_q, self._mask_indices(indices, batch), commit_loss

    @torch.no_grad()
    def encode_indices(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Tokens only, via the quantizer's inference path (no training state updates).

        Returns:
            indices: [batch_size, n_csts, num_quantizers], -1 where masked
        """
        indices = self.vector_quantization.encode_indices(self.encoder(batch))
        return self._mask_indices(indices, batch)

    def decode(self, z_q: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode quantized embeddings.

        Args:
            z_q: Quantized embeddings [n_valid, codebook_dim]
            batch: Original batch dict with 'csts' and 'mask'

        Returns:
            reconstructed_csts
        """
        # Decode
        x_hat_valid = self.decoder(z_q, batch)  # [n_valid, d_vector]
        return x_hat_valid

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict cluster labels for input data.

        Args:
            batch: Dictionary containing 'csts' and 'mask'

        Returns:
            Indices of shape [batch_size, n_csts, num_quantizers]
            Masked positions will have index -1
        """
        return self.encode(batch)[1]

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step with reconstruction and commitment loss.

        Args:
            batch: Dictionary containing data tensors
            batch_idx: Index of current batch

        Returns:
            Total loss tensor
        """
        # Encode
        z_q, indices, commit_loss = self.encode(batch)

        # Compute reconstruction loss
        recon_loss = self.decoder.compute_loss(z_q, batch)

        # Total loss
        total_loss = self.reconstruction_weight * recon_loss + commit_loss

        # Log metrics
        self.log("train/total_loss", total_loss, prog_bar=True)
        self.log("train/recon_loss", recon_loss, prog_bar=True)
        self.log("train/commit_loss", commit_loss, prog_bar=True)

        # Log any per-decoder loss breakdown (e.g. CellDecoder's weighted vs
        # unweighted energy MSE and the configured energy_loss_alpha).
        self._log_decoder_metrics("train")

        # Log codebook utilization per quantizer
        utils = compute_codebook_utilization(indices, batch.get(self.mask_key), self.codebook_size)
        for q, u in enumerate(utils):
            self.log(f"train/codebook_util_q{q}", u)
        self.log("train/codebook_util_avg", sum(utils) / len(utils))

        return total_loss

    def _log_decoder_metrics(self, stage: str) -> None:
        """Log the decoder's optional ``last_metrics`` breakdown, if present."""
        extra = getattr(self.decoder, "last_metrics", None)
        if extra:
            for name, value in extra.items():
                self.log(f"{stage}/{name}", value)

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Validation step."""
        # Encode
        z_q, indices, commit_loss = self.encode(batch)

        # Compute reconstruction loss
        recon_loss = self.decoder.compute_loss(z_q, batch)

        # TODO: plot some reconstructions?
        # reconstruction = self.decode(z_q, batch)

        # Total loss
        total_loss = self.reconstruction_weight * recon_loss + commit_loss

        # Log metrics
        self.log("val/total_loss", total_loss, prog_bar=True)
        self.log("val/recon_loss", recon_loss, prog_bar=True)
        self.log("val/commit_loss", commit_loss, prog_bar=True)

        # Log any per-decoder loss breakdown (see training_step).
        self._log_decoder_metrics("val")

        # Log codebook utilization per quantizer
        utils = compute_codebook_utilization(indices, batch.get(self.mask_key), self.codebook_size)
        for q, u in enumerate(utils):
            self.log(f"val/codebook_util_q{q}", u)
        self.log("val/codebook_util_avg", sum(utils) / len(utils))

        return {"val_loss": total_loss, "indices": indices}

    def predict_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Predict step returns indices, labels, and eventNumber if available."""
        result = {"indices": self.encode_indices(batch), "labels": batch["labels"]}
        if "eventNumber" in batch:
            result["eventNumber"] = batch["eventNumber"]
        return result
