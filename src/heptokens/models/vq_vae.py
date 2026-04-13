"""PyTorch Lightning wrapper for VQ-VAE with ResidualVQ."""

import logging
from typing import Dict, Tuple

import torch
from lightning import LightningModule
from vector_quantize_pytorch import ResidualVQ

from heptokens.models.coders import Decoder, Encoder
from heptokens.models.utils import ScheduledOptimiserMixin, compute_codebook_utilization

log = logging.getLogger(__name__)


class LitVqVae(ScheduledOptimiserMixin, LightningModule):
    """Lightning Module wrapper for VQ-VAE with ResidualVQ.

    Args:
        encoder_hidden_dims: List of hidden dimensions for encoder MLP
        decoder_hidden_dims: List of hidden dimensions for decoder MLP
        codebook_size: Number of codes in each codebook
        codebook_dim: Dimension of each code
        num_quantizers: Number of residual quantizers
        commitment_weight: Weight for commitment loss
        learning_rate: Learning rate for optimizer
        reconstruction_weight: Weight for reconstruction loss
        data_sample: Sample data for initialization (optional, for compatibility)
        n_classes: Number of classes (optional, for compatibility)
        **kwargs: Additional arguments passed to ResidualVQ
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        codebook_size: int = 1024,
        codebook_dim: int = 256,
        num_quantizers: int = 8,
        commitment_weight: float = 1.0,
        learning_rate: float = 1e-3,
        reconstruction_weight: float = 1.0,
        optimizer=None,
        scheduler=None,
        data_sample: torch.Tensor = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.reconstruction_weight = reconstruction_weight
        self.codebook_size = codebook_size

        # Infer input dimension from data_sample if provided
        if data_sample is not None:
            input_dim = data_sample["csts"].shape[-1]
        else:
            input_dim = 3  # Default for now

        # Declare encoder
        self.encoder = encoder(input_dim=input_dim, output_dim=codebook_dim)
        # Declare decoder
        self.decoder = decoder(input_dim=codebook_dim, output_dim=input_dim)

        # Vector quantization
        self.vector_quantization = ResidualVQ(
            dim=codebook_dim,
            codebook_size=codebook_size,
            num_quantizers=num_quantizers,
            commitment=commitment_weight,
        )

    def encode(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode input data to quantized embeddings.

        Args:
            batch: Dictionary containing 'csts' tensor of shape [batch_size, n_csts, d_vector]
                and 'mask' tensor of shape [batch_size, n_csts]

        Returns:
            z_q: Quantized embeddings of shape [n_valid, codebook_dim]
            indices: Indices of shape [n_valid, num_quantizers]
            commit_loss: Commitment loss tensor
        """

        # Encode
        z_e = self.encoder(batch)  # [batch_size, n_csts, codebook_dim]

        # Quantize
        z_q, indices_batched, commit_loss = self.vector_quantization(z_e)

        # Move dimensions [n_codes, batch_dim, n_csts] -> [batch_dim, n_csts, n_codes]
        indices = indices_batched.permute(1, 2, 0).contiguous()
        # Set masked positions to -1
        indices = indices.masked_fill(~batch["mask"].unsqueeze(-1), -1)

        return z_q, indices, commit_loss.mean()

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

        # Log codebook utilization per quantizer
        utils = compute_codebook_utilization(indices, batch["mask"], self.codebook_size)
        for q, u in enumerate(utils):
            self.log(f"train/codebook_util_q{q}", u)
        self.log("train/codebook_util_avg", sum(utils) / len(utils))

        return total_loss

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

        # Log codebook utilization per quantizer
        utils = compute_codebook_utilization(indices, batch["mask"], self.codebook_size)
        for q, u in enumerate(utils):
            self.log(f"val/codebook_util_q{q}", u)
        self.log("val/codebook_util_avg", sum(utils) / len(utils))

        return {"val_loss": total_loss, "indices": indices}

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Predict step returns indices."""
        return self(batch)
