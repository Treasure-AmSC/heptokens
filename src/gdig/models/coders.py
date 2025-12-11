# Simple encoders and decoders for use in autoencoder models.
import logging
from typing import Dict, Optional

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class CoderModel(nn.Module):
    """
    Base class for Encoder and Decoder models.
    Default is a simple MLP with LayerNorm and ReLU activations.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int] = [128, 256, 512]):
        super(CoderModel, self).__init__()
        # Build model
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the model."""
        if mask is not None:
            # Select valid inputs
            x = x[mask]
        # Encode valid inputs
        encoding = self.model(x)
        if mask is not None:
            # Create full output tensor with zeros for invalid positions
            full_encoding = torch.zeros(
                (*mask.shape, encoding.shape[1]), device=encoding.device, dtype=encoding.dtype
            )
            full_encoding[mask] = encoding
            return full_encoding
        else:
            return encoding


class Coder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, model: nn.Module = CoderModel):
        super(Coder, self).__init__()
        # Build coder (encoder/decoder)
        self.coder = model(input_dim=input_dim, output_dim=output_dim)


class Encoder(Coder):
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode input data to latent embeddings.

        Args:
            batch: Dictionary containing 'csts' tensor of shape [batch_size, n_csts, d_vector]
                and 'mask' tensor of shape [batch_size, n_csts]

        Returns:
            Latent embeddings of shape [n_valid, codebook_dim]
        """

        # Encode
        z_e = self.coder(batch["csts"], batch["mask"])

        return z_e


class Decoder(Coder):
    def forward(self, z_q: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode quantized embeddings.

        Args:
            z_q: Quantized embeddings [n_valid, codebook_dim]
            batch: Original batch dict with 'csts' and 'mask'

        Returns:
            Tuple of (reconstructed_csts, reconstruction_loss)
        """
        # Decode
        return self.coder(z_q, batch["mask"])  # [n_valid, d_vector]

    def compute_loss(self, z_q: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute reconstruction loss given quantized embeddings and original batch.
        Default is L1 loss.

        Args:
            z_q: Quantized embeddings [n_valid, codebook_dim]
            batch: Original batch dict with 'csts' and 'mask'

        Returns:
            Reconstruction loss tensor
        """
        csts = batch["csts"]
        mask = batch["mask"]

        # Decode
        reconstructed_csts = self.forward(z_q, batch)

        # Compute reconstruction loss (MSE)
        valid_csts = csts[mask]  # [n_valid, d_vector]
        recon_loss = nn.functional.l1_loss(reconstructed_csts[mask], valid_csts)
        return recon_loss
