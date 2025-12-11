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
            x = x[mask]  # Select valid inputs
        # TODO: if a mask is passed do you need to do a reshape?
        return self.model(x)


class Coder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int] = [128, 256, 512]):
        super(Coder, self).__init__()
        # Build decoder
        decoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            decoder_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                ]
            )
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, output_dim))
        self.coder = nn.Sequential(*decoder_layers)
        # TODO: replace this with semantics like the below and pass model class to use as argument
        # self.coder = model(input_dim=input_dim, output_dim=output_dim)


class Encoder(Coder):
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode input data to latent embeddings.

        Args:
            batch: Dictionary containing 'csts' tensor of shape [batch_size, n_csts, d_vector]
                and 'mask' tensor of shape [batch_size, n_csts]

        Returns:
            Latent embeddings of shape [n_valid, codebook_dim]
        """
        csts = batch["csts"]
        mask = batch["mask"]

        # Select valid constituents
        valid_csts = csts[mask]  # [n_valid, d_vector]

        # Encode
        z_e = self.coder(valid_csts)  # [n_valid, codebook_dim]
        # TODO: this should be something like
        # z_e = self.coder(csts, mask)

        return z_e


class Decoder(Coder):
    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decode quantized embeddings.

        Args:
            z_q: Quantized embeddings [n_valid, codebook_dim]
            batch: Original batch dict with 'csts' and 'mask'

        Returns:
            Tuple of (reconstructed_csts, reconstruction_loss)
        """
        # Decode
        return self.coder(z_q)  # [n_valid, d_vector]

    def compute_loss(self, z_q: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute reconstruction loss given quantized embeddings and original batch.
        Default is MSE loss.

        Args:
            z_q: Quantized embeddings [n_valid, codebook_dim]
            batch: Original batch dict with 'csts' and 'mask'

        Returns:
            Reconstruction loss tensor
        """
        csts = batch["csts"]
        mask = batch["mask"]

        # Decode
        x_hat_valid = self.coder(z_q)  # [n_valid, d_vector]

        # Compute reconstruction loss (MSE)
        valid_csts = csts[mask]  # [n_valid, d_vector]
        recon_loss = nn.functional.mse_loss(x_hat_valid, valid_csts)

        return recon_loss
