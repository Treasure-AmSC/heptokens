"""Template encoder and decoder for a new modality.

Copy this file, rename it to <your_modality>_coders.py, and fill in the
sections marked with TODO.

Both classes extend the abstract base classes BaseEncoder / BaseDecoder,
which enforce the interface expected by LitVqVae (vq_vae.py).

Interfaces (do not change the signatures):
  Encoder.forward(batch: dict)            -> Tensor [batch, n_tokens, codebook_dim]
  Decoder.forward(z_q, batch)             -> Tensor [batch, n_tokens, d_features]
  Decoder.compute_loss(z_q, batch)        -> scalar Tensor
"""

import torch
import torch.nn as nn

from heptokens.models.coders import BaseDecoder, BaseEncoder

# TODO: set these to match the keys used in your Dataset.__getitem__
INPUT_KEY = "csts"  # primary input tensor key
MASK_KEY = "mask"  # validity mask key (set to None if unused)


# ---------------------------------------------------------------------------
# 1. Encoder
# ---------------------------------------------------------------------------


class MyEncoder(BaseEncoder):
    """Encode <modality> tokens to a latent space suitable for VQ-VAE.

    The encoder maps each token independently (set-based) or as a sequence
    to a vector of size ``codebook_dim``.

    Args:
        input_dim: Dimensionality of each input token (d_features).
        codebook_dim: Dimensionality of the output latent space.
    """

    def __init__(self, input_dim: int, codebook_dim: int):
        super().__init__()
        # TODO: define your encoder network.
        # Options:
        #   - MLP (token-wise, position-invariant):
        #       from heptokens.models.coders import CoderModel
        #       self.net = CoderModel(input_dim, codebook_dim)
        #   - Transformer (attends across tokens):
        #       from heptokens.models.transformer import Transformer
        #       self.net = Transformer(input_dim, codebook_dim, ...)
        raise NotImplementedError("Define your encoder network.")

        self.input_key = INPUT_KEY
        self.mask_key = MASK_KEY

    def forward(self, batch: dict) -> torch.Tensor:
        """Encode tokens to latent embeddings.

        Args:
            batch: dict with self.input_key -> [B, N, d] and optionally
                   self.mask_key -> [B, N] boolean.

        Returns:
            Tensor of shape [B, N, codebook_dim] with invalid positions zeroed.
        """
        # x = batch[self.input_key]  # [B, N, d_features]
        # mask = batch.get(self.mask_key)  # [B, N] or None

        # TODO: run your network.
        # Token-wise MLP example (CoderModel handles masking internally):
        #   z = self.net(x, mask)
        # Transformer example:
        #   z = self.net(x, mask)
        raise NotImplementedError("Implement encoder forward pass.")


# ---------------------------------------------------------------------------
# 2. Decoder
# ---------------------------------------------------------------------------


class MyDecoder(BaseDecoder):
    """Decode VQ-VAE quantized embeddings back to <modality> feature space.

    Args:
        codebook_dim: Dimensionality of quantized embeddings (input to decoder).
        output_dim: Dimensionality of each reconstructed token (d_features).
    """

    def __init__(self, codebook_dim: int, output_dim: int):
        super().__init__()
        # TODO: define your decoder network (mirrors the encoder structure).
        raise NotImplementedError("Define your decoder network.")

        self.input_key = INPUT_KEY
        self.mask_key = MASK_KEY

    def forward(self, z_q: torch.Tensor, batch: dict) -> torch.Tensor:
        """Decode quantized embeddings to data space.

        Args:
            z_q: [B, N, codebook_dim] quantized latent vectors.
            batch: original batch dict (used for mask if needed).

        Returns:
            Tensor of shape [B, N, output_dim].
        """
        # mask = batch.get(self.mask_key)

        # TODO: run your decoder.
        raise NotImplementedError("Implement decoder forward pass.")

    def compute_loss(self, z_q: torch.Tensor, batch: dict) -> torch.Tensor:
        """Compute reconstruction loss.

        Args:
            z_q: [B, N, codebook_dim] quantized latent vectors.
            batch: original batch dict with ground-truth at self.input_key.

        Returns:
            Scalar loss tensor.
        """
        targets = batch[self.input_key]
        mask = batch.get(self.mask_key)

        recon = self.forward(z_q, batch)

        # TODO: choose an appropriate loss.
        # L1 on valid positions (variable-length):
        if mask is not None:
            loss = nn.functional.l1_loss(recon[mask], targets[mask])
        else:
            loss = nn.functional.l1_loss(recon, targets)
        return loss
