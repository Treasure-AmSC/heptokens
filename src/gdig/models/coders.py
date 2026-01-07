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


class DiffusionDecoder(nn.Module):
    """Decoder using diffusion model for reconstruction loss."""

    def __init__(
        self,
        input_dim: int,  # codebook_dim
        output_dim: int,  # d_vector (data dimension)
        model: nn.Module = CoderModel,
        num_timesteps: int = 1000,
        beta_schedule: str = "linear",
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.output_dim = output_dim
        self.latent_dim = input_dim

        # Model takes: [noisy_data (output_dim) + latent (latent_dim) + time_embed] -> output_dim
        self.time_embed_dim = 32
        self.time_embed = nn.Sequential(
            nn.Linear(1, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        # Input is: noisy_csts + z_q + time_embedding
        coder_input_dim = output_dim + input_dim + self.time_embed_dim
        self.coder = model(input_dim=coder_input_dim, output_dim=output_dim)

        # Initialize beta schedule
        if beta_schedule == "linear":
            betas = torch.linspace(0.0001, 0.02, num_timesteps)
        elif beta_schedule == "cosine":
            betas = self._cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

        # Precompute alphas
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine annealing schedule."""
        steps = torch.arange(timesteps + 1, dtype=torch.float32)
        alphas_cumprod = torch.cos(((steps / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def forward(
        self, z_q: torch.Tensor, batch: Dict[str, torch.Tensor], t: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass for inference - performs DDPM sampling.

        Args:
            z_q: Quantized embeddings [batch, n_csts, codebook_dim]
            batch: Batch dict (uses 'mask' if present)
            t: Ignored (full denoising always performed)

        Returns:
            Generated samples [batch, n_csts, d_vector]
        """
        mask = batch.get("mask")
        device = z_q.device
        batch_size, n_csts = z_q.shape[0], z_q.shape[1]

        # Start from pure noise
        x = torch.randn(batch_size, n_csts, self.output_dim, device=device)

        # Denoise step by step from T-1 to 0
        for t_idx in reversed(range(self.num_timesteps)):
            t_batch = torch.full((batch_size,), t_idx, device=device, dtype=torch.long)

            # Time embedding
            t_normalized = (t_batch.float() / self.num_timesteps).view(-1, 1)
            t_embed = self.time_embed(t_normalized)
            t_embed = t_embed.unsqueeze(1).expand(-1, n_csts, -1)

            # Predict noise
            model_input = torch.cat([x, z_q, t_embed], dim=-1)
            predicted_noise = self.coder(model_input, mask)

            # Compute denoising step
            alpha_cumprod = self.alphas_cumprod[t_idx]
            beta = self.betas[t_idx]

            # DDPM sampling formula
            if t_idx > 0:
                noise = torch.randn_like(x)
                alpha_cumprod_prev = self.alphas_cumprod[t_idx - 1]
            else:
                noise = torch.zeros_like(x)
                alpha_cumprod_prev = torch.tensor(1.0, device=device)

            # Posterior mean
            pred_x0 = (x - torch.sqrt(1 - alpha_cumprod) * predicted_noise) / torch.sqrt(
                alpha_cumprod
            )

            # Posterior variance
            posterior_variance = beta * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod)

            # Update x
            x = (
                torch.sqrt(alpha_cumprod_prev) * pred_x0
                + torch.sqrt(1 - alpha_cumprod_prev - posterior_variance) * predicted_noise
                + torch.sqrt(posterior_variance) * noise
            )

            # Mask invalid positions
            if mask is not None:
                x = x.masked_fill(~mask.unsqueeze(-1), 0.0)

        return x

    def compute_loss(self, z_q: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute diffusion loss (noise prediction loss).

        Args:
            z_q: Quantized embeddings [batch, n_csts, codebook_dim]
            batch: Original batch dict with 'csts' [batch, n_csts, d_vector] and 'mask'

        Returns:
            Reconstruction loss tensor
        """
        csts = batch["csts"]  # [batch, n_csts, d_vector]
        mask = batch["mask"]  # [batch, n_csts]

        # Sample random timestep per batch element
        t = torch.randint(0, self.num_timesteps, (csts.shape[0],), device=csts.device)

        # Sample noise
        noise = torch.randn_like(csts)  # [batch, n_csts, d_vector]

        # Add noise to clean data
        sqrt_alphas = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alphas = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        noisy_csts = sqrt_alphas * csts + sqrt_one_minus_alphas * noise

        # Time embedding
        t_normalized = (t.float() / self.num_timesteps).view(-1, 1)  # [batch, 1]
        t_embed = self.time_embed(t_normalized)  # [batch, time_embed_dim]
        t_embed = t_embed.unsqueeze(1).expand(
            -1, csts.shape[1], -1
        )  # [batch, n_csts, time_embed_dim]

        # Concatenate inputs: [noisy_data, latent, time]
        model_input = torch.cat([noisy_csts, z_q, t_embed], dim=-1)  # [batch, n_csts, input_dim]

        # Predict noise
        predicted_noise = self.coder(model_input, mask)  # [batch, n_csts, d_vector]

        # L2 loss on predicted noise (only on valid positions)
        valid_noise = noise[mask]
        valid_predicted = predicted_noise[mask]
        recon_loss = nn.functional.mse_loss(valid_predicted, valid_noise)

        return recon_loss
