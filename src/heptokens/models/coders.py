# Simple encoders and decoders for use in autoencoder models.
import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Dict, Optional

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class BaseEncoder(nn.Module, ABC):
    """Abstract interface for VQ-VAE encoders.

    All encoders must accept a batch dict and return latent embeddings.
    The batch dict must contain at minimum the primary input tensor and,
    for variable-length data, a boolean validity mask.

    Contract:
        forward(batch: dict) -> Tensor of shape [batch, n_tokens, codebook_dim]
            where invalid (masked) positions are zeroed out.
    """

    @abstractmethod
    def forward(self, batch: dict) -> torch.Tensor:
        """Encode a batch to latent embeddings."""
        ...


class BaseDecoder(nn.Module, ABC):
    """Abstract interface for VQ-VAE decoders.

    All decoders must accept quantized embeddings + the original batch and
    return reconstructed data, as well as expose a compute_loss method.

    Contract:
        forward(z_q, batch) -> reconstructed Tensor
        compute_loss(z_q, batch) -> scalar loss Tensor
    """

    @abstractmethod
    def forward(self, z_q: torch.Tensor, batch: dict) -> torch.Tensor:
        """Decode quantized embeddings to data space."""
        ...

    @abstractmethod
    def compute_loss(self, z_q: torch.Tensor, batch: dict) -> torch.Tensor:
        """Compute reconstruction loss between decoded output and batch targets."""
        ...


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
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        model: nn.Module = CoderModel,
        input_key: str = "csts",
        mask_key: str = "mask",
        data_sample=None,
    ):
        super(Coder, self).__init__()
        # Build coder (encoder/decoder)
        self.coder = model(input_dim=input_dim, output_dim=output_dim)
        self.input_key = input_key
        self.mask_key = mask_key


class Encoder(Coder, BaseEncoder):
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode input data to latent embeddings.

        Args:
            batch: Dictionary containing the primary input tensor (key given by
                ``self.input_key``, default ``'csts'``) of shape
                [batch_size, n_tokens, d_vector] and an optional boolean validity
                mask (key given by ``self.mask_key``, default ``'mask'``) of
                shape [batch_size, n_tokens].

        Returns:
            Latent embeddings of shape [batch_size, n_tokens, codebook_dim]
            with invalid (masked) positions zeroed out.
        """
        mask = batch.get(self.mask_key)
        z_e = self.coder(batch[self.input_key], mask)
        return z_e


class Decoder(Coder, BaseDecoder):
    def forward(self, z_q: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode quantized embeddings.

        Args:
            z_q: Quantized embeddings [batch_size, n_tokens, codebook_dim]
            batch: Original batch dict; uses the mask at ``self.mask_key``
                to process only valid positions.

        Returns:
            Reconstructed tensor of shape [batch_size, n_tokens, d_vector]
            with invalid positions zeroed out.
        """
        mask = batch.get(self.mask_key)
        return self.coder(z_q, mask)

    def compute_loss(self, z_q: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute reconstruction loss given quantized embeddings and original batch.
        Default is L1 loss on valid (unmasked) positions.

        Args:
            z_q: Quantized embeddings [batch_size, n_tokens, codebook_dim]
            batch: Original batch dict with the primary input at ``self.input_key``
                and an optional mask at ``self.mask_key``.

        Returns:
            Reconstruction loss tensor (scalar).
        """
        targets = batch[self.input_key]
        mask = batch.get(self.mask_key)

        reconstructed = self.forward(z_q, batch)

        if mask is not None:
            recon_loss = nn.functional.l1_loss(reconstructed[mask], targets[mask])
        else:
            recon_loss = nn.functional.l1_loss(reconstructed, targets)
        return recon_loss


class CellDecoder(Decoder):
    """Decoder for calorimeter cell patch data with pixel-level sparsity handling.

    Expects the input key to hold a tensor of shape ``[B, T, 2*n_pixels]`` where
    the first half is the energy (energy^0.3) and the second half is a binary
    pixel-level occupancy indicator — both computed in ``CellDataset.__getitem__``.

    Loss:
      - Binary cross-entropy on the indicator for all pixels in non-empty patches
      - MSE on energy for pixels where the indicator is 1 (occupied pixels only)
    """

    def compute_loss(self, z_q: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        targets = batch[self.input_key]  # [B, T, 2*n_pixels]
        mask = batch.get(self.mask_key)  # [B, T] bool or None
        recon = self.forward(z_q, batch)  # [B, T, 2*n_pixels]

        n = targets.shape[-1] // 2
        energy_recon = recon[..., :n]
        ind_logits = recon[..., n:]
        energy_true = targets[..., :n]
        ind_true = targets[..., n:]  # binary float {0., 1.}

        if mask is not None:
            loss_ind = nn.functional.binary_cross_entropy_with_logits(
                ind_logits[mask], ind_true[mask]
            )
            pixel_mask = mask.unsqueeze(-1) & (ind_true > 0.5)
        else:
            loss_ind = nn.functional.binary_cross_entropy_with_logits(ind_logits, ind_true)
            pixel_mask = ind_true > 0.5

        if pixel_mask.any():
            loss_energy = nn.functional.mse_loss(energy_recon[pixel_mask], energy_true[pixel_mask])
        else:
            loss_energy = recon.sum() * 0.0  # differentiable zero

        return loss_energy + loss_ind


class DiffusionDecoder(BaseDecoder):
    """Decoder using diffusion model for reconstruction loss."""

    def __init__(
        self,
        input_dim: int,  # codebook_dim
        output_dim: int,  # d_vector (data dimension)
        model: nn.Module = CoderModel,
        num_timesteps: int = 1000,
        beta_schedule: str = "linear",
        input_key: str = "csts",
        mask_key: str = "mask",
        data_sample=None,
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

        self.input_key = input_key
        self.mask_key = mask_key

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
        mask = batch.get(self.mask_key)
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
        csts = batch[self.input_key]  # [batch, n_tokens, d_vector]
        mask = batch.get(self.mask_key)  # [batch, n_tokens] or None

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

        # L2 loss on predicted noise (only on valid positions if mask provided)
        if mask is not None:
            valid_noise = noise[mask]
            valid_predicted = predicted_noise[mask]
        else:
            valid_noise = noise.reshape(-1, noise.shape[-1])
            valid_predicted = predicted_noise.reshape(-1, predicted_noise.shape[-1])
        recon_loss = nn.functional.mse_loss(valid_predicted, valid_noise)

        return recon_loss


class ConditionalEncoder(BaseEncoder):
    """Encoder that concatenates a context stream from the batch before encoding.

    The inner model receives input of dimension (input_dim + context_dim).
    Context is read from a configurable batch key and concatenated with the
    primary input along the feature dimension.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        model: nn.Module = CoderModel,
        input_key: str = "x_cont",
        mask_key: str = "x_mask",
        context_key: str = "positions",
        data_sample: dict | None = None,
        context_dim: int | None = None,
    ):
        super().__init__()

        if context_dim is None:
            if data_sample is not None and isinstance(data_sample, Mapping):
                context_dim = data_sample[context_key].shape[-1]
            else:
                raise ValueError(
                    "context_dim must be provided or inferrable from a dict data_sample"
                )

        self.input_key = input_key
        self.mask_key = mask_key
        self.context_key = context_key
        self.coder = model(input_dim=input_dim + context_dim, output_dim=output_dim)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch[self.input_key]
        ctx = batch[self.context_key]
        mask = batch.get(self.mask_key)
        combined = torch.cat([x, ctx], dim=-1)
        return self.coder(combined, mask)


class ConditionalDecoder(BaseDecoder):
    """Decoder that concatenates a context stream from the batch with z_q before decoding.

    The inner model receives input of dimension (input_dim + context_dim) and
    outputs the primary feature dimension (output_dim). Loss is computed against
    the primary input key only.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        model: nn.Module = CoderModel,
        input_key: str = "x_cont",
        mask_key: str = "x_mask",
        context_key: str = "positions",
        data_sample: dict | None = None,
        context_dim: int | None = None,
    ):
        super().__init__()
        from collections.abc import Mapping

        if context_dim is None:
            if data_sample is not None and isinstance(data_sample, Mapping):
                context_dim = data_sample[context_key].shape[-1]
            else:
                raise ValueError(
                    "context_dim must be provided or inferrable from a dict data_sample"
                )

        self.input_key = input_key
        self.mask_key = mask_key
        self.context_key = context_key
        self.coder = model(input_dim=input_dim + context_dim, output_dim=output_dim)

    def forward(self, z_q: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        ctx = batch[self.context_key]
        mask = batch.get(self.mask_key)
        combined = torch.cat([z_q, ctx], dim=-1)
        return self.coder(combined, mask)

    def compute_loss(self, z_q: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        targets = batch[self.input_key]
        mask = batch.get(self.mask_key)
        reconstructed = self.forward(z_q, batch)
        if mask is not None:
            return nn.functional.l1_loss(reconstructed[mask], targets[mask])
        return nn.functional.l1_loss(reconstructed, targets)
