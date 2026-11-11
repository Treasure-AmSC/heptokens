"""LitVqVae with an injected Quantizer: behaviour shared by every quantizer implementation."""

from functools import partial

import pytest
import torch
from hydra import compose, initialize_config_module
from hydra.utils import instantiate

from heptokens.export_tokens import extract_codebooks
from heptokens.models.coders import CoderModel, Decoder, Encoder
from heptokens.models.quantizers import LatentBins, Quantizer, ResidualVQQuantizer
from heptokens.models.vq_vae import LitVqVae
from heptokens.traced_modules import build_inference_modules

B, T, D_IN, D = 3, 5, 3, 4


def make_batch() -> dict:
    torch.manual_seed(1)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[0, -2:] = False
    return {"csts": torch.randn(B, T, D_IN), "mask": mask}


def make_model(quantizer) -> LitVqVae:
    torch.manual_seed(0)
    return LitVqVae(
        encoder=partial(Encoder, model=partial(CoderModel, hidden_dims=[8])),
        decoder=partial(Decoder, model=partial(CoderModel, hidden_dims=[8])),
        quantizer=quantizer,
        codebook_dim=D,
        data_sample=make_batch(),
    )


def test_injected_residual_vq_tokenizes_each_element():
    model = make_model(partial(ResidualVQQuantizer, codebook_size=8, num_quantizers=2)).eval()
    batch = make_batch()

    z_q, indices, loss = model.encode(batch)

    assert z_q.shape == (B, T, D) and loss.ndim == 0
    assert indices.shape == (B, T, 2)
    assert (indices[~batch["mask"]] == -1).all()
    assert ((indices[batch["mask"]] >= 0) & (indices[batch["mask"]] < 8)).all()
    assert torch.equal(model.encode_indices(batch), indices)


def test_injected_latent_bins_tokenize_each_latent_dimension():
    model = make_model(partial(LatentBins, num_bins=16, bin_range=4.0, noise_std_bins=0.29)).eval()
    batch = make_batch()

    z_q, indices, loss = model.encode(batch)

    assert z_q.shape == (B, T, D) and loss.ndim == 0
    assert indices.shape == (B, T, D)  # one token per latent dimension
    assert (indices[~batch["mask"]] == -1).all()
    assert ((indices[batch["mask"]] >= 0) & (indices[batch["mask"]] < 16)).all()
    assert torch.equal(model.encode_indices(batch), indices)
    assert model.codebook_size == 16


VQVAE_CONFIGS = [
    "vqvae",
    "transformer_vqvae",
    "set2vec_vqvae",
    "new_modality_vqvae",
    "transformer_latent_bins",
]


@pytest.mark.parametrize("name", VQVAE_CONFIGS)
def test_packaged_configs_build_their_quantizer(name):
    with initialize_config_module(config_module="heptokens.conf", version_base=None):
        cfg = compose(config_name=f"model/{name}").model
    quantizer = instantiate(cfg.quantizer)(dim=cfg.codebook_dim)
    assert isinstance(quantizer, Quantizer)
    _, indices, _ = quantizer(torch.randn(2, 5, cfg.codebook_dim))
    assert indices.shape == (2, 5, quantizer.num_quantizers)


def test_residual_vq_revival_never_turns_padding_into_codes():
    """With dead-code revival on, padded positions are never sampled as new codes."""
    x = torch.zeros(1, 64, D)  # padding is exactly zero
    x[0, :4] = 3.0  # four valid elements sharing one vector
    mask = torch.zeros(1, 64, dtype=torch.bool)
    mask[0, :4] = True

    def codebook_after_one_step(mask):
        torch.manual_seed(0)
        quantizer = ResidualVQQuantizer(
            dim=D, codebook_size=8, num_quantizers=1, threshold_ema_dead_code=1.0
        ).train()
        quantizer(x, mask=mask)
        return quantizer.codebooks()[0]  # [codebook_size, dim]

    is_padding = (codebook_after_one_step(None) == 0).all(-1)
    assert is_padding.any(), "without the mask, revival should pick up padding"
    assert not (codebook_after_one_step(mask) == 0).all(-1).any()


class SignQuantizer(Quantizer):
    """A minimal custom quantizer: one binary token per latent dimension (its sign)."""

    def __init__(self, *, dim: int):
        super().__init__()
        self.dim, self.num_quantizers, self.codebook_size = dim, dim, 2

    def forward(self, z_e, mask=None):
        indices = self.encode_indices(z_e)
        z_q = z_e + (indices.to(z_e) * 2 - 1 - z_e).detach()  # straight-through to +-1
        return z_q, indices, (z_e.abs() - 1).pow(2).mean()

    def index_encoder(self):
        return SignIndexEncoder()

    def codebooks(self):
        values = torch.tensor([-1.0, 1.0])  # [codebook_size]
        return torch.eye(self.dim)[:, None, :] * values[None, :, None]  # [Q, 2, dim]


class SignIndexEncoder(torch.nn.Module):
    def forward(self, z_e: torch.Tensor) -> torch.Tensor:
        return (z_e > 0).long()


def test_a_custom_quantizer_trains_tokenizes_and_traces_without_touching_litvqvae():
    model = make_model(SignQuantizer)
    batch = make_batch()
    assert torch.isfinite(model.training_step(batch, 0))

    model.eval()
    z_q, indices, _ = model.encode(batch)
    valid = batch["mask"]
    codebooks = torch.from_numpy(extract_codebooks(model))  # what token export saves
    z_export = sum(codebooks[q, indices[valid][:, q]] for q in range(D))
    assert torch.equal(z_export, z_q[valid])

    modules = build_inference_modules(model)
    assert torch.equal(modules["index_encoder"](batch["csts"], valid), indices)
