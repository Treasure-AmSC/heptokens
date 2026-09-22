"""Unit tests for CellDecoder's per-pixel energy-loss weighting (energy_loss_alpha)."""

import torch

from heptokens.models.coders import CellDecoder, CoderModel


def _original_cell_loss(recon, targets, mask):
    """Reference implementation of the pre-weighting CellDecoder loss."""
    n = targets.shape[-1] // 2
    energy_recon = recon[..., :n]
    ind_logits = recon[..., n:]
    energy_true = targets[..., :n]
    ind_true = targets[..., n:]

    if mask is not None:
        loss_ind = torch.nn.functional.binary_cross_entropy_with_logits(
            ind_logits[mask], ind_true[mask]
        )
        pixel_mask = mask.unsqueeze(-1) & (ind_true > 0.5)
    else:
        loss_ind = torch.nn.functional.binary_cross_entropy_with_logits(ind_logits, ind_true)
        pixel_mask = ind_true > 0.5

    if pixel_mask.any():
        loss_energy = torch.nn.functional.mse_loss(
            energy_recon[pixel_mask], energy_true[pixel_mask]
        )
    else:
        loss_energy = recon.sum() * 0.0
    return loss_energy + loss_ind


def _make_batch(seed=0, n_pixels=8, batch=4, tokens=6):
    rng = torch.Generator().manual_seed(seed)
    lin_energy = torch.rand(batch, tokens, n_pixels, generator=rng) * 50.0
    # Randomly zero out some pixels so the occupancy indicator is nontrivial.
    keep = torch.rand(batch, tokens, n_pixels, generator=rng) > 0.3
    lin_energy = lin_energy * keep
    energy = lin_energy.pow(0.3)  # stored as E**p (p=0.3), as in preprocessing
    indicator = (lin_energy > 0).float()
    targets = torch.cat([energy, indicator], dim=-1)  # [B, T, 2*n_pixels]
    # Some patches empty (all-zero energy) -> masked out.
    mask = energy.sum(dim=-1) > 0
    return targets, mask


def test_alpha_zero_is_bitwise_identical():
    """energy_loss_alpha=0 must reproduce the original loss exactly."""
    torch.manual_seed(0)
    n_pixels, codebook_dim = 8, 5
    decoder = CellDecoder(
        input_dim=codebook_dim,
        output_dim=2 * n_pixels,
        model=CoderModel,
        input_key="energy",
        mask_key="mask",
        energy_loss_alpha=0.0,
    )
    decoder.eval()

    targets, mask = _make_batch(seed=1, n_pixels=n_pixels)
    batch = {"energy": targets, "mask": mask}
    z_q = torch.randn(targets.shape[0], targets.shape[1], codebook_dim)

    with torch.no_grad():
        recon = decoder.forward(z_q, batch)
        expected = _original_cell_loss(recon, targets, mask)
        got = decoder.compute_loss(z_q, batch)

    assert torch.equal(got, expected)


def test_alpha_positive_upweights_high_energy_pixel():
    """With a mis-reconstructed high-energy pixel (others perfect), the
    weighted energy MSE (alpha>0) must exceed the unweighted one."""
    n_pixels = 4
    p = 0.3
    lin_energy = torch.tensor([[[100.0, 1.0, 1.0, 1.0]]])  # [1, 1, 4]
    energy_true = lin_energy.pow(p)
    indicator = torch.ones_like(energy_true)
    targets = torch.cat([energy_true, indicator], dim=-1)  # [1, 1, 8]
    mask = torch.ones(1, 1, dtype=torch.bool)

    # Perfect reconstruction except an error on the high-energy pixel.
    energy_recon = energy_true.clone()
    energy_recon[..., 0] -= 0.5
    ind_logits = torch.full_like(energy_true, 10.0)  # confident, ~0 BCE
    fixed_recon = torch.cat([energy_recon, ind_logits], dim=-1)

    batch = {"energy": targets, "mask": mask}
    z_q = torch.zeros(1, 1, 5)

    for alpha in (0.0, 0.7, 1.4):
        decoder = CellDecoder(
            input_dim=5,
            output_dim=2 * n_pixels,
            model=CoderModel,
            input_key="energy",
            mask_key="mask",
            energy_loss_alpha=alpha,
            energy_power=p,
        )
        decoder.forward = lambda _z, _b: fixed_recon  # noqa: B023 -- deterministic recon
        decoder.compute_loss(z_q, batch)
        m = decoder.last_metrics
        if alpha == 0.0:
            assert torch.isclose(m["energy_mse_weighted"], m["energy_mse"])
        else:
            assert m["energy_mse_weighted"] > m["energy_mse"]
