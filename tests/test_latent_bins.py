"""Latent-binning bottleneck (LatentBins) and its LitVqVae integration."""

from functools import partial
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from omegaconf import OmegaConf

from heptokens.export_tokens import export, extract_codebooks
from heptokens.models.coders import CellDecoder, Encoder
from heptokens.models.quantizers import LatentBins, ResidualVQQuantizer
from heptokens.models.transformer import Transformer
from heptokens.models.vq_vae import LitVqVae
from heptokens.train import train


def test_eval_latent_is_the_bin_centre():
    bins = LatentBins(dim=3, num_bins=4, bin_range=1.0, noise_std_bins=0.29).eval()
    h = torch.tensor([[[0.0, 50.0, -50.0]]])  # mu = 0, ~1, ~-1
    z, indices, loss = bins(h)
    assert indices[0, 0].tolist() == [2, 3, 0]
    assert torch.allclose(z, torch.tensor([[[0.25, 0.75, -0.75]]]))
    assert indices.shape == (1, 1, 3) and loss.ndim == 0 and not loss.any()

    h = 3 * torch.randn(8, 5, 3)
    z, indices, _ = bins(h)
    mu = torch.tanh(h)
    assert (z - mu).abs().max() <= 0.25 + 1e-6
    assert torch.equal(bins.get_output_from_indices(indices), z)


def test_training_noise_is_noise_std_bins_bin_widths():
    bins = LatentBins(dim=4, num_bins=8, bin_range=2.0, noise_std_bins=0.29).train()
    z, indices, _ = bins(torch.zeros(1000, 50, 4))
    assert abs(z.std().item() / (0.29 * 0.5) - 1) < 0.02
    assert (indices == 4).all()  # tokens are mu's bins, noise-free


def make_model(quantizer) -> LitVqVae:
    coder = partial(Transformer, d_model=16, n_heads=2, num_layers=1, dim_feedforward=32, pos_dim=3)
    keys = dict(input_key="energy", mask_key="mask", positions_key="positions")
    torch.manual_seed(0)
    return LitVqVae(
        encoder=partial(Encoder, model=coder, **keys),
        decoder=partial(CellDecoder, model=coder, **keys),
        codebook_dim=4,
        input_key="energy",
        data_sample=make_batch(),
        quantizer=quantizer,
    )


def make_batch() -> dict:
    torch.manual_seed(1)
    mask = torch.rand(2, 6) > 0.3
    energy = torch.rand(2, 6, 128) * mask.unsqueeze(-1)
    energy[..., 64:] = (energy[..., :64] > 0.5).float()
    return {"energy": energy, "mask": mask, "positions": torch.randn(2, 6, 3)}


def test_tokens_decode_as_the_eval_latent():
    model = make_model(partial(LatentBins, num_bins=4096, bin_range=4.0, noise_std_bins=0.29))
    assert isinstance(model.vector_quantization, LatentBins) and model.codebook_size == 4096
    batch = make_batch()
    assert torch.isfinite(model.training_step(batch, 0))

    model.eval()
    z_q, indices, _ = model.encode(batch)
    assert torch.equal(model.encode_indices(batch), indices)
    assert (indices[~batch["mask"]] == -1).all() and (indices[batch["mask"]] >= 0).all()
    z_tokens = model.vector_quantization.get_output_from_indices(indices.clamp(min=0))
    valid = batch["mask"]
    assert torch.equal(model.decode(z_tokens, batch)[valid], model.decode(z_q, batch)[valid])


def test_exported_codebooks_reproduce_the_eval_latent():
    model = make_model(partial(LatentBins, num_bins=64, bin_range=4.0, noise_std_bins=0.29)).eval()
    batch = make_batch()
    z_q, indices, _ = model.encode(batch)  # [B, P, dim] each
    codebooks = torch.from_numpy(extract_codebooks(model))  # [dim, num_bins, dim]
    assert codebooks.shape == (4, 64, 4)
    valid = batch["mask"]
    idx = indices[valid]  # [n_valid, dim]
    z_export = sum(codebooks[q, idx[:, q]] for q in range(codebooks.shape[0]))  # [n_valid, dim]
    assert torch.equal(z_export, z_q[valid])


def test_transformer_latent_bins_config():
    with initialize_config_module(config_module="heptokens.conf", version_base=None):
        cfg = compose(config_name="model/transformer_latent_bins").model
    model = instantiate(cfg)
    assert isinstance(model.vector_quantization, LatentBins)
    assert model.vector_quantization.dim == cfg.codebook_dim
    assert model.codebook_size == 4096
    batch = {"csts": torch.randn(2, 5, 3), "mask": torch.ones(2, 5, dtype=torch.bool)}
    assert torch.isfinite(model.training_step(batch, 0))


def _datamodule_cfg(hdf5_file: Path, train_frac: float, val_frac: float, test_frac: float):
    return {
        "_target_": "heptokens.data.structured_array.StructuredArrayModule",
        "dataset": {
            "_target_": "heptokens.data.structured_array.StructuredArrayHDF5Dataset",
            "_partial_": True,
            "file_path": str(hdf5_file),
            "obj_group": "jets",
            "set_group": "tracks",
            "obj_features": ["pt", "eta"],
            "set_features": ["pt", "eta", "phi"],
            "label_key": "label",
            "mask_key": "valid",
        },
        "data_path": str(hdf5_file),  # export names its output after this file
        "train_frac": train_frac,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "batch_size": 32,
        "num_workers": 0,
    }


# quantizer config, quantizer class, tokens per element, values per token
QUANTIZERS = {
    "latent_bins": (
        {
            "_target_": "heptokens.models.quantizers.LatentBins",
            "_partial_": True,
            "num_bins": 64,
            "bin_range": 4.0,
            "noise_std_bins": 0.29,
        },
        LatentBins,
        4,
        64,
    ),
    "residual_vq": (
        {
            "_target_": "heptokens.models.quantizers.ResidualVQQuantizer",
            "_partial_": True,
            "codebook_size": 8,
            "num_quantizers": 2,
        },
        ResidualVQQuantizer,
        2,
        8,
    ),
}


@pytest.mark.parametrize("quantizer", QUANTIZERS)
def test_checkpoint_reloads_and_exports(hdf5_file, tmp_path, quantizer):
    settings, vq_class, n_tokens, n_values = QUANTIZERS[quantizer]
    full_path = tmp_path / "run"
    coder = {"_target_": "heptokens.models.coders.CoderModel", "_partial_": True}
    train(
        OmegaConf.create(
            {
                "seed": 42,
                "precision": "medium",
                "compile": None,
                "full_resume": False,
                "ckpt_path": None,
                "weight_ckpt_path": None,
                "full_path": str(full_path),
                "datamodule": _datamodule_cfg(hdf5_file, 0.7, 0.15, 0.15),
                "model": {
                    "_target_": "heptokens.models.vq_vae.LitVqVae",
                    "encoder": {
                        "_target_": "heptokens.models.coders.Encoder",
                        "_partial_": True,
                        "model": {**coder, "hidden_dims": [32, 64]},
                    },
                    "decoder": {
                        "_target_": "heptokens.models.coders.Decoder",
                        "_partial_": True,
                        "model": {**coder, "hidden_dims": [64, 32]},
                    },
                    "codebook_dim": 4,
                    "quantizer": settings,
                },
                "trainer": {
                    "_target_": "lightning.Trainer",
                    "max_epochs": 1,
                    "limit_train_batches": 2,
                    "limit_val_batches": 1,
                    "accelerator": "cpu",
                    "enable_progress_bar": False,
                    "default_root_dir": str(full_path),
                },
                "callbacks": {
                    "checkpoint": {
                        "_target_": "lightning.pytorch.callbacks.ModelCheckpoint",
                        "dirpath": str(full_path / "checkpoints"),
                        "filename": "last",
                    },
                },
                "logger": {
                    "_target_": "lightning.pytorch.loggers.CSVLogger",
                    "save_dir": str(full_path),
                },
            }
        )
    )
    ckpt_path = full_path / "checkpoints" / "last.ckpt"

    model = LitVqVae.load_from_checkpoint(ckpt_path, map_location="cpu").eval()
    assert isinstance(model.vector_quantization, vq_class)
    assert model.codebook_size == n_values

    export(
        OmegaConf.create(
            {
                "ckpt_path": str(ckpt_path),
                "output_dir": str(tmp_path / "export"),
                "datamodule": _datamodule_cfg(hdf5_file, 0.0, 0.0, 1.0),
                "trainer": {
                    "_target_": "lightning.Trainer",
                    "accelerator": "cpu",
                    "enable_progress_bar": False,
                },
            }
        )
    )
    data = np.load(tmp_path / "export" / f"{hdf5_file.stem}.npz")
    indices = torch.from_numpy(data["indices"]).long()  # [jets, elements, n_tokens]
    codebooks = torch.from_numpy(data["codebooks"])  # [n_tokens, n_values, codebook_dim]
    assert indices.shape == (100, 20, n_tokens)
    assert codebooks.shape == (n_tokens, n_values, 4)

    # The export's predict order: test set of the same split, not shuffled.
    datamodule = instantiate(OmegaConf.create(_datamodule_cfg(hdf5_file, 0.0, 0.0, 1.0)))
    datamodule.setup(stage="predict")
    with torch.no_grad():
        encoded = [model.encode(batch)[:2] for batch in datamodule.predict_dataloader()]
    z_q = torch.cat([z for z, _ in encoded])  # [jets, elements, codebook_dim]
    assert torch.equal(indices, torch.cat([i for _, i in encoded]))
    valid = indices[..., 0] >= 0
    z_export = sum(codebooks[q, indices[..., q][valid]] for q in range(n_tokens))
    assert torch.allclose(z_export, z_q[valid], atol=1e-5)
