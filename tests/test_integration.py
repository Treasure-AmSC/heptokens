"""End-to-end integration test: train a VQ-VAE then export tokens."""

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from heptokens.export_tokens import export
from heptokens.train import train


@pytest.fixture
def train_cfg(hdf5_file, tmp_path):
    output_dir = tmp_path / "results"
    full_path = output_dir / "test_project" / "test_run"
    full_path.mkdir(parents=True)

    cfg = OmegaConf.create(
        {
            "seed": 42,
            "precision": "medium",
            "compile": None,
            "tags": None,
            "full_resume": False,
            "ckpt_flag": "last.ckpt",
            "ckpt_path": None,
            "weight_ckpt_path": None,
            "test_only": False,
            "full_path": str(full_path),
            "datamodule": {
                "_target_": "heptokens.data.structured_array.StructuredArrayModule",
                "data_path": str(hdf5_file),
                "obj_group": "jets",
                "set_group": "tracks",
                "obj_features": ["pt", "eta"],
                "set_features": ["pt", "eta", "phi"],
                "label_key": "label",
                "mask_key": "valid",
                "train_frac": 0.7,
                "val_frac": 0.15,
                "test_frac": 0.15,
                "batch_size": 32,
                "num_workers": 0,
            },
            "model": {
                "_target_": "heptokens.models.vq_vae.LitVqVae",
                "encoder": {
                    "_target_": "heptokens.models.coders.Encoder",
                    "_partial_": True,
                    "model": {
                        "_target_": "heptokens.models.coders.CoderModel",
                        "_partial_": True,
                        "hidden_dims": [32, 64],
                    },
                },
                "decoder": {
                    "_target_": "heptokens.models.coders.Decoder",
                    "_partial_": True,
                    "model": {
                        "_target_": "heptokens.models.coders.CoderModel",
                        "_partial_": True,
                        "hidden_dims": [64, 32],
                    },
                },
                "codebook_size": 8,
                "codebook_dim": 4,
                "num_quantizers": 2,
                "learning_rate": 1e-3,
            },
            "trainer": {
                "_target_": "lightning.Trainer",
                "max_epochs": 1,
                "limit_train_batches": 2,
                "limit_val_batches": 1,
                "accelerator": "cpu",
                "devices": 1,
                "enable_progress_bar": False,
                "default_root_dir": str(full_path),
                "reload_dataloaders_every_n_epochs": 0,
            },
            "callbacks": {
                "checkpoint": {
                    "_target_": "lightning.pytorch.callbacks.ModelCheckpoint",
                    "dirpath": str(full_path / "checkpoints"),
                    "filename": "last",
                    "enable_version_counter": False,
                    "auto_insert_metric_name": False,
                },
            },
            "logger": {
                "_target_": "lightning.pytorch.loggers.CSVLogger",
                "save_dir": str(full_path),
                "name": "test",
            },
        }
    )
    return cfg


def test_train_then_export(train_cfg, hdf5_file, tmp_path):
    # Run training
    train(train_cfg)

    ckpt_path = Path(train_cfg.full_path) / "checkpoints" / "last.ckpt"
    assert ckpt_path.exists(), f"Checkpoint not found at {ckpt_path}"

    export_dir = tmp_path / "export_output"
    export_cfg = OmegaConf.create(
        {
            "ckpt_path": str(ckpt_path),
            "output_dir": str(export_dir),
            "full_path": str(export_dir),
            "datamodule": {
                "_target_": "heptokens.data.structured_array.StructuredArrayModule",
                "data_path": str(hdf5_file),
                "obj_group": "jets",
                "set_group": "tracks",
                "obj_features": ["pt", "eta"],
                "set_features": ["pt", "eta", "phi"],
                "label_key": "label",
                "mask_key": "valid",
                "train_frac": 0.0,
                "val_frac": 0.0,
                "test_frac": 1.0,
                "batch_size": 32,
                "num_workers": 0,
            },
            "trainer": {
                "_target_": "lightning.Trainer",
                "accelerator": "cpu",
                "devices": 1,
                "enable_progress_bar": False,
            },
        }
    )
    # Run export
    export(export_cfg)

    stem = hdf5_file.stem
    npz_path = export_dir / f"{stem}.npz"
    assert npz_path.exists(), f"Export output not found at {npz_path}"

    data = np.load(npz_path)
    assert "indices" in data
    assert "labels" in data
    assert "codebooks" in data

    assert data["indices"].ndim == 3
    assert data["indices"].shape[2] == 2  # num_quantizers
    assert data["codebooks"].shape == (2, 8, 4)  # [num_quantizers, codebook_size, codebook_dim]
