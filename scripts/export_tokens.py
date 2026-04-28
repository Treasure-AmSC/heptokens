"""Tokenize an HDF5 dataset with a trained VQ-VAE.

Reuses the same data module and preprocessing configs as train.py so that
feature lists, transforms, pT cuts, etc. stay in sync automatically.

Usage:
    pixi run python scripts/export_tokens.py \
        ckpt_path=/path/to/vqvae/best.ckpt \
        output_dir=results/tokenized/model_name \
        datamodule.data_path=/path/to/file.h5

    # Override preprocessing transforms if needed:
    pixi run python scripts/export_tokens.py \
        ckpt_path=... output_dir=... datamodule.data_path=... \
        datamodule.transforms.preprocess.cst_fn.filename=resources/cst_quantiles.joblib \
        datamodule.transforms.preprocess.jet_fn.filename=resources/jet_quantiles.joblib

Output npz contains:
    indices     : [N, num_csts, num_quantizers]  int16  (-1 = masked)
    labels      : [N]                            int8
    codebooks   : [num_quantizers, codebook_size, codebook_dim]  float32
    eventNumber : [N]                            int64

Reconstruction:
    z_q[i, j, :] = sum over q of codebooks[q, indices[i, j, q], :]
"""

import logging
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from heptokens.models.vq_vae import LitVqVae

log = logging.getLogger(__name__)


def extract_codebooks(model: LitVqVae) -> np.ndarray:
    """Return codebook embeddings as [num_quantizers, codebook_size, codebook_dim]."""
    vq = model.vector_quantization
    codebooks = torch.stack([layer.codebook for layer in vq.layers])
    return codebooks.detach().cpu().numpy()


@hydra.main(version_base=None, config_path="../configs", config_name="tokenize")
def main(cfg: DictConfig) -> None:
    log.info("Instantiating data module from config")
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Loading VQ-VAE checkpoint: {cfg.ckpt_path} (device={device})")
    model = LitVqVae.load_from_checkpoint(cfg.ckpt_path, map_location=device)
    model.eval()

    log.info("Running tokenization via trainer.predict()")
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=[], logger=False)
    predictions = trainer.predict(model, datamodule=datamodule)

    # Gather results from all batches
    all_indices = torch.cat([p["indices"] for p in predictions]).cpu().numpy().astype(np.int16)
    all_labels = torch.cat([p["labels"] for p in predictions]).cpu().numpy().astype(np.int8)
    codebooks = extract_codebooks(model)

    save_dict = dict(indices=all_indices, labels=all_labels, codebooks=codebooks)

    # Include eventNumber if available
    if "eventNumber" in predictions[0]:
        all_event_numbers = torch.cat([p["eventNumber"] for p in predictions]).cpu().numpy()
        save_dict["eventNumber"] = all_event_numbers
        log.info(f"Including eventNumber array of length {len(all_event_numbers):,}")

    log.info(
        f"Tokenized {len(all_labels):,} jets -> "
        f"indices {all_indices.shape}, codebooks {codebooks.shape}"
    )

    # Save compressed npz
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(cfg.datamodule.data_path).stem
    out_path = output_dir / f"{stem}.npz"

    log.info(f"Saving {out_path}")
    np.savez_compressed(out_path, **save_dict)
    log.info(f"Saved {out_path}: {out_path.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
