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
import os
import tempfile
from pathlib import Path

import hydra
import numpy as np
import torch
from lightning.pytorch.callbacks import BasePredictionWriter
from omegaconf import DictConfig

from heptokens.models.vq_vae import LitVqVae

log = logging.getLogger(__name__)


def extract_codebooks(model: LitVqVae) -> np.ndarray:
    """Return codebook embeddings as [num_quantizers, codebook_size, codebook_dim]."""
    vq = model.vector_quantization
    codebooks = torch.stack([layer.codebook for layer in vq.layers])
    return codebooks.detach().cpu().numpy()


class MemmapPredictionWriter(BasePredictionWriter):
    """Write predictions to memory-mapped files batch-by-batch."""

    def __init__(self, tmp_dir: str, total_jets: int, num_csts: int, num_quantizers: int):
        super().__init__(write_interval="batch")
        self.tmp_dir = Path(tmp_dir)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.total_jets = total_jets
        self.pos = 0  # current write position

        # Pre-allocate memory-mapped output arrays
        self.indices_path = self.tmp_dir / "indices.npy"
        self.labels_path = self.tmp_dir / "labels.npy"
        self.events_path = self.tmp_dir / "eventNumber.npy"

        self.indices_mmap = np.lib.format.open_memmap(
            str(self.indices_path), mode="w+", dtype=np.int16,
            shape=(total_jets, num_csts, num_quantizers),
        )
        self.labels_mmap = np.lib.format.open_memmap(
            str(self.labels_path), mode="w+", dtype=np.int8,
            shape=(total_jets,),
        )
        self.events_mmap = np.lib.format.open_memmap(
            str(self.events_path), mode="w+", dtype=np.int64,
            shape=(total_jets,),
        )
        self.has_events = False

    def write_on_batch_end(
        self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx
    ):
        indices = prediction["indices"].cpu().numpy().astype(np.int16)
        labels = prediction["labels"].cpu().numpy().astype(np.int8)
        n = len(labels)

        self.indices_mmap[self.pos : self.pos + n] = indices
        self.labels_mmap[self.pos : self.pos + n] = labels

        if "eventNumber" in prediction:
            self.has_events = True
            events = prediction["eventNumber"].cpu().numpy()
            self.events_mmap[self.pos : self.pos + n] = events

        self.pos += n

    def finalize(self):
        """Flush mmaps and truncate to actual size if fewer jets than expected."""
        del self.indices_mmap, self.labels_mmap, self.events_mmap
        return self.pos


@hydra.main(version_base=None, config_path="../configs", config_name="tokenize")
def main(cfg: DictConfig) -> None:
    log.info("Instantiating data module from config")
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Loading VQ-VAE checkpoint: {cfg.ckpt_path} (device={device})")
    model = LitVqVae.load_from_checkpoint(cfg.ckpt_path, map_location=device)
    model.eval()

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine dataset size for memmap pre-allocation
    import h5py
    with h5py.File(cfg.datamodule.data_path, "r") as f:
        total_jets = len(f["jets"])
    # Infer shape from model
    num_csts = getattr(datamodule.test_set, "num_csts", 40)
    num_quantizers = model.hparams.get("num_quantizers", 4)

    # Write memmap files to local scratch (fast NVMe, plenty of space)
    tmp_base = os.environ.get("LOCAL_TMPDIR", os.environ.get("TMPDIR", None))
    if tmp_base is None or not Path(tmp_base).exists():
        tmp_base = str(output_dir)

    with tempfile.TemporaryDirectory(dir=tmp_base) as tmp_dir:
        writer = MemmapPredictionWriter(tmp_dir, total_jets, num_csts, num_quantizers)

        log.info(
            f"Running tokenization via trainer.predict() "
            f"({total_jets:,} jets, memmap to {tmp_dir})"
        )
        trainer = hydra.utils.instantiate(
            cfg.trainer, callbacks=[writer], logger=False
        )
        trainer.predict(model, datamodule=datamodule, return_predictions=False)

        # Finalize and read back (memmap = no extra RAM, just maps the file)
        actual_jets = writer.finalize()
        log.info(f"Wrote {actual_jets:,} jets to memmap")

        indices = np.lib.format.open_memmap(str(writer.indices_path), mode="r")[:actual_jets]
        labels = np.lib.format.open_memmap(str(writer.labels_path), mode="r")[:actual_jets]
        codebooks = extract_codebooks(model)

        stem = Path(cfg.datamodule.data_path).stem
        out_path = output_dir / f"{stem}.npz"

        save_dict = dict(indices=indices, labels=labels, codebooks=codebooks)
        if writer.has_events:
            event_numbers = np.lib.format.open_memmap(
                str(writer.events_path), mode="r"
            )[:actual_jets]
            save_dict["eventNumber"] = event_numbers
            log.info(f"Including eventNumber array of length {actual_jets:,}")

        log.info(
            f"Tokenized {actual_jets:,} jets -> "
            f"indices {indices.shape}, codebooks {codebooks.shape}"
        )

        log.info(f"Saving compressed {out_path} ...")
        np.savez_compressed(out_path, **save_dict)
        log.info(f"Saved {out_path}: {out_path.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
