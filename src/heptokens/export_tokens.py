"""Export VQ-VAE token indices for a dataset — usable as `python -m heptokens.export_tokens`.

Reuses the same data module and preprocessing configs as train.py so that
feature lists, transforms, pT cuts, etc. stay in sync automatically.

Usage (from an experiment repo with local configs):
    python -m heptokens.export_tokens -cp configs \\
        ckpt_path=/path/to/vqvae/best.ckpt \\
        output_dir=results/tokenized/model_name \\
        datamodule.data_path=/path/to/file.h5

Output npz contains:
    indices     : [N, num_elements, num_quantizers]  int16  (-1 = masked)
    labels      : [N]                                int8
    codebooks   : [num_quantizers, codebook_size, codebook_dim]  float32
    eventNumber : [N]                            int64  (if present in dataset)

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
    """Write predictions to memory-mapped files batch-by-batch to avoid OOM.

    Optional position tokenization (opt-in, disabled by default):
        If ``pos_tokenizer`` is provided, this writer additionally reads the
        raw ``positions`` tensor directly from the dataloader ``batch`` (NOT
        from ``prediction`` — VQ-VAE ``predict_step``/indices are completely
        untouched by this) and writes tokenized ``pos_tokens`` to a separate
        memmap/npz key. When ``pos_tokenizer`` is None (the default), no
        position-related files/keys are created at all.
    """

    def __init__(
        self,
        tmp_dir: str,
        total_jets: int,
        num_elements: int,
        num_quantizers: int,
        pos_tokenizer=None,
        num_pos_features: int = 0,
        positions_key: str = "positions",
        mask_key: str = "mask",
    ):
        super().__init__(write_interval="batch")
        self.tmp_dir = Path(tmp_dir)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.total_jets = total_jets
        self.pos = 0

        self.pos_tokenizer = pos_tokenizer
        self.positions_key = positions_key
        self.mask_key = mask_key

        self.indices_path = self.tmp_dir / "indices.npy"
        self.labels_path = self.tmp_dir / "labels.npy"
        self.events_path = self.tmp_dir / "eventNumber.npy"

        self.indices_mmap = np.lib.format.open_memmap(
            str(self.indices_path),
            mode="w+",
            dtype=np.int16,
            shape=(total_jets, num_elements, num_quantizers),
        )
        self.labels_mmap = np.lib.format.open_memmap(
            str(self.labels_path),
            mode="w+",
            dtype=np.int8,
            shape=(total_jets,),
        )
        self.events_mmap = np.lib.format.open_memmap(
            str(self.events_path),
            mode="w+",
            dtype=np.int64,
            shape=(total_jets,),
        )
        self.has_events = False

        self.has_pos_tokens = self.pos_tokenizer is not None
        self.pos_tokens_path = None
        self.pos_tokens_mmap = None
        if self.has_pos_tokens:
            if num_pos_features <= 0:
                raise ValueError(
                    "num_pos_features must be > 0 when pos_tokenizer is provided"
                )
            self.pos_tokens_path = self.tmp_dir / "pos_tokens.npy"
            self.pos_tokens_mmap = np.lib.format.open_memmap(
                str(self.pos_tokens_path),
                mode="w+",
                dtype=np.int16,
                shape=(total_jets, num_elements, num_pos_features),
            )

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
            self.events_mmap[self.pos : self.pos + n] = prediction["eventNumber"].cpu().numpy()

        if self.pos_tokenizer is not None:
            positions = batch.get(self.positions_key)
            if positions is not None:
                pos_idx = self.pos_tokenizer(positions)
                mask = batch.get(self.mask_key)
                if mask is not None:
                    pos_idx = pos_idx.masked_fill(~mask.unsqueeze(-1).bool(), -1)
                self.pos_tokens_mmap[self.pos : self.pos + n] = (
                    pos_idx.cpu().numpy().astype(np.int16)
                )

        self.pos += n

    def finalize(self) -> int:
        """Flush mmaps and return actual number of jets written."""
        del self.indices_mmap, self.labels_mmap, self.events_mmap
        if self.pos_tokens_mmap is not None:
            del self.pos_tokens_mmap
        return self.pos


def export(cfg: DictConfig) -> None:
    """Export logic — importable by experiment repos with their own Hydra entrypoints."""
    log.info("Instantiating data module from config")
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup(stage="predict")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Loading VQ-VAE checkpoint: {cfg.ckpt_path} (device={device})")
    model = LitVqVae.load_from_checkpoint(cfg.ckpt_path, map_location=device)
    model.eval()

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_jets = len(datamodule.test_set)
    num_elements = getattr(datamodule.test_set, "num_elements", 40)
    num_quantizers = model.hparams.get("num_quantizers", 4)

    # Optional, opt-in export-layer position tokenization. Disabled unless the
    # caller's config explicitly provides a `pos_tokenizer` group. This never
    # touches the VQ-VAE model, its encode/decode/loss, or ResidualVQ indices —
    # positions are read directly from the dataloader batch by the writer.
    pos_tokenizer = None
    num_pos_features = 0
    if cfg.get("pos_tokenizer") is not None:
        pos_tokenizer = hydra.utils.instantiate(cfg.pos_tokenizer)
        num_pos_features = getattr(pos_tokenizer, "n_features", None)
        if not num_pos_features:
            raise ValueError(
                "cfg.pos_tokenizer was provided but the instantiated object has no "
                "positive 'n_features' attribute"
            )
        log.info(f"Position tokenization enabled: {num_pos_features} feature(s)")
    positions_key = cfg.get("pos_tokenizer_positions_key", "positions")
    mask_key = cfg.get("pos_tokenizer_mask_key", "mask")

    tmp_base = os.environ.get("LOCAL_TMPDIR", os.environ.get("TMPDIR", None))
    if tmp_base is None or not Path(tmp_base).exists():
        tmp_base = str(output_dir)

    with tempfile.TemporaryDirectory(dir=tmp_base) as tmp_dir:
        writer = MemmapPredictionWriter(
            tmp_dir,
            total_jets,
            num_elements,
            num_quantizers,
            pos_tokenizer=pos_tokenizer,
            num_pos_features=num_pos_features,
            positions_key=positions_key,
            mask_key=mask_key,
        )

        log.info(
            f"Running tokenization via trainer.predict() ({total_jets:,} jets, memmap to {tmp_dir})"
        )
        trainer = hydra.utils.instantiate(cfg.trainer, callbacks=[writer], logger=False)
        trainer.predict(model, datamodule=datamodule, return_predictions=False)

        actual_jets = writer.finalize()
        log.info(f"Wrote {actual_jets:,} jets to memmap")

        indices = np.lib.format.open_memmap(str(writer.indices_path), mode="r")[:actual_jets]
        labels = np.lib.format.open_memmap(str(writer.labels_path), mode="r")[:actual_jets]
        codebooks = extract_codebooks(model)

        stem = Path(cfg.datamodule.data_path).stem
        out_path = output_dir / f"{stem}.npz"

        save_dict = dict(indices=indices, labels=labels, codebooks=codebooks)
        if writer.has_events:
            event_numbers = np.lib.format.open_memmap(str(writer.events_path), mode="r")[
                :actual_jets
            ]
            save_dict["eventNumber"] = event_numbers
            log.info(f"Including eventNumber array of length {actual_jets:,}")
        if writer.has_pos_tokens:
            pos_tokens = np.lib.format.open_memmap(str(writer.pos_tokens_path), mode="r")[
                :actual_jets
            ]
            save_dict["pos_tokens"] = pos_tokens
            log.info(f"Including pos_tokens array of shape {pos_tokens.shape}")

        log.info(
            f"Tokenized {actual_jets:,} jets -> "
            f"indices {indices.shape}, codebooks {codebooks.shape}"
        )
        log.info(f"Saving compressed {out_path} ...")
        np.savez_compressed(out_path, **save_dict)
        log.info(f"Saved {out_path}: {out_path.stat().st_size / 1e9:.2f} GB")


@hydra.main(version_base=None, config_path="pkg://heptokens.conf", config_name="tokenize.yaml")
def main(cfg: DictConfig) -> None:
    export(cfg)


if __name__ == "__main__":
    main()
