"""Self-contained VQ-VAE tokenizer training script.

This script demonstrates the heptokens low-level API without any Hydra
configuration.  It is intentionally verbose so that each step is clear.
For production use with config management, experiment tracking, and
Snakemake orchestration, see scripts/train.py instead.

Usage:
    pixi run python scripts/train_simple.py \
        --data_path /path/to/mc-ttbar.h5 \
        --output_dir ./results/simple_run \
        --max_epochs 5
"""

import argparse
from functools import partial
from pathlib import Path

import lightning.pytorch as pl
import torch

from heptokens.data.atlas_mappable import SingleFileMapModule
from heptokens.models.coders import Decoder, Encoder
from heptokens.models.vq_vae import LitVqVae


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a VQ-VAE tokenizer on jet data.")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/sdf/scratch/users/s/samklein/data/atlas/mc-flavtag-ttbar-small.h5",
        help="Path to the ATLAS HDF5 data file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/simple_run",
        help="Directory where checkpoints will be saved.",
    )
    parser.add_argument(
        "--num_jets", type=int, default=None, help="Cap on number of jets to load (default: all)."
    )
    parser.add_argument(
        "--num_csts", type=int, default=40, help="Max constituents per jet (default: 40)."
    )
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--codebook_size", type=int, default=512)
    parser.add_argument("--codebook_dim", type=int, default=6)
    parser.add_argument("--num_quantizers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------ #
    # Reproducibility
    # ------------------------------------------------------------------ #
    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("medium")

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    # Each batch is a dict with keys:
    #   "csts"   : float32 [batch, num_csts, n_features]  constituent features
    #   "mask"   : bool    [batch, num_csts]               True for valid constituents
    #   "jets"   : float32 [batch, n_jet_features]         jet-level features
    #   "labels" : int64   [batch]                         flavour label (0-3)
    datamodule = SingleFileMapModule(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_frac=0.7,
        val_frac=0.15,
        test_frac=0.15,
        n_classes=4,
        cst_features=["pt", "deta", "dphi", "d0"],
        jet_features=["pt", "mass"],
        num_jets=args.num_jets,
        num_csts=args.num_csts,
        seed=args.seed,
    )

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    # LitVqVae wraps a ResidualVQ (vector-quantize-pytorch) with a simple
    # MLP encoder and decoder.  data_sample is used only to infer the
    # input feature dimension automatically.
    model = LitVqVae(
        encoder=partial(Encoder),  # MLP: input_dim -> codebook_dim
        decoder=partial(Decoder),  # MLP: codebook_dim -> input_dim
        codebook_size=args.codebook_size,  # codes per quantizer
        codebook_dim=args.codebook_dim,  # latent dimension
        num_quantizers=args.num_quantizers,
        learning_rate=args.learning_rate,
        data_sample=datamodule.get_data_sample(),
    )

    print(model)

    # ------------------------------------------------------------------ #
    # Trainer
    # ------------------------------------------------------------------ #
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="best",
        monitor="val/total_loss",
        mode="min",
        save_last=True,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1,
        gradient_clip_val=1.0,
        default_root_dir=str(output_dir),
        callbacks=[checkpoint_callback],
        enable_progress_bar=True,
    )

    # ------------------------------------------------------------------ #
    # Train
    # ------------------------------------------------------------------ #
    trainer.fit(model, datamodule=datamodule)

    print(f"\nTraining complete. Checkpoint saved to: {output_dir}/checkpoints/")

    # ------------------------------------------------------------------ #
    # Quick inference demo
    # ------------------------------------------------------------------ #
    # Load the best checkpoint and run forward pass to get token indices.
    best_ckpt = output_dir / "checkpoints" / "best.ckpt"
    if best_ckpt.exists():
        print("\n--- Inference demo ---")
        loaded_model = LitVqVae.load_from_checkpoint(str(best_ckpt), map_location="cpu")
        loaded_model.eval()

        datamodule.setup(stage="fit")
        sample_batch = next(iter(datamodule.val_dataloader()))

        with torch.no_grad():
            # indices shape: [batch, num_csts, num_quantizers]
            # Masked (invalid) positions have index -1
            indices = loaded_model(sample_batch)

        print(f"Batch size         : {indices.shape[0]}")
        print(f"Max constituents   : {indices.shape[1]}")
        print(f"Quantizer stages   : {indices.shape[2]}")
        print("Token indices (first jet, first 5 constituents):")
        print(indices[0, :5])


if __name__ == "__main__":
    main()
