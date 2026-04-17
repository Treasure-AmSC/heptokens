"""VQ-VAE tokenizer training on Z'->ttbar (tops) + QCD jets.

Drop-in replacement for scripts/simple_demo.py that uses the new
ZPrimeMapModule instead of SingleFileMapModule.  The model and training
loop are identical; only the datamodule changes.

Usage:
    pixi run python scripts/simple_demo_zprime.py \
        --tops_paths data/tops_ZPrime_M1000_W300_batch0.h5 \
                     data/tops_ZPrime_M1000_W100_batch0.h5 \
        --qcd_paths  data/Pt300_merge/batch0.h5 \
                     data/Pt470_merge/batch0.h5 \
        --output_dir ./results/zprime_run

The datamodule emits the same batch format as the ATLAS loaders:
    csts   : float32 [batch, num_csts, 10]    PFCands features (px..pdgId)
    mask   : bool    [batch, num_csts]         True = real particle
    jets   : float32 [batch, 4]               [pt, eta, phi, mass]
    labels : int64   [batch]                  0 = QCD, 1 = tops

So the LitVqVae model and Trainer are used exactly as before.
"""

import argparse
from functools import partial
from pathlib import Path

import lightning.pytorch as pl
import torch

from heptokens.data.zprime_mappable import ZPrimeMapModule
from heptokens.models.coders import Decoder, Encoder
from heptokens.models.vq_vae import LitVqVae


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a VQ-VAE tokenizer on Z'->ttbar tops + QCD jets."
    )
    parser.add_argument(
        "--tops_paths",
        nargs="+",
        required=True,
        help="One or more tops HDF5 files (Z'->ttbar jets, label=1).",
    )
    parser.add_argument(
        "--qcd_paths",
        nargs="+",
        required=True,
        help="One or more QCD HDF5 files (label=0).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/zprime_run",
    )
    parser.add_argument(
        "--num_jets_per_file",
        type=int,
        default=None,
        help="Cap on jets loaded from each file (default: all).",
    )
    parser.add_argument(
        "--num_csts",
        type=int,
        default=40,
        help="Max constituents per jet (default: 40, max: 150).",
    )
    # QCD options
    parser.add_argument(
        "--use_weights_qcd",
        action="store_true",
        help="Load per-jet cross-section weights from QCD files.",
    )
    parser.add_argument(
        "--weighted_sampling",
        action="store_true",
        help="Use WeightedRandomSampler to reweight QCD pt spectrum during training.",
    )
    # Training hyperparameters (same as simple_demo.py)
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

    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("medium")

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    datamodule = ZPrimeMapModule(
        tops_paths=args.tops_paths,
        qcd_paths=args.qcd_paths,
        num_jets_per_file=args.num_jets_per_file,
        num_csts=args.num_csts,
        use_weights_qcd=args.use_weights_qcd,
        weighted_sampling=args.weighted_sampling,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_frac=0.7,
        val_frac=0.15,
        test_frac=0.15,
        seed=args.seed,
    )

    # ------------------------------------------------------------------ #
    # Model  (identical to simple_demo.py — input_dim auto-detected)
    # ------------------------------------------------------------------ #
    model = LitVqVae(
        encoder=partial(Encoder),
        decoder=partial(Decoder),
        codebook_size=args.codebook_size,
        codebook_dim=args.codebook_dim,
        num_quantizers=args.num_quantizers,
        learning_rate=args.learning_rate,
        data_sample=datamodule.get_data_sample(),
    )

    print(model)

    # ------------------------------------------------------------------ #
    # Trainer  (identical to simple_demo.py)
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

    trainer.fit(model, datamodule=datamodule)

    print(f"\nTraining complete. Checkpoint saved to: {output_dir}/checkpoints/")

    # ------------------------------------------------------------------ #
    # Quick inference demo  (identical to simple_demo.py)
    # ------------------------------------------------------------------ #
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
