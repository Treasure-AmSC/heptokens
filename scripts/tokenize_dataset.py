"""Tokenize HDF5 datasets with a trained VQ-VAE and save to compressed npz.

Saves token indices (int16), event numbers, labels, and the codebook
embeddings — so z_q can be reconstructed from indices alone without
needing the checkpoint.  Masked positions have index -1.

Usage:
    pixi run python scripts/tokenize_dataset.py \
        --ckpt results/vqvae_scan/vqvae_scan/<model>/checkpoints/best.ckpt \
        --input /path/to/small.h5 /path/to/medium.h5 /path/to/large.h5 \
        --output_dir results/tokenized/<model>

Reconstruction (after loading):
    data = np.load("tokenized.npz")
    codebooks = data["codebooks"]        # [nq, cb_size, cb_dim]
    indices   = data["indices"]          # [N, max_csts, nq]
    mask      = indices[:, :, 0] >= 0    # derive mask from indices

    # z_q[i, j, :] = sum over q of codebooks[q, indices[i,j,q], :]
    z_q = np.zeros((*indices.shape[:2], codebooks.shape[-1]), dtype=np.float32)
    for q in range(codebooks.shape[0]):
        valid = indices[:, :, q] >= 0
        z_q[valid] += codebooks[q, indices[:, :, q][valid]]
"""

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import torch
from joblib import load as joblib_load
from tqdm import tqdm

from heptokens.data.collation import preprocess_batch
from heptokens.models.vq_vae import LitVqVae

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Must match training config (configs/datamodule/atlas_mappable.yaml)
CST_FEATURES = [
    "pt", "deta", "dphi", "d0", "d0RelativeToBeamspot",
    "d0Uncertainty", "d0RelativeToBeamspotUncertainty",
    "z0RelativeToBeamspot", "z0RelativeToBeamspotUncertainty",
    "z0SinTheta", "z0SinThetaUncertainty",
    "lifetimeSignedD0", "lifetimeSignedD0Significance",
    "lifetimeSignedZ0SinTheta", "lifetimeSignedZ0SinThetaSignificance",
    "theta", "thetaUncertainty", "qOverP", "qOverPUncertainty", "ptfrac",
]
JET_FEATURES = ["pt", "mass", "eta", "phi"]
LABEL_KEY = "HadronConeExclTruthLabelID"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tokenize HDF5 jet datasets with a trained VQ-VAE.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt", type=str, required=True, help="VQ-VAE checkpoint path")
    parser.add_argument(
        "--input", type=str, nargs="+", required=True, help="Input h5 file(s)"
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--batch_size", type=int, default=100_000,
                        help="H5 read chunk size (larger = fewer I/O round-trips)")
    parser.add_argument("--num_csts", type=int, default=40)
    parser.add_argument("--num_jets", type=int, default=None, help="Max jets per file (None=all)")
    parser.add_argument(
        "--cst_fn",
        type=str,
        default="resources/cst_quantiles.joblib",
        help="Path to constituent preprocessing transform",
    )
    parser.add_argument(
        "--jet_fn",
        type=str,
        default="resources/jet_quantiles.joblib",
        help="Path to jet preprocessing transform",
    )
    parser.add_argument(
        "--max_jet_pt", type=float, default=7.0e6, help="Maximum jet pT cut"
    )
    parser.add_argument(
        "--max_cst_pt", type=float, default=1.0e6, help="Maximum constituent pT cut"
    )
    return parser.parse_args()


def extract_codebooks(model: LitVqVae) -> np.ndarray:
    """Extract codebook embeddings from the VQ-VAE.

    Returns:
        codebooks: [num_quantizers, codebook_size, codebook_dim] float32
    """
    vq = model.vector_quantization
    codebooks = torch.stack([layer.codebook for layer in vq.layers])
    return codebooks.detach().cpu().numpy()


# Known truth labels for ATLAS flavour-tagging: b=5, c=4, tau=15, light=0
DEFAULT_LABEL_MAP = {0: 0, 4: 1, 5: 2, 15: 3}


def _read_h5_chunk(f, start, end, num_csts):
    """Read one chunk of jets + tracks from an open h5 file (CPU-bound I/O)."""
    bs = end - start

    # Read jets compound slice ONCE (avoids re-decompressing chunks per field)
    jets_slice = f["jets"][start:end]
    jets_np = np.empty((bs, len(JET_FEATURES)), dtype=np.float32)
    for i, feat in enumerate(JET_FEATURES):
        jets_np[:, i] = jets_slice[feat]
    event_nums = jets_slice["eventNumber"].astype(np.int64)
    labels_raw = jets_slice[LABEL_KEY]

    # Global h5 row indices for unique jet identification
    jet_idx = np.arange(start, end, dtype=np.int64)

    # Read tracks compound slice ONCE
    tracks_slice = f["tracks"][start:end, :num_csts]
    csts_np = np.empty((bs, num_csts, len(CST_FEATURES)), dtype=np.float32)
    for i, feat in enumerate(CST_FEATURES):
        csts_np[:, :, i] = tracks_slice[feat]
    mask_np = tracks_slice["valid"].astype(bool)

    return jets_np, event_nums, labels_raw, csts_np, mask_np, jet_idx


def tokenize_file(
    model: LitVqVae,
    h5_path: str,
    cst_fn,
    jet_fn,
    *,
    num_csts: int,
    batch_size: int,
    num_jets: int | None,
    max_jet_pt: float | None,
    max_cst_pt: float | None,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Tokenize one h5 file, returning arrays for npz.

    Returns dict with keys: indices, labels, eventNumber
    """
    label_map = dict(DEFAULT_LABEL_MAP)
    codebook_size = model.codebook_size

    # Scale encode sub-batch to codebook size:
    # Two GPU memory constraints:
    # 1. VQ distance matrix: [encode_bs * num_csts, codebook_size] floats
    # 2. Transformer activations: ~[encode_bs, num_csts, d_model] × num_layers
    # Scale cap with inverse of codebook size: small cb → bigger batches
    max_gpu_bytes = 4e9
    vq_limit = int(max_gpu_bytes / (num_csts * codebook_size * 4))
    # For small codebooks (<=1024), transformer memory is the constraint
    # A100 80GB can handle ~16k jets through a small transformer
    max_encode = 16384 if codebook_size <= 1024 else 2048
    encode_bs = max(64, min(vq_limit, max_encode))
    log.info(f"Encode sub-batch size: {encode_bs} (codebook_size={codebook_size})")

    cst_pt_idx = CST_FEATURES.index("pt") if "pt" in CST_FEATURES else None
    jet_pt_idx = JET_FEATURES.index("pt") if "pt" in JET_FEATURES else None

    with h5py.File(h5_path, "r") as f:
        total = len(f["jets"])
        if num_jets is not None:
            total = min(total, num_jets)

        log.info(f"Processing {h5_path}: {total:,} jets (batch_size={batch_size})")

        all_indices = []
        all_labels = []
        all_event_numbers = []
        all_jet_idx = []

        chunk_starts = list(range(0, total, batch_size))
        pbar = tqdm(chunk_starts, desc=Path(h5_path).stem)

        # Prefetch first chunk
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_read_h5_chunk, f, chunk_starts[0],
                                 min(chunk_starts[0] + batch_size, total), num_csts)

        t_io = t_pre = t_enc = 0.0

        for ci, start in enumerate(pbar):
            end = min(start + batch_size, total)

            # Wait for prefetched data
            t0 = time.perf_counter()
            jets_np, event_nums, labels_raw, csts_np, mask_np, jet_idx = future.result()
            t_io += time.perf_counter() - t0

            # Start prefetching next chunk while we process this one
            if ci + 1 < len(chunk_starts):
                next_start = chunk_starts[ci + 1]
                next_end = min(next_start + batch_size, total)
                future = executor.submit(_read_h5_chunk, f, next_start, next_end, num_csts)

            t0 = time.perf_counter()
            bs = end - start
            # Vectorized label mapping (avoid Python loop over 100K elements)
            label_keys = np.array(list(label_map.keys()))
            label_vals = np.array(list(label_map.values()), dtype=np.int8)
            labels_mapped = np.full(bs, -1, dtype=np.int8)
            for k, v in zip(label_keys, label_vals):
                labels_mapped[labels_raw == k] = v

            # --- Apply pT cuts ---
            keep = np.ones(bs, dtype=bool)
            if max_jet_pt is not None and jet_pt_idx is not None:
                keep &= jets_np[:, jet_pt_idx] <= max_jet_pt
            if max_cst_pt is not None and cst_pt_idx is not None:
                cst_pts = csts_np[:, :, cst_pt_idx]
                max_per_jet = np.where(mask_np, cst_pts, 0).max(axis=1)
                keep &= max_per_jet <= max_cst_pt

            if not keep.any():
                continue

            jets_np = jets_np[keep]
            event_nums = event_nums[keep]
            jet_idx = jet_idx[keep]
            labels_mapped = labels_mapped[keep]
            csts_np = csts_np[keep]
            mask_np = mask_np[keep]

            all_event_numbers.append(event_nums)
            all_jet_idx.append(jet_idx)
            all_labels.append(labels_mapped)

            # --- Preprocess for VQ-VAE ---
            batch = {
                "csts": torch.from_numpy(csts_np).float(),
                "mask": torch.from_numpy(mask_np),
                "jets": torch.from_numpy(jets_np).float(),
            }
            batch = preprocess_batch(batch, cst_fn, jet_fn)
            t_pre += time.perf_counter() - t0

            # --- Tokenize (sub-batch to fit VQ distance matrix in GPU) ---
            t0 = time.perf_counter()
            batch_indices = []
            n = batch["csts"].shape[0]
            for i in range(0, n, encode_bs):
                sub = {
                    k: v[i : i + encode_bs].to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                with torch.no_grad():
                    _z_q, idx, _commit = model.encode(sub)
                batch_indices.append(idx.cpu())

            indices_np = torch.cat(batch_indices, dim=0).numpy().astype(np.int16)
            all_indices.append(indices_np)
            t_enc += time.perf_counter() - t0

            # Log timing every 10 iterations
            if (ci + 1) % 10 == 0 or ci == 0:
                n_done = ci + 1
                log.info(
                    f"  [{n_done}/{len(chunk_starts)}] "
                    f"io={t_io/n_done:.1f}s  pre={t_pre/n_done:.1f}s  "
                    f"enc={t_enc/n_done:.1f}s  total={t_io+t_pre+t_enc:.0f}s"
                )

        executor.shutdown(wait=False)

    # Concatenate
    result = {
        "indices": np.concatenate(all_indices, axis=0),
        "labels": np.concatenate(all_labels, axis=0),
        "eventNumber": np.concatenate(all_event_numbers, axis=0),
        "jet_idx": np.concatenate(all_jet_idx, axis=0),
    }
    log.info(
        f"  -> {result['indices'].shape[0]:,} jets after cuts, "
        f"indices shape {result['indices'].shape}"
    )
    return result


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Loading checkpoint: {args.ckpt} (device={device})")
    model = LitVqVae.load_from_checkpoint(args.ckpt, map_location=device)
    model.to(device)
    model.eval()

    # Extract codebooks once
    codebooks = extract_codebooks(model)
    log.info(
        f"Codebooks: {codebooks.shape} "
        f"(nq={codebooks.shape[0]}, cb_size={codebooks.shape[1]}, cb_dim={codebooks.shape[2]})"
    )

    # Load preprocessing transforms
    cst_fn = joblib_load(args.cst_fn)
    jet_fn = joblib_load(args.jet_fn)
    log.info(f"Preprocessing: cst_fn ({cst_fn.n_features_in_} feats), jet_fn ({jet_fn.n_features_in_} feats)")

    # Process each input file
    for h5_path in args.input:
        stem = Path(h5_path).stem  # e.g. "mc-flavtag-ttbar-small"
        out_path = output_dir / f"{stem}.npz"

        arrays = tokenize_file(
            model,
            h5_path,
            cst_fn,
            jet_fn,
            num_csts=args.num_csts,
            batch_size=args.batch_size,
            num_jets=args.num_jets,
            max_jet_pt=args.max_jet_pt,
            max_cst_pt=args.max_cst_pt,
            device=device,
        )

        # Save to compressed npz
        log.info(f"Saving {out_path} ...")
        np.savez_compressed(
            out_path,
            # Per-jet arrays
            indices=arrays["indices"],          # [N, 40, nq] int16
            labels=arrays["labels"],            # [N] int8
            eventNumber=arrays["eventNumber"],  # [N] int64
            jet_idx=arrays["jet_idx"],          # [N] int64 — row index in source h5
            # Codebook for z_q reconstruction (shared across files)
            codebooks=codebooks,                # [nq, cb_size, cb_dim] float32
        )

        # Report file size
        size_gb = out_path.stat().st_size / 1e9
        log.info(f"  -> {out_path}: {size_gb:.2f} GB")

    log.info("Done.")


if __name__ == "__main__":
    main()
