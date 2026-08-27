"""Compute codebook utilization for trained VQ-VAE models.

Loads each checkpoint, runs inference on a data sample, and reports
per-quantizer utilization (fraction of unique codes used).

Usage
-----
# Single run:
pixi run python scripts/codebook_utilization.py \
    --run_dirs results/vqvae_scan/vqvae_scan/transformer_vqvae_enc_large_cb32768_cd8_nq4_lr0.0003_cw1.0_nj10000000

# All tested runs (auto-discover):
pixi run python scripts/codebook_utilization.py --auto

# With custom data:
pixi run python scripts/codebook_utilization.py --auto --num_jets 100000
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from joblib import load as joblib_load

log = logging.getLogger(__name__)

SCAN_ROOT = Path("results/vqvae_scan/vqvae_scan")
RESOURCES = Path("resources")
DATA_PATH = Path("/sdf/scratch/users/j/jkrupa/data/atlas/mc-flavtag-ttbar-small.h5")


def load_model(ckpt_path: Path):
    """Load a VQ-VAE model from checkpoint."""
    from heptokens.models.vq_vae import LitVqVae

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LitVqVae.load_from_checkpoint(str(ckpt_path), map_location=device)
    model.to(device)
    model.eval()
    return model, device


def load_data(num_jets: int, num_csts: int = 40):
    """Load test data and preprocessing transforms."""
    import h5py

    cst_fn = joblib_load(RESOURCES / "cst_quantiles.joblib")

    with h5py.File(DATA_PATH, "r") as f:
        jets_raw = f["jets"][:num_jets]
        tracks_raw = f["tracks"][:num_jets]

    # Extract constituent features (same order as atlas_mappable.yaml)
    cst_features = [
        "pt", "deta", "dphi",
        "d0", "d0RelativeToBeamspot", "d0Uncertainty", "d0RelativeToBeamspotUncertainty",
        "z0RelativeToBeamspot", "z0RelativeToBeamspotUncertainty",
        "z0SinTheta", "z0SinThetaUncertainty",
        "lifetimeSignedD0", "lifetimeSignedD0Significance",
        "lifetimeSignedZ0SinTheta", "lifetimeSignedZ0SinThetaSignificance",
        "theta", "thetaUncertainty",
        "qOverP", "qOverPUncertainty",
        "ptfrac",
    ]
    valid = tracks_raw["valid"].astype(bool)

    n_feat = len(cst_features)
    csts = np.zeros((num_jets, num_csts, n_feat), dtype=np.float32)
    mask = np.zeros((num_jets, num_csts), dtype=bool)

    for i, feat in enumerate(cst_features):
        csts[:, :, i] = tracks_raw[feat][:, :num_csts]
    mask = valid[:, :num_csts]

    # Apply preprocessing
    csts_flat = csts[mask]
    if (feat_diff := cst_fn.n_features_in_ - n_feat) > 0:
        pad = np.zeros((csts_flat.shape[0], feat_diff), dtype=np.float32)
        csts_flat = np.concatenate((csts_flat, pad), axis=-1)

    csts_flat = cst_fn.transform(csts_flat)

    if feat_diff > 0:
        csts_flat = csts_flat[:, :-feat_diff]

    csts[mask] = csts_flat.astype(np.float32)

    return {
        "csts": torch.from_numpy(csts).float(),
        "mask": torch.from_numpy(mask).bool(),
    }


def compute_utilization(
    model, device, data: dict, batch_size: int = 1024
) -> dict:
    """Run inference and compute per-quantizer codebook utilization."""
    csts = data["csts"]
    mask = data["mask"]
    n_jets = csts.shape[0]

    codebook_size = model.codebook_size
    num_quantizers = len(model.vector_quantization.layers)

    # Collect unique codes per quantizer
    unique_codes = [set() for _ in range(num_quantizers)]
    total_tokens = 0

    with torch.no_grad():
        for start in range(0, n_jets, batch_size):
            end = min(start + batch_size, n_jets)
            batch = {
                "csts": csts[start:end].to(device),
                "mask": mask[start:end].to(device),
            }
            _, indices, _ = model.encode(batch)
            # indices: [B, n_csts, num_quantizers], -1 for masked
            valid_mask = batch["mask"]  # [B, n_csts]
            valid_indices = indices[valid_mask]  # [n_valid, num_quantizers]
            total_tokens += valid_indices.shape[0]

            for q in range(num_quantizers):
                codes = valid_indices[:, q].cpu().numpy()
                unique_codes[q].update(codes.tolist())

    # Compute utilization
    result = {
        "codebook_size": codebook_size,
        "num_quantizers": num_quantizers,
        "total_tokens": total_tokens,
        "per_quantizer": {},
    }
    for q in range(num_quantizers):
        n_unique = len(unique_codes[q])
        result["per_quantizer"][q] = {
            "unique_codes": n_unique,
            "utilization": n_unique / codebook_size,
        }
    result["avg_utilization"] = np.mean(
        [v["utilization"] for v in result["per_quantizer"].values()]
    )

    return result


def find_tested_runs() -> list[Path]:
    """Find all runs with TEST_SUCCESS.txt."""
    runs = []
    for d in sorted(SCAN_ROOT.iterdir()):
        if d.is_dir() and (d / "TEST_SUCCESS.txt").exists():
            ckpt = d / "checkpoints" / "best.ckpt"
            if ckpt.exists():
                runs.append(d)
    return runs


def parse_run_name(name: str) -> dict:
    """Extract hyperparams from run directory name."""
    import re
    m = re.search(r"cb(\d+)_cd(\d+)_nq(\d+)_lr([\d.e-]+)", name)
    if m:
        return {
            "cb": int(m.group(1)),
            "cd": int(m.group(2)),
            "nq": int(m.group(3)),
            "lr": m.group(4),
        }
    return {}


def main():
    parser = argparse.ArgumentParser(description="Compute codebook utilization.")
    parser.add_argument("--run_dirs", nargs="+", default=None, help="Run directories")
    parser.add_argument("--auto", action="store_true", help="Auto-discover all tested runs")
    parser.add_argument("--num_jets", type=int, default=50000, help="Number of test jets")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--output", type=str, default=None, help="Save results to CSV")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.auto:
        run_dirs = find_tested_runs()
        log.info(f"Found {len(run_dirs)} tested runs")
    elif args.run_dirs:
        run_dirs = [Path(d) for d in args.run_dirs]
    else:
        parser.error("Specify --run_dirs or --auto")

    log.info(f"Loading {args.num_jets} test jets...")
    data = load_data(args.num_jets)
    log.info(f"Data loaded: {data['csts'].shape[0]} jets, "
             f"{data['mask'].sum().item()} valid constituents")

    results = []
    for run_dir in run_dirs:
        name = run_dir.name
        ckpt = run_dir / "checkpoints" / "best.ckpt"
        log.info(f"Processing {name}...")

        try:
            model, device = load_model(ckpt)
            # Detect model input_dim from first linear layer
            input_dim = None
            for _, mod in model.encoder.named_modules():
                if hasattr(mod, 'in_features'):
                    input_dim = mod.in_features
                    break
            if input_dim is None:
                input_dim = data["csts"].shape[-1]
            run_data = {
                "csts": data["csts"][:, :, :input_dim],
                "mask": data["mask"],
            }
            util = compute_utilization(model, device, run_data, args.batch_size)

            hp = parse_run_name(name)
            row = {
                "run": name,
                **hp,
                "codebook_size": util["codebook_size"],
                "num_quantizers": util["num_quantizers"],
                "total_tokens": util["total_tokens"],
                "avg_utilization": util["avg_utilization"],
            }
            for q, v in util["per_quantizer"].items():
                row[f"util_q{q}"] = v["utilization"]
                row[f"unique_q{q}"] = v["unique_codes"]

            results.append(row)

            # Print
            q_str = "  ".join(
                f"q{q}={v['unique_codes']}/{util['codebook_size']} ({v['utilization']:.1%})"
                for q, v in util["per_quantizer"].items()
            )
            log.info(f"  avg={util['avg_utilization']:.1%}  {q_str}")

            # Free GPU memory
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            log.error(f"  FAILED: {e}")

    # Print summary table
    if results:
        print("\n" + "=" * 100)
        print(f"{'Run':<75s} {'cb':>6s} {'nq':>3s} {'avg%':>6s}  per-quantizer")
        print("-" * 100)
        for r in sorted(results, key=lambda x: x.get("avg_utilization", 0)):
            q_parts = []
            for i in range(r["num_quantizers"]):
                u = r.get(f"util_q{i}", 0)
                q_parts.append(f"q{i}={u:.0%}")
            print(f"{r['run']:<75s} {r['codebook_size']:>6d} {r['num_quantizers']:>3d} "
                  f"{r['avg_utilization']:>5.1%}  {' '.join(q_parts)}")

    # Optionally save CSV
    if args.output and results:
        import csv
        # Collect all keys across all rows (different nq → different columns)
        all_keys = dict.fromkeys(k for r in results for k in r.keys())
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        log.info(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
