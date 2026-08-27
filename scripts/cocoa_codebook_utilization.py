"""Full-test-set codebook utilization for COCOA tracks & topo-cluster VQ-VAEs.

Mirrors the JetSet accumulation method in ``scripts/codebook_vqvae_utilization.py``:
for each residual-quantizer stage, accumulate the set of *unique* codes used across the
ENTIRE COCOA test split (``test_1M_cells256_isInfFalse``), then divide by the codebook
size. This makes COCOA utilization directly comparable to the JetSet
``results/codebook_utilization/utilization.csv`` (unlike the per-batch
``val/codebook_util_*`` logged during training, which is averaged over validation batches
and therefore biased low for large codebooks).

The COCOA test data is loaded ONCE per modality (via the same datamodule + preprocessing
used in training) and reused across every run of that modality.

Usage
-----
    pixi run python scripts/cocoa_codebook_utilization.py --modality both
    pixi run python scripts/cocoa_codebook_utilization.py --modality tracks --num_events 5000
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

log = logging.getLogger(__name__)

COCOA_SCAN = Path("results/cocoa_scan")
DM_CONFIG = {
    "tracks": "configs/datamodule/cocoa_tracks.yaml",
    "topos": "configs/datamodule/cocoa_topos.yaml",
}


def load_model(ckpt_path: Path, device: torch.device):
    from heptokens.models.vq_vae import LitVqVae

    model = LitVqVae.load_from_checkpoint(str(ckpt_path), map_location=device)
    model.to(device)
    model.eval()
    return model


def cache_test_batches(modality: str, num_events: int | None) -> list[dict]:
    """Load the full COCOA test split once (preprocessed) and cache batches on CPU."""
    cfg = OmegaConf.load(DM_CONFIG[modality])
    if num_events is not None:
        cfg.num_events = num_events
    cfg.pin_memory = False
    dm = instantiate(cfg)
    dm.setup("test")
    loader = dm.test_dataloader()

    batches, n_events = [], 0
    for b in loader:
        batches.append({"csts": b["csts"].cpu(), "mask": b["mask"].cpu()})
        n_events += int(b["mask"].shape[0])
    log.info("[%s] cached %d batches, %d events", modality, len(batches), n_events)
    return batches


@torch.no_grad()
def utilization_for_run(model, batches: list[dict], device: torch.device) -> dict:
    """Accumulate unique codes per quantizer across all cached batches."""
    cb = int(model.codebook_size)
    nq = len(model.vector_quantization.layers)
    seen = [np.zeros(cb, dtype=bool) for _ in range(nq)]
    total = 0
    for b in batches:
        batch = {"csts": b["csts"].to(device), "mask": b["mask"].to(device)}
        _, indices, _ = model.encode(batch)          # [B, n_csts, nq], -1 for masked
        valid = indices[batch["mask"]]               # [n_valid, nq]
        total += int(valid.shape[0])
        valid = valid.cpu().numpy()
        for q in range(nq):
            seen[q][valid[:, q]] = True
    counts = [int(s.sum()) for s in seen]
    per_q = [c / cb for c in counts]
    return {
        "codebook_size": cb,
        "codebook_dim": int(getattr(model, "codebook_dim", 0)),
        "num_quantizers": nq,
        "total_tokens": total,
        "avg_utilization": float(np.mean(per_q)),
        "per_q": per_q,
        "counts": counts,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modality", choices=["tracks", "topos", "both"], default="both")
    ap.add_argument("--num_events", type=int, default=None,
                    help="Limit events per split (default: full test set).")
    ap.add_argument("--output", default="results/codebook_utilization/cocoa_utilization.csv")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device: %s", device)

    modalities = ["tracks", "topos"] if args.modality == "both" else [args.modality]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    max_nq = 0
    for mod in modalities:
        ckpts = sorted(COCOA_SCAN.joinpath(mod).glob("*/checkpoints/best.ckpt"))
        if not ckpts:
            log.warning("no runs found for modality %s", mod)
            continue
        batches = cache_test_batches(mod, args.num_events)
        for ckpt in ckpts:
            run = ckpt.parent.parent.name
            try:
                model = load_model(ckpt, device)
                res = utilization_for_run(model, batches, device)
                max_nq = max(max_nq, res["num_quantizers"])
                row = {
                    "modality": mod, "run": run,
                    "cb": res["codebook_size"], "cd": res["codebook_dim"],
                    "nq": res["num_quantizers"], "total_tokens": res["total_tokens"],
                    "avg_utilization": round(res["avg_utilization"], 6),
                }
                for q in range(res["num_quantizers"]):
                    row[f"util_q{q}"] = round(res["per_q"][q], 6)
                    row[f"unique_q{q}"] = res["counts"][q]
                rows.append(row)
                log.info("%s: avg=%.4f per_q=%s", run, res["avg_utilization"],
                         [round(x, 3) for x in res["per_q"]])
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            except Exception as e:  # noqa: BLE001 - keep going over the scan
                log.error("%s FAILED: %s", run, e)

    if not rows:
        log.error("no rows produced; nothing written")
        return

    base_cols = ["modality", "run", "cb", "cd", "nq", "total_tokens", "avg_utilization"]
    q_cols = [f"util_q{q}" for q in range(max_nq)] + [f"unique_q{q}" for q in range(max_nq)]
    fieldnames = base_cols + q_cols
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    log.info("wrote %d rows to %s", len(rows), out)


if __name__ == "__main__":
    main()
