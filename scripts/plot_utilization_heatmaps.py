"""Average codebook-utilization heatmaps on the cb x cd grid (one panel per nq).

Standalone script mirroring the `_plot_grid_heatmap` styling of
scripts/compare_scan.py / scripts/compare_cocoa_scan.py, but for codebook
utilization (higher is better) using `viridis_r` so that darker = better,
matching the "darker = better" convention of the MAE/resolution heatmaps.

Three modalities are produced from pre-computed CSVs:
  - JetSet          -> results/vqvae_scan/plots_jetset/jetset_utilization_heatmap.pdf
  - COCOA tracks    -> results/cocoa_scan/plots/cocoa_utilization_heatmap.pdf
  - COCOA topos     -> results/cocoa_scan/plots_topos/cocoa_utilization_heatmap.pdf

Usage
-----
pixi run python scripts/plot_utilization_heatmaps.py
"""
from __future__ import annotations

import csv
import logging
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from matplotlib.colors import Normalize

log = logging.getLogger(__name__)

UTIL_DIR = Path("results/codebook_utilization")
JETSET_CSV = UTIL_DIR / "jetset_utilization.csv"
COCOA_CSV = UTIL_DIR / "cocoa_utilization.csv"

# JetSet large-scan grid (cd=2 excluded to match resolution/MAE heatmaps).
JETSET_CBS = [1024, 8192, 16384, 32768]
JETSET_CDS = [4, 8, 16]
JETSET_NQS = [3, 4]

# COCOA grid (cd=2 kept, matching COCOA resolution/MAE heatmaps).
COCOA_CBS = [64, 256, 1024, 4096]
COCOA_CDS = {"tracks": [2, 4, 8], "topos": [2, 4, 8, 12]}
COCOA_NQS = [2, 3, 4]

CBAR_LABEL = "Avg. codebook util."


def _plot_grid_heatmap(results, out_path, cbs, cds, nqs, cbar_label, fmt="{:.3f}"):
    """cb x cd heatmap of avg utilization, one panel per nq, shared color scale.

    Uses viridis_r so darker = higher utilization = better, consistent with the
    MAE/resolution heatmaps where darker = lower error = better.
    Label colour is luminance-adaptive (white on dark cells, black on light).
    """
    plt.style.use(hep.style.CMS)
    cmap = plt.get_cmap("viridis_r")

    all_vals = [r["value"] for r in results if np.isfinite(r["value"])]
    vmin, vmax = (float(np.min(all_vals)), float(np.max(all_vals))) if all_vals else (None, None)
    norm = Normalize(vmin=vmin, vmax=vmax) if (vmin is not None and vmax > vmin) else None

    fig, axes = plt.subplots(1, len(nqs), figsize=(6 * len(nqs), 5), squeeze=False)
    axes = axes.flatten()
    for ax_idx, nq in enumerate(nqs):
        ax = axes[ax_idx]
        grid = np.full((len(cds), len(cbs)), np.nan)
        for r in results:
            if r["nq"] != nq:
                continue
            if r["cb"] in cbs and r["cd"] in cds:
                grid[cds.index(r["cd"]), cbs.index(r["cb"])] = r["value"]
        im = ax.imshow(grid, cmap=cmap, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(cbs)))
        ax.set_xticklabels([rf"$2^{{{int(round(np.log2(cb)))}}}$" for cb in cbs])
        ax.set_yticks(range(len(cds)))
        ax.set_yticklabels(cds)
        ax.set_xlabel("Codebook size", fontsize=18)
        ax.set_ylabel("Codebook dim", fontsize=18)
        ax.set_title(f"nq = {nq}", fontsize=13)
        for i in range(len(cds)):
            for j in range(len(cbs)):
                if not np.isnan(grid[i, j]):
                    lum = 1.0
                    if norm is not None:
                        rr, gg, bb, _ = cmap(norm(grid[i, j]))
                        lum = 0.299 * rr + 0.587 * gg + 0.114 * bb
                    ax.text(j, i, fmt.format(grid[i, j]), ha="center", va="center",
                            fontsize=9, color="white" if lum < 0.5 else "black")
        plt.colorbar(im, ax=ax, shrink=0.8, label=cbar_label)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {out_path}")


def _print_grid(title, results, cbs, cds, nqs):
    print(f"\n=== {title}: avg_utilization grid (rows=cd, cols=cb) ===")
    for nq in nqs:
        print(f"\n  nq = {nq}")
        header = "    cd \\ cb | " + " | ".join(f"{cb:>8d}" for cb in cbs)
        print(header)
        print("    " + "-" * (len(header) - 4))
        for cd in cds:
            cells = []
            for cb in cbs:
                v = next((r["value"] for r in results
                          if r["nq"] == nq and r["cb"] == cb and r["cd"] == cd), None)
                cells.append(f"{v:8.3f}" if v is not None else "     ---")
            print(f"    {cd:>7d} | " + " | ".join(cells))


def load_jetset():
    results = []
    with open(JETSET_CSV) as f:
        for row in csv.DictReader(f):
            run = row["run"]
            if not ("enc_large" in run and "lr0.0003" in run and "nj10000000" in run):
                continue
            cb, cd, nq = int(row["cb"]), int(row["cd"]), int(row["nq"])
            av = row["avg_utilization"]
            if av == "":
                continue
            results.append({"cb": cb, "cd": cd, "nq": nq, "value": float(av)})
    return results


def load_cocoa(modality):
    results = []
    with open(COCOA_CSV) as f:
        for row in csv.DictReader(f):
            if row["modality"] != modality:
                continue
            m = re.search(r"cd(\d+)", row["run"])
            if not m:
                log.warning(f"No cd in run name, skipping: {row['run']}")
                continue
            cd = int(m.group(1))  # cd column is buggy (0) for COCOA -> parse from name
            cb, nq = int(row["cb"]), int(row["nq"])
            av = row["avg_utilization"]
            if av == "":
                continue
            results.append({"cb": cb, "cd": cd, "nq": nq, "value": float(av)})
    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # ── JetSet ────────────────────────────────────────────────────────────
    jetset = load_jetset()
    _plot_grid_heatmap(
        jetset,
        Path("results/vqvae_scan/plots_jetset/jetset_utilization_heatmap.pdf"),
        JETSET_CBS, JETSET_CDS, JETSET_NQS, CBAR_LABEL,
    )
    _print_grid("JetSet", jetset, JETSET_CBS, JETSET_CDS, JETSET_NQS)

    # ── COCOA tracks ──────────────────────────────────────────────────────
    tracks = load_cocoa("tracks")
    _plot_grid_heatmap(
        tracks,
        Path("results/cocoa_scan/plots/cocoa_utilization_heatmap.pdf"),
        COCOA_CBS, COCOA_CDS["tracks"], COCOA_NQS, CBAR_LABEL,
    )
    _print_grid("COCOA tracks", tracks, COCOA_CBS, COCOA_CDS["tracks"], COCOA_NQS)

    # ── COCOA topos ───────────────────────────────────────────────────────
    topos = load_cocoa("topos")
    _plot_grid_heatmap(
        topos,
        Path("results/cocoa_scan/plots_topos/cocoa_utilization_heatmap.pdf"),
        COCOA_CBS, COCOA_CDS["topos"], COCOA_NQS, CBAR_LABEL,
    )
    _print_grid("COCOA topos", topos, COCOA_CBS, COCOA_CDS["topos"], COCOA_NQS)


if __name__ == "__main__":
    main()
