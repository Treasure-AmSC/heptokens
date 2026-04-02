"""Plot 2D histogram correlation of original vs reconstructed constituent features."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import mplhep as hep
import numpy as np

CST_FEATURE_NAMES = [
    r"$p_T$", r"$\Delta\eta$", r"$\Delta\phi$",
    r"$d_0$", r"$d_0$ (beamspot)", r"$\sigma(d_0)$", r"$\sigma(d_0^{\mathrm{BS}})$",
    r"$z_0^{\mathrm{BS}}$", r"$\sigma(z_0^{\mathrm{BS}})$",
    r"$z_0 \sin\theta$", r"$\sigma(z_0 \sin\theta)$",
    r"signed $d_0$", r"signed $d_0$ signif.",
    r"signed $z_0 \sin\theta$", r"signed $z_0 \sin\theta$ signif.",
    r"$\theta$", r"$\sigma(\theta)$",
    r"$q/p$", r"$\sigma(q/p)$",
    r"$p_T$ frac",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--output_dir", default="results/recon_comparison")
    parser.add_argument("--output_name", default="orig_vs_reco_2d")
    args = parser.parse_args()

    plt.style.use(hep.style.CMS)

    metrics_path = Path(args.run_dir) / "test_metrics" / "test_metrics.npz"
    d = np.load(metrics_path)
    orig = d["cst_original"]
    reco = d["cst_recon"]

    n_features = orig.shape[1]
    n_cols = 4
    n_rows = int(np.ceil(n_features / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
    axes = axes.flatten()

    for i in range(n_features):
        ax = axes[i]
        o = orig[:, i]
        r = reco[:, i]
        valid = np.isfinite(o) & np.isfinite(r)
        o, r = o[valid], r[valid]

        lo, hi = np.percentile(np.concatenate([o, r]), [0.5, 99.5])
        bins = np.linspace(lo, hi, 101)

        h, xedges, yedges = np.histogram2d(o, r, bins=[bins, bins])
        h = np.ma.masked_where(h == 0, h)

        ax.pcolormesh(xedges, yedges, h.T, cmap="viridis", norm=mcolors.LogNorm())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, alpha=0.7)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")

        name = CST_FEATURE_NAMES[i] if i < len(CST_FEATURE_NAMES) else f"feature {i}"
        ax.set_xlabel(f"Original {name}", fontsize=12)
        ax.set_ylabel(f"Reconstructed {name}", fontsize=12)

    for i in range(n_features, len(axes)):
        axes[i].axis("off")

    title = args.label or Path(args.run_dir).parent.name
    fig.suptitle(f"{title}: Original vs Reconstructed", fontsize=16, y=1.01)
    fig.tight_layout()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    outpath = out / f"{args.output_name}.pdf"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outpath}")


if __name__ == "__main__":
    main()
