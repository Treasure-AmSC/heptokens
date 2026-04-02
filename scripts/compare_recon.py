"""Compare VQ-VAE reconstruction quality across multiple runs.

Loads test_metrics.npz files produced by the ReconstructionMonitor callback
and overlays jet residual histograms and constituent feature scatter plots.

Usage
-----
pixi run python scripts/compare_recon.py \
    --run_dirs results/vqvae_scan/vqvae_scan/enc_medium_cb512_cd8_nq3_lr0.001_cw0.1_nj10000000 \
               results/vqvae_scan/vqvae_scan/enc_medium_cb512_cd8_nq3_lr0.001_cw1.0_nj10000000 \
    --run_labels "cw=0.1" "cw=1.0" \
    --output_dir results/recon_comparison \
    --output_name cw_comparison
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

log = logging.getLogger(__name__)

# Jet residual variables to overlay
JET_RESIDUAL_KEYS = [
    "pt", "mass", "eta", "phi",
    "pt_rel", "mass_rel", "eta_rel", "phi_rel",
    "radial_dist",
]

JET_RESIDUAL_LABELS = {
    "pt": r"Jet $p_T$ residual (truth $-$ reco) [MeV]",
    "mass": r"Jet mass residual (truth $-$ reco) [MeV]",
    "eta": r"Jet $\eta$ residual (truth $-$ reco)",
    "phi": r"Jet $\phi$ residual (truth $-$ reco)",
    "pt_rel": r"Jet $p_T$ relative residual",
    "mass_rel": r"Jet mass relative residual",
    "eta_rel": r"Jet $\eta$ relative residual",
    "phi_rel": r"Jet $\phi$ relative residual",
    "radial_dist": r"Constituent radial distance",
}

# Constituent feature indices (matching default cst ordering from atlas_iterable.yaml)
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

# Line styles to cycle through when the color palette repeats
_LINESTYLES = ["-", "--", "-.", ":"]


def _style_cycle(n: int) -> list[tuple[str, str]]:
    """Return (color, linestyle) pairs for *n* runs, cycling linestyles on color repeats."""
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not prop_cycle:
        prop_cycle = [f"C{i}" for i in range(10)]
    n_colors = len(prop_cycle)
    styles = []
    for i in range(n):
        color = prop_cycle[i % n_colors]
        ls = _LINESTYLES[(i // n_colors) % len(_LINESTYLES)]
        styles.append((color, ls))
    return styles


def load_metrics(run_dir: Path) -> dict[str, np.ndarray]:
    """Load test_metrics.npz from a run directory."""
    metrics_path = run_dir / "test_metrics" / "test_metrics.npz"
    if not metrics_path.exists():
        raise FileNotFoundError(f"No test metrics found at {metrics_path}")
    return dict(np.load(metrics_path))


def plot_jet_residuals(
    all_metrics: list[dict[str, np.ndarray]],
    labels: list[str],
    output_path: Path,
) -> None:
    """Overlay jet residual histograms across runs."""
    plt.style.use(hep.style.CMS)

    keys = [k for k in JET_RESIDUAL_KEYS if k in all_metrics[0]]
    n_vars = len(keys)
    n_cols = min(3, n_vars)
    n_rows = int(np.ceil(n_vars / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    styles = _style_cycle(len(all_metrics))

    for i, key in enumerate(keys):
        ax = axes[i]
        # Compute shared bin edges from all runs
        all_vals = [m[key][np.isfinite(m[key])] for m in all_metrics]
        combined = np.concatenate(all_vals)
        lo, hi = np.percentile(combined, [1, 99])
        bins = np.linspace(lo, hi, 51)

        for vals, label, (color, ls) in zip(all_vals, labels, styles):
            ax.hist(vals, bins=bins, histtype="step", linewidth=1.5,
                    label=label, density=True, color=color, linestyle=ls)

        ax.set_xlabel(JET_RESIDUAL_LABELS.get(key, key), fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    for i in range(n_vars, len(axes)):
        axes[i].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved jet residual comparison to {output_path}")


def plot_cst_features(
    all_metrics: list[dict[str, np.ndarray]],
    labels: list[str],
    output_path: Path,
) -> None:
    """Overlay constituent feature residual histograms across runs."""
    if "cst_original" not in all_metrics[0]:
        log.info("No constituent features saved; skipping cst comparison plot.")
        return

    plt.style.use(hep.style.CMS)
    styles = _style_cycle(len(all_metrics))

    n_features = all_metrics[0]["cst_original"].shape[1]
    n_cols = min(4, n_features)
    n_rows = int(np.ceil(n_features / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for feat_idx in range(n_features):
        ax = axes[feat_idx]
        feat_name = CST_FEATURE_NAMES[feat_idx] if feat_idx < len(CST_FEATURE_NAMES) else f"feature {feat_idx}"

        # Compute per-constituent residuals per run, overlay histograms
        all_residuals = []
        for m in all_metrics:
            res = m["cst_original"][:, feat_idx] - m["cst_recon"][:, feat_idx]
            all_residuals.append(res[np.isfinite(res)])

        combined = np.concatenate(all_residuals)
        lo, hi = np.percentile(combined, [1, 99])
        bins = np.linspace(lo, hi, 51)

        for res, label, (color, ls) in zip(all_residuals, labels, styles):
            ax.hist(res, bins=bins, histtype="step", linewidth=1.5,
                    label=label, density=True, color=color, linestyle=ls)

        ax.set_xlabel(f"{feat_name} residual (orig $-$ reco)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    for i in range(n_features, len(axes)):
        axes[i].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved constituent feature comparison to {output_path}")


def plot_iqr_vs_pt(
    all_metrics: list[dict[str, np.ndarray]],
    labels: list[str],
    output_path: Path,
    pt_min: float = 5_000.0,
    pt_max: float = 80_000.0,
    n_bins: int = 15,
    min_entries: int = 10,
) -> None:
    """Overlay IQR/median of pT response vs truth pT."""
    if "truth_pt" not in all_metrics[0] or "pt_ratio" not in all_metrics[0]:
        return

    plt.style.use(hep.style.CMS)
    styles = _style_cycle(len(all_metrics))
    fig, ax = plt.subplots(figsize=(7, 6))
    bin_edges = np.linspace(pt_min, pt_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    for metrics, label, (color, ls) in zip(all_metrics, labels, styles):
        truth_pt = metrics["truth_pt"].ravel()
        pt_ratio = metrics["pt_ratio"].ravel()
        valid = np.isfinite(truth_pt) & np.isfinite(pt_ratio) & (pt_ratio > 0)
        truth_pt, pt_ratio = truth_pt[valid], pt_ratio[valid]

        iqr_med = np.full(n_bins, np.nan)
        for i in range(n_bins):
            lo = truth_pt >= bin_edges[i]
            hi = truth_pt <= bin_edges[i + 1] if i == n_bins - 1 else truth_pt < bin_edges[i + 1]
            ratios = pt_ratio[lo & hi]
            if ratios.size < min_entries:
                continue
            median = np.median(ratios)
            if not np.isfinite(median) or median <= 0:
                continue
            q25, q75 = np.percentile(ratios, [25, 75])
            iqr_med[i] = (q75 - q25) / median

        ok = np.isfinite(iqr_med)
        if np.any(ok):
            ax.plot(bin_centers[ok], iqr_med[ok], marker="o", linewidth=1.5,
                   label=label, color=color, linestyle=ls)

    ax.set_xlabel(r"Truth jet $p_T$ [MeV]", fontsize=12)
    ax.set_ylabel(r"IQR / median of $p_T^{\mathrm{reco}} / p_T^{\mathrm{truth}}$", fontsize=12)
    ax.set_xlim(pt_min, pt_max)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved IQR vs pT comparison to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare VQ-VAE reconstruction across runs.")
    parser.add_argument("--run_dirs", nargs="+", required=True, help="Paths to run directories")
    parser.add_argument("--run_labels", nargs="+", default=None, help="Labels for each run (defaults to directory names)")
    parser.add_argument("--output_dir", type=str, default="results/recon_comparison")
    parser.add_argument("--output_name", type=str, default="recon_comparison")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    run_dirs = [Path(d) for d in args.run_dirs]
    labels = args.run_labels or [d.name for d in run_dirs]
    if len(labels) != len(run_dirs):
        raise ValueError(f"Got {len(run_dirs)} run_dirs but {len(labels)} labels")

    all_metrics = [load_metrics(d) for d in run_dirs]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = args.output_name

    plot_jet_residuals(all_metrics, labels, output_dir / f"{name}_jet_residuals.pdf")
    plot_cst_features(all_metrics, labels, output_dir / f"{name}_cst_features.pdf")
    plot_iqr_vs_pt(all_metrics, labels, output_dir / f"{name}_iqr_vs_pt.pdf")


if __name__ == "__main__":
    main()
