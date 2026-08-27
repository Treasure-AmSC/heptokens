"""Compare VQ-VAE vs Transformer VQ-VAE residuals.

Solid lines = MLP VQ-VAE,  dotted lines = Transformer VQ-VAE.
Same color = same scan-parameter value.

Usage
-----
pixi run python scripts/compare_models.py --mode nq
pixi run python scripts/compare_models.py --mode codebook_size
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

log = logging.getLogger(__name__)

SCAN_ROOT = Path("results/vqvae_scan/vqvae_scan")

JET_RESIDUAL_KEYS = ["pt_rel", "mass_rel", "eta_rel", "phi_rel", "radial_dist"]
JET_RESIDUAL_LABELS = {
    "pt_rel": r"Jet $p_T$ relative residual",
    "mass_rel": r"Jet mass relative residual",
    "eta_rel": r"Jet $\eta$ relative residual",
    "phi_rel": r"Jet $\phi$ relative residual",
    "radial_dist": r"Constituent radial distance",
}

# ── Scan mode definitions ─────────────────────────────────────────────────────
SCAN_MODES = {
    "nq": {
        "values": [1, 2, 3, 4],
        "label_prefix": "nq",
        "vqvae_tpl":       "enc_medium_cb512_cd8_nq{v}_lr0.001_cw1.0_nj10000000",
        "transformer_tpl": "transformer_vqvae_enc_medium_cb512_cd8_nq{v}_lr0.001_cw1.0_nj1000000",
        "subtitle": "MLP VQ-VAE (solid, 10M jets) vs Transformer VQ-VAE (dotted, 1M jets)",
        "vqvae_bar_label": "MLP VQ-VAE (10M)",
        "transformer_bar_label": "Transformer VQ-VAE (1M)",
    },
    "codebook_size": {
        "values": [32, 64, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
        "label_prefix": "cb",
        "vqvae_tpl":       "enc_medium_cb{v}_cd8_nq3_lr0.001_cw1.0_nj10000000",
        "transformer_tpl": "transformer_vqvae_enc_medium_cb{v}_cd8_nq3_lr0.001_cw1.0_nj10000000",
        "subtitle": "MLP VQ-VAE (solid) vs Transformer VQ-VAE (dotted)  —  10M jets, nq=3",
        "vqvae_bar_label": "MLP VQ-VAE",
        "transformer_bar_label": "Transformer VQ-VAE",
    },
}


def load_metrics(run_name):
    path = SCAN_ROOT / run_name / "test" / "test_metrics" / "test_metrics.npz"
    if not path.exists():
        return None
    return dict(np.load(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="results/recon_comparison")
    parser.add_argument("--output_name", default=None,
                        help="Output file prefix (default: vqvae_vs_transformer_<mode>)")
    parser.add_argument("--mode", choices=list(SCAN_MODES), default="nq",
                        help="Scan dimension to compare")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    plt.style.use(hep.style.CMS)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mode = SCAN_MODES[args.mode]
    values = mode["values"]
    prefix = mode["label_prefix"]
    subtitle = mode["subtitle"]
    output_name = args.output_name or f"vqvae_vs_transformer_{args.mode}"

    # Assign colors from the default prop cycle
    prop_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [f"C{i}" for i in range(10)])

    # ── Load all metrics ──────────────────────────────────────────────────
    entries = []  # (value, label, linestyle, color, metrics)
    for i, v in enumerate(values):
        color = prop_colors[i % len(prop_colors)]
        m = load_metrics(mode["vqvae_tpl"].format(v=v))
        if m is not None:
            entries.append((v, f"MLP {prefix}={v}", "-", color, m))
        m = load_metrics(mode["transformer_tpl"].format(v=v))
        if m is not None:
            entries.append((v, f"Transf. {prefix}={v}", ":", color, m))

    if not entries:
        log.error("No metrics found")
        return

    # ── Plot 1: Jet residual overlays ─────────────────────────────────────
    keys = [k for k in JET_RESIDUAL_KEYS if k in entries[0][4]]
    n = len(keys)
    n_cols = min(3, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for i, key in enumerate(keys):
        ax = axes[i]
        # Shared bins
        all_vals = []
        for _, _, _, _, m in entries:
            v = m[key]
            all_vals.append(v[np.isfinite(v)])
        combined = np.concatenate(all_vals)
        lo, hi = np.percentile(combined, [1, 99])
        bins = np.linspace(lo, hi, 51)

        for nq, label, ls, color, m in entries:
            v = m[key][np.isfinite(m[key])]
            ax.hist(v, bins=bins, histtype="step", linewidth=2,
                    label=label, density=True, color=color, linestyle=ls)

        ax.set_xlabel(JET_RESIDUAL_LABELS.get(key, key), fontsize=14)
        ax.set_ylabel("Density", fontsize=14)
        ax.legend(fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)

    for i in range(n, len(axes)):
        axes[i].axis("off")

    fig.suptitle(subtitle, fontsize=16, y=1.01)
    fig.tight_layout()
    out = output_dir / f"{output_name}_residuals.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {out}")

    # ── Plot 2: IQR vs pT ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    pt_min, pt_max, n_bins = 5_000., 80_000., 15
    bin_edges = np.linspace(pt_min, pt_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    for nq, label, ls, color, m in entries:
        truth_pt = m["truth_pt"].ravel()
        pt_ratio = m["pt_ratio"].ravel()
        valid = np.isfinite(truth_pt) & np.isfinite(pt_ratio) & (pt_ratio > 0)
        truth_pt, pt_ratio = truth_pt[valid], pt_ratio[valid]

        iqr_med = np.full(n_bins, np.nan)
        for j in range(n_bins):
            lo_mask = truth_pt >= bin_edges[j]
            hi_mask = truth_pt <= bin_edges[j + 1] if j == n_bins - 1 else truth_pt < bin_edges[j + 1]
            ratios = pt_ratio[lo_mask & hi_mask]
            if ratios.size < 10:
                continue
            median = np.median(ratios)
            if not np.isfinite(median) or median <= 0:
                continue
            q25, q75 = np.percentile(ratios, [25, 75])
            iqr_med[j] = (q75 - q25) / median

        ok = np.isfinite(iqr_med)
        if np.any(ok):
            ax.plot(bin_centers[ok], iqr_med[ok], marker="o", linewidth=2,
                    label=label, color=color, linestyle=ls)

    ax.set_xlabel(r"Truth jet $p_T$ [MeV]", fontsize=14)
    ax.set_ylabel(r"IQR / median of $p_T^{\rm reco}/p_T^{\rm truth}$", fontsize=14)
    ax.set_xlim(pt_min, pt_max)
    ax.set_title(subtitle, fontsize=13)
    ax.legend(fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = output_dir / f"{output_name}_iqr_vs_pt.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {out}")

    # ── Plot 3: Summary bar chart ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    mlp_entries = [(nq, l, m) for nq, l, ls, c, m in entries if ls == "-"]
    trn_entries = [(nq, l, m) for nq, l, ls, c, m in entries if ls == ":"]

    x = np.arange(len(values))
    width = 0.35

    for ax_idx, (metric_key, ylabel) in enumerate([
        ("pt_rel", r"Median $|p_T$ rel. residual$|$"),
        ("radial_dist", r"Median radial distance"),
    ]):
        ax = axes[ax_idx]
        mlp_vals, trn_vals = [], []
        for v in values:
            # MLP
            mlp_m = next((m for n, _, m in mlp_entries if n == v), None)
            if mlp_m is not None:
                d = mlp_m[metric_key]; d = d[np.isfinite(d)]
                mlp_vals.append(np.median(np.abs(d)) if metric_key == "pt_rel" else np.median(d))
            else:
                mlp_vals.append(0)
            # Transformer
            trn_m = next((m for n, _, m in trn_entries if n == v), None)
            if trn_m is not None:
                d = trn_m[metric_key]; d = d[np.isfinite(d)]
                trn_vals.append(np.median(np.abs(d)) if metric_key == "pt_rel" else np.median(d))
            else:
                trn_vals.append(0)

        bar_colors = [prop_colors[i % len(prop_colors)] for i in range(len(values))]
        ax.bar(x - width/2, mlp_vals, width, label=mode["vqvae_bar_label"],
               color=bar_colors, edgecolor="black", linewidth=0.8)
        ax.bar(x + width/2, trn_vals, width, label=mode["transformer_bar_label"],
               color=bar_colors, edgecolor="black", linewidth=0.8,
               hatch="///")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{prefix}={v}" for v in values], fontsize=10,
                           rotation=45 if len(values) > 6 else 0, ha="right" if len(values) > 6 else "center")
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(subtitle, fontsize=14)
    fig.tight_layout()
    out = output_dir / f"{output_name}_summary.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {out}")


if __name__ == "__main__":
    main()
