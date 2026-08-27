"""Compare medium vs large transformer VQ-VAE across matching hyperparams.

Produces:
- IQR vs pT overlay (medium=solid, large=dashed, color=codebook size)
- Jet pT relative residual overlay
- Summary bar chart

Usage
-----
pixi run python scripts/compare_med_vs_large.py
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

# Matched configs: cd=8, nq=3 at three codebook sizes
CONFIGS = [
    {"cb": 8192,  "cd": 8, "nq": 3},
    {"cb": 16384, "cd": 8, "nq": 3},
    {"cb": 32768, "cd": 8, "nq": 3},
]

MED_TPL = "transformer_vqvae_enc_medium_cb{cb}_cd{cd}_nq{nq}_lr0.001_cw1.0_nj10000000"
LRG_TPL = "transformer_vqvae_enc_large_cb{cb}_cd{cd}_nq{nq}_lr0.0003_cw1.0_nj10000000"

PT_MIN, PT_MAX, N_BINS = 5_000.0, 80_000.0, 15

JET_KEYS = ["pt_rel", "mass_rel", "eta_rel", "phi_rel"]
JET_LABELS = {
    "pt_rel": r"Jet $p_T$ relative residual",
    "mass_rel": r"Jet mass relative residual",
    "eta_rel": r"Jet $\eta$ relative residual",
    "phi_rel": r"Jet $\phi$ relative residual",
}


def load_metrics(name):
    for sub in ("test/test_metrics/test_metrics.npz", "test_metrics/test_metrics.npz"):
        p = SCAN_ROOT / name / sub
        if p.exists():
            return dict(np.load(p))
    return None


def compute_iqr_vs_pt(metrics):
    bin_edges = np.linspace(PT_MIN, PT_MAX, N_BINS + 1)
    bc = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    tp = metrics["truth_pt"].ravel()
    pr = metrics["pt_ratio"].ravel()
    ok = np.isfinite(tp) & np.isfinite(pr) & (pr > 0)
    tp, pr = tp[ok], pr[ok]
    iqr = np.full(N_BINS, np.nan)
    for i in range(N_BINS):
        lo = tp >= bin_edges[i]
        hi = tp <= bin_edges[i + 1] if i == N_BINS - 1 else tp < bin_edges[i + 1]
        r = pr[lo & hi]
        if r.size < 10:
            continue
        med = np.median(r)
        if not np.isfinite(med) or med <= 0:
            continue
        q25, q75 = np.percentile(r, [25, 75])
        iqr[i] = (q75 - q25) / med
    return bc, iqr


def compute_overall_iqr(metrics):
    tp = metrics["truth_pt"].ravel()
    pr = metrics["pt_ratio"].ravel()
    ok = np.isfinite(tp) & np.isfinite(pr) & (pr > 0) & (tp >= PT_MIN) & (tp <= PT_MAX)
    pr = pr[ok]
    if pr.size < 10:
        return np.nan
    med = np.median(pr)
    q25, q75 = np.percentile(pr, [25, 75])
    return (q75 - q25) / med


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="results/med_vs_large")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    plt.style.use(hep.style.CMS)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prop_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [f"C{i}" for i in range(10)])

    # Load all
    entries = []  # (label, linestyle, color, metrics, cb)
    for i, cfg in enumerate(CONFIGS):
        color = prop_colors[i % len(prop_colors)]
        cb = cfg["cb"]
        m_med = load_metrics(MED_TPL.format(**cfg))
        m_lrg = load_metrics(LRG_TPL.format(**cfg))
        if m_med is not None:
            entries.append((f"Medium cb={cb}", "-", color, m_med, cb))
        if m_lrg is not None:
            entries.append((f"Large cb={cb}", "--", color, m_lrg, cb))

    if not entries:
        log.error("No metrics found")
        return

    # ── Plot 1: IQR vs pT ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    for label, ls, color, m, _ in entries:
        bc, iqr = compute_iqr_vs_pt(m)
        ok = np.isfinite(iqr)
        if np.any(ok):
            ax.plot(bc[ok], iqr[ok], marker="o", linewidth=2,
                    label=label, color=color, linestyle=ls)

    ax.set_xlabel(r"Truth jet $p_T$ [MeV]", fontsize=14)
    ax.set_ylabel(r"IQR / median of $p_T^{\rm reco}/p_T^{\rm truth}$", fontsize=14)
    ax.set_xlim(PT_MIN, PT_MAX)
    ax.set_title("Medium vs Large Transformer  (cd=8, nq=3)\nsolid = medium (lr=1e-3), dashed = large (lr=3e-4)",
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = output_dir / "med_vs_large_iqr_vs_pt.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {out}")

    # ── Plot 2: Jet residual overlays ─────────────────────────────────────
    keys = [k for k in JET_KEYS if k in entries[0][3]]
    n = len(keys)
    n_cols = min(2, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5.5 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for i, key in enumerate(keys):
        ax = axes[i]
        all_v = [m[key][np.isfinite(m[key])] for _, _, _, m, _ in entries]
        combined = np.concatenate(all_v)
        lo, hi = np.percentile(combined, [1, 99])
        bins = np.linspace(lo, hi, 51)

        for label, ls, color, m, _ in entries:
            v = m[key][np.isfinite(m[key])]
            ax.hist(v, bins=bins, histtype="step", linewidth=2,
                    label=label, density=True, color=color, linestyle=ls)

        ax.set_xlabel(JET_LABELS.get(key, key), fontsize=13)
        ax.set_ylabel("Density", fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    for i in range(n, len(axes)):
        axes[i].axis("off")

    fig.suptitle("Medium vs Large Transformer  (cd=8, nq=3)\nsolid = medium, dashed = large",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    out = output_dir / "med_vs_large_residuals.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {out}")

    # ── Plot 3: Summary bar chart ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    labels_bar = []
    iqr_vals = []
    bar_colors = []
    hatches = []
    for label, ls, color, m, cb in entries:
        labels_bar.append(label)
        iqr_vals.append(compute_overall_iqr(m))
        bar_colors.append(color)
        hatches.append("///" if ls == "--" else "")

    x = np.arange(len(labels_bar))
    bars = ax.bar(x, iqr_vals, color=bar_colors, edgecolor="black", linewidth=0.8)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_bar, fontsize=10, rotation=30, ha="right")
    ax.set_ylabel(r"IQR / median of $p_T^{\rm reco}/p_T^{\rm truth}$", fontsize=13)
    ax.set_title("Medium vs Large Transformer  (cd=8, nq=3)\nsolid = medium, hatched = large",
                 fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out = output_dir / "med_vs_large_summary.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {out}")

    # Print ranking
    log.info("=== Medium vs Large comparison ===")
    for label, _, _, m, _ in entries:
        iqr = compute_overall_iqr(m)
        log.info(f"  {label:30s}  IQR/med={iqr:.5f}")


if __name__ == "__main__":
    main()
