"""Compare COCOA track VQ-VAE scan: jet pT resolution, MAE heatmaps, per-feature plots.

Usage
-----
pixi run python scripts/compare_cocoa_scan.py
pixi run python scripts/compare_cocoa_scan.py --scan_dir results/cocoa_scan/tracks --output_dir results/cocoa_scan/plots
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

log = logging.getLogger(__name__)

FEATURE_NAMES = ["pT [GeV]", r"$\eta$", r"$\phi$", r"$d_0$ [mm]", r"$z_0$ [mm]"]
PT_MIN, PT_MAX, N_BINS = 1.0, 100.0, 15

# Per-feature labels and which feature indices feed the jet 4-vector (jet pT).
FEATURE_LABELS = {
    "Track": [r"$p_T$ [GeV]", r"$\eta$", r"$\phi$", r"$d_0$ [mm]", r"$z_0$ [mm]"],
    "Topo": [r"$\eta$", r"$\phi$", r"$E$ [GeV]", r"$\rho$",
             r"$\sigma_\eta$", r"$\sigma_\phi$", r"$E_{\mathrm{ECAL}}$", r"$E_{\mathrm{HCAL}}$"],
}
JET_PT_FEATURES = {"Track": {0, 1, 2}, "Topo": {0, 1, 2}}


def load_scan_results(scan_dir: Path) -> list[dict]:
    results = []
    for d in sorted(scan_dir.iterdir()):
        if not d.is_dir() or not re.match(r"cocoa_\w+_cb", d.name):
            continue
        metrics_files = list(d.rglob("test_metrics.npz"))
        if not metrics_files:
            continue
        metrics_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        data = np.load(metrics_files[0])

        m_cb = re.search(r"cb(\d+)", d.name)
        m_cd = re.search(r"cd(\d+)", d.name)
        m_nq = re.search(r"nq(\d+)", d.name)
        if not all([m_cb, m_cd, m_nq]):
            continue

        cb, cd, nq = int(m_cb.group(1)), int(m_cd.group(1)), int(m_nq.group(1))
        orig = data["cst_original"]
        reco = data["cst_recon"]
        mae_per_feat = np.mean(np.abs(orig - reco), axis=0)

        entry = {"name": d.name, "cb": cb, "cd": cd, "nq": nq,
                 "mae": np.mean(mae_per_feat), "mae_per_feat": mae_per_feat}

        if "truth_pt" in data and "pt_ratio" in data:
            entry["truth_pt"] = data["truth_pt"].ravel()
            pr = data["pt_ratio"].ravel()
            entry["pt_ratio"] = pr
            valid = pr[np.isfinite(pr) & (pr > 0)]
            if valid.size:
                med = np.median(valid)
                q25, q75 = np.percentile(valid, [25, 75])
                entry["jet_res"] = (q75 - q25) / med if med > 0 else np.nan

        results.append(entry)

    results.sort(key=lambda r: r["mae"])
    return results


def compute_iqr_vs_pt(truth_pt, pt_ratio):
    bin_edges = np.linspace(PT_MIN, PT_MAX, N_BINS + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    valid = np.isfinite(truth_pt) & np.isfinite(pt_ratio) & (pt_ratio > 0)
    truth_pt, pt_ratio = truth_pt[valid], pt_ratio[valid]
    iqr_med = np.full(N_BINS, np.nan)
    for i in range(N_BINS):
        lo = truth_pt >= bin_edges[i]
        hi = truth_pt <= bin_edges[i + 1] if i == N_BINS - 1 else truth_pt < bin_edges[i + 1]
        ratios = pt_ratio[lo & hi]
        if ratios.size < 10:
            continue
        median = np.median(ratios)
        if not np.isfinite(median) or median <= 0:
            continue
        q25, q75 = np.percentile(ratios, [25, 75])
        iqr_med[i] = (q75 - q25) / median
    return bin_centers, iqr_med


def plot_resolution_vs_pt(results, output_dir, modality="Track"):
    plt.style.use(hep.style.CMS)
    cbs = sorted(set(r["cb"] for r in results))
    cds = sorted(set(r["cd"] for r in results))
    nqs = sorted(set(r["nq"] for r in results))

    palette = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd", "#8c564b"]
    cb_colors = dict(zip(cbs, palette))

    # Facet by codebook dim (rows) x num quantizers (cols); color encodes codebook size.
    # This avoids overloading linestyle, which is hard to read when curves overlap.
    fig, axes = plt.subplots(len(cds), len(nqs),
                             figsize=(5.0 * len(nqs), 3.6 * len(cds)),
                             sharex=True, sharey=True, squeeze=False)

    for i, cd in enumerate(cds):
        for j, nq in enumerate(nqs):
            ax = axes[i][j]
            for r in results:
                if r["cd"] != cd or r["nq"] != nq or "truth_pt" not in r:
                    continue
                bc, iqr = compute_iqr_vs_pt(r["truth_pt"], r["pt_ratio"])
                ok = np.isfinite(iqr)
                if not ok.any():
                    continue
                ax.plot(bc[ok], iqr[ok], marker="o", markersize=4, linewidth=1.5,
                        color=cb_colors[r["cb"]], label=f"cb={r['cb']}")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3, which="both")
            ax.set_xlim(PT_MIN, PT_MAX)
            ax.set_title(f"nq = {nq}, cd = {cd}", fontsize=12)
            if j == 0:
                ax.set_ylabel(r"IQR / median" + "\n" + r"$p_T^{\rm reco}/p_T^{\rm truth}$", fontsize=11)
            if i == len(cds) - 1:
                ax.set_xlabel(r"Truth jet $p_T$ [GeV]", fontsize=12)

    # Single shared legend (codebook size only).
    handles = [plt.Line2D([], [], color=cb_colors[cb], marker="o", linewidth=1.5,
                          label=f"cb = {cb}") for cb in cbs]
    fig.legend(handles=handles, loc="upper center", ncol=len(cbs),
               fontsize=11, frameon=False, bbox_to_anchor=(0.5, 1.0))

    fig.suptitle(rf"COCOA {modality} Tokenization: Jet $p_T$ Resolution vs Truth $p_T$",
                 fontsize=14, y=1.04)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = output_dir / "cocoa_jet_pt_resolution_vs_pt.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {out}")


def _plot_grid_heatmap(results, output_dir, value_key, cbar_label, suptitle, out_name,
                       fmt="{:.4f}", shared_scale=False, vmin=None, vmax=None):
    """cb x cd heatmap of `value_key`, one panel per nq. Lower = greener (RdYlGn_r).

    If shared_scale, all panels use a common color range (and text-color threshold) so
    cells can be compared directly across subplots. Explicit vmin/vmax (if given)
    take precedence over shared_scale's auto-computed range — use this to force the
    same color scale across separate invocations of this script (e.g. comparing two
    scan_dirs side by side), since shared_scale alone only shares within one call.
    """
    plt.style.use(hep.style.CMS)
    cbs = sorted(set(r["cb"] for r in results))
    cds = sorted(set(r["cd"] for r in results))
    nqs = sorted(set(r["nq"] for r in results))

    from matplotlib.colors import Normalize
    cmap = plt.get_cmap("viridis")

    if vmin is None and vmax is None and shared_scale:
        all_vals = [r[value_key] for r in results if value_key in r]
        if all_vals:
            vmin, vmax = float(np.min(all_vals)), float(np.max(all_vals))

    fig, axes = plt.subplots(1, len(nqs), figsize=(6 * len(nqs), 5), squeeze=False)
    axes = axes.flatten()

    for ax_idx, nq in enumerate(nqs):
        ax = axes[ax_idx]
        grid = np.full((len(cds), len(cbs)), np.nan)
        for r in results:
            if r["nq"] != nq or value_key not in r:
                continue
            grid[cds.index(r["cd"]), cbs.index(r["cb"])] = r[value_key]
        im = ax.imshow(grid, cmap=cmap, aspect="auto", origin="lower",
                       vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(cbs)))
        ax.set_xticklabels([rf"$2^{{{int(round(np.log2(cb)))}}}$" for cb in cbs])
        ax.set_yticks(range(len(cds)))
        ax.set_yticklabels(cds)
        ax.set_xlabel("Codebook size", fontsize=18)
        ax.set_ylabel("Codebook dim", fontsize=18)
        ax.set_title(f"nq = {nq}", fontsize=13)
        gvmin = vmin if vmin is not None else np.nanmin(grid)
        gvmax = vmax if vmax is not None else np.nanmax(grid)
        pnorm = Normalize(gvmin, gvmax) if np.isfinite(gvmin) and np.isfinite(gvmax) and gvmax > gvmin else None
        for i in range(len(cds)):
            for j in range(len(cbs)):
                if not np.isnan(grid[i, j]):
                    # Luminance-adaptive label colour: white on dark viridis, black on light.
                    lum = 1.0
                    if pnorm is not None:
                        rr, gg, bb, _ = cmap(pnorm(grid[i, j]))
                        lum = 0.299 * rr + 0.587 * gg + 0.114 * bb
                    ax.text(j, i, fmt.format(grid[i, j]), ha="center", va="center",
                            fontsize=9, color="white" if lum < 0.5 else "black")
        plt.colorbar(im, ax=ax, shrink=0.8, label=cbar_label)

    fig.tight_layout()
    out = output_dir / out_name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {out}")


def plot_heatmap(results, output_dir, modality="Track"):
    _plot_grid_heatmap(results, output_dir, value_key="mae", cbar_label="MAE",
                       suptitle=f"COCOA {modality} Reconstruction MAE",
                       out_name="cocoa_scan_heatmap.pdf", shared_scale=True)


def plot_resolution_heatmap(results, output_dir, modality="Track", vmin=None, vmax=None):
    if not any("jet_res" in r for r in results):
        log.info("No jet_res available; skipping resolution heatmap.")
        return
    _plot_grid_heatmap(results, output_dir, value_key="jet_res",
                       cbar_label=r"IQR / median",
                       suptitle=rf"COCOA {modality} Jet $p_T$ Resolution",
                       out_name="cocoa_scan_resolution_heatmap.pdf",
                       shared_scale=True, vmin=vmin, vmax=vmax)


def plot_mae_per_feature(results, output_dir, modality="Track"):
    """Per-feature MAE vs codebook dim, isolating which features cd helps/hurts.

    One subplot per feature (own y-scale). x = codebook dim, color = codebook size,
    at the largest num_quantizers. Titles of jet-pT-relevant features are highlighted.
    """
    plt.style.use(hep.style.CMS)
    cbs = sorted(set(r["cb"] for r in results))
    cds = sorted(set(r["cd"] for r in results))
    nq = max(set(r["nq"] for r in results))

    labels = FEATURE_LABELS.get(modality)
    jet_feats = JET_PT_FEATURES.get(modality, set())
    n_feat = len(next(iter(results))["mae_per_feat"])
    if labels is None or len(labels) != n_feat:
        labels = [f"feature {i}" for i in range(n_feat)]

    palette = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd", "#8c564b"]
    cb_colors = dict(zip(cbs, palette))

    ncol = 4 if n_feat > 5 else n_feat
    nrow = int(np.ceil(n_feat / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.4 * nrow), squeeze=False)
    axes = axes.flatten()

    for f in range(n_feat):
        ax = axes[f]
        for cb in cbs:
            xs, ys = [], []
            for cd in cds:
                vals = [r["mae_per_feat"][f] for r in results
                        if r["cb"] == cb and r["cd"] == cd and r["nq"] == nq]
                if vals:
                    xs.append(cd)
                    ys.append(np.mean(vals))
            if xs:
                ax.plot(xs, ys, marker="o", markersize=5, linewidth=1.5,
                        color=cb_colors[cb], label=f"cb = {cb}")
        ax.set_xticks(cds)
        ax.set_xlabel("Codebook dim", fontsize=11)
        ax.set_ylabel("MAE (physical)", fontsize=10)
        in_jet = f in jet_feats
        ax.set_title(labels[f] + ("  (jet $p_T$)" if in_jet else ""),
                     fontsize=12, color="#2ca02c" if in_jet else "#555555",
                     fontweight="bold" if in_jet else "normal")
        ax.grid(True, alpha=0.3)

    for f in range(n_feat, len(axes)):
        axes[f].axis("off")

    handles = [plt.Line2D([], [], color=cb_colors[cb], marker="o", linewidth=1.5,
                          label=f"cb = {cb}") for cb in cbs]
    fig.legend(handles=handles, loc="upper center", ncol=len(cbs),
               fontsize=11, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"COCOA {modality} Per-Feature Reconstruction MAE (nq = {nq})",
                 fontsize=14, y=1.04)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = output_dir / "cocoa_mae_per_feature.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {out}")


def plot_mae_vs_cb(results, output_dir):
    plt.style.use(hep.style.CMS)
    cbs = sorted(set(r["cb"] for r in results))
    cds = sorted(set(r["cd"] for r in results))
    nqs = sorted(set(r["nq"] for r in results))

    cd_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    colors = {cd: cd_palette[i % len(cd_palette)] for i, cd in enumerate(cds)}
    nq_markers = ["o", "s", "^", "D", "v", "P"]
    markers = {nq: nq_markers[i % len(nq_markers)] for i, nq in enumerate(nqs)}

    fig, ax = plt.subplots(figsize=(8, 5))
    for cd in cds:
        for nq in nqs:
            vals = [(r["cb"], r["mae"]) for r in results if r["cd"] == cd and r["nq"] == nq]
            if not vals:
                continue
            vals.sort()
            xs, ys = zip(*vals)
            ax.plot(xs, ys, marker=markers[nq], color=colors[cd], linewidth=1.5,
                    label=f"cd={cd}, nq={nq}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Codebook size", fontsize=12)
    ax.set_ylabel("MAE (physical space)", fontsize=12)
    ax.set_title("Reconstruction MAE vs Codebook Size", fontsize=13)
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = output_dir / "cocoa_mae_vs_cb_size.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser(description="Compare COCOA track scan results.")
    parser.add_argument("--scan_dir", type=str, default="results/cocoa_scan/tracks")
    parser.add_argument("--output_dir", type=str, default="results/cocoa_scan/plots")
    parser.add_argument("--res_vmin", type=float, default=None,
                        help="Fixed lower bound for the jet pT resolution heatmap's color scale "
                             "(overrides auto shared_scale). Use with --res_vmax to make two "
                             "separate scan_dir runs directly comparable.")
    parser.add_argument("--res_vmax", type=float, default=None,
                        help="Fixed upper bound for the jet pT resolution heatmap's color scale.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    scan_dir = Path(args.scan_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_scan_results(scan_dir)
    log.info(f"Loaded {len(results)} models")

    log.info("Ranking (best to worst MAE):")
    for i, r in enumerate(results):
        log.info(f"  {i+1:>2}. cb={r['cb']:>4d} cd={r['cd']} nq={r['nq']} MAE={r['mae']:.5f}")

    modality = "Topo" if "topo" in scan_dir.name.lower() else "Track"
    plot_resolution_vs_pt(results, output_dir, modality=modality)
    plot_heatmap(results, output_dir, modality=modality)
    plot_resolution_heatmap(results, output_dir, modality=modality, vmin=args.res_vmin, vmax=args.res_vmax)
    plot_mae_vs_cb(results, output_dir)
    plot_mae_per_feature(results, output_dir, modality=modality)

    log.info("Done!")


if __name__ == "__main__":
    main()
