"""Compare large-transformer VQ-VAE scan: codebook size, codebook dim, num quantizers.

Produces three sets of plots, one per scan dimension (cb, cd, nq).
Each plot shows IQR/median of pT response vs truth pT, with lines for each
value of the scanned parameter, grouped by the other two parameters.

Also produces a summary bar chart ranking all 12 configurations by
overall median IQR/median.

Usage
-----
pixi run python scripts/compare_scan.py
pixi run python scripts/compare_scan.py --output_dir results/scan_comparison
"""
from __future__ import annotations

import argparse
import logging
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

log = logging.getLogger(__name__)

SCAN_ROOT = Path("results/vqvae_scan/vqvae_scan")

# Grid definition
CODEBOOK_SIZES = [1024, 8192, 16384, 32768]
# cd=2 excluded: it is clearly worse (capacity-limited / undertrained points).
CODEBOOK_DIMS = [4, 8, 16]
NUM_QUANTIZERS = [3, 4]

TPL = "transformer_vqvae_enc_large_cb{cb}_cd{cd}_nq{nq}_lr0.0003_cw1.0_nj10000000"

# IQR binning
PT_MIN, PT_MAX, N_BINS = 5_000.0, 80_000.0, 15


def load_metrics(cb, cd, nq):
    name = TPL.format(cb=cb, cd=cd, nq=nq)
    path = SCAN_ROOT / name / "test" / "test_metrics" / "test_metrics.npz"
    if not path.exists():
        # Fallback: metrics directly under run dir
        path = SCAN_ROOT / name / "test_metrics" / "test_metrics.npz"
    if not path.exists():
        # Last resort: search anywhere under the run dir (handles doubly-nested
        # default_root_dir paths from manual one-off submissions). Pick the newest.
        hits = sorted((SCAN_ROOT / name).rglob("test_metrics.npz"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        if hits:
            path = hits[0]
    if not path.exists():
        return None
    return dict(np.load(path))


def compute_iqr_vs_pt(metrics):
    """Return (bin_centers, iqr_med) arrays."""
    bin_edges = np.linspace(PT_MIN, PT_MAX, N_BINS + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    truth_pt = metrics["truth_pt"].ravel()
    pt_ratio = metrics["pt_ratio"].ravel()
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


def compute_overall_iqr(metrics):
    """Single scalar IQR/median for the full pT range."""
    truth_pt = metrics["truth_pt"].ravel()
    pt_ratio = metrics["pt_ratio"].ravel()
    valid = (
        np.isfinite(truth_pt) & np.isfinite(pt_ratio) & (pt_ratio > 0)
        & (truth_pt >= PT_MIN) & (truth_pt <= PT_MAX)
    )
    pt_ratio = pt_ratio[valid]
    if pt_ratio.size < 10:
        return np.nan
    median = np.median(pt_ratio)
    q25, q75 = np.percentile(pt_ratio, [25, 75])
    return (q75 - q25) / median


# ── Scan-dimension plot helpers ───────────────────────────────────────────────


def plot_scan_iqr(
    scan_name: str,
    scan_values: list,
    fixed_combos: list[dict],
    output_dir: Path,
):
    """IQR vs pT plot scanning one dimension, with a subplot per fixed combo."""
    plt.style.use(hep.style.CMS)
    prop_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [f"C{i}" for i in range(10)])

    n_combos = len(fixed_combos)
    n_cols = min(3, n_combos)
    n_rows = int(np.ceil(n_combos / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5.5 * n_rows),
                             squeeze=False)
    axes = axes.flatten()

    for panel_idx, combo in enumerate(fixed_combos):
        ax = axes[panel_idx]
        # Build panel title from fixed params
        fixed_label = ", ".join(f"{k}={v}" for k, v in combo.items() if k != scan_name)

        for i, sv in enumerate(scan_values):
            params = {**combo, scan_name: sv}
            m = load_metrics(**params)
            if m is None:
                continue
            bc, iqr = compute_iqr_vs_pt(m)
            ok = np.isfinite(iqr)
            if np.any(ok):
                ax.plot(bc[ok], iqr[ok], marker="o", linewidth=2,
                        color=prop_colors[i % len(prop_colors)],
                        label=f"{scan_name}={sv}")

        ax.set_xlabel(r"Truth jet $p_T$ [MeV]", fontsize=13)
        ax.set_ylabel(r"IQR / median of $p_T^{\rm reco}/p_T^{\rm truth}$", fontsize=13)
        ax.set_xlim(PT_MIN, PT_MAX)
        ax.set_title(fixed_label, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    for i in range(n_combos, len(axes)):
        axes[i].axis("off")

    fig.suptitle(f"Scan: {scan_name}  (large transformer, lr=3e-4)", fontsize=15, y=1.01)
    fig.tight_layout()
    out = output_dir / f"scan_{scan_name}_iqr_vs_pt.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {out}")


def plot_scan_residuals(
    scan_name: str,
    scan_values: list,
    fixed_combos: list[dict],
    output_dir: Path,
):
    """Jet pT relative residual histograms scanning one dimension."""
    plt.style.use(hep.style.CMS)
    prop_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [f"C{i}" for i in range(10)])

    n_combos = len(fixed_combos)
    n_cols = min(3, n_combos)
    n_rows = int(np.ceil(n_combos / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5.5 * n_rows),
                             squeeze=False)
    axes = axes.flatten()

    for panel_idx, combo in enumerate(fixed_combos):
        ax = axes[panel_idx]
        fixed_label = ", ".join(f"{k}={v}" for k, v in combo.items() if k != scan_name)

        all_vals = []
        entries = []
        for i, sv in enumerate(scan_values):
            params = {**combo, scan_name: sv}
            m = load_metrics(**params)
            if m is None:
                continue
            v = m["pt_rel"]
            v = v[np.isfinite(v)]
            all_vals.append(v)
            entries.append((sv, v, prop_colors[i % len(prop_colors)]))

        if not all_vals:
            continue

        combined = np.concatenate(all_vals)
        lo, hi = np.percentile(combined, [1, 99])
        bins = np.linspace(lo, hi, 51)

        for sv, v, color in entries:
            ax.hist(v, bins=bins, histtype="step", linewidth=2,
                    label=f"{scan_name}={sv}", density=True, color=color)

        ax.set_xlabel(r"Jet $p_T$ relative residual", fontsize=13)
        ax.set_ylabel("Density", fontsize=13)
        ax.set_title(fixed_label, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    for i in range(n_combos, len(axes)):
        axes[i].axis("off")

    fig.suptitle(f"Scan: {scan_name}  (large transformer, lr=3e-4)", fontsize=15, y=1.01)
    fig.tight_layout()
    out = output_dir / f"scan_{scan_name}_pt_residual.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {out}")


def plot_summary_bar(output_dir: Path):
    """Summary bar chart of overall IQR/median for all 12 configurations, sorted."""
    plt.style.use(hep.style.CMS)

    results = []
    for cb, cd, nq in product(CODEBOOK_SIZES, CODEBOOK_DIMS, NUM_QUANTIZERS):
        m = load_metrics(cb, cd, nq)
        if m is None:
            continue
        iqr = compute_overall_iqr(m)
        results.append({"cb": cb, "cd": cd, "nq": nq, "iqr": iqr})

    if not results:
        log.error("No metrics found for summary")
        return

    # Sort by IQR (best = lowest first)
    results.sort(key=lambda r: r["iqr"])

    labels = [f"cb={r['cb']} cd={r['cd']} nq={r['nq']}" for r in results]
    iqr_vals = [r["iqr"] for r in results]

    # Color-code by codebook size (palette-driven so any cb set works)
    palette = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd", "#8c564b"]
    cbs_sorted = sorted({r["cb"] for r in results})
    cb_colors = {cb: palette[i % len(palette)] for i, cb in enumerate(cbs_sorted)}
    bar_colors = [cb_colors[r["cb"]] for r in results]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(results))
    ax.bar(x, iqr_vals, color=bar_colors, edgecolor="black", linewidth=0.8)

    # Custom legend (codebook size only)
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=cb_colors[cb], edgecolor="black", label=f"cb={cb}")
                       for cb in cbs_sorted]
    ax.legend(handles=legend_elements, fontsize=10, loc="upper left")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=45, ha="right")
    ax.set_ylabel(r"IQR / median of $p_T^{\rm reco}/p_T^{\rm truth}$", fontsize=13)
    ax.set_title("Large Transformer VQ-VAE  (lr=3e-4)  — sorted by IQR", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    out = output_dir / "scan_summary_bar.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {out}")

    # Print ranking
    log.info("=== Performance ranking (best → worst) ===")
    for i, r in enumerate(results):
        log.info(f"  {i+1}. cb={r['cb']:>5d}  cd={r['cd']}  nq={r['nq']}  IQR/med={r['iqr']:.5f}")


# ── COCOA-style summary plots (heatmaps + faceted resolution) ─────────────────


def build_results():
    """Load the grid into per-config dicts: cb, cd, nq, mae, jet_res, truth_pt, pt_ratio."""
    results = []
    for cb, cd, nq in product(CODEBOOK_SIZES, CODEBOOK_DIMS, NUM_QUANTIZERS):
        m = load_metrics(cb, cd, nq)
        if m is None:
            continue
        entry = {"cb": cb, "cd": cd, "nq": nq}
        if "cst_original" in m and "cst_recon" in m:
            entry["mae"] = float(np.mean(np.abs(m["cst_original"] - m["cst_recon"])))
        entry["jet_res"] = compute_overall_iqr(m)
        if "truth_pt" in m and "pt_ratio" in m:
            entry["truth_pt"] = m["truth_pt"].ravel()
            entry["pt_ratio"] = m["pt_ratio"].ravel()
        results.append(entry)
    return results


def plot_resolution_vs_pt(results, output_dir):
    """Jet pT resolution vs truth pT, faceted by (cd rows, nq cols), colored by cb, log y."""
    plt.style.use(hep.style.CMS)
    cbs = sorted(set(r["cb"] for r in results))
    cds = sorted(set(r["cd"] for r in results))
    nqs = sorted(set(r["nq"] for r in results))
    palette = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd", "#8c564b"]
    cb_colors = dict(zip(cbs, palette))

    fig, axes = plt.subplots(len(cds), len(nqs), figsize=(5.0 * len(nqs), 3.6 * len(cds)),
                             sharex=True, sharey=True, squeeze=False)
    for i, cd in enumerate(cds):
        for j, nq in enumerate(nqs):
            ax = axes[i][j]
            for r in results:
                if r["cd"] != cd or r["nq"] != nq or "truth_pt" not in r:
                    continue
                bc, iqr = compute_iqr_vs_pt({"truth_pt": r["truth_pt"], "pt_ratio": r["pt_ratio"]})
                ok = np.isfinite(iqr)
                if ok.any():
                    ax.plot(bc[ok], iqr[ok], marker="o", markersize=4, linewidth=1.5,
                            color=cb_colors[r["cb"]], label=f"cb={r['cb']}")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3, which="both")
            ax.set_xlim(PT_MIN, PT_MAX)
            ax.set_title(f"nq = {nq}, cd = {cd}", fontsize=12)
            if j == 0:
                ax.set_ylabel(r"IQR / median" + "\n" + r"$p_T^{\rm reco}/p_T^{\rm truth}$", fontsize=11)
            if i == len(cds) - 1:
                ax.set_xlabel(r"Truth jet $p_T$ [MeV]", fontsize=12)

    handles = [plt.Line2D([], [], color=cb_colors[cb], marker="o", linewidth=1.5,
                          label=f"cb = {cb}") for cb in cbs]
    fig.legend(handles=handles, loc="upper center", ncol=len(cbs), fontsize=11,
               frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(r"JetSet Tokenization: Jet $p_T$ Resolution vs Truth $p_T$", fontsize=14, y=1.04)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = output_dir / "jetset_jet_pt_resolution_vs_pt.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {out}")


def _plot_grid_heatmap(results, output_dir, value_key, cbar_label, suptitle, out_name, fmt="{:.4f}"):
    """cb x cd heatmap of `value_key`, one panel per nq, shared color scale. Lower = greener."""
    plt.style.use(hep.style.CMS)
    cbs = sorted(set(r["cb"] for r in results))
    cds = sorted(set(r["cd"] for r in results))
    nqs = sorted(set(r["nq"] for r in results))
    all_vals = [r[value_key] for r in results if value_key in r and np.isfinite(r[value_key])]
    vmin, vmax = (float(np.min(all_vals)), float(np.max(all_vals))) if all_vals else (None, None)

    from matplotlib.colors import Normalize
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=vmin, vmax=vmax) if vmin is not None else None

    fig, axes = plt.subplots(1, len(nqs), figsize=(6 * len(nqs), 5), squeeze=False)
    axes = axes.flatten()
    for ax_idx, nq in enumerate(nqs):
        ax = axes[ax_idx]
        grid = np.full((len(cds), len(cbs)), np.nan)
        for r in results:
            if r["nq"] != nq or value_key not in r:
                continue
            grid[cds.index(r["cd"]), cbs.index(r["cb"])] = r[value_key]
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
                    # Luminance-adaptive label colour: white on dark viridis, black on light.
                    lum = 1.0
                    if norm is not None:
                        rr, gg, bb, _ = cmap(norm(grid[i, j]))
                        lum = 0.299 * rr + 0.587 * gg + 0.114 * bb
                    ax.text(j, i, fmt.format(grid[i, j]), ha="center", va="center",
                            fontsize=9, color="white" if lum < 0.5 else "black")
        plt.colorbar(im, ax=ax, shrink=0.8, label=cbar_label)
    fig.tight_layout()
    out = output_dir / out_name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser(description="Compare large-transformer scan results.")
    parser.add_argument("--output_dir", type=str, default="results/scan_comparison")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── COCOA-style summary plots ─────────────────────────────────────────
    results = build_results()
    if results:
        plot_resolution_vs_pt(results, output_dir)
        _plot_grid_heatmap(results, output_dir, "mae", "MAE",
                           "JetSet Reconstruction MAE", "jetset_mae_heatmap.pdf")
        _plot_grid_heatmap(results, output_dir, "jet_res", r"IQR / median",
                           r"JetSet Jet $p_T$ Resolution", "jetset_resolution_heatmap.pdf")

    # ── Scan: codebook_size ───────────────────────────────────────────────
    combos_cb = [{"cb": None, "cd": cd, "nq": nq}
                 for cd, nq in product(CODEBOOK_DIMS, NUM_QUANTIZERS)]
    plot_scan_iqr("cb", CODEBOOK_SIZES, combos_cb, output_dir)
    plot_scan_residuals("cb", CODEBOOK_SIZES, combos_cb, output_dir)

    # ── Scan: codebook_dim ────────────────────────────────────────────────
    combos_cd = [{"cb": cb, "cd": None, "nq": nq}
                 for cb, nq in product(CODEBOOK_SIZES, NUM_QUANTIZERS)]
    plot_scan_iqr("cd", CODEBOOK_DIMS, combos_cd, output_dir)
    plot_scan_residuals("cd", CODEBOOK_DIMS, combos_cd, output_dir)

    # ── Scan: num_quantizers ──────────────────────────────────────────────
    combos_nq = [{"cb": cb, "cd": cd, "nq": None}
                 for cb, cd in product(CODEBOOK_SIZES, CODEBOOK_DIMS)]
    plot_scan_iqr("nq", NUM_QUANTIZERS, combos_nq, output_dir)
    plot_scan_residuals("nq", NUM_QUANTIZERS, combos_nq, output_dir)

    # ── Summary ───────────────────────────────────────────────────────────
    plot_summary_bar(output_dir)


if __name__ == "__main__":
    main()
