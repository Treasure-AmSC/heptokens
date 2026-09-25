#!/usr/bin/env python
"""Design-rules figure (four panels), d-crossing appendix plots, and Fig. 1 (rate-distortion Pareto).

All values are read from the test_metrics.npz files of repo A at run time, so new runs are picked up
automatically; missing runs are simply absent from the plot. Metric: IQR/median of per-jet pt_ratio.

  (a) resolution vs Q at the selected (K, d)
  (b) resolution vs log2 K at the selected (d, Q)
  (c) d of the Pareto-optimal configuration vs bits per element (Q log2 K)
  (d) positions in (solid) vs positions out (dashed, hollow) vs total bits per element,
      best d at each K, Q = 4

Where several seeds exist for one configuration the mean is plotted with a min-max error bar.
JetSet d = 2 is excluded from the main figures and shown (marked) in the appendix d-crossing plot.

usage: python scripts/plot_design_rules.py [--outdir paper/figures]
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mplhep as hep
import numpy as np

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "results"

MODALITIES = {
    "JetSet": dict(
        color="#1f77b4", D=20, pos_bits=20, selected=(32768, 8, 4), slug="jetset",
        posout_grid=([8192, 16384, 32768], [4, 8, 16]),
        dirs=[RES / "vqvae_scan/vqvae_scan"],
        rx=r"^transformer_vqvae_enc_large_cb(\d+)_cd(\d+)_nq(\d+)_lr0\.0003_cw1\.0_nj10000000"
           r"(?:_(posout)|_seed(\d+))?$",
    ),
    "COCOA tracks": dict(
        color="#ff7f0e", D=5, pos_bits=30, selected=(4096, 4, 4), slug="tracks",
        posout_grid=([256, 1024, 4096], [2, 4, 8]),
        dirs=[RES / "cocoa_scan/tracks", RES / "cocoa_scan/tracks_posout"],
        rx=r"^cocoa_tracks(?:_(posout))?_cb(\d+)_cd(\d+)_nq(\d+)_lr0\.001(?:_seed(\d+))?$",
    ),
    "COCOA topoclusters": dict(
        color="#2ca02c", D=8, pos_bits=30, selected=(4096, 8, 4), slug="topos",
        posout_grid=([256, 1024, 4096], [4, 8, 12]),
        dirs=[RES / "cocoa_scan/topos", RES / "cocoa_scan/topos_posout", RES / "cocoa_scan/topos_seed"],
        rx=r"^cocoa_topos(?:_(posout))?_cb(\d+)_cd(\d+)_nq(\d+)_lr0\.001(?:_seed(\d+))?$",
    ),
}
Q_VALUES = [2, 3, 4, 6, 8]
EVAL_LIMIT = 1e-3  # Fig. 1: below this the tracks result depends on evaluation numerics


def iqr_over_median(path):
    pr = np.load(path)["pt_ratio"]
    pr = pr[np.isfinite(pr) & (pr > 0)]
    q25, q50, q75 = np.percentile(pr, [25, 50, 75])
    return (q75 - q25) / q50


def parse(name, spec):
    m = re.match(spec["rx"], name)
    if m is None:
        return None
    g = m.groups()
    if spec["slug"] == "jetset":
        K, d, Q, posout, seed = g
    else:
        posout, K, d, Q, seed = g
    return int(K), int(d), int(Q), ("out" if posout else "in"), int(seed) if seed else 42


def load(spec):
    """{(K, d, Q, 'in'|'out'): [values over seeds]}"""
    runs = {}
    for base in spec["dirs"]:
        for run in sorted(base.glob("*")):
            key = parse(run.name, spec)
            if key is None:
                continue
            files = sorted(set(glob.glob(str(run / "test/**/test_metrics.npz"), recursive=True)))
            if len(files) > 1:
                raise RuntimeError(f"several test_metrics.npz under {run}: {files}")
            if files:
                runs.setdefault(key[:4], []).append(iqr_over_median(files[0]))
    return runs


def central(vals):
    v = np.asarray(vals)
    return v.mean(), (v.mean() - v.min(), v.max() - v.mean())


def plot_series(ax, xs, vals, color, hollow=False, ls="-", label=None, marker="o"):
    """Line through the per-configuration means; min-max error bars where several seeds exist."""
    if not xs:
        return
    order = np.argsort(xs)
    xs = np.asarray(xs)[order]
    cv = [central(vals[i]) for i in order]
    y = np.array([c[0] for c in cv])
    err = np.array([c[1] for c in cv]).T
    ax.errorbar(xs, y, yerr=err if err.any() else None, fmt=marker, ls=ls, color=color, ms=6,
                mfc="white" if hollow else color, mec=color, lw=1.6, capsize=3, label=label, zorder=3)


def pareto(points):
    """points: [(bits, y, cfg)], lower y and fewer bits better. Best per bits, then running min."""
    best = {}
    for b, y, cfg in points:
        if b not in best or y < best[b][0]:
            best[b] = (y, cfg)
    out, run_min = [], np.inf
    for b in sorted(best):
        y, cfg = best[b]
        if y < run_min:
            out.append((b, y, cfg))
            run_min = y
    return out


def main_figure(data, outdir):
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.6))
    (ax_a, ax_b), (ax_c, ax_d) = axes
    c_offsets = {"JetSet": 1.05, "COCOA tracks": 1.0, "COCOA topoclusters": 0.95}  # separate overlapping steps

    for name, spec in MODALITIES.items():
        runs, col = data[name], spec["color"]
        K0, d0, Q0 = spec["selected"]
        inn = {k[:3]: v for k, v in runs.items() if k[3] == "in" and not (spec["slug"] == "jetset" and k[1] == 2)}

        # (a) resolution vs Q
        qs = [Q for Q in Q_VALUES if (K0, d0, Q) in inn]
        plot_series(ax_a, qs, [inn[(K0, d0, Q)] for Q in qs], col, label=f"{name} ({K0}, {d0})")

        # (b) resolution vs log2 K
        ks = sorted(K for (K, d, Q) in inn if d == d0 and Q == Q0)
        plot_series(ax_b, [np.log2(K) for K in ks], [inn[(K, d0, Q0)] for K in ks], col,
                    label=f"{name} (d={d0}, Q={Q0})")

        # (c) d of the Pareto-optimal configuration vs bits
        front = pareto([(Q * np.log2(K), central(v)[0], (K, d, Q)) for (K, d, Q), v in inn.items()])
        if front:
            b = [p[0] for p in front]
            dstar = [p[2][1] * c_offsets[name] for p in front]
            ax_c.step(b, dstar, where="mid", color=col, lw=1.6)
            ax_c.plot(b, dstar, "o", color=col, ms=4)

        # (d) positions in vs out, Q=4, best d at each K over the positions-out d grid;
        # a point is drawn only once every d of the grid exists at that K
        out = {k[:3]: v for k, v in runs.items() if k[3] == "out"}
        K_grid, d_grid = spec["posout_grid"]
        for table, hollow, ls, extra in ((inn, False, "-", 0), (out, True, "--", spec["pos_bits"])):
            xs, vals = [], []
            for K in K_grid:
                if all((K, d, 4) in table for d in d_grid):
                    xs.append(4 * np.log2(K) + extra)
                    vals.append(min((table[(K, d, 4)] for d in d_grid), key=lambda v: central(v)[0]))
            plot_series(ax_d, xs, vals, col, hollow=hollow, ls=ls)

    for ax in (ax_a, ax_b, ax_d):
        ax.set_yscale("log")
        ax.set_ylabel(r"Jet $p_T$ IQR / median")
        ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs="auto"))
    ax_a.set_xlabel(r"Number of quantizers $Q$")
    ax_a.set_xticks(Q_VALUES)
    ax_a.set_title("(a) Resolution vs $Q$", loc="left", fontsize=12)
    ax_a.legend(fontsize=8.5, loc="upper right", frameon=False)
    ax_b.set_xlabel(r"$\log_2 K$")
    ax_b.set_title("(b) Resolution vs codebook size", loc="left", fontsize=12)
    ax_c.set_yscale("log", base=2)
    ax_c.yaxis.set_major_locator(mticker.FixedLocator([2, 4, 8, 16]))
    ax_c.yaxis.set_major_formatter(mticker.FixedFormatter(["2", "4", "8", "16"]))
    ax_c.yaxis.set_minor_locator(mticker.NullLocator())
    ax_c.set_ylim(1.6, 20)
    ax_c.set_xlabel(r"Bits per element $Q\,\log_2 K$")
    ax_c.set_ylabel(r"Pareto-optimal $d^\ast$")
    ax_c.set_title(r"(c) Optimal codebook dimension", loc="left", fontsize=12)
    ax_d.set_xlabel("Total bits per element")
    ax_d.set_title("(d) Positions in vs out ($Q=4$)", loc="left", fontsize=12)
    ax_d.legend(handles=[plt.Line2D([], [], color="0.3", marker="o", ls="-", label="positions in"),
                         plt.Line2D([], [], color="0.3", marker="o", mfc="white", ls="--", label="positions out")],
                fontsize=8.5, loc="upper right", frameon=False)
    for ax in axes.flat:
        ax.tick_params(axis="both", which="both", direction="in", top=True, right=True, labelsize=10)
        ax.xaxis.label.set_size(11)
        ax.yaxis.label.set_size(11)
    fig.tight_layout()
    save(fig, outdir, "design_rules")


def d_crossing(data, outdir):
    for name, spec in MODALITIES.items():
        inn = {k[:3]: v for k, v in data[name].items() if k[3] == "in"}
        Ks = sorted({K for K, _, _ in inn})
        Qs = sorted({Q for _, _, Q in inn})
        cmap = plt.get_cmap("viridis")
        fig, ax = plt.subplots(figsize=(6.4, 5.0))
        for i, K in enumerate(Ks):
            for Q, ls in zip(Qs, ["-", "--", ":", "-.", (0, (5, 1)), (0, (1, 3))]):
                ds = sorted(d for (k, d, q) in inn if k == K and q == Q)
                if not ds:
                    continue
                c = cmap(i / max(len(Ks) - 1, 1))
                y = [central(inn[(K, d, Q)])[0] for d in ds]
                ax.plot(ds, y, ls=ls, color=c, marker="o", ms=5, label=f"K={K}, Q={Q}")
                if spec["slug"] == "jetset" and 2 in ds:
                    ax.plot(2, y[0], "o", ms=11, mfc="none", mec="red", mew=1.4, zorder=4)
        if spec["slug"] == "jetset" and any(d == 2 for _, d, _ in inn):
            ax.plot([], [], "o", ms=9, mfc="none", mec="red", mew=1.4, label="d = 2 (appendix only)")
        ax.set_xscale("log", base=2)
        ds_all = sorted({d for _, d, _ in inn})
        ax.xaxis.set_major_locator(mticker.FixedLocator(ds_all))
        ax.xaxis.set_major_formatter(mticker.FixedFormatter([str(d) for d in ds_all]))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.set_yscale("log")
        ax.set_xlabel("Codebook dimension $d$")
        ax.set_ylabel(r"Jet $p_T$ IQR / median")
        ax.set_title(name, loc="left", fontsize=13)
        ax.legend(fontsize=7.5, ncol=2, frameon=False)
        ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
        fig.tight_layout()
        save(fig, outdir, f"design_rules_d_crossing_{spec['slug']}")


def fig1(data, outdir):
    """Rate-distortion Pareto (style of heptokens-cocoa scripts/plot_rate_distortion.py) from repo A,
    with the evaluation-limited region shaded and the tracks points inside it faded."""
    marker_by_d = {2: "v", 4: "o", 8: "s", 12: "^", 16: "^"}
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ylim = (5e-4, 0.16)
    ax.axhspan(ylim[0], EVAL_LIMIT, color="0.85", zorder=0, lw=0)
    ax.text(0.03, np.sqrt(ylim[0] * EVAL_LIMIT), "evaluation-limited", transform=ax.get_yaxis_transform(),
            va="center", ha="left", fontsize=13, color="0.35")
    for name, spec in MODALITIES.items():
        col = spec["color"]
        inn = {k[:3]: central(v)[0] for k, v in data[name].items() if k[3] == "in"}
        # Fig. 1 keeps its existing grid: JetSet without d = 2, topoclusters Q = 4 only
        if spec["slug"] == "jetset":
            inn = {k: v for k, v in inn.items() if k[1] != 2}
        if spec["slug"] == "topos":
            inn = {k: v for k, v in inn.items() if k[2] == 4}
        fade = spec["slug"] == "tracks"
        pts = [(Q * np.log2(K) / (32 * spec["D"]), y, (K, d, Q)) for (K, d, Q), y in inn.items()]
        for x, y, (K, d, Q) in pts:
            low = fade and y < EVAL_LIMIT
            ax.scatter(x, y, marker=marker_by_d[d], s=44, facecolor=col, edgecolor="none",
                       alpha=0.12 if low else 0.35, zorder=2)
        front = pareto(pts)  # fewer bits (smaller information fraction) and lower y are better
        ax.plot([p[0] for p in front], [p[1] for p in front], "-", color=col, lw=2.4, zorder=3, label=name)
        if spec["selected"] in inn:
            K, d, Q = spec["selected"]
            y = inn[spec["selected"]]
            low = fade and y < EVAL_LIMIT
            ax.plot(Q * np.log2(K) / (32 * spec["D"]), y, marker="*", ms=22, mfc=col, mec="black", mew=0.7,
                    ls="none", zorder=5, alpha=0.4 if low else 1.0)
    ax.set_yscale("log")
    ax.set_xlim(0.025, 0.315)
    ax.set_ylim(*ylim)
    ax.axhline(0.01, color="0.3", ls=":", lw=1.3, zorder=1)
    ax.text(0.62, 0.01, "1% resolution goal", color="0.3", fontsize=13, va="bottom", ha="left",
            transform=ax.get_yaxis_transform())
    star = plt.Line2D([], [], marker="*", ms=13 * 15 / 10.5, mfc="0.5", mec="black", mew=0.7, ls="none",
                      label="optimal config")
    h, lab = ax.get_legend_handles_labels()
    leg = ax.legend(h + [star], lab + ["optimal config"], loc="upper right", frameon=True, framealpha=0.9,
                    edgecolor="none", fontsize=15)
    ax.add_artist(leg)
    shapes = [plt.Line2D([], [], marker=m, ms=10, mfc="0.4", mec="none", ls="none", label=l)
              for m, l in (("v", "d = 2"), ("o", "d = 4"), ("s", "d = 8"), ("^", "d = 12/16"))]
    ax.legend(handles=shapes, loc="lower left", bbox_to_anchor=(0.0, 0.13), frameon=True, framealpha=0.9,
              edgecolor="none", fontsize=15, title="codebook dim.", title_fontsize=15)
    ax.set_ylabel(r"Jet $p_T$ IQR / median", fontsize=20)
    ax.set_xlabel(r"Information fraction  ($Q\,\log_2 K \,/\, 32\,D$)", fontsize=20)
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True, labelsize=16)
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs="auto"))
    fig.tight_layout()
    save(fig, outdir, "rate_distortion_pareto")


def save(fig, outdir, stem):
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    print(f"wrote {outdir / stem}.pdf")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=REPO / "paper/figures")
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    plt.style.use(hep.style.CMS)
    plt.rcParams.update({"font.size": 11, "legend.handlelength": 2.8})
    data = {name: load(spec) for name, spec in MODALITIES.items()}
    for name, runs in data.items():
        n_in = sum(k[3] == "in" for k in runs)
        n_out = sum(k[3] == "out" for k in runs)
        seeds = {k: len(v) for k, v in runs.items() if len(v) > 1}
        print(f"{name}: {n_in} positions-in, {n_out} positions-out configurations; multi-seed: {seeds}")
    main_figure(data, args.outdir)
    d_crossing(data, args.outdir)
    fig1(data, args.outdir)


if __name__ == "__main__":
    main()
