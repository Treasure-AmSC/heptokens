"""Plot codebook utilization vs hyperparameters from scan CSV.

Reads the CSV produced by ``codebook_utilization.py --auto --output``.

Produces:
  1. Utilization vs codebook size, lines = (nq, cd) combos, per architecture.
  2. Utilization vs num_quantizers, lines = (cb, cd) combos.
  3. Per-quantizer utilization breakdown for each model.

Usage
-----
pixi run python scripts/plot_codebook_utilization.py
pixi run python scripts/plot_codebook_utilization.py --input results/codebook_utilization/utilization.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd

plt.style.use(hep.style.CMS)

INPUT_CSV = Path("results/codebook_utilization/utilization.csv")
OUTPUT_DIR = Path("results/codebook_utilization/plots")

MARKERS = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h"]
COLORS = plt.cm.tab10.colors


def parse_run_details(name: str) -> dict:
    """Extract all hyperparameters from a run directory name."""
    info = {}
    # Model type
    if name.startswith("transformer_vqvae"):
        info["model"] = "transformer"
    elif name.startswith("vqvae") or name.startswith("enc_"):
        info["model"] = "mlp"
    else:
        info["model"] = "unknown"

    # Encoder size
    m = re.search(r"enc_(small|medium|large)", name)
    info["enc"] = m.group(1) if m else "unknown"

    # Codebook params
    m = re.search(r"cb(\d+)", name)
    info["cb"] = int(m.group(1)) if m else 0

    m = re.search(r"cd(\d+)", name)
    info["cd"] = int(m.group(1)) if m else 0

    m = re.search(r"nq(\d+)", name)
    info["nq"] = int(m.group(1)) if m else 0

    m = re.search(r"lr([\d.e-]+)", name)
    info["lr"] = float(m.group(1)) if m else 0

    m = re.search(r"cw([\d.]+)", name)
    info["cw"] = float(m.group(1)) if m else 0

    m = re.search(r"nj(\d+)", name)
    info["nj"] = int(m.group(1)) if m else 0

    # Suffix
    info["suffix"] = "_onlyptetaphi" if "onlyptetaphi" in name else ""

    return info


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Parse run names for fields not already in the CSV (model, enc, cw, nj, suffix)
    details = df["run"].apply(parse_run_details).apply(pd.Series)
    # Only add columns that don't already exist
    for col in details.columns:
        if col not in df.columns:
            df[col] = details[col]
    return df


def plot_util_vs_codebook_size(df: pd.DataFrame, output_dir: Path):
    """Plot avg utilization vs codebook size, with lines for each (nq, cd, lr) combo.

    Separate panels for each (model, enc) architecture.
    """
    # Filter to standard runs (10M jets, cw=1.0, no suffix)
    mask = (df["nj"] == 10_000_000) & (df["cw"] == 1.0) & (df["suffix"] == "")
    sub = df[mask].copy()

    # Group by architecture
    for (model, enc), grp in sub.groupby(["model", "enc"]):
        # Need at least 2 different cb values
        if grp["cb"].nunique() < 2:
            continue

        fig, ax = plt.subplots(figsize=(10, 7))
        combos = sorted(grp.groupby(["nq", "cd", "lr"]).groups.keys())

        for i, (nq, cd, lr) in enumerate(combos):
            sel = grp[(grp["nq"] == nq) & (grp["cd"] == cd) & (grp["lr"] == lr)].sort_values("cb")
            if len(sel) < 2:
                continue
            label = f"nq={nq}, cd={cd}"
            if grp["lr"].nunique() > 1:
                label += f", lr={lr}"
            ax.plot(
                sel["cb"], sel["avg_utilization"] * 100,
                marker=MARKERS[i % len(MARKERS)],
                color=COLORS[i % len(COLORS)],
                label=label,
                linewidth=2, markersize=8,
            )

        ax.set_xscale("log", base=2)
        ax.set_xlabel("Codebook Size")
        ax.set_ylabel("Average Utilization [%]")
        ax.set_title(f"{model} encoder={enc}")
        ax.set_ylim(0, 105)
        ax.legend(fontsize=14)
        ax.grid(True, alpha=0.3)

        fname = output_dir / f"util_vs_cb_{model}_{enc}.pdf"
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {fname}")


def plot_util_vs_nq(df: pd.DataFrame, output_dir: Path):
    """Plot avg utilization vs num_quantizers, with lines for each (cb, cd) combo."""
    mask = (df["nj"] == 10_000_000) & (df["cw"] == 1.0) & (df["suffix"] == "")
    sub = df[mask].copy()

    for (model, enc), grp in sub.groupby(["model", "enc"]):
        if grp["nq"].nunique() < 2:
            continue

        fig, ax = plt.subplots(figsize=(10, 7))
        combos = sorted(grp.groupby(["cb", "cd"]).groups.keys())

        for i, (cb, cd) in enumerate(combos):
            sel = grp[(grp["cb"] == cb) & (grp["cd"] == cd)].sort_values("nq")
            if len(sel) < 2:
                continue
            ax.plot(
                sel["nq"], sel["avg_utilization"] * 100,
                marker=MARKERS[i % len(MARKERS)],
                color=COLORS[i % len(COLORS)],
                label=f"cb={cb}, cd={cd}",
                linewidth=2, markersize=8,
            )

        ax.set_xlabel("Number of Quantizers")
        ax.set_ylabel("Average Utilization [%]")
        ax.set_title(f"{model} encoder={enc}")
        ax.set_ylim(0, 105)
        ax.set_xticks(sorted(grp["nq"].unique()))
        ax.legend(fontsize=14)
        ax.grid(True, alpha=0.3)

        fname = output_dir / f"util_vs_nq_{model}_{enc}.pdf"
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {fname}")


def plot_util_vs_cd(df: pd.DataFrame, output_dir: Path):
    """Plot avg utilization vs codebook dim, with lines for each (cb, nq) combo."""
    mask = (df["nj"] == 10_000_000) & (df["cw"] == 1.0) & (df["suffix"] == "")
    sub = df[mask].copy()

    for (model, enc), grp in sub.groupby(["model", "enc"]):
        if grp["cd"].nunique() < 2:
            continue

        fig, ax = plt.subplots(figsize=(10, 7))
        combos = sorted(grp.groupby(["cb", "nq"]).groups.keys())

        for i, (cb, nq) in enumerate(combos):
            sel = grp[(grp["cb"] == cb) & (grp["nq"] == nq)].sort_values("cd")
            if len(sel) < 2:
                continue
            ax.plot(
                sel["cd"], sel["avg_utilization"] * 100,
                marker=MARKERS[i % len(MARKERS)],
                color=COLORS[i % len(COLORS)],
                label=f"cb={cb}, nq={nq}",
                linewidth=2, markersize=8,
            )

        ax.set_xscale("log", base=2)
        ax.set_xlabel("Codebook Dimension")
        ax.set_ylabel("Average Utilization [%]")
        ax.set_title(f"{model} encoder={enc}")
        ax.set_ylim(0, 105)
        ax.legend(fontsize=14)
        ax.grid(True, alpha=0.3)

        fname = output_dir / f"util_vs_cd_{model}_{enc}.pdf"
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {fname}")


def plot_per_quantizer_breakdown(df: pd.DataFrame, output_dir: Path):
    """Bar chart showing per-quantizer utilization for each run, grouped by architecture."""
    mask = (df["nj"] == 10_000_000) & (df["cw"] == 1.0) & (df["suffix"] == "")
    sub = df[mask].copy()

    for (model, enc), grp in sub.groupby(["model", "enc"]):
        # Sort by avg utilization
        grp = grp.sort_values("avg_utilization", ascending=True)

        # Find per-quantizer columns
        q_cols = sorted([c for c in grp.columns if c.startswith("util_q")])
        n_q_max = len(q_cols)
        if n_q_max == 0:
            continue

        n_runs = len(grp)
        fig, ax = plt.subplots(figsize=(max(12, n_runs * 0.6), 7))

        x = np.arange(n_runs)
        width = 0.8 / n_q_max

        for qi, col in enumerate(q_cols):
            vals = grp[col].fillna(0).values * 100
            ax.bar(
                x + qi * width - 0.4 + width / 2,
                vals,
                width=width,
                label=f"q{qi}",
                alpha=0.85,
            )

        # Labels
        labels = []
        for _, row in grp.iterrows():
            labels.append(f"cb={int(row['cb'])}\ncd={int(row['cd'])}\nnq={int(row['nq'])}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("Utilization [%]")
        ax.set_title(f"Per-Quantizer Utilization — {model} encoder={enc}")
        ax.set_ylim(0, 105)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, axis="y")

        fname = output_dir / f"per_quantizer_{model}_{enc}.pdf"
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {fname}")


def plot_summary_all_architectures(df: pd.DataFrame, output_dir: Path):
    """Comparison of utilization across architectures for matching configs."""
    mask = (df["nj"] == 10_000_000) & (df["cw"] == 1.0) & (df["suffix"] == "")
    sub = df[mask].copy()

    # Find configs that exist in multiple (model, enc) combos
    sub["config"] = sub.apply(
        lambda r: f"cb={int(r['cb'])}_cd={int(r['cd'])}_nq={int(r['nq'])}", axis=1
    )
    sub["arch"] = sub["model"] + "_" + sub["enc"]

    # Pivot: rows=config, cols=arch, values=avg_utilization
    pivot = sub.pivot_table(
        index="config", columns="arch", values="avg_utilization", aggfunc="first"
    )
    # Keep only configs present in 2+ architectures
    pivot = pivot.dropna(thresh=2).sort_index()

    if len(pivot) == 0:
        print("No matching configs across architectures for summary plot.")
        return

    fig, ax = plt.subplots(figsize=(max(12, len(pivot) * 0.8), 7))
    n_arch = len(pivot.columns)
    width = 0.8 / n_arch
    x = np.arange(len(pivot))

    for i, arch in enumerate(pivot.columns):
        vals = pivot[arch].fillna(0).values * 100
        ax.bar(x + i * width - 0.4 + width / 2, vals, width=width, label=arch, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, fontsize=10, rotation=45, ha="right")
    ax.set_ylabel("Average Utilization [%]")
    ax.set_title("Codebook Utilization: architecture comparison")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")

    fname = output_dir / "util_arch_comparison.pdf"
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")


def main():
    parser = argparse.ArgumentParser(description="Plot codebook utilization scan.")
    parser.add_argument("--input", type=str, default=str(INPUT_CSV))
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    csv_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(csv_path)
    print(f"Loaded {len(df)} runs from {csv_path}")
    print(f"  Models: {sorted(df['model'].unique())}")
    print(f"  Encoders: {sorted(df['enc'].unique())}")
    print(f"  CB sizes: {sorted(df['cb'].unique())}")
    print(f"  CB dims: {sorted(df['cd'].unique())}")
    print(f"  Num quantizers: {sorted(df['nq'].unique())}")

    plot_util_vs_codebook_size(df, output_dir)
    plot_util_vs_nq(df, output_dir)
    plot_util_vs_cd(df, output_dir)
    plot_per_quantizer_breakdown(df, output_dir)
    plot_summary_all_architectures(df, output_dir)

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
