#!/usr/bin/env python
"""Compare pflow performance for tokenized input sweeps vs the continuous-input baseline.

Reads each sweep's sweep_metrics.csv (from pflow_sweep_panel.py) and the continuous run's
eval/{metrics,token_metrics}.json, and plots resolution, median response and truthpart token
accuracy vs input information fraction. Continuous inputs are float32, i.e. fraction 1.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

DDN = Path("/fs/ddn/sdf/group/atlas/d/jkrupa/hep4m_pflow_sweep")
WS = Path("/sdf/data/atlas/u/jkrupa/heptokens")

PANELS = [
    ("resolution_iqr_over_median", "jet $p_T$ resolution\n(IQR / median)"),
    ("median_jet_pt_response", "jet $p_T$ bias\n(median response)"),
    ("token_acc_overall", "truthpart token\naccuracy"),
]


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep", nargs=2, action="append", metavar=("LABEL", "CSV"),
                   help="tokenized sweep: label and sweep_metrics.csv (repeatable)")
    p.add_argument("--cont_eval", type=Path, default=DDN / "cont_mlp256/cs1024/eval")
    p.add_argument("--out_png", type=Path, default=WS / "results/pflow_input_comparison/pflow_inputs_vs_info.png")
    args = p.parse_args()
    if not args.sweep:
        args.sweep = [["nq3 (cd8/cd8)", str(WS / "results/hep4m_pflow_sweep/sweep_metrics.csv")],
                      ["nq4 (trk cd4 / topo cd8)", str(DDN / "nq4_trkcd4_topocd8/sweep_metrics.csv")]]
    return args


def main():
    args = get_args()
    cont = json.loads((args.cont_eval / "metrics.json").read_text())
    cont.update(json.loads((args.cont_eval / "token_metrics.json").read_text()))

    fig, axes = plt.subplots(len(PANELS), 1, figsize=(6.5, 10), sharex=True)
    for i, (label, path) in enumerate(args.sweep):
        rows = sorted(csv.DictReader(open(path)), key=lambda r: float(r["info_fraction"]))
        x = [float(r["info_fraction"]) for r in rows]
        for ax, (key, _) in zip(axes, PANELS):
            yerr = [float(r[key + "_err"]) for r in rows] if key + "_err" in rows[0] else None
            ax.errorbar(x, [float(r[key]) for r in rows], yerr=yerr, fmt="o-", color=f"C{i}",
                        capsize=3, label=label)
        for xi, r in zip(x, rows):
            axes[0].annotate(f"K={r['cs']}", (xi, float(r[PANELS[0][0]])), fontsize=7, color=f"C{i}",
                             textcoords="offset points", xytext=(4, 4))

    for ax, (key, ylabel) in zip(axes, PANELS):
        err = cont.get(key + "_err", 0.0)
        ax.axhline(cont[key], color="k", ls="--", lw=1)
        ax.axhspan(cont[key] - err, cont[key] + err, color="k", alpha=0.12, lw=0)
        ax.errorbar([1.0], [cont[key]], yerr=[err], fmt="*", ms=11, color="k", capsize=3,
                    label="continuous (MLP input)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axes[1].axhline(1.0, ls=":", lw=0.8, color="gray")
    axes[0].set_title("4M particle flow: tokenized vs continuous inputs")
    axes[0].legend(fontsize=8)
    axes[-1].set_xscale("log")
    axes[-1].set_xlabel(r"input information fraction  $n_q\log_2 K / (32\,D)$")

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out_png}")


if __name__ == "__main__":
    main()
