#!/usr/bin/env python
"""Aggregate the input-tokenizer sweep eval into a CSV + panel figure.

For each cs point it reads <sweep_root>/cs<cs>/eval/{metrics.json, token_metrics.json},
computes the input information fraction (see info_fraction below), and plots the
pflow metrics vs information fraction:
  (1) jet pT resolution  = IQR / median of the jet-pT response,
  (2) jet pT bias        = median jet-pT response,
  (3) truthpart token accuracy (if available).
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep_root", type=Path,
                   default=Path("/sdf/data/atlas/u/jkrupa/heptokens/results/hep4m_pflow_sweep"))
    p.add_argument("--cs", type=int, nargs="+", default=[64, 256, 1024, 4096])
    p.add_argument("--cd", type=int, default=8)
    p.add_argument("--nq", type=int, default=3)
    p.add_argument("--out_csv", type=Path, default=None)
    p.add_argument("--out_png", type=Path, default=None)
    return p.parse_args()


# Information fraction convention (matches the paper's Fig. 1). D is the number of
# CONTINUOUS INPUT FEATURES per element (the float32 cost of the ORIGINAL data),
# NOT the codebook/latent dimension d: in Fig. 1, points at fixed (Q, K) but
# different d share the same x. The PF input mixes two modalities with different
# feature counts (tracks D=5, topoclusters D=8), so the combined fraction is total
# token bits over total original bits, weighted by the mean per-event multiplicity:
#     x = (N_trk + N_topo) * Q*log2(K) / (32 * (5*N_trk + 8*N_topo))
# N_trk, N_topo are the mean numbers of VALID tracks/topoclusters per event in the
# PF training sample (20M events), computed from the token export offsets (NOT the
# padding maxima 15/50, which truncate <0.02% of events):
N_TRK, N_TOPO = 3.618, 12.948   # mean valid multiplicities, 20M PF train export
D_TRK, D_TOPO = 5, 8            # continuous input features per track / topocluster


def info_fraction(cs, nq, n_trk=N_TRK, n_topo=N_TOPO):
    return (n_trk + n_topo) * nq * math.log2(cs) / (32 * (D_TRK * n_trk + D_TOPO * n_topo))


def main():
    args = get_args()
    out_csv = args.out_csv or (args.sweep_root / "sweep_metrics.csv")
    out_png = args.out_png or (args.sweep_root / "sweep_panel.png")

    rows = []
    for cs in args.cs:
        evaldir = args.sweep_root / f"cs{cs}" / "eval"
        mfile = evaldir / "metrics.json"
        if not mfile.exists():
            print(f"[skip] cs{cs}: {mfile} missing")
            continue
        m = json.loads(mfile.read_text())
        tfile = evaldir / "token_metrics.json"
        t = json.loads(tfile.read_text()) if tfile.exists() else {}
        median = m.get("median_jet_pt_response", float("nan"))
        iqr = m.get("iqr_jet_pt_response", float("nan"))
        rows.append({
            "cs": cs,
            "info_fraction": info_fraction(cs, args.nq),
            "median_jet_pt_response": median,
            "median_jet_pt_response_err": m.get("median_jet_pt_response_err", float("nan")),
            "iqr_jet_pt_response": iqr,
            "iqr_jet_pt_response_err": m.get("iqr_jet_pt_response_err", float("nan")),
            "resolution_iqr_over_median": m.get("resolution_iqr_over_median",
                                                (iqr / median) if median else float("nan")),
            "resolution_iqr_over_median_err": m.get("resolution_iqr_over_median_err", float("nan")),
            "mean_jet_pt_response": m.get("mean_jet_pt_response", float("nan")),
            "std_jet_pt_response": m.get("std_jet_pt_response", float("nan")),
            "mean_reco_cardinality_at_threshold": m.get("mean_reco_cardinality_at_threshold", float("nan")),
            "n_events": m.get("n_events", float("nan")),
            "token_acc_overall": t.get("token_acc_overall", float("nan")),
            "token_acc_overall_err": t.get("token_acc_overall_err", float("nan")),
            "token_acc_exact": t.get("token_acc_exact", float("nan")),
        })

    if not rows:
        raise SystemExit("No eval metrics found; run scripts/pflow_eval_point.sh for each cs first.")

    rows.sort(key=lambda r: r["info_fraction"])

    # write CSV
    cols = list(rows[0].keys())
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(f"{r[c]}" for c in cols) + "\n")
    print(f"wrote {out_csv}")

    x = [r["info_fraction"] for r in rows]
    labels = [f"cs{r['cs']}" for r in rows]

    fig, axes = plt.subplots(3, 1, figsize=(6, 10), sharex=True)
    axes[0].errorbar(x, [r["resolution_iqr_over_median"] for r in rows],
                     yerr=[r["resolution_iqr_over_median_err"] for r in rows],
                     fmt="o-", color="C0", capsize=3)
    axes[0].set_ylabel("jet $p_T$ resolution\n(IQR / median)")
    axes[0].set_title("4M truthpart reconstruction vs input information")

    axes[1].errorbar(x, [r["median_jet_pt_response"] for r in rows],
                     yerr=[r["median_jet_pt_response_err"] for r in rows],
                     fmt="s-", color="C1", capsize=3)
    axes[1].axhline(1.0, ls="--", lw=0.8, color="gray")
    axes[1].set_ylabel("jet $p_T$ bias\n(median response)")

    tok = [r["token_acc_overall"] for r in rows]
    if any(not (isinstance(v, float) and math.isnan(v)) for v in tok):
        axes[2].errorbar(x, tok, yerr=[r["token_acc_overall_err"] for r in rows],
                         fmt="^-", color="C2", capsize=3)
    axes[2].set_ylabel("truthpart token\naccuracy")
    axes[2].set_xlabel(r"input information fraction  $n_q\log_2 K / (32\,D)$")

    for ax in axes:
        ax.grid(alpha=0.3)
    for xi, li in zip(x, labels):
        axes[0].annotate(li, (xi, axes[0].get_ylim()[1]), fontsize=7, ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
