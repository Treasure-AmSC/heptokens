"""Bootstrap uncertainty on background rejection at fixed signal efficiency.

For each run, loads predictions.npz (same layout as compare_roc.py --from_predictions),
computes rejection = 1 / mis-ID-rate for a background class at one or more fixed
signal-efficiency working points, and estimates uncertainty via bootstrap
resampling (with replacement) of the signal/background score arrays.

The threshold at a given signal efficiency is computed exactly via a quantile
of the signal scores (no 20000-point grid scan), so this is fast enough to
bootstrap hundreds of times even on tens of millions of jets.

Usage
-----
pixi run python scripts/bootstrap_rejection.py \\
    --run_dirs results/vqvae_scan/classifiers/token_clf_full_cb1024_nq3 \\
               results/vqvae_scan/classifiers/token_clf_full_cb1024_nq4 \\
               results/vqvae_scan/classifiers/token_clf_full_cb8192_nq4 \\
               results/vqvae_scan/classifiers/token_clf_full_v2/test_full/test_predictions \\
               results/vqvae_scan/classifiers/feature_clf_full_v3/test_full/test_predictions \\
    --run_labels cb1024_nq3 cb1024_nq4 cb8192_nq4 token_clf_full_v2 feature_clf_full_v3 \\
    --targets 0.50 0.70 \\
    --n_boot 200 \\
    --max_jets 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_roc import DEFAULT_CLASS_NAMES, _pretty_run_label  # noqa: E402


def rejection_at_eff(sig_scores: np.ndarray, bkg_scores: np.ndarray, target_eff: float) -> float:
    """Background rejection (1 / mis-ID rate) at a fixed signal efficiency.

    The threshold is the exact quantile of ``sig_scores`` such that
    ``target_eff`` fraction of signal jets pass it.
    """
    thr = np.quantile(sig_scores, 1.0 - target_eff)
    misid = (bkg_scores >= thr).mean()
    return 1.0 / misid if misid > 0 else np.inf


def bootstrap_rejection(
    sig_scores: np.ndarray,
    bkg_scores: np.ndarray,
    targets: list[float],
    n_boot: int,
    rng: np.random.Generator,
) -> dict[float, np.ndarray]:
    """Bootstrap-resample rejection@target_eff for each target in *targets*."""
    n_sig, n_bkg = len(sig_scores), len(bkg_scores)
    out = {t: np.empty(n_boot) for t in targets}
    for b in range(n_boot):
        s = sig_scores[rng.integers(0, n_sig, n_sig)]
        k = bkg_scores[rng.integers(0, n_bkg, n_bkg)]
        for t in targets:
            out[t][b] = rejection_at_eff(s, k, t)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap uncertainty on background rejection at fixed signal efficiency."
    )
    parser.add_argument("--run_dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--run_labels", nargs="+", required=True)
    parser.add_argument(
        "--targets", type=float, nargs="+", default=[0.50, 0.70],
        help="Signal-efficiency working points (default: 0.50 0.70).",
    )
    parser.add_argument("--n_boot", type=int, default=200, help="Number of bootstrap resamples.")
    parser.add_argument(
        "--max_jets", type=int, default=0,
        help="Cap jets loaded per run (0 = load all, default: 0).",
    )
    parser.add_argument("--signal_class", type=str, default="b", help="Signal class name (default: b).")
    parser.add_argument("--bkg_class", type=str, default="light", help="Background class name (default: light).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if len(args.run_dirs) != len(args.run_labels):
        raise SystemExit("--run_dirs and --run_labels must have the same length.")

    class_names = dict(DEFAULT_CLASS_NAMES)
    name_to_idx = {v.replace("$", "").replace(r"\tau", "tau").lower(): k for k, v in class_names.items()}
    signal_idx = name_to_idx[args.signal_class.lower()]
    bkg_idx = name_to_idx[args.bkg_class.lower()]

    rng = np.random.default_rng(args.seed)

    header = f"{'run':<20s} {'label':<25s}  " + "  ".join(
        f"rej@{t:.0%}{args.signal_class}-eff (mean +/- std)" for t in args.targets
    )
    print(header, flush=True)

    for run_dir, run_label in zip(args.run_dirs, args.run_labels):
        run_dir = run_dir.resolve()
        npz_path = run_dir / "test" / "test_predictions" / "predictions.npz"
        if not npz_path.exists():
            npz_path = run_dir / "predictions.npz"
        if not npz_path.exists():
            print(f"Skipping {run_label}: no predictions.npz found in {run_dir}", flush=True)
            continue

        data = np.load(npz_path)
        probs, labels = data["probs"], data["labels"]
        if args.max_jets and args.max_jets > 0:
            probs, labels = probs[: args.max_jets], labels[: args.max_jets]

        sig_scores = probs[labels == signal_idx, signal_idx]
        bkg_scores = probs[labels == bkg_idx, signal_idx]

        nominal = {t: rejection_at_eff(sig_scores, bkg_scores, t) for t in args.targets}
        boot_vals = bootstrap_rejection(sig_scores, bkg_scores, args.targets, args.n_boot, rng)

        pretty = _pretty_run_label(run_label)
        row = "  ".join(f"{nominal[t]:8.1f} +/- {boot_vals[t].std():6.1f}" for t in args.targets)
        print(f"{run_label:<20s} {pretty:<25s}  {row}", flush=True)


if __name__ == "__main__":
    main()
