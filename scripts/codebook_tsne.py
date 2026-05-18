"""Codebook interpretability via t-SNE.

Visualizes the learned codebook vectors in 2D, colored by which jet flavor
each code is most associated with. Also produces a "purity" variant where
opacity encodes how specialized each code is to a single flavor.

Uses the pre-exported tokenized NPZ file (avoids re-running inference).

Usage
-----
pixi run python scripts/codebook_tsne.py

# Custom paths:
pixi run python scripts/codebook_tsne.py \
    --npz results/vqvae_scan/tokenized/transformer_vqvae_enc_large_cb32768_cd8_nq4_lr0.0003_cw1.0_nj10000000/mc-flavtag-ttbar-small.npz \
    --output_dir results/codebook_interpretability \
    --max_jets 1000000
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
import mplhep as hep
import numpy as np
from sklearn.manifold import TSNE

log = logging.getLogger(__name__)

BEST_MODEL = "transformer_vqvae_enc_large_cb32768_cd8_nq4_lr0.0003_cw1.0_nj10000000"
DEFAULT_NPZ = Path(f"results/vqvae_scan/tokenized/{BEST_MODEL}/mc-flavtag-ttbar-small.npz")

FLAVOR_NAMES = ["light", "c", "b", r"$\tau$"]
FLAVOR_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def compute_code_flavor_fractions(
    indices: np.ndarray, labels: np.ndarray, codebook_size: int, num_quantizers: int
) -> np.ndarray:
    """Compute flavor frequency for each code in each quantizer.

    Args:
        indices: [N_jets, num_csts, num_quantizers] int16, -1 = masked
        labels: [N_jets] int8
        codebook_size: number of codes per quantizer
        num_quantizers: number of residual quantizers

    Returns:
        fractions: [num_quantizers, codebook_size, num_classes] float32
            Normalized frequency of each flavor for each code.
    """
    num_classes = int(labels.max()) + 1
    counts = np.zeros((num_quantizers, codebook_size, num_classes), dtype=np.float64)

    for q in range(num_quantizers):
        idx_q = indices[:, :, q]  # [N_jets, num_csts]
        for cls in range(num_classes):
            cls_mask = labels == cls
            cls_indices = idx_q[cls_mask]  # [N_cls_jets, num_csts]
            valid = cls_indices >= 0
            codes = cls_indices[valid]
            np.add.at(counts[q, :, cls], codes, 1)

    totals = counts.sum(axis=-1, keepdims=True)
    totals = np.where(totals == 0, 1, totals)
    fractions = counts / totals
    return fractions.astype(np.float32)


def compute_code_usage_counts(
    indices: np.ndarray, codebook_size: int, num_quantizers: int
) -> np.ndarray:
    """Total usage count per code per quantizer."""
    usage = np.zeros((num_quantizers, codebook_size), dtype=np.int64)
    for q in range(num_quantizers):
        idx_q = indices[:, :, q]
        valid = idx_q >= 0
        codes = idx_q[valid]
        np.add.at(usage[q], codes, 1)
    return usage


def run_tsne(codebooks: np.ndarray, perplexity: float = 50, seed: int = 42) -> list:
    """Run t-SNE on each quantizer's codebook independently.

    Args:
        codebooks: [num_quantizers, codebook_size, codebook_dim]

    Returns:
        List of [codebook_size, 2] arrays (one per quantizer)
    """
    embeddings = []
    num_quantizers = codebooks.shape[0]
    for q in range(num_quantizers):
        log.info(f"  Running t-SNE for quantizer {q} ({codebooks.shape[1]} codes, dim={codebooks.shape[2]})")
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=seed,
            max_iter=1000,
            init="pca",
            learning_rate="auto",
        )
        emb = tsne.fit_transform(codebooks[q])
        embeddings.append(emb)
    return embeddings


def plot_tsne_by_flavor(
    embeddings: list,
    fractions: np.ndarray,
    usage: np.ndarray,
    output_dir: Path,
    min_usage: int = 100,
):
    """Plot t-SNE colored by dominant flavor, one subplot per quantizer."""
    plt.style.use(hep.style.CMS)
    num_quantizers = len(embeddings)

    fig, axes = plt.subplots(1, num_quantizers, figsize=(5.5 * num_quantizers, 5), squeeze=False)
    axes = axes.flatten()

    for q in range(num_quantizers):
        ax = axes[q]
        emb = embeddings[q]
        frac = fractions[q]  # [codebook_size, num_classes]
        used = usage[q] >= min_usage

        dominant_class = frac.argmax(axis=-1)
        purity = frac.max(axis=-1)

        for cls in range(frac.shape[1]):
            mask = (dominant_class == cls) & used
            if not mask.any():
                continue
            colors = [to_rgba(FLAVOR_COLORS[cls], alpha=0.3 + 0.7 * purity[i])
                      for i in np.where(mask)[0]]
            sizes = np.clip(np.log1p(usage[q][mask]) * 2, 3, 40)
            ax.scatter(
                emb[mask, 0], emb[mask, 1],
                c=colors, s=sizes,
                edgecolors="none", rasterized=True,
            )

        # Plot unused codes as gray x's
        unused = ~used
        if unused.any():
            ax.scatter(
                emb[unused, 0], emb[unused, 1],
                c="gray", s=8, marker="x", alpha=0.3,
                label=f"unused (<{min_usage})",
            )

        ax.set_title(f"Quantizer {q}", fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])

    legend_elements = [
        Patch(facecolor=FLAVOR_COLORS[i], label=FLAVOR_NAMES[i])
        for i in range(len(FLAVOR_NAMES))
    ]
    legend_elements.append(
        plt.Line2D([0], [0], marker="x", color="gray", linestyle="None",
                   markersize=6, label=f"unused (<{min_usage})")
    )
    axes[-1].legend(handles=legend_elements, loc="upper right", fontsize=10)

    fig.suptitle(
        r"Codebook t-SNE — color = dominant jet flavor, opacity = purity",
        fontsize=14, y=1.02,
    )
    fig.tight_layout()
    out = output_dir / "codebook_tsne_by_flavor.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {out}")


def plot_tsne_purity(
    embeddings: list,
    fractions: np.ndarray,
    usage: np.ndarray,
    output_dir: Path,
    min_usage: int = 100,
):
    """Plot t-SNE where color = purity (how specialized each code is)."""
    plt.style.use(hep.style.CMS)
    num_quantizers = len(embeddings)

    fig, axes = plt.subplots(1, num_quantizers, figsize=(5.5 * num_quantizers, 5), squeeze=False)
    axes = axes.flatten()

    for q in range(num_quantizers):
        ax = axes[q]
        emb = embeddings[q]
        frac = fractions[q]
        used = usage[q] >= min_usage

        purity = frac.max(axis=-1)

        sc = ax.scatter(
            emb[used, 0], emb[used, 1],
            c=purity[used], cmap="RdYlBu_r", vmin=0.25, vmax=1.0,
            s=10, edgecolors="none", rasterized=True,
        )
        ax.set_title(f"Quantizer {q}", fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])

    cbar = fig.colorbar(sc, ax=axes.tolist(), shrink=0.8, pad=0.02)
    cbar.set_label("Flavor purity (max class fraction)", fontsize=11)

    fig.suptitle("Codebook t-SNE — flavor purity", fontsize=14, y=1.02)
    fig.tight_layout()
    out = output_dir / "codebook_tsne_purity.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {out}")


def plot_flavor_entropy(
    embeddings: list,
    fractions: np.ndarray,
    usage: np.ndarray,
    output_dir: Path,
    min_usage: int = 100,
):
    """Plot t-SNE colored by Shannon entropy of flavor distribution per code."""
    plt.style.use(hep.style.CMS)
    num_quantizers = len(embeddings)

    fig, axes = plt.subplots(1, num_quantizers, figsize=(5.5 * num_quantizers, 5), squeeze=False)
    axes = axes.flatten()

    for q in range(num_quantizers):
        ax = axes[q]
        emb = embeddings[q]
        frac = fractions[q]
        used = usage[q] >= min_usage

        # Shannon entropy (0 = pure, log2(4) = 2 = uniform)
        frac_safe = np.where(frac > 0, frac, 1)
        entropy = -np.sum(frac * np.log2(frac_safe), axis=-1)

        sc = ax.scatter(
            emb[used, 0], emb[used, 1],
            c=entropy[used], cmap="viridis", vmin=0, vmax=2.0,
            s=10, edgecolors="none", rasterized=True,
        )
        ax.set_title(f"Quantizer {q}", fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])

    cbar = fig.colorbar(sc, ax=axes.tolist(), shrink=0.8, pad=0.02)
    cbar.set_label("Shannon entropy [bits] (0 = pure, 2 = uniform)", fontsize=11)

    fig.suptitle("Codebook t-SNE — flavor entropy", fontsize=14, y=1.02)
    fig.tight_layout()
    out = output_dir / "codebook_tsne_entropy.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {out}")


def print_summary_stats(fractions: np.ndarray, usage: np.ndarray, min_usage: int = 100):
    """Print summary statistics about codebook specialization."""
    num_quantizers = fractions.shape[0]
    num_classes = fractions.shape[2]

    log.info("=" * 60)
    log.info("Codebook specialization summary")
    log.info("=" * 60)

    for q in range(num_quantizers):
        used = usage[q] >= min_usage
        n_used = used.sum()
        frac = fractions[q][used]
        purity = frac.max(axis=-1)
        dominant = frac.argmax(axis=-1)

        log.info(f"\nQuantizer {q}: {n_used} codes used (of {fractions.shape[1]})")
        log.info(f"  Mean purity: {purity.mean():.3f}")
        log.info(f"  Codes with purity > 0.5: {(purity > 0.5).sum()} ({100*(purity > 0.5).mean():.1f}%)")
        log.info(f"  Codes with purity > 0.75: {(purity > 0.75).sum()} ({100*(purity > 0.75).mean():.1f}%)")

        for cls in range(num_classes):
            n_cls = (dominant == cls).sum()
            cls_purity = purity[dominant == cls].mean() if (dominant == cls).any() else 0
            log.info(f"  {FLAVOR_NAMES[cls]:>6s}-dominated: {n_cls:5d} codes, mean purity {cls_purity:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Codebook t-SNE interpretability.")
    parser.add_argument("--npz", type=str, default=str(DEFAULT_NPZ),
                        help="Path to tokenized NPZ file")
    parser.add_argument("--output_dir", type=str, default="results/codebook_interpretability")
    parser.add_argument("--max_jets", type=int, default=1_000_000,
                        help="Max jets to use for computing flavor fractions")
    parser.add_argument("--perplexity", type=float, default=50,
                        help="t-SNE perplexity")
    parser.add_argument("--min_usage", type=int, default=100,
                        help="Minimum usage count to include a code in plots")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenized data
    log.info(f"Loading tokenized data from {args.npz}")
    data = np.load(args.npz)
    indices = data["indices"]  # [N, num_csts, num_quantizers]
    labels = data["labels"]    # [N]
    codebooks = data["codebooks"]  # [num_quantizers, codebook_size, codebook_dim]

    num_quantizers, codebook_size, codebook_dim = codebooks.shape
    log.info(f"  {len(labels):,} jets, {num_quantizers} quantizers, "
             f"codebook: {codebook_size} codes x {codebook_dim}d")
    log.info(f"  Label distribution: {dict(zip(FLAVOR_NAMES, np.bincount(labels)))}")

    # Subsample for flavor fraction computation if needed
    n = min(args.max_jets, len(labels))
    if n < len(labels):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(labels), n, replace=False)
        indices_sub = indices[idx]
        labels_sub = labels[idx]
        log.info(f"  Subsampled to {n:,} jets for flavor fraction computation")
    else:
        indices_sub = indices
        labels_sub = labels

    # Compute flavor fractions per code
    log.info("Computing per-code flavor fractions...")
    fractions = compute_code_flavor_fractions(indices_sub, labels_sub, codebook_size, num_quantizers)

    # Compute usage counts
    log.info("Computing code usage counts...")
    usage = compute_code_usage_counts(indices_sub, codebook_size, num_quantizers)

    # Print summary
    print_summary_stats(fractions, usage, min_usage=args.min_usage)

    # Run t-SNE on codebook vectors
    log.info("Running t-SNE on codebook vectors...")
    embeddings = run_tsne(codebooks, perplexity=args.perplexity, seed=args.seed)

    # Generate plots
    log.info("Generating plots...")
    plot_tsne_by_flavor(embeddings, fractions, usage, output_dir, min_usage=args.min_usage)
    plot_tsne_purity(embeddings, fractions, usage, output_dir, min_usage=args.min_usage)
    plot_flavor_entropy(embeddings, fractions, usage, output_dir, min_usage=args.min_usage)

    log.info("Done!")


if __name__ == "__main__":
    main()
