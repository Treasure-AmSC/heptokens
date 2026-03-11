#!/usr/bin/env python3
"""Produce jet/track histograms split by flavour labels.

Optionally, plot features after applying user-provided preprocessing transformers:
- constituent transformer (e.g. ``cst_quantiles.joblib``)
- jet transformer (e.g. ``jet_quantiles.joblib``)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load as joblib_load

from heptokens.utils.plot_physics import (
    FLAVOUR_LABELS,
    TRUTH_LABEL_COL,
    ensure_output_dir,
    load_jets_table,
    make_histograms,
    plot_track_histograms,
)

log = logging.getLogger(__name__)


def _parse_feature_list(raw_features: str) -> list[str]:
    return [feature.strip() for feature in raw_features.split(",") if feature.strip()]


def _slice_length(total: int, max_jets: int | None) -> int:
    if max_jets is None:
        return total
    if max_jets <= 0:
        raise ValueError("--transform-max-jets must be positive.")
    return min(max_jets, total)


def plot_transformed_jet_histograms(
    jets_df: pd.DataFrame,
    transformer_path: Path,
    jet_features: list[str],
    output_dir: Path,
    bins: int,
    density: bool,
    dpi: int,
    dataset_name: str,
) -> None:
    log.info("Loading jet transformer from %s", transformer_path)
    jet_transformer = joblib_load(transformer_path)

    missing = [feature for feature in jet_features if feature not in jets_df.columns]
    if missing:
        raise KeyError(f"Jet features not found in jets table: {missing}")

    jet_values = jets_df[jet_features].to_numpy(dtype=np.float64, copy=True)
    finite_rows = np.all(np.isfinite(jet_values), axis=1)
    if not np.any(finite_rows):
        raise ValueError("No finite jet rows available for transformation.")
    if np.any(~finite_rows):
        log.warning("Dropping %d jet rows with non-finite values before transform.", np.sum(~finite_rows))

    transformed = jet_transformer.transform(jet_values[finite_rows])
    transformed_df = pd.DataFrame(
        transformed, columns=[f"{feature}_transformed" for feature in jet_features]
    )
    transformed_df[TRUTH_LABEL_COL] = (
        jets_df.loc[finite_rows, TRUTH_LABEL_COL].astype(int).to_numpy(copy=False)
    )

    make_histograms(
        transformed_df,
        output_dir,
        bins,
        density,
        dpi,
        f"{dataset_name} (jet transformed)",
    )


def plot_transformed_constituent_histograms(
    tracks_ds: h5py.Dataset,
    jet_labels: np.ndarray,
    transformer_path: Path,
    cst_features: list[str],
    output_dir: Path,
    bins: int,
    density: bool,
    dpi: int,
    dataset_name: str,
    chunk_size: int,
    max_jets: int | None,
) -> None:
    log.info("Loading constituent transformer from %s", transformer_path)
    cst_transformer = joblib_load(transformer_path)

    available_fields = set(tracks_ds.dtype.names or ())
    required = {"valid", *cst_features}
    missing = sorted(required - available_fields)
    if missing:
        raise KeyError(f"Required track fields not found in tracks table: {missing}")

    total_jets = _slice_length(len(jet_labels), max_jets)
    if total_jets == 0:
        return

    jet_labels = jet_labels[:total_jets]
    n_features = len(cst_features)
    n_csts = tracks_ds.shape[1]

    vmin = np.full(n_features, np.inf, dtype=np.float64)
    vmax = np.full(n_features, -np.inf, dtype=np.float64)

    for start in range(0, total_jets, chunk_size):
        end = min(start + chunk_size, total_jets)
        valid_mask = tracks_ds["valid"][start:end].astype(bool)
        if not np.any(valid_mask):
            continue

        feature_cube = np.empty((end - start, n_csts, n_features), dtype=np.float32)
        for i, feature in enumerate(cst_features):
            feature_cube[:, :, i] = tracks_ds[feature][start:end]

        flat_values = feature_cube[valid_mask]
        finite_rows = np.all(np.isfinite(flat_values), axis=1)
        if not np.any(finite_rows):
            continue

        transformed = cst_transformer.transform(flat_values[finite_rows])
        vmin = np.minimum(vmin, transformed.min(axis=0))
        vmax = np.maximum(vmax, transformed.max(axis=0))

    bin_edges_per_feature: list[np.ndarray | None] = []
    for i in range(n_features):
        lo = vmin[i]
        hi = vmax[i]
        if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(lo, hi):
            bin_edges_per_feature.append(None)
            continue
        bin_edges_per_feature.append(np.linspace(lo, hi, bins + 1))

    if all(edges is None for edges in bin_edges_per_feature):
        log.warning("No finite transformed constituent values found; skipping transformed track plots.")
        return

    counts = {
        pdg: [np.zeros(bins, dtype=np.float64) if edges is not None else None for edges in bin_edges_per_feature]
        for pdg in FLAVOUR_LABELS
    }
    totals = {pdg: 0 for pdg in FLAVOUR_LABELS}

    for start in range(0, total_jets, chunk_size):
        end = min(start + chunk_size, total_jets)
        valid_mask = tracks_ds["valid"][start:end].astype(bool)
        if not np.any(valid_mask):
            continue

        feature_cube = np.empty((end - start, n_csts, n_features), dtype=np.float32)
        for i, feature in enumerate(cst_features):
            feature_cube[:, :, i] = tracks_ds[feature][start:end]

        labels_chunk = jet_labels[start:end]
        for pdg in FLAVOUR_LABELS:
            jet_mask = labels_chunk == pdg
            if not np.any(jet_mask):
                continue

            masked_values = feature_cube[jet_mask]
            masked_valid = valid_mask[jet_mask]
            if not np.any(masked_valid):
                continue

            flat_values = masked_values[masked_valid]
            finite_rows = np.all(np.isfinite(flat_values), axis=1)
            if not np.any(finite_rows):
                continue

            transformed = cst_transformer.transform(flat_values[finite_rows])
            totals[pdg] += transformed.shape[0]
            for i, edges in enumerate(bin_edges_per_feature):
                if edges is None:
                    continue
                hist, _ = np.histogram(transformed[:, i], bins=edges)
                counts[pdg][i] += hist

    ensure_output_dir(output_dir)
    for i, feature in enumerate(cst_features):
        edges = bin_edges_per_feature[i]
        if edges is None:
            continue

        bin_widths = np.diff(edges)
        bin_centers = edges[:-1] + 0.5 * bin_widths

        fig, ax = plt.subplots()
        ymax = 0.0
        ymin_positive = np.inf

        for pdg, label_name in FLAVOUR_LABELS.items():
            if totals[pdg] == 0:
                continue
            feature_counts = counts[pdg][i]
            if feature_counts is None:
                continue

            heights = feature_counts.copy()
            if density:
                heights /= totals[pdg]
                heights /= bin_widths

            ax.step(bin_centers, heights, where="mid", label=label_name, linewidth=2)
            if heights.size:
                ymax = max(ymax, float(np.max(heights)))
                positives = heights[heights > 0]
                if positives.size:
                    ymin_positive = min(ymin_positive, float(np.min(positives)))

        if not ax.has_data():
            plt.close(fig)
            continue

        ax.set_xlabel(f"{feature}_transformed")
        ax.set_ylabel("Probability density" if density else "Entries")
        ax.text(
            0.02,
            0.98,
            dataset_name + r"$\sqrt{s}=$ 13.6 TeV $t\bar{t}$",
            transform=ax.transAxes,
            fontsize=24,
            va="top",
        )
        if ymax > 0:
            ax.set_ylim(top=ymax * 1.4)
        ax.legend()
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        fig.tight_layout()

        output_path = output_dir / f"{feature}_transformed.png"
        fig.savefig(output_path, dpi=dpi)
        ax.set_yscale("log")
        if np.isfinite(ymin_positive) and ymin_positive > 0 and ymax > 0:
            ax.set_ylim(bottom=ymin_positive / 1.2, top=ymax * 1.4)
        output_path_log = output_dir / f"{feature}_transformed_log.png"
        fig.savefig(output_path_log, dpi=dpi)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot jet and track feature histograms split by truth labels."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the HDF5 file (e.g. data/mc-flavtag-ttbar-small.h5).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plot") / "figures" / "jets",
        help="Directory where histograms will be written.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of bins to use for the histograms.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="JetSet",
        help="Name of dataset.",
    )
    parser.add_argument(
        "--density",
        action="store_true",
        help="Normalise histograms to show probability density.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for the saved figures.",
    )
    parser.add_argument(
        "--track-output-dir",
        type=Path,
        default=Path("plot") / "figures" / "tracks",
        help="Directory where track histograms will be written.",
    )
    parser.add_argument(
        "--track-chunk-size",
        type=int,
        default=250_000,
        help="Number of jets to process per chunk when streaming track data.",
    )
    parser.add_argument(
        "--track-max-jets",
        type=int,
        default=None,
        help="Limit the number of leading jets considered for track histograms.",
    )
    parser.add_argument(
        "--jet-transformer",
        type=Path,
        default=None,
        help="Optional path to a jet transformer joblib (e.g. jet_quantiles.joblib).",
    )
    parser.add_argument(
        "--cst-transformer",
        type=Path,
        default=None,
        help="Optional path to a constituent transformer joblib (e.g. cst_quantiles.joblib).",
    )
    parser.add_argument(
        "--jet-features",
        type=str,
        default="pt,mass,eta,phi",
        help="Comma-separated jet features expected by --jet-transformer.",
    )
    parser.add_argument(
        "--cst-features",
        type=str,
        default="pt,deta,dphi,d0",
        help="Comma-separated constituent features expected by --cst-transformer.",
    )
    parser.add_argument(
        "--track-features",
        type=str,
        default=None,
        help="Comma-separated raw track features to plot (default: all from TRACK_FEATURES).",
    )
    parser.add_argument(
        "--jet-transformed-output-dir",
        type=Path,
        default=None,
        help="Output directory for transformed jet histograms (default: <output-dir>/transformed).",
    )
    parser.add_argument(
        "--cst-transformed-output-dir",
        type=Path,
        default=None,
        help="Output directory for transformed track histograms (default: <track-output-dir>/transformed).",
    )
    parser.add_argument(
        "--transform-chunk-size",
        type=int,
        default=50_000,
        help="Chunk size used when transforming track features.",
    )
    parser.add_argument(
        "--transform-max-jets",
        type=int,
        default=None,
        help="Optional limit on jets used for transformed constituent histograms.",
    )
    parser.add_argument(
        "--skip-jet-histograms",
        action="store_true",
        help="Skip raw jet feature histograms.",
    )
    parser.add_argument(
        "--skip-raw-track-histograms",
        action="store_true",
        help="Skip raw track feature histograms.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    jets_df = load_jets_table(args.input)
    if not args.skip_jet_histograms:
        make_histograms(jets_df, args.output_dir, args.bins, args.density, args.dpi, args.dataset_name)

    if args.jet_transformer is not None:
        jet_transformed_output_dir = args.jet_transformed_output_dir
        if jet_transformed_output_dir is None:
            jet_transformed_output_dir = args.output_dir / "transformed"

        plot_transformed_jet_histograms(
            jets_df=jets_df,
            transformer_path=args.jet_transformer,
            jet_features=_parse_feature_list(args.jet_features),
            output_dir=jet_transformed_output_dir,
            bins=args.bins,
            density=args.density,
            dpi=args.dpi,
            dataset_name=args.dataset_name,
        )

    # Get the jet labels
    jet_labels = jets_df[TRUTH_LABEL_COL].astype(int).to_numpy()

    with h5py.File(args.input, "r") as handle:
        tracks_ds = handle["tracks"]

        # Plot raw track histograms
        if not args.skip_raw_track_histograms:
            track_feature_filter = _parse_feature_list(args.track_features) if args.track_features else None
            plot_track_histograms(
                tracks_ds,
                jet_labels,
                args.track_output_dir,
                args.bins,
                args.density,
                args.dpi,
                args.dataset_name,
                args.track_chunk_size,
                args.track_max_jets,
                feature_filter=track_feature_filter,
            )

        if args.cst_transformer is not None:
            cst_transformed_output_dir = args.cst_transformed_output_dir
            if cst_transformed_output_dir is None:
                cst_transformed_output_dir = args.track_output_dir / "transformed"

            plot_transformed_constituent_histograms(
                tracks_ds=tracks_ds,
                jet_labels=jet_labels,
                transformer_path=args.cst_transformer,
                cst_features=_parse_feature_list(args.cst_features),
                output_dir=cst_transformed_output_dir,
                bins=args.bins,
                density=args.density,
                dpi=args.dpi,
                dataset_name=f"{args.dataset_name} (constituent transformed)",
                chunk_size=args.transform_chunk_size,
                max_jets=args.transform_max_jets,
            )


if __name__ == "__main__":
    main()
