#!/usr/bin/env python3
"""Produce jet and track feature histograms split by flavour labels."""

from __future__ import annotations

import argparse
from pathlib import Path
import h5py
from gdig.utils import (
    TRUTH_LABEL_COL,
    load_jets_table,
    make_histograms,
    plot_track_histograms,
)


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jets_df = load_jets_table(args.input)
    make_histograms(
        jets_df, args.output_dir, args.bins, args.density, args.dpi, args.dataset_name
    )
    # Get the jet lables
    jet_labels = jets_df[TRUTH_LABEL_COL].astype(int).to_numpy()
    # Load the
    with h5py.File(args.input, "r") as handle:
        tracks_ds = handle["tracks"]
        # Plot track histograms
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
        )


if __name__ == "__main__":
    main()
