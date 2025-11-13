from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mplhep as hep

plt.style.use(hep.style.ROOT)


FLAVOUR_LABELS: Dict[int, str] = {
    5: "b",
    4: "c",
    0: "light",
}
TRUTH_LABEL_COL = "HadronConeExclTruthLabelID"
TRACK_FEATURES: Tuple[str, ...] = (
    "valid",
    "d0",
    "d0RelativeToBeamspot",
    "d0RelativeToBeamspotUncertainty",
    "d0Uncertainty",
    "phiUncertainty",
    "pt",
    "qOverP",
    "qOverPUncertainty",
    "theta",
    "thetaUncertainty",
    "z0RelativeToBeamspot",
    "z0RelativeToBeamspotUncertainty",
    "z0SinTheta",
    "z0SinThetaUncertainty",
    "numberOfInnermostPixelLayerHits",
    "numberOfInnermostPixelLayerSharedHits",
    "numberOfInnermostPixelLayerSplitHits",
    "numberOfNextToInnermostPixelLayerHits",
    "numberOfPixelHits",
    "numberOfPixelSharedHits",
    "numberOfPixelSplitHits",
    "numberOfSCTHits",
    "numberOfSCTSharedHits",
    "deta",
    "dphi",
    "ptfrac",
    "lifetimeSignedD0",
    "lifetimeSignedD0Significance",
    "lifetimeSignedZ0SinTheta",
    "lifetimeSignedZ0SinThetaSignificance",
    "ftagTruthOriginLabel",
    "ftagTruthParentBarcode",
    "ftagTruthVertexIndex",
    "GN2v01_trackOrigin",
    "GN2v01_vertexIndex",
)


def load_jets_table(h5_path: Path) -> pd.DataFrame:
    """Load the jets dataset into a DataFrame."""
    try:
        import h5py  # Imported lazily to avoid unnecessary hard dependency on CLI usage.
    except ImportError as exc:
        raise ImportError(
            "h5py is required to read the jets table but is not installed."
        ) from exc

    with h5py.File(h5_path, "r") as handle:
        jets_table = handle["jets"]
        data = jets_table[:]

    df = pd.DataFrame.from_records(data)

    # Promote float16 columns to float32 for more stable plotting.
    float16_cols = df.select_dtypes(include=["float16"]).columns
    df[float16_cols] = df[float16_cols].astype("float32")

    return df


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def iter_features(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Yield numeric feature columns except the truth label."""
    numeric_df = df.select_dtypes(include=[np.number])
    for column in numeric_df.columns:
        # if column == TRUTH_LABEL_COL:
        #    continue
        yield column, numeric_df[column]


def make_histograms(
    df: pd.DataFrame,
    output_dir: Path,
    bins: int,
    density: bool,
    dpi: int,
    dataset_name: str,
) -> None:
    """Create one histogram per feature, split by jet flavour."""
    if TRUTH_LABEL_COL not in df:
        raise KeyError(f"Column '{TRUTH_LABEL_COL}' not found in jets table.")

    label_series = df[TRUTH_LABEL_COL].astype(int)
    mask = label_series.isin(FLAVOUR_LABELS.keys())
    filtered = df.loc[mask].copy()
    filtered[TRUTH_LABEL_COL] = label_series.loc[mask]

    if filtered.empty:
        raise ValueError("No jets with desired truth labels were found.")

    ensure_output_dir(output_dir)

    for feature, series in iter_features(filtered):
        joint = pd.DataFrame({"feature": series, "label": filtered[TRUTH_LABEL_COL]})
        joint = joint.replace([np.inf, -np.inf], np.nan).dropna()
        if joint.empty:
            continue

        values = joint["feature"].to_numpy()
        vmin = values.min()
        vmax = values.max()
        if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
            continue

        # Use a shared set of bins across flavours for readability.
        bin_edges = np.linspace(vmin, vmax, bins + 1)

        fig, ax = plt.subplots()
        ymax = 0.0
        ymin_positive = np.inf
        for pdg, label_name in FLAVOUR_LABELS.items():
            sample = joint.loc[joint["label"] == pdg, "feature"].to_numpy()
            if sample.size == 0:
                continue
            counts, _, _ = ax.hist(
                sample,
                bins=bin_edges,
                histtype="step",
                density=density,
                label=f"{label_name}",
                linewidth=2,
            )
            if counts.size:
                ymax = max(ymax, float(np.max(counts)))
                positives = counts[counts > 0]
                if positives.size:
                    ymin_positive = min(ymin_positive, float(np.min(positives)))

        ax.set_xlabel(feature)
        ax.set_ylabel("Probability density" if density else "Entries")
        # ax.set_title(feature)
        ax.legend()
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        fig.tight_layout()
        ax.text(
            0.02,
            0.98,
            dataset_name + r"$\sqrt{s}=$ 13.6 TeV $t\bar{t}$",
            transform=ax.transAxes,
            fontsize=24,
            va="top",
        )
        if ymax > 0:
            ax.set_ylim(top=ymax * 1.2)

        output_path = output_dir / f"{feature}.png"
        fig.savefig(output_path, dpi=dpi)
        ax.set_yscale("log")
        if np.isfinite(ymin_positive) and ymin_positive > 0 and ymax > 0:
            ax.set_ylim(bottom=ymin_positive / 1.2, top=ymax * 3.0)
        output_path_log = output_dir / f"{feature}_log.png"
        fig.savefig(output_path_log, dpi=dpi)
        plt.close(fig)


def compute_numeric_bin_edges(
    vmin: float, vmax: float, dtype: np.dtype, bins: int
) -> np.ndarray:
    if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
        return np.empty(0)

    if np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.bool_):
        lo = int(np.floor(vmin))
        hi = int(np.ceil(vmax))
        return np.arange(lo - 0.5, hi + 1.5)

    return np.linspace(vmin, vmax, bins + 1)


def plot_track_histograms(
    tracks_ds: np.ndarray,
    jet_labels: np.ndarray,
    output_dir: Path,
    bins: int,
    density: bool,
    dpi: int,
    dataset_name: str,
    chunk_size: int,
    max_jets: Optional[int],
) -> None:
    """Utility to plot histograms split by jet flavour."""
    ensure_output_dir(output_dir)
    if tracks_ds.shape[0] != len(jet_labels):
        raise ValueError(
            "Tracks dataset length does not match number of jets; cannot map tracks to jet labels."
        )

    if max_jets is not None:
        max_jets = int(max_jets)
        if max_jets <= 0:
            raise ValueError("--track-max-jets must be positive.")
        slice_length = min(max_jets, len(jet_labels))
    else:
        slice_length = len(jet_labels)

    if slice_length == 0:
        return

    jet_labels = jet_labels[:slice_length]

    available_fields = tracks_ds.dtype.names or ()
    feature_list = [name for name in TRACK_FEATURES if name in available_fields]
    if not feature_list:
        raise ValueError("No expected track features were found in the tracks table.")

    for feature in feature_list:
        if feature == "valid":
            continue

        vmin = np.inf
        vmax = -np.inf

        for start in range(0, slice_length, chunk_size):
            end = min(start + chunk_size, slice_length)
            values = tracks_ds[feature][start:end]
            valid_mask = tracks_ds["valid"][start:end].astype(bool)
            sample = np.asarray(values[valid_mask])
            if sample.size == 0:
                continue
            sample = sample[np.isfinite(sample)]
            if sample.size == 0:
                continue
            vmin = min(vmin, float(sample.min()))
            vmax = max(vmax, float(sample.max()))

        bin_edges = compute_numeric_bin_edges(
            vmin, vmax, tracks_ds.dtype[feature], bins
        )
        if bin_edges.size == 0:
            continue

        bin_widths = np.diff(bin_edges)
        counts = {
            pdg: np.zeros(len(bin_edges) - 1, dtype=np.float64)
            for pdg in FLAVOUR_LABELS
        }
        totals = {pdg: 0 for pdg in FLAVOUR_LABELS}

        for start in range(0, slice_length, chunk_size):
            end = min(start + chunk_size, slice_length)
            values = tracks_ds[feature][start:end]
            valid_mask = tracks_ds["valid"][start:end].astype(bool)
            labels_chunk = jet_labels[start:end]

            for pdg in FLAVOUR_LABELS:
                jet_mask = labels_chunk == pdg
                if not np.any(jet_mask):
                    continue
                feature_slice = np.asarray(values[jet_mask])
                valid_slice = valid_mask[jet_mask]
                if feature_slice.size == 0:
                    continue
                sample = np.asarray(feature_slice[valid_slice], dtype=np.float64)
                sample = sample[np.isfinite(sample)]
                if sample.size == 0:
                    continue
                hist, _ = np.histogram(sample, bins=bin_edges)
                counts[pdg] += hist
                totals[pdg] += int(sample.size)

        fig, ax = plt.subplots()
        bin_centers = bin_edges[:-1] + 0.5 * bin_widths
        ymax = 0.0
        ymin_positive = np.inf

        for pdg, label_name in FLAVOUR_LABELS.items():
            if totals[pdg] == 0:
                continue
            heights = counts[pdg].copy()
            if density:
                heights /= totals[pdg]
                heights /= bin_widths
            ax.step(
                bin_centers,
                heights,
                where="mid",
                label=f"{label_name}",
                linewidth=2,
            )
            if heights.size:
                ymax = max(ymax, float(np.max(heights)))
                positives = heights[heights > 0]
                if positives.size:
                    ymin_positive = min(ymin_positive, float(np.min(positives)))

        if not ax.has_data():
            plt.close(fig)
            continue

        ax.set_xlabel(feature)
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

        output_path = output_dir / f"{feature}.png"
        fig.savefig(output_path, dpi=dpi)
        ax.set_yscale("log")
        if np.isfinite(ymin_positive) and ymin_positive > 0 and ymax > 0:
            ax.set_ylim(bottom=ymin_positive / 1.2, top=ymax * 1.4)
        output_path_log = output_dir / f"{feature}_log.png"
        fig.savefig(output_path_log, dpi=dpi)
        plt.close(fig)
