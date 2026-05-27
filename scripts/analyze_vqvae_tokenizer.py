"""Make post-training diagnostics for an event/object VQ-VAE tokenizer."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from heptokens.data.atlas_event_mappable import AtlasEventMapDataset
from heptokens.models.vq_vae import LitVqVae

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger(__name__)


DATAMODULE_KEYS = {
    "_target_",
    "data_path",
    "data_paths",
    "train_frac",
    "val_frac",
    "test_frac",
    "seed",
    "n_classes",
    "num_workers",
    "batch_size",
    "pin_memory",
    "persistent_workers",
    "multiprocessing_context",
    "transforms",
}

DEFAULT_OBJECT_FEATURES = {
    "jets": [
        "pt",
        "eta",
        "phi",
        "mass",
        "n_trk",
        "QG_nTracks",
        "QG_tracksWidth",
        "QG_tracksC1",
        "DL1d_pb",
        "DL1d_pc",
        "DL1d_pu",
        "GN2_pb",
        "GN2_pc",
        "GN2_pu",
    ],
    "electrons": [
        "pt",
        "eta",
        "phi",
        "charge",
        "trk_iso03",
        "LHLoose",
        "LHMedium",
        "LHTight",
    ],
    "muons": ["pt", "eta", "phi", "charge", "trk_iso03", "quality", "muonType"],
    "taus": [
        "pt",
        "eta",
        "phi",
        "charge",
        "is_1prong",
        "NNDecayMode",
        "RNNJetScore",
        "RNNEleScore",
    ],
    "photons": ["pt", "eta", "phi", "trk_iso03", "isLoose", "isTight"],
}

FEATURE_LABELS = {
    "pt": r"$p_T$",
    "eta": r"$\eta$",
    "phi": r"$\phi$",
    "mass": "mass",
    "met": "MET",
    "sumet": r"$\Sigma E_T$",
    "n_trk": "n tracks",
    "qg_ntracks": "QG nTracks",
    "qg_trackswidth": "QG tracksWidth",
    "qg_tracksc1": "QG tracksC1",
    "dl1d_pb": r"DL1d $p_b$",
    "dl1d_pc": r"DL1d $p_c$",
    "dl1d_pu": r"DL1d $p_u$",
    "gn2_pb": r"GN2 $p_b$",
    "gn2_pc": r"GN2 $p_c$",
    "gn2_pu": r"GN2 $p_u$",
    "trk_iso03": "track isolation",
    "lhloose": "LHLoose",
    "lhmedium": "LHMedium",
    "lhtight": "LHTight",
    "quality": "quality",
    "muontype": "muonType",
    "is_1prong": "is_1prong",
    "nndecaymode": "NNDecayMode",
    "rnnjetscore": "RNNJetScore",
    "rnnelescore": "RNNEleScore",
    "isloose": "isLoose",
    "istight": "isTight",
    "charge": "charge",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create reconstruction and codebook-usage plots from a trained "
            "heptokens VQ-VAE tokenizer run."
        )
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Training run directory containing full_config.yaml and checkpoints/.",
    )
    parser.add_argument(
        "--checkpoint",
        help="Checkpoint to analyze. Defaults to best.ckpt, then last.ckpt.",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for plots. Defaults to <run-dir>/figures/tokenizer_analysis.",
    )
    parser.add_argument(
        "--num-events-per-file",
        type=int,
        default=20_000,
        help="Cap events loaded per H5 file for diagnostics.",
    )
    parser.add_argument(
        "--max-valid-objects",
        type=int,
        default=200_000,
        help="Stop after this many valid reconstructed objects.",
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--h5-files",
        nargs="+",
        help="Optional H5 files to analyze instead of files stored in full_config.yaml.",
    )
    return parser.parse_args()


def choose_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def find_checkpoint(run_dir: Path, checkpoint: str | None) -> Path:
    if checkpoint is not None:
        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    for name in ("best.ckpt", "last.ckpt"):
        path = run_dir / "checkpoints" / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No best.ckpt or last.ckpt found in {run_dir / 'checkpoints'}")


def dataset_kwargs_from_cfg(
    cfg,
    args: argparse.Namespace,
    run_dir: Path,
) -> tuple[list[str], dict]:
    datamodule = OmegaConf.to_container(cfg.datamodule, resolve=True)
    if args.h5_files:
        data_paths = list(args.h5_files)
    elif datamodule.get("data_paths"):
        data_paths = list(datamodule["data_paths"])
    else:
        data_paths = [datamodule["data_path"]]

    data_paths = [
        str((run_dir / path).resolve()) if not Path(path).is_absolute() else str(path)
        for path in data_paths
    ]

    dataset_kwargs = {
        key: value for key, value in datamodule.items() if key not in DATAMODULE_KEYS
    }
    dataset_kwargs["num_events"] = args.num_events_per_file
    return data_paths, dataset_kwargs


def feature_names_from_cfg(cfg, n_features: int) -> list[str]:
    datamodule = OmegaConf.to_container(cfg.datamodule, resolve=True)
    output_mode = datamodule.get("output_mode")
    collections = datamodule.get("object_collections") or []
    object_type = datamodule.get("object_type")

    inputs = []
    if output_mode == "object":
        for collection in collections:
            if collection.get("object_name") == object_type:
                inputs = collection.get("inputs") or []
                break
    elif collections:
        inputs = collections[0].get("inputs") or []

    names = [Path(path).name for path in inputs]
    if not names and object_type in DEFAULT_OBJECT_FEATURES:
        names = DEFAULT_OBJECT_FEATURES[object_type]
    if len(names) != n_features:
        if len(names) > n_features:
            names = names[:n_features]
        else:
            names = names + [f"feature_{idx}" for idx in range(len(names), n_features)]
    return names


def object_name_from_cfg(cfg) -> str:
    datamodule = OmegaConf.to_container(cfg.datamodule, resolve=True)
    if datamodule.get("output_mode") == "object" and datamodule.get("object_type"):
        return str(datamodule["object_type"]).rstrip("s")
    return "object"


def to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def collect_diagnostics(
    *,
    model: LitVqVae,
    data_paths: list[str],
    dataset_kwargs: dict,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    max_valid_objects: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    originals = []
    recons = []
    all_indices = []
    n_seen = 0

    with torch.no_grad():
        for data_path in data_paths:
            log.info("Analyzing %s", data_path)
            dataset = AtlasEventMapDataset(data_path, **dataset_kwargs)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False,
            )
            for batch in loader:
                batch = to_device(batch, device)
                z_q, indices, _ = model.encode(batch)
                recon = model.decode(z_q, batch)

                mask = batch["mask"].bool()
                original_valid = batch["csts"][mask].detach().cpu().float().numpy()
                recon_valid = recon[mask].detach().cpu().float().numpy()
                indices_valid = indices[mask].detach().cpu().long().numpy()

                originals.append(original_valid)
                recons.append(recon_valid)
                all_indices.append(indices_valid)
                n_seen += len(original_valid)
                if n_seen >= max_valid_objects:
                    original = np.concatenate(originals, axis=0)[:max_valid_objects]
                    reconstruction = np.concatenate(recons, axis=0)[:max_valid_objects]
                    code_indices = np.concatenate(all_indices, axis=0)[:max_valid_objects]
                    return original, reconstruction, code_indices, n_seen

    if not originals:
        raise RuntimeError("No valid objects found. Check the object mask and input paths.")
    return (
        np.concatenate(originals, axis=0),
        np.concatenate(recons, axis=0),
        np.concatenate(all_indices, axis=0),
        n_seen,
    )


def apply_hep_style(ax) -> None:
    ax.tick_params(direction="in", which="both", top=True, right=True, width=1.2, labelsize=13)
    ax.tick_params(which="major", length=7)
    ax.tick_params(which="minor", length=3.5)
    ax.minorticks_on()
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)


def feature_label(name: str) -> str:
    lower = Path(name).name.lower()
    if lower in FEATURE_LABELS:
        return FEATURE_LABELS[lower]
    return name.replace("_", " ")


def safe_filename(text: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in text)


def display_values(
    original: np.ndarray,
    reconstruction: np.ndarray,
    feature_name: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    label = feature_label(feature_name)
    lower = Path(feature_name).name.lower()
    scale = 1.0
    unit = ""
    if lower in {"pt", "met", "sumet", "mass"} and np.nanpercentile(original, 95) > 1000:
        scale = 1000.0
        unit = " [GeV]"
    return original / scale, reconstruction / scale, f"{label}{unit}"


def finite_pair(original: np.ndarray, reconstruction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(original) & np.isfinite(reconstruction)
    return original[finite], reconstruction[finite]


def histogram_bins(values: np.ndarray, n_bins: int = 75) -> np.ndarray:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.linspace(0.0, 1.0, n_bins)

    unique = np.unique(values)
    integer_like = np.allclose(unique, np.round(unique), atol=1e-6)
    if len(unique) <= 20 and integer_like:
        lo = int(np.floor(unique.min()))
        hi = int(np.ceil(unique.max()))
        return np.arange(lo - 0.5, hi + 1.5, 1.0)

    lo, hi = np.percentile(values, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(values.min()), float(values.max())
    if lo == hi:
        width = abs(lo) * 0.05 if lo != 0 else 1.0
        lo, hi = lo - width, hi + width
    return np.linspace(lo, hi, n_bins)


def plot_feature_triptychs(
    original: np.ndarray,
    reconstruction: np.ndarray,
    feature_names: list[str],
    object_name: str,
    output_dir: Path,
) -> None:
    rng = np.random.default_rng(42)
    for feature_idx, feature_name in enumerate(feature_names):
        if feature_idx >= original.shape[1] or feature_idx >= reconstruction.shape[1]:
            log.warning("Skipping %s: feature is not in the reconstructed arrays", feature_name)
            continue
        orig, reco, axis_label = display_values(
            original[:, feature_idx],
            reconstruction[:, feature_idx],
            feature_name,
        )
        orig, reco = finite_pair(orig, reco)
        if len(orig) == 0:
            log.warning("Skipping %s: no finite values", feature_name)
            continue

        combined = np.concatenate([orig, reco])
        bins = histogram_bins(combined, n_bins=70)
        residual = reco - orig
        res_bins = histogram_bins(residual, n_bins=70)

        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
        ax = axes[0]
        ax.hist(
            orig,
            bins=bins,
            histtype="step",
            density=False,
            linewidth=1.9,
            color="#1f77b4",
            label="original",
        )
        ax.hist(
            reco,
            bins=bins,
            histtype="step",
            density=False,
            linewidth=1.9,
            color="#ff7f0e",
            label="reconstructed",
        )
        ax.set_title(f"{object_name} {feature_label(feature_name)}", fontsize=16)
        ax.set_xlabel(axis_label, fontsize=13)
        ax.set_ylabel("objects", fontsize=13)
        ax.legend(frameon=True, fontsize=11)
        apply_hep_style(ax)

        ax = axes[1]
        ax.hist(residual, bins=res_bins, color="#1f77b4", alpha=0.75)
        ax.axvline(0, color="black", linewidth=1.2)
        ax.set_title("residual", fontsize=16)
        ax.set_xlabel("reco - original", fontsize=13)
        ax.set_ylabel("objects", fontsize=13)
        apply_hep_style(ax)

        ax = axes[2]
        n_plot = min(len(orig), 50_000)
        idx = rng.choice(len(orig), size=n_plot, replace=False)
        ax.scatter(orig[idx], reco[idx], s=3, alpha=0.22, rasterized=True)
        line_lo, line_hi = histogram_bins(np.concatenate([orig[idx], reco[idx]])).take([0, -1])
        ax.plot([line_lo, line_hi], [line_lo, line_hi], color="black", linewidth=1.2)
        ax.set_xlim(line_lo, line_hi)
        ax.set_ylim(line_lo, line_hi)
        ax.set_title("correlation", fontsize=16)
        ax.set_xlabel("original", fontsize=13)
        ax.set_ylabel("reconstructed", fontsize=13)
        apply_hep_style(ax)

        fig.tight_layout()
        fig.savefig(
            output_dir / f"{safe_filename(feature_name)}_reconstruction_triptych.png",
            dpi=180,
        )
        plt.close(fig)


def plot_feature_overlay(
    original: np.ndarray,
    reconstruction: np.ndarray,
    feature_names: list[str],
    object_name: str,
    output_dir: Path,
) -> None:
    for feature_idx, feature_name in enumerate(feature_names):
        if feature_idx >= original.shape[1] or feature_idx >= reconstruction.shape[1]:
            log.warning("Skipping %s: feature is not in the reconstructed arrays", feature_name)
            continue
        orig, reco, axis_label = display_values(
            original[:, feature_idx],
            reconstruction[:, feature_idx],
            feature_name,
        )
        orig, reco = finite_pair(orig, reco)
        if len(orig) == 0:
            log.warning("Skipping %s: no finite values", feature_name)
            continue
        combined = np.concatenate([orig, reco])
        bins = histogram_bins(combined, n_bins=75)

        fig, ax = plt.subplots(figsize=(8.0, 6.0))
        ax.hist(
            orig,
            bins=bins,
            histtype="step",
            density=True,
            linewidth=2.0,
            color="black",
            label="Original",
        )
        ax.hist(
            reco,
            bins=bins,
            histtype="step",
            density=True,
            linewidth=2.0,
            color="#00A000",
            label="VQVAE",
        )
        ax.set_title(f"{object_name.capitalize()} {feature_label(feature_name)}", fontsize=17)
        ax.set_xlabel(axis_label, fontsize=14)
        ax.set_ylabel("Normalized", fontsize=14)
        ax.legend(frameon=False, fontsize=12)
        apply_hep_style(ax)
        fig.tight_layout()
        fig.savefig(output_dir / f"{safe_filename(feature_name)}_overlay.png", dpi=180)
        plt.close(fig)


def codebook_counts(indices: np.ndarray, codebook_size: int) -> np.ndarray:
    n_quantizers = indices.shape[1]
    counts = np.zeros((n_quantizers, codebook_size), dtype=np.int64)
    for quantizer_idx in range(n_quantizers):
        values = indices[:, quantizer_idx]
        values = values[values >= 0]
        counts[quantizer_idx] = np.bincount(values, minlength=codebook_size)[:codebook_size]
    return counts


def plot_codebook_usage(counts: np.ndarray, output_path: Path) -> None:
    n_quantizers = counts.shape[0]
    fig, axes = plt.subplots(n_quantizers, 1, figsize=(10, 2.6 * n_quantizers), sharex=True)
    axes = np.atleast_1d(axes)
    for quantizer_idx, ax in enumerate(axes):
        ax.bar(np.arange(counts.shape[1]), counts[quantizer_idx], width=1.0)
        used = int(np.count_nonzero(counts[quantizer_idx]))
        dead = int(counts.shape[1] - used)
        ax.set_ylabel(f"q{quantizer_idx}")
        ax.set_title(f"quantizer {quantizer_idx}: used={used}, dead={dead}")
        ax.grid(axis="y", alpha=0.2)
    axes[-1].set_xlabel("code index")
    fig.suptitle("Codebook usage frequency", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_codebook_frequency_hist(
    counts: np.ndarray,
    object_name: str,
    output_path: Path,
) -> None:
    flat_counts = counts.reshape(-1)
    used_fraction = np.count_nonzero(flat_counts) / len(flat_counts)
    hi = np.percentile(flat_counts, 99.5)
    if hi <= 0:
        hi = max(1, flat_counts.max())
    bins = np.linspace(0, hi, 80)

    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    ax.hist(flat_counts, bins=bins, color="#1f77b4", alpha=0.78)
    ax.set_title(f"{object_name} codebook: {100 * used_fraction:.1f}% used", fontsize=20)
    ax.set_xlabel("token frequency", fontsize=16)
    ax.set_ylabel("tokens", fontsize=16)
    apply_hep_style(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_dead_tokens(counts: np.ndarray, output_path: Path) -> None:
    used = np.count_nonzero(counts, axis=1)
    dead = counts.shape[1] - used
    labels = [f"q{idx}" for idx in range(counts.shape[0])]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, used, label="used", color="#315C9E")
    ax.bar(labels, dead, bottom=used, label="dead", color="#B84A4A")
    ax.set_ylabel("codes")
    ax.set_title("Used vs dead codes")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def reconstruction_metrics(
    original: np.ndarray,
    reconstruction: np.ndarray,
    feature_names: list[str],
) -> dict:
    metrics = {}
    for idx, name in enumerate(feature_names):
        orig, reco = finite_pair(original[:, idx], reconstruction[:, idx])
        if len(orig) == 0:
            continue
        residual = reco - orig
        metrics[name] = {
            "n": int(len(orig)),
            "mean_original": float(np.mean(orig)),
            "mean_reconstructed": float(np.mean(reco)),
            "bias": float(np.mean(residual)),
            "std": float(np.std(residual)),
            "mae": float(np.mean(np.abs(residual))),
            "rmse": float(np.sqrt(np.mean(residual**2))),
            "p16": float(np.percentile(residual, 16)),
            "p50": float(np.percentile(residual, 50)),
            "p84": float(np.percentile(residual, 84)),
        }
    return metrics


def codebook_summary(counts: np.ndarray) -> dict:
    summary = {}
    for quantizer_idx, quantizer_counts in enumerate(counts):
        used = int(np.count_nonzero(quantizer_counts))
        dead = int(len(quantizer_counts) - used)
        assignments = int(quantizer_counts.sum())
        used_counts = quantizer_counts[quantizer_counts > 0]
        summary[f"quantizer_{quantizer_idx}"] = {
            "assignments": assignments,
            "used_codes": used,
            "dead_codes": dead,
            "total_codes": int(len(quantizer_counts)),
            "used_fraction": float(used / len(quantizer_counts)),
            "percent_used": float(100 * used / len(quantizer_counts)),
            "max_frequency": int(quantizer_counts.max()) if len(quantizer_counts) else 0,
            "mean_frequency_used": float(used_counts.mean()) if len(used_counts) else 0.0,
        }
    return summary


def plot_residual_summary(metrics: dict, output_path: Path) -> None:
    if not metrics:
        return
    names = list(metrics)
    rmse = [metrics[name]["rmse"] for name in names]
    mae = [metrics[name]["mae"] for name in names]
    bias = [metrics[name]["bias"] for name in names]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, values, title in [
        (axes[0], rmse, "RMSE"),
        (axes[1], mae, "MAE"),
        (axes[2], bias, "bias"),
    ]:
        ax.bar(names, values, color="#4C78A8")
        ax.axhline(0, color="black", linewidth=1.1)
        ax.set_title(title, fontsize=15)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=35)
        apply_hep_style(ax)
    fig.suptitle("Reconstruction Residual Summary", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary(
    output_path: Path,
    *,
    counts: np.ndarray,
    original: np.ndarray,
    reconstruction: np.ndarray,
    feature_names: list[str],
    n_seen: int,
) -> None:
    lines = []
    lines.append(f"valid_objects_analyzed: {len(original)}")
    lines.append(f"valid_objects_seen_before_stop: {n_seen}")
    lines.append("")
    lines.append("reconstruction:")
    for idx, name in enumerate(feature_names):
        diff = reconstruction[:, idx] - original[:, idx]
        lines.append(f"  {name}:")
        lines.append(f"    mae: {np.mean(np.abs(diff)):.8g}")
        lines.append(f"    rmse: {np.sqrt(np.mean(diff**2)):.8g}")
        lines.append(f"    bias: {np.mean(diff):.8g}")
    lines.append("")
    lines.append("codebook:")
    for quantizer_idx, quantizer_counts in enumerate(counts):
        used = int(np.count_nonzero(quantizer_counts))
        dead = int(len(quantizer_counts) - used)
        total = int(quantizer_counts.sum())
        lines.append(f"  quantizer_{quantizer_idx}:")
        lines.append(f"    total_assignments: {total}")
        lines.append(f"    used_codes: {used}")
        lines.append(f"    dead_codes: {dead}")
        lines.append(f"    used_fraction: {used / len(quantizer_counts):.6f}")
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    run_dir = Path(args.run_dir).resolve()
    cfg_path = run_dir / "full_config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)

    cfg = OmegaConf.load(cfg_path)
    data_paths, dataset_kwargs = dataset_kwargs_from_cfg(cfg, args, run_dir)
    checkpoint = find_checkpoint(run_dir, args.checkpoint)
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else run_dir / "figures" / "tokenizer_analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    log.info("Loading %s", checkpoint)
    model = LitVqVae.load_from_checkpoint(checkpoint, map_location=device)
    model.to(device)
    model.eval()

    original, reconstruction, indices, n_seen = collect_diagnostics(
        model=model,
        data_paths=data_paths,
        dataset_kwargs=dataset_kwargs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        max_valid_objects=args.max_valid_objects,
    )

    feature_names = feature_names_from_cfg(cfg, original.shape[1])
    object_name = object_name_from_cfg(cfg)
    codebook_size = int(getattr(model.hparams, "codebook_size", cfg.model.codebook_size))
    counts = codebook_counts(indices, codebook_size)
    reco_metrics = reconstruction_metrics(original, reconstruction, feature_names)
    usage_summary = codebook_summary(counts)

    np.save(output_dir / "codebook_counts.npy", counts)
    (output_dir / "reconstruction_metrics.json").write_text(
        json.dumps(reco_metrics, indent=2) + "\n"
    )
    (output_dir / "codebook_usage_summary.json").write_text(
        json.dumps(usage_summary, indent=2) + "\n"
    )
    plot_feature_triptychs(
        original,
        reconstruction,
        feature_names,
        object_name,
        output_dir,
    )
    plot_feature_overlay(
        original,
        reconstruction,
        feature_names,
        object_name,
        output_dir,
    )
    plot_codebook_usage(counts, output_dir / "codebook_frequency.png")
    plot_codebook_frequency_hist(
        counts,
        object_name,
        output_dir / "codebook_token_frequency_hist.png",
    )
    plot_dead_tokens(counts, output_dir / "dead_tokens.png")
    plot_residual_summary(reco_metrics, output_dir / "residual_summary.png")
    write_summary(
        output_dir / "summary.txt",
        counts=counts,
        original=original,
        reconstruction=reconstruction,
        feature_names=feature_names,
        n_seen=n_seen,
    )
    log.info("Wrote tokenizer diagnostics to %s", output_dir)


if __name__ == "__main__":
    main()
