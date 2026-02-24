"""Compare ROC curves across feature classifiers and token (VQ-VAE) classifiers.

For each preprocessing configuration (e.g. quantile_500), loads both a
feature_classifier and a token classifier checkpoint, runs inference on the
validation split, and saves a HEP-style signal efficiency vs background
rejection plot.

Usage
-----
pixi run python scripts/compare_roc.py \
    --results_dir results/preprocessing_better \
    --data_path /sdf/scratch/users/s/samklein/data/atlas/mc-flavtag-ttbar-small.h5

One PDF per preprocessing is written to --output_dir
(default: {results_dir}/roc_comparison/).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import torch as T
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import auc, roc_curve

# Register custom OmegaConf resolvers (int_div, min, max, if, …) used in saved configs
import qhep.utils.hydra as _qhep_hydra  # noqa: F401  (side-effect import)
from qhep.utils.hydra import reload_original_config

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------
# MapDataset remaps sorted unique PDG IDs to contiguous indices.
# For the ATLAS ttbar dataset with n_classes=4 the classes are:
#   PDG 0  -> index 0  (light-flavour jets)
#   PDG 4  -> index 1  (charm jets)
#   PDG 5  -> index 2  (b jets)
#   PDG 15 -> index 3  (tau jets)
# This map is used only for axis labels; it does not affect inference.
DEFAULT_CLASS_NAMES: dict[int, str] = {
    0: "light",
    1: "c",
    2: "b",
    3: r"$\tau$",
}


def get_class_names(
    data_path: str, label_key: str = "HadronConeExclTruthLabelID"
) -> dict[int, str]:
    """Derive the class-index-to-name mapping from the raw HDF5 label column.

    Falls back to DEFAULT_CLASS_NAMES on any error.
    """
    pdg_to_name = {0: "light", 4: "c", 5: "b", 15: r"$\tau$"}
    try:
        with h5py.File(data_path, "r") as f:
            raw = f["jets"][label_key][:10_000]
        unique_pdg = sorted(np.unique(raw).tolist())
        return {i: pdg_to_name.get(int(v), str(int(v))) for i, v in enumerate(unique_pdg)}
    except Exception as exc:
        log.warning("Could not derive class names from data: %s. Using defaults.", exc)
        return DEFAULT_CLASS_NAMES


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def override_data_path(cfg: DictConfig, data_path: str) -> DictConfig:
    """Return a copy of cfg with the datamodule data_path overridden."""
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg["datamodule"]["data_path"] = data_path
    return OmegaConf.create(cfg)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def run_inference(
    cfg: DictConfig,
    data_path: str,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Instantiate datamodule + model from saved config, load weights, run
    inference over the full validation split.

    ``cfg.ckpt_path`` must already be set (done by ``load_config``).

    Returns
    -------
    probs : np.ndarray, shape (N, n_classes)  — softmax probabilities
    labels : np.ndarray, shape (N,)            — integer class indices
    """
    import hydra.utils as hu

    cfg = override_data_path(cfg, data_path)

    # ------------------------------------------------------------------
    # Datamodule
    # ------------------------------------------------------------------
    datamodule = hu.instantiate(cfg.datamodule)
    datamodule.setup("fit")

    data_sample = datamodule.get_data_sample()
    n_classes = datamodule.get_n_classes()

    # ------------------------------------------------------------------
    # Model — instantiate with fresh weights then load saved state_dict
    # ------------------------------------------------------------------
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model: T.nn.Module = hu.instantiate(
        model_cfg,
        data_sample=data_sample,
        n_classes=n_classes,
    )

    ckpt_path = cfg.ckpt_path
    if ckpt_path is None:
        raise FileNotFoundError("No checkpoint path found in config (cfg.ckpt_path is None).")
    ckpt = T.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    model = model.to(device)

    # ------------------------------------------------------------------
    # Inference loop over validation DataLoader
    # ------------------------------------------------------------------
    val_loader = datamodule.val_dataloader()
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with T.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) if isinstance(v, T.Tensor) else v for k, v in batch.items()}
            logits = model(batch)  # [B, n_classes]
            probs = T.softmax(logits, dim=-1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels)

    return np.concatenate(all_probs, axis=0), np.concatenate(all_labels, axis=0)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
# Line styles and labels for each model type
MODEL_STYLES: dict[str, dict] = {
    "feature": {"linestyle": "-", "label_suffix": "feature"},
    "token": {"linestyle": "--", "label_suffix": "token"},
}


def plot_comparison(
    preprocess_name: str,
    results: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: Path,
    n_classes: int,
    class_names: dict[int, str] | None = None,
) -> None:
    """Produce a single HEP-style ROC plot comparing feature and token classifiers.

    Parameters
    ----------
    preprocess_name : str
        Used in the figure title and file name.
    results : dict
        ``{"feature": (probs, labels), "token": (probs, labels)}``
    output_path : Path
        Where to write the PDF.
    n_classes : int
        Number of jet classes.
    class_names : dict[int, str], optional
        Maps class index to physics label string.
    """
    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    plt.style.use(hep.style.CMS)
    fig, ax = plt.subplots(figsize=(8, 8))

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for class_idx in range(n_classes):
        color = colors[class_idx % len(colors)]
        class_name = class_names.get(class_idx, str(class_idx))

        for model_key, (probs, labels) in results.items():
            style = MODEL_STYLES.get(model_key, {"linestyle": ":", "label_suffix": model_key})

            y_true = (labels == class_idx).astype(int)
            y_score = probs[:, class_idx]

            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)

            # HEP convention: signal efficiency (TPR) vs background rejection (1/FPR)
            with np.errstate(divide="ignore", invalid="ignore"):
                rejection = np.where(fpr > 0, 1.0 / fpr, np.nan)

            valid = np.isfinite(rejection) & (rejection < 1e4) & (tpr > 0)

            ax.plot(
                tpr[valid],
                rejection[valid],
                color=color,
                linestyle=style["linestyle"],
                linewidth=2,
                label=f"{class_name} — {style['label_suffix']} (AUC={roc_auc:.3f})",
            )

    ax.set_xlabel("Signal Efficiency", fontsize=14)
    ax.set_ylabel("Background Rejection (1/FPR)", fontsize=14)
    ax.set_yscale("log")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([1, 1e4])
    ax.legend(loc="upper right", fontsize=9, ncol=1)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"ROC comparison — {preprocess_name}", fontsize=13)
    hep.cms.label(ax=ax, label="Simulation", data=False, fontsize=12)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved ROC plot → %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare ROC curves: feature classifier vs token classifier."
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        required=True,
        help="Parent directory containing classifier/ and feature_classifier/ subdirs.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the ATLAS HDF5 file used for inference (validation split).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Where to write comparison PDFs. Defaults to {results_dir}/roc_comparison/.",
    )
    parser.add_argument(
        "--preprocessings",
        nargs="+",
        default=None,
        help="Subset of preprocessing names to compare. Defaults to all found.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if T.cuda.is_available() else "cpu",
        help="Torch device for inference.",
    )
    args = parser.parse_args()

    results_dir: Path = args.results_dir.resolve()
    output_dir: Path = (args.output_dir or results_dir / "roc_comparison").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    classifier_dir = results_dir / "classifier"
    feature_classifier_dir = results_dir / "feature_classifier"

    # Discover preprocessing names
    if args.preprocessings:
        preprocessings = args.preprocessings
    else:
        if not classifier_dir.exists():
            raise SystemExit(f"classifier/ subdir not found under {results_dir}")
        preprocessings = sorted(d.name for d in classifier_dir.iterdir() if d.is_dir())

    if not preprocessings:
        raise SystemExit("No preprocessing configurations found.")

    log.info("Comparing preprocessings: %s", preprocessings)

    # Derive class labels from data once
    class_names = get_class_names(args.data_path)
    log.info("Class names: %s", class_names)

    for preprocess in preprocessings:
        log.info("=== %s ===", preprocess)

        tok_result_dir = classifier_dir / preprocess
        feat_result_dir = feature_classifier_dir / preprocess

        tok_cfg = reload_original_config(
            path=str(tok_result_dir),
            file_name="full_config.yaml",
            set_ckpt_path=True,
            ckpt_flag="last",
            set_wandb_resume=False,
        )
        feat_cfg = reload_original_config(
            path=str(feat_result_dir),
            file_name="full_config.yaml",
            set_ckpt_path=True,
            ckpt_flag="last",
            set_wandb_resume=False,
        )
        if tok_cfg is None or feat_cfg is None:
            log.warning(
                "Skipping %s: full_config.yaml not found in one or both result dirs.", preprocess
            )
            continue

        if tok_cfg.ckpt_path is None:
            log.warning("No token classifier checkpoint found for %s, skipping.", preprocess)
            continue
        if feat_cfg.ckpt_path is None:
            log.warning("No feature classifier checkpoint found for %s, skipping.", preprocess)
            continue
        n_classes = int(feat_cfg.datamodule.n_classes)

        results: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        log.info("  [feature] running inference on %s ...", feat_cfg.ckpt_path)
        results["feature"] = run_inference(feat_cfg, args.data_path, args.device)

        log.info("  [token]   running inference on %s ...", tok_cfg.ckpt_path)
        results["token"] = run_inference(tok_cfg, args.data_path, args.device)

        out_pdf = output_dir / f"{preprocess}.pdf"
        plot_comparison(preprocess, results, out_pdf, n_classes, class_names)

    log.info("All done. Plots written to %s", output_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
