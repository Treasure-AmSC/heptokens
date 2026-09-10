"""CLI entry point for fitting preprocessing scalers — usable as `heptokens-fit-preprocessors`."""

import logging
from pathlib import Path

import hydra
import joblib
import numpy as np
from omegaconf import DictConfig

from heptokens.data.transforms import create_preprocessing_transformer

log = logging.getLogger(__name__)


def _resolve_indices(indices, feature_names: list[str] | None, label: str) -> list[int] | None:
    if not indices:
        return None
    resolved = []
    for idx in indices:
        if isinstance(idx, str):
            if feature_names is None:
                raise ValueError(f"{label}: got feature name '{idx}' but {label}_features not set")
            resolved.append(list(feature_names).index(idx))
        else:
            resolved.append(int(idx))
    return resolved


def fit_preprocessors(cfg: DictConfig) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Instantiating the data module")
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup("fit")

    log.info("Collecting training data")
    cst_batches, jet_batches = [], []
    loader = datamodule.train_dataloader()
    for i, batch in enumerate(loader):
        if cfg.max_batches and i >= cfg.max_batches:
            break
        csts = batch["csts"]
        mask = batch["mask"]
        cst_batches.append(csts[mask].numpy())
        if cfg.fit_jets and "jets" in batch:
            jet_batches.append(batch["jets"].numpy())

    cst_array = np.concatenate(cst_batches, axis=0)
    log.info(f"Collected {len(cst_array)} constituent samples, {cst_array.shape[1]} features")

    saved_files: dict[str, str] = {}

    cst_log_indices = _resolve_indices(
        cfg.get("cst_log_feature_indices"), cfg.get("cst_features"), "cst"
    )

    for mode in cfg.cst_modes:
        if mode in ("log_quantile", "log_standard") and not cst_log_indices:
            log.warning(f"Skipping cst mode '{mode}': cst_log_feature_indices not set")
            continue
        transformer = create_preprocessing_transformer(
            mode=mode,
            log_feature_indices=cst_log_indices,
            n_quantiles=cfg.n_quantiles,
            log_offset=cfg.log_offset,
            n_features=cst_array.shape[1],
        )
        transformer.fit(cst_array)
        path = output_dir / f"cst_{mode}.joblib"
        joblib.dump(transformer, path)
        saved_files[f"cst_{mode}"] = str(path.resolve())
        log.info(f"Saved {path}")

    if jet_batches:
        jet_array = np.concatenate(jet_batches, axis=0)
        log.info(f"Collected {len(jet_array)} jet samples, {jet_array.shape[1]} features")
        jet_log_indices = _resolve_indices(
            cfg.get("jet_log_feature_indices"), cfg.get("jet_features"), "jet"
        )

        for mode in cfg.jet_modes:
            if mode in ("log_quantile", "log_standard") and not jet_log_indices:
                log.warning(f"Skipping jet mode '{mode}': jet_log_feature_indices not set")
                continue
            transformer = create_preprocessing_transformer(
                mode=mode,
                log_feature_indices=jet_log_indices,
                n_quantiles=cfg.n_quantiles,
                log_offset=cfg.log_offset,
                n_features=jet_array.shape[1],
            )
            transformer.fit(jet_array)
            path = output_dir / f"jet_{mode}.joblib"
            joblib.dump(transformer, path)
            saved_files[f"jet_{mode}"] = str(path.resolve())
            log.info(f"Saved {path}")

    _write_config_snippet(output_dir, saved_files)


def _write_config_snippet(output_dir: Path, saved_files: dict[str, str]) -> None:
    lines = [
        "# Copy the paths below into your datamodule config.",
        "# Load with: joblib.load('<path>') and pass as cst_fn / jet_fn to preprocess_batch.",
        "#",
    ]
    for key, path in saved_files.items():
        lines.append(f"# {key}: {path}")

    snippet_path = output_dir / "preprocessor_config.yaml"
    snippet_path.write_text("\n".join(lines) + "\n")
    log.info(f"Saved config snippet to {snippet_path}")


@hydra.main(
    version_base=None,
    config_path="pkg://heptokens.conf",
    config_name="fit_preprocessors.yaml",
)
def main(cfg: DictConfig) -> None:
    fit_preprocessors(cfg)


if __name__ == "__main__":
    main()
