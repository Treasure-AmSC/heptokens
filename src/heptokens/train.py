"""Training entry point — usable as `python -m heptokens.train` or `heptokens-train`."""

import logging
import warnings
from pathlib import Path

import hydra
import lightning.pytorch as pl
import torch as T
from omegaconf import DictConfig

from heptokens.utils.hydra import (
    instantiate_collection,
    log_hyperparameters,
    print_config,
    reload_original_config,
    save_config,
    save_declaration,
)

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message="Failed to load image Python extension")


def train(cfg: DictConfig) -> None:
    """Training logic — importable by experiment repos with their own Hydra entrypoints."""
    log.info("Setting up full job config")

    if cfg.full_resume:
        log.info("Attempting to resume previous job")
        old_cfg = reload_original_config(ckpt_flag=cfg.ckpt_flag)
        if old_cfg is not None:
            cfg = old_cfg
    print_config(cfg)

    log.info(f"Setting seed to: {cfg.seed}")
    pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Setting matrix precision to: {cfg.precision}")
    T.set_float32_matmul_precision(cfg.precision)

    log.info("Instantiating the data module")
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    log.info("Instantiating the model")
    if cfg.weight_ckpt_path:
        log.info(f"Loading model weights from checkpoint: {cfg.ckpt_path}")
        model_class = hydra.utils.get_class(cfg.model._target_)
        model = model_class.load_from_checkpoint(cfg.ckpt_path, map_location="cpu")
    else:
        model = hydra.utils.instantiate(
            cfg.model,
            data_sample=datamodule.get_data_sample(),
            n_classes=datamodule.get_n_classes(),
        )

    if cfg.compile:
        log.info(f"Compiling the model using torch 2.0: {cfg.compile}")
        model = T.compile(model, mode=cfg.compile)

    log.info("Instantiating all callbacks")
    callbacks = instantiate_collection(cfg.callbacks)

    log.info("Instantiating the logger")
    logger = hydra.utils.instantiate(cfg.logger)

    log.info("Instantiating the trainer")
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    log.info("Logging all hyperparameters")
    log_hyperparameters(cfg, model, trainer)
    log.info(model)

    log.info("Saving config so job can be resumed")
    save_config(cfg)

    if cfg.get("test_only", False):
        log.info("Running test-only evaluation via validate on test data")
        for cb in trainer.callbacks:
            if hasattr(cb, "save_metrics"):
                cb.save_metrics = True
            if hasattr(cb, "save_predictions"):
                cb.save_predictions = True
        datamodule.setup(stage="test")
        trainer.validate(model, dataloaders=datamodule.test_dataloader(), ckpt_path=cfg.ckpt_path)
        save_declaration(filename=str(Path("..") / "TEST_SUCCESS.txt"))
    else:
        log.info("Starting training!")
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

        log.info("Checking if training finished correctly")
        if trainer.state.status == "finished":
            log.info(" -- YES!! -- ")
            save_declaration()


@hydra.main(version_base=None, config_path="pkg://heptokens.conf", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
