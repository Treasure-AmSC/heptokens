"""General utilities for lightning modules."""

import logging
import math
from pathlib import Path

import h5py
import torch as T
from lightning import LightningModule, Trainer
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, OneCycleLR

from .torch_utils import (
    get_sched,
    to_numpy,
)

log = logging.getLogger(__name__)


def save_predictions(
    model: LightningModule,
    datamodule: LightningModule,
    trainer: Trainer,
    file_path: str,
    ckpt_path: str | None = None,
) -> None:
    log.info("Running inference on test set")
    outputs = trainer.predict(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    log.info("Combining predictions across dataset")
    keys = list(outputs[0].keys())
    score_dict = {k: T.vstack([o[k] for o in outputs]) for k in keys}
    score_dict = to_numpy(score_dict)

    log.info("Saving outputs")
    output_dir = Path(file_path, "outputs")
    print(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_dir / "test_set.h5", mode="w") as file:
        for k in keys:
            file.create_dataset(k, data=score_dict[k])


def get_max_steps(model: LightningModule) -> int:
    """Get the maximum number of steps from the model trainer."""
    try:
        log.info("Attempting to get the max steps from the model trainer")
        max_steps = model.trainer.max_steps
        if max_steps < 1:
            steps_per_epoch = len(model.trainer.datamodule.train_dataloader())
            max_epochs = model.trainer.max_epochs
            max_steps = steps_per_epoch * max_epochs
        log.info(f"Success:  max_steps = {max_steps}")
    except Exception as e:
        log.info(f"Failed to get max steps from the model trainer: {e}")
        max_steps = 0
    return max_steps


def linear_warmup(
    optimizer: Optimizer,
    model: LightningModule,  # noqa: ARG001
    warmup_steps: int = 1000,
    init_factor: float = 1e-2,
) -> LambdaLR:
    """Return a scheduler with a linear warmup."""

    def fn(x: int) -> float:
        return min(1, init_factor + x * (1 - init_factor) / max(1, warmup_steps))

    return LambdaLR(optimizer, fn)


def linear_warmup_exp_decay(
    optimizer: Optimizer,
    model: LightningModule,  # noqa: ARG001
    warmup_steps: int = 1000,
    half_life: int = 1000,
    final_factor: float = 1e-3,
    init_factor: float = 1e-1,
) -> LambdaLR:
    """Return a scheduler with a linear warmup and a sqrt decay."""

    def fn(x: int) -> float:
        if x < warmup_steps:
            return init_factor + x * (1 - init_factor) / max(1, warmup_steps)
        decay = -math.log(2) / half_life
        return max(math.exp(decay * (x - warmup_steps)), final_factor)

    return LambdaLR(optimizer, fn)


def linear_warmup_cosine_decay(
    optimizer: Optimizer,
    warmup_steps: int = 1000,
    total_steps: int = 10000,
    final_factor: float = 5e-2,
    init_factor: float = 1e-5,
    model: LightningModule | None = None,
) -> LambdaLR:
    """Return a scheduler with a linear warmup and a cosine decay."""
    # Attempt to get the max steps from the model trainer
    if total_steps == -1 and model is not None:
        total_steps = get_max_steps(model)

    warmup_steps = max(1, warmup_steps)  # Avoid division by zero
    assert 0 < final_factor < 1, "Final factor must be less than 1"
    assert 0 < init_factor < 1, "Initial factor must be less than 1"
    assert 0 < warmup_steps < total_steps, "Total steps must be greater than warmup"

    def fn(x: int) -> float:
        if x <= warmup_steps:
            return init_factor + x * (1 - init_factor) / warmup_steps
        if x >= total_steps:
            return final_factor
        t = (x - warmup_steps) / (total_steps - warmup_steps) * math.pi
        return (1 + math.cos(t)) * (1 - final_factor) / 2 + final_factor

    return LambdaLR(optimizer, fn)


def one_cycle(
    model: LightningModule,
    optimizer: Optimizer,
    total_steps: int = 1000,
    **kwargs,
) -> OneCycleLR:
    """Get the learning rate scheduler."""
    total_steps = get_max_steps(model) or total_steps
    return OneCycleLR(
        optimizer,
        **kwargs,
        total_steps=total_steps,
        max_lr=optimizer.param_groups[0]["lr"],
    )


def standard_optim_sched(model: LightningModule) -> dict:
    """Configure the optimizers and learning rate sheduler.

    In favour of deprecating this in the future, as it is overly verbose.
    """
    # Finish initialising the partialy created methods
    opt = model.hparams.optimizer(filter(lambda p: p.requires_grad, model.parameters()))

    # Use mltools to initialise the scheduler
    # as we can sync the cycle length with the number of steps per epoch
    sched = get_sched(
        model.hparams.sched_config.mltools,
        opt,
        steps_per_epoch=len(model.trainer.datamodule.train_dataloader()),
        max_epochs=model.trainer.max_epochs,
        max_steps=model.trainer.max_steps,
    )

    # Return the dict for the lightning trainer
    return {
        "optimizer": opt,
        "lr_scheduler": {"scheduler": sched, **model.hparams.sched_config.lightning},
    }


def simple_optim_sched(model: LightningModule) -> dict:
    """Configure the optimizers and learning rate sheduler."""
    opt = model.hparams.optimizer(filter(lambda p: p.requires_grad, model.parameters()))
    scheduler = {
        "scheduler": model.hparams.scheduler(optimizer=opt, model=model),
        "interval": "step",
    }
    return [opt], [scheduler]


def multi_optim_sched(model: LightningModule, modules: list) -> dict:
    """Configure multiple optimizers and learning rate sheduler."""
    opt_params = model.hparams.optimizer
    sched_params = model.hparams.scheduler
    assert len(modules) == len(opt_params) == len(sched_params)
    opts = []
    scheds = []
    for module, opt_param, sched_param in zip(modules, opt_params, sched_params, strict=False):
        opt = opt_param(filter(lambda p: p.requires_grad, module.parameters()))
        sched = {
            "scheduler": sched_param(optimizer=opt, model=model),
            "interval": "step",
        }
        opts.append(opt)
        scheds.append(sched)
    return opts, scheds
