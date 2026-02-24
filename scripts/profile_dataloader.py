"""A script to profile the data loading performance of a PyTorch DataLoader"""

import json
import logging
import time
import warnings
from pathlib import Path

import hydra
import lightning.pytorch as pl
import torch as T
import torch.nn as nn
from lightning import LightningModule
from omegaconf import DictConfig
from tqdm import tqdm

from heptokens.utils.hydra import (
    print_config,
    reload_original_config,
)

log = logging.getLogger(__name__)
# Suppress torchvision image library warnings (we don't use image functionality)
warnings.filterwarnings("ignore", message="Failed to load image Python extension")


class DummyModel(LightningModule):
    """A minimal model that does nothing for profiling dataloader performance."""

    def __init__(self):
        super().__init__()
        # Add a dummy parameter so the optimizer has something to optimize
        self.dummy_param = nn.Parameter(T.zeros(1))

    def forward(self, x):
        return x

    def training_step(self, batch, batch_idx):
        # Just return a dummy loss
        return T.tensor(0.0, device=self.device, requires_grad=True)

    def validation_step(self, batch, batch_idx):
        return T.tensor(0.0, device=self.device)

    def configure_optimizers(self):
        # Dummy optimizer
        return T.optim.SGD(self.parameters(), lr=0.001)


class ProfilingCallback(pl.Callback):
    """Callback to track timing metrics during training."""

    def __init__(self):
        super().__init__()
        self.epochs = []
        self.current_epoch_data = {}

    def on_train_epoch_start(self, trainer, pl_module):
        self.current_epoch_data = {
            "epoch": trainer.current_epoch + 1,
            "epoch_start": time.perf_counter(),
        }

    def on_train_epoch_end(self, trainer, pl_module):
        if "epoch_start" not in self.current_epoch_data:
            return
        self.current_epoch_data["train_end"] = time.perf_counter()
        train_time = self.current_epoch_data["train_end"] - self.current_epoch_data["epoch_start"]
        self.current_epoch_data["train_time"] = train_time

    def on_validation_epoch_start(self, trainer, pl_module):
        # Only track if we're in actual training (not sanity check)
        if "epoch_start" not in self.current_epoch_data:
            self.current_epoch_data["epoch_start"] = time.perf_counter()
        self.current_epoch_data["val_start"] = time.perf_counter()

    def on_validation_epoch_end(self, trainer, pl_module):
        # Skip if no epoch_start (sanity check)
        if "epoch_start" not in self.current_epoch_data:
            return

        val_end = time.perf_counter()
        val_time = val_end - self.current_epoch_data.get("val_start", val_end)
        epoch_time = val_end - self.current_epoch_data["epoch_start"]

        self.current_epoch_data.update(
            {
                "val_time": val_time,
                "total_time": epoch_time,
            }
        )

        self.epochs.append(self.current_epoch_data.copy())

        # Only log if we have epoch number (not during sanity check)
        if "epoch" in self.current_epoch_data:
            log.info(
                f"Epoch {self.current_epoch_data['epoch']}: {epoch_time:.2f}s total | "
                f"Train: {self.current_epoch_data.get('train_time', 0):.2f}s | "
                f"Val: {val_time:.2f}s"
            )


def profile_dataloader_iteration(
    dataloader, device, name: str, num_epochs: int = 3, track_transfer: bool = True
):
    """Profile dataloader iteration with optional device transfer tracking.

    Args:
        dataloader: The dataloader to profile
        device: Device to transfer data to (use 'cpu' for CPU-only profiling)
        name: Name of the dataloader (e.g., 'train', 'val')
        num_epochs: Number of epochs to profile
        track_transfer: Whether to separately track transfer time
    """
    device_str = str(device)
    test_name = "with_device_transfer" if device_str != "cpu" else "cpu_only"

    log.info(f"\n{'=' * 60}")
    log.info(f"Profiling: {name} - {test_name} ({device_str})")
    log.info(f"{'=' * 60}")

    timings = []
    transfer_timings = []
    batch_counts = []

    for epoch in range(num_epochs):
        start_time = time.perf_counter()
        transfer_time = 0
        batch_count = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            batch_count += 1

            # Time the transfer
            if track_transfer:
                t0 = time.perf_counter()

            if isinstance(batch, dict):
                batch = {
                    k: v.to(device) if isinstance(v, T.Tensor) else v for k, v in batch.items()
                }
            elif isinstance(batch, (list, tuple)):
                batch = [v.to(device) if isinstance(v, T.Tensor) else v for v in batch]
            else:
                batch = batch.to(device)

            if track_transfer:
                transfer_time += time.perf_counter() - t0

        epoch_time = time.perf_counter() - start_time
        timings.append(epoch_time)
        transfer_timings.append(transfer_time)
        batch_counts.append(batch_count)

        throughput = batch_count / epoch_time
        if track_transfer and transfer_time > 0:
            transfer_pct = (transfer_time / epoch_time) * 100
            log.info(
                f"Epoch {epoch + 1}: {epoch_time:.2f}s total, "
                f"{transfer_time:.2f}s transfer ({transfer_pct:.1f}%), "
                f"{throughput:.2f} batches/sec"
            )
        else:
            log.info(
                f"Epoch {epoch + 1}: {epoch_time:.2f}s, "
                f"{batch_count} batches, {throughput:.2f} batches/sec"
            )

    avg_time = sum(timings) / len(timings)
    avg_transfer_time = sum(transfer_timings) / len(transfer_timings)
    avg_throughput = sum(batch_counts) / sum(timings)

    result = {
        "name": name,
        "test": test_name,
        "device": device_str,
        "num_epochs": num_epochs,
        "timings": timings,
        "batch_counts": batch_counts,
        "avg_epoch_time": avg_time,
        "avg_throughput_batches_per_sec": avg_throughput,
    }

    if track_transfer:
        result["transfer_timings"] = transfer_timings
        result["avg_transfer_time"] = avg_transfer_time

    return result


def profile_with_trainer(cfg, datamodule, num_epochs: int = 5):
    """Profile using actual Lightning Trainer for realistic training loop."""
    log.info(f"\n{'=' * 60}")
    log.info("Profiling: Full training loop with Lightning Trainer")
    log.info(f"{'=' * 60}")

    # Create dummy model
    model = DummyModel()

    # Create profiling callback
    profiling_callback = ProfilingCallback()

    # Use actual trainer config from Hydra for realistic profiling
    # Disable checkpointing and logger to focus on dataloader performance
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        max_epochs=num_epochs,
        callbacks=profiling_callback,
        enable_checkpointing=False,
        logger=False,
    )

    # Run training
    trainer.fit(model, datamodule=datamodule)

    return {
        "test": "full_training_loop_with_trainer",
        "device": str(trainer.strategy.root_device),
        "num_epochs": num_epochs,
        "epochs": profiling_callback.epochs,
    }


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    """Profile dataloader performance with various tests."""
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

    # Determine device
    device = T.device("cuda" if T.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    log.info("Instantiating the data module")
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup("fit")

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    # Collect all results
    all_results = []

    # Test 1: CPU-only iteration
    cpu_device = T.device("cpu")
    all_results.append(
        profile_dataloader_iteration(
            train_loader, cpu_device, "train", num_epochs=3, track_transfer=False
        )
    )
    all_results.append(
        profile_dataloader_iteration(
            val_loader, cpu_device, "val", num_epochs=3, track_transfer=False
        )
    )

    # Test 2: With device transfer (if GPU available)
    if device.type != "cpu":
        all_results.append(
            profile_dataloader_iteration(
                train_loader, device, "train", num_epochs=3, track_transfer=True
            )
        )
        all_results.append(
            profile_dataloader_iteration(
                val_loader, device, "val", num_epochs=3, track_transfer=True
            )
        )

    # Test 3: Full training loop with actual Trainer
    all_results.append(profile_with_trainer(cfg, datamodule, num_epochs=5))

    # Save results
    output_file = Path("dataloader_profile_results.json")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    log.info(f"\n{'=' * 60}")
    log.info(f"Profiling complete! Results saved to: {output_file}")
    log.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
