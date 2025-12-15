# A callback to monitor reconstruction quality in original (unscaled) space.
import logging

import torch as T
from lightning.pytorch.callbacks import Callback
from sklearn.base import BaseEstimator

from gdig.data.preprocessing import inverse_preprocess_batch
from gdig.utils.torch_utils import dict_to_device

log = logging.getLogger(__name__)


class ReconstructionMonitor(Callback):
    """Monitor reconstruction quality in original (unscaled) space.

    Args:
        cst_fn: Fitted transformer for constituents (e.g., QuantileTransformer)
        jet_fn: Fitted transformer for jets (e.g., QuantileTransformer)
        log_every_n_epochs: How often to compute unscaled metrics (default: 1)
    """

    def __init__(
        self,
        cst_fn: BaseEstimator,
        jet_fn: BaseEstimator,
        log_every_n_epochs: int = 1,
        max_batches: int = 1,
    ):
        super().__init__()
        self.cst_fn = cst_fn
        self.jet_fn = jet_fn
        self.log_every_n_epochs = log_every_n_epochs
        self.max_batches = max_batches

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Compute metrics in original space during validation."""
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return

        # Limit to max_batches for efficiency
        if batch_idx >= self.max_batches:
            return

        # Get reconstructions (model-specific, adjust as needed)
        with T.no_grad():
            encoding = pl_module.encode(batch)[0]
            recon_scaled = pl_module.decode(encoding, batch)

        # Create reconstruction dict in scaled space
        recon_dict_scaled = {
            "csts": recon_scaled,
            "jets": batch["jets"].clone(),
            "mask": batch["mask"],
        }

        # Inverse transform both original and reconstruction
        original_unscaled = inverse_preprocess_batch(
            dict_to_device(batch, "cpu"), self.cst_fn, self.jet_fn
        )
        recon_unscaled = inverse_preprocess_batch(
            dict_to_device(recon_dict_scaled, "cpu"), self.cst_fn, self.jet_fn
        )

        # Compute loss in original space
        mask = batch["mask"].cpu()
        original_csts = original_unscaled["csts"][mask]
        recon_csts = recon_unscaled["csts"][mask]

        unscaled_mae = T.mean(T.abs(original_csts - recon_csts))
        unscaled_mse = T.mean((original_csts - recon_csts) ** 2)

        # Log metrics
        pl_module.log("val/unscaled_mae", unscaled_mae, prog_bar=False)
        pl_module.log("val/unscaled_mse", unscaled_mse, prog_bar=False)

        # Make plots of reconstructed features vs original and log with wandb
        if trainer.logger is not None and hasattr(trainer.logger, "experiment"):
            import matplotlib.pyplot as plt
            import wandb

            logger = trainer.logger.experiment

            # Plot each feature against its reconstruction
            for feature_idx in range(original_csts.shape[1]):
                original_feat = original_csts[:, feature_idx].cpu().numpy()
                recon_feat = recon_csts[:, feature_idx].cpu().numpy()

                plt.figure(figsize=(6, 6))
                plt.scatter(original_feat, recon_feat, alpha=0.5, s=1)
                plt.xlabel("Original Feature")
                plt.ylabel("Reconstructed Feature")
                plt.title(feature_idx)
                plt.plot(
                    [original_feat.min(), original_feat.max()],
                    [original_feat.min(), original_feat.max()],
                    "r--",
                )  # y=x line

                logger.log({f"val/recon_feature_{feature_idx}": wandb.Image(plt)})
                plt.close()
