# A callback to monitor reconstruction quality in original (unscaled) space.
import logging

import numpy as np
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
        pt_idx: int = 0,
        deta_idx: int = 1,
        dphi_idx: int = 2,
        compute_jet_metrics: bool = True,
    ):
        super().__init__()
        self.cst_fn = cst_fn
        self.jet_fn = jet_fn
        self.log_every_n_epochs = log_every_n_epochs
        self.max_batches = max_batches
        self.pt_idx = pt_idx
        self.deta_idx = deta_idx
        self.dphi_idx = dphi_idx
        self.compute_jet_metrics = compute_jet_metrics

        if compute_jet_metrics:
            # Store jet metrics across batches for epoch-level summary
            self.jet_residuals = []

    # Add helper method to compute jets from constituents
    def _compute_jet_from_constituents(self, csts, mask):
        """Compute jet pt from constituent pts (relative coordinates).

        Since constituents are in relative (deta, dphi) coordinates,
        we only sum their pt values. The eta/phi are relative to jet axis,
        so reconstructed jets have centroids at (0,0) in eta/phi space.

        Args:
            csts: [batch, n_constituents, features] where features are [pt, deta, dphi, ...]
            mask: [batch_size, n_constituents] boolean mask of valid constituents

        Returns:
            dict with jet pt (sum of constituent pts)
        """
        # Extract constituent pt
        pt = csts[:, :, self.pt_idx].cpu().numpy()
        mask_np = mask.cpu().numpy()

        # Sum pt for valid constituents only
        jet_pt = np.sum(np.where(mask_np, pt, 0), axis=1)

        return {"pt": jet_pt}

    def _delta_phi(self, phi1, phi2):
        """Compute delta phi wrapped to [-pi, pi]."""
        dphi = phi1 - phi2
        return np.arctan2(np.sin(dphi), np.cos(dphi))

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Compute metrics in original space during validation."""
        if trainer.sanity_checking:
            return
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return
        if batch_idx >= self.max_batches:
            return

        # Get reconstructions
        with T.no_grad():
            encoding = pl_module.encode(batch)[0]
            recon_scaled = pl_module.decode(encoding, batch)

        # Create reconstruction dict in scaled space
        recon_dict_scaled = {
            "csts": recon_scaled,
            "jets": batch["jets"].clone(),
            "mask": batch["mask"],
        }

        # Inverse transform
        original_unscaled = inverse_preprocess_batch(
            dict_to_device(batch, "cpu"), self.cst_fn, self.jet_fn
        )
        recon_unscaled = inverse_preprocess_batch(
            dict_to_device(recon_dict_scaled, "cpu"), self.cst_fn, self.jet_fn
        )

        # Constituent-level metrics
        mask = batch["mask"].cpu()
        original_csts = original_unscaled["csts"][mask]
        recon_csts = recon_unscaled["csts"][mask]

        unscaled_mae = T.mean(T.abs(original_csts - recon_csts))
        unscaled_mse = T.mean((original_csts - recon_csts) ** 2)

        pl_module.log("val/unscaled_mae", unscaled_mae, prog_bar=False)
        pl_module.log("val/unscaled_mse", unscaled_mse, prog_bar=False)

        # Jet-level metrics (pt-based only, since coordinates are relative)
        if self.compute_jet_metrics:
            # TODO: Jeff please check if this makes sense (probably doesn't)
            # Compute jets from original and reconstructed constituents
            jet_truth = self._compute_jet_from_constituents(original_unscaled["csts"], mask)
            jet_reco = self._compute_jet_from_constituents(recon_unscaled["csts"], mask)

            # Compute pt residuals and radial distance
            pt_residuals = jet_truth["pt"] - jet_reco["pt"]

            # Compute mean radial distance in (deta, dphi) space
            mask_np = mask.cpu().numpy()
            original_valid = original_unscaled["csts"][mask_np]
            recon_valid = recon_unscaled["csts"][mask_np]
            original_deta = original_valid[:, self.deta_idx].cpu().numpy()
            original_dphi = original_valid[:, self.dphi_idx].cpu().numpy()
            recon_deta = recon_valid[:, self.deta_idx].cpu().numpy()
            recon_dphi = recon_valid[:, self.dphi_idx].cpu().numpy()

            radial_distance = np.sqrt(
                (original_deta - recon_deta) ** 2 + (original_dphi - recon_dphi) ** 2
            )

            residuals = {
                "pt": pt_residuals,
                "radial_dist": radial_distance,
            }

            # Store for epoch-level aggregation
            self.jet_residuals.append(residuals)

            # Log immediate metrics
            pl_module.log("val/jet_pt_bias", float(np.mean(pt_residuals)), prog_bar=False)
            pl_module.log("val/jet_pt_resolution", float(np.std(pt_residuals)), prog_bar=False)
            pl_module.log(
                "val/constituent_radial_dist", float(np.mean(radial_distance)), prog_bar=False
            )

        # Feature plots (existing code)
        if trainer.logger is not None and hasattr(trainer.logger, "experiment"):
            import matplotlib.pyplot as plt
            import wandb

            logger = trainer.logger.experiment

            for feature_idx in range(original_csts.shape[1]):
                original_feat = original_csts[:, feature_idx].cpu().numpy()
                recon_feat = recon_csts[:, feature_idx].cpu().numpy()

                fig, ax = plt.subplots(figsize=(6, 6))
                ax.scatter(original_feat, recon_feat, alpha=0.5, s=1)
                ax.set_xlabel("Original Feature")
                ax.set_ylabel("Reconstructed Feature")
                ax.set_title(f"Feature {feature_idx}")
                ax.plot(
                    [original_feat.min(), original_feat.max()],
                    [original_feat.min(), original_feat.max()],
                    "r--",
                )

                logger.log({f"val/recon_feature_{feature_idx}": wandb.Image(fig)})
                plt.close(fig)

    # Add method to plot jet residuals at end of validation epoch
    def on_validation_epoch_end(self, trainer, pl_module):
        """Create summary plots of jet reconstruction at end of epoch."""
        if not self.compute_jet_metrics or not self.jet_residuals:
            return

        # Concatenate all batch residuals
        all_residuals = {
            key: np.concatenate([batch[key] for batch in self.jet_residuals])
            for key in self.jet_residuals[0].keys()
        }

        # Clear for next epoch
        self.jet_residuals.clear()

        if trainer.logger is not None and hasattr(trainer.logger, "experiment"):
            import matplotlib.pyplot as plt
            import mplhep as hep
            import wandb

            # Use HEP style
            plt.style.use(hep.style.CMS)

            # Create residual plots
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            axes = axes.flatten()

            labels = ["Jet pt residual (truth - reco)", "Constituent radial distance"]
            for i, (var, label) in enumerate(zip(all_residuals.keys(), labels)):
                ax = axes[i]
                residuals = all_residuals[var]

                # Plot histogram
                if np.sum(~np.isfinite(residuals)) != 0:
                    # Plot warning message
                    ax.text(
                        0.5,
                        0.5,
                        f"Non-finite values in {var} residuals",
                        ha="center",
                        va="center",
                        fontsize=14,
                        transform=ax.transAxes,
                        color="red",
                        weight="bold",
                    )
                else:
                    # Plot histogram
                    ax.hist(residuals, bins=50, histtype="step", linewidth=2)

                    # Add statistics text
                    mean_val = np.mean(residuals)
                    std_val = np.std(residuals)
                    median_val = np.median(residuals)
                    ax.text(
                        0.05,
                        0.95,
                        f"Mean: {mean_val:.3f}\nMedian: {median_val:.3f}\nStd: {std_val:.3f}",
                        transform=ax.transAxes,
                        verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                    )
                ax.set_xlabel(label, fontsize=12)
                ax.set_ylabel("Count", fontsize=12)
                ax.grid(True, alpha=0.3)

            fig.tight_layout()
            trainer.logger.experiment.log(
                {"val/jet_residuals": wandb.Image(fig), "epoch": trainer.current_epoch}
            )
            plt.close(fig)
