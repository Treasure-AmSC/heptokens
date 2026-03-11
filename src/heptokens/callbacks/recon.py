# A callback to monitor reconstruction quality in original (unscaled) space.
import logging

import numpy as np
import torch as T
from lightning.pytorch.callbacks import Callback
from sklearn.base import BaseEstimator

from heptokens.data.collation import inverse_preprocess_batch
from heptokens.utils.torch_utils import dict_to_device

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
        jet_pt_idx: int = 0,
        jet_mass_idx: int = 1,
        jet_eta_idx: int = 2,
        jet_phi_idx: int = 3,
        compute_jet_metrics: bool = True,
        iqr_med_pt_min: float = 5_000.0,
        iqr_med_pt_max: float = 140_000.0,
        iqr_med_n_bins: int = 20,
        iqr_med_min_entries_per_bin: int = 10,
    ):
        super().__init__()
        self.cst_fn = cst_fn
        self.jet_fn = jet_fn
        self.log_every_n_epochs = log_every_n_epochs
        self.max_batches = max_batches
        self.pt_idx = pt_idx
        self.deta_idx = deta_idx
        self.dphi_idx = dphi_idx
        self.jet_pt_idx = jet_pt_idx
        self.jet_mass_idx = jet_mass_idx
        self.jet_eta_idx = jet_eta_idx
        self.jet_phi_idx = jet_phi_idx
        self.compute_jet_metrics = compute_jet_metrics
        self.iqr_med_pt_min = iqr_med_pt_min
        self.iqr_med_pt_max = iqr_med_pt_max
        self.iqr_med_n_bins = iqr_med_n_bins
        self.iqr_med_min_entries_per_bin = iqr_med_min_entries_per_bin

        # Store constituent features across batches for epoch-level plots
        self.cst_originals = []
        self.cst_recons = []

        if compute_jet_metrics:
            # Store jet metrics across batches for epoch-level summary
            self.jet_residuals = []

    @staticmethod
    def wrap_phi(phi):
        """Wrap angle(s) to [-pi, pi)."""
        return (phi + np.pi) % (2 * np.pi) - np.pi

    # Add helper method to compute jets from constituents
    def _compute_jet_from_constituents(self, csts, mask, jets):
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
        # Extract jet eta, phi
        jet_etas = jets[:, self.jet_eta_idx].cpu().numpy()
        jet_phis = jets[:, self.jet_phi_idx].cpu().numpy()

        # assume csts masses are zero, so we can compute jet mass from constituent pts and jet pt
        csts_pts = csts[:, :, self.pt_idx].cpu().numpy()
        csts_detas = csts[:, :, self.deta_idx].cpu().numpy()
        csts_dphis = csts[:, :, self.dphi_idx].cpu().numpy()

        # add back jet coords to get absolute csts coords
        csts_phis_unbounded = csts_dphis + jet_phis[:, None]
        csts_phis = self._delta_phi(csts_phis_unbounded, 0.0)
        csts_etas = csts_detas + jet_etas[:, None]

        pxs = csts_pts * np.cos(csts_phis)
        pys = csts_pts * np.sin(csts_phis)
        pzs = csts_pts * np.sinh(csts_etas)
        energies = csts_pts * np.cosh(csts_etas)

        mask_np = mask.cpu().numpy()

        # Sum for valid constituents only
        reco_jet_pxs = np.sum(np.where(mask_np, pxs, 0), axis=-1)
        reco_jet_pys = np.sum(np.where(mask_np, pys, 0), axis=-1)
        reco_jet_pzs = np.sum(np.where(mask_np, pzs, 0), axis=-1)
        reco_jet_pts = np.sqrt(reco_jet_pxs**2 + reco_jet_pys**2)
        reco_jet_pts = np.where(reco_jet_pts == 0, 1e-10, reco_jet_pts)
        reco_jet_etas = np.arcsinh(reco_jet_pzs / reco_jet_pts)
        reco_jet_phis = np.arctan2(reco_jet_pys, reco_jet_pxs)

        reco_jet_energies = np.sum(np.where(mask_np, energies, 0), axis=-1)
        reco_jet_m2s = reco_jet_energies**2 - (reco_jet_pxs**2 + reco_jet_pys**2 + reco_jet_pzs**2)
        reco_jet_masses = np.sqrt(np.clip(reco_jet_m2s, 0.0, None))

        return {
            "pt": reco_jet_pts,
            "mass": reco_jet_masses,
            "eta": reco_jet_etas,
            "phi": reco_jet_phis,
        }

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

        # Store constituent features for epoch-level scatter plots
        self.cst_originals.append(original_csts.cpu().numpy())
        self.cst_recons.append(recon_csts.cpu().numpy())

        # Jet-level metrics (pt-based only, since coordinates are relative)
        if self.compute_jet_metrics:
            # Compute jets from original and reconstructed constituents

            # ToDo: decide whether to get the truth jets from the jet array, or to compute
            # them from unscaled constituents.
            # the latter might assume less about the reconstruction?
            """
            jet_truth = {
                "pt": original_unscaled["jets"][:, self.jet_pt_idx].cpu().numpy(),
                "mass": original_unscaled["jets"][:, self.jet_mass_idx].cpu().numpy(),
                "eta": original_unscaled["jets"][:, self.jet_eta_idx].cpu().numpy(),
                "phi": self._delta_phi(
                    original_unscaled["jets"][:, self.jet_phi_idx].cpu().numpy(), 0.0
                ),
            }
            """
            jet_truth = self._compute_jet_from_constituents(
                original_unscaled["csts"], mask, original_unscaled["jets"]
            )
            jet_reco = self._compute_jet_from_constituents(
                recon_unscaled["csts"], mask, original_unscaled["jets"]
            )

            # Compute pt residuals and radial distance
            pt_residuals = jet_truth["pt"] - jet_reco["pt"]

            pt_ratio = np.divide(
                jet_reco["pt"],
                jet_truth["pt"],
                out=np.full_like(jet_reco["pt"], np.nan),
                where=jet_truth["pt"] > 0,
            )

            mass_residuals = jet_truth["mass"] - jet_reco["mass"]
            eta_residuals = jet_truth["eta"] - jet_reco["eta"]
            phi_residuals = self._delta_phi(jet_truth["phi"], jet_reco["phi"])

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
                "mass": mass_residuals,
                "eta": eta_residuals,
                "phi": phi_residuals,
                "radial_dist": radial_distance,
                "truth_pt": jet_truth["pt"],
                "pt_ratio": pt_ratio,
                "reco_pt": jet_reco["pt"],
            }

            # Store for epoch-level aggregation
            self.jet_residuals.append(residuals)

            # Log immediate metrics
            pl_module.log("val/jet_pt_bias", float(np.mean(pt_residuals)), prog_bar=False)
            pl_module.log("val/jet_pt_resolution", float(np.std(pt_residuals)), prog_bar=False)
            pl_module.log(
                "val/constituent_radial_dist", float(np.mean(radial_distance)), prog_bar=False
            )



    # Add method to plot jet residuals at end of validation epoch
    def on_validation_epoch_end(self, trainer, pl_module):
        """Create summary plots at end of epoch."""
        # Feature scatter plots from accumulated batches
        if self.cst_originals and trainer.logger is not None and hasattr(trainer.logger, "experiment"):
            import matplotlib.pyplot as plt
            import wandb

            all_original = np.concatenate(self.cst_originals, axis=0)
            all_recon = np.concatenate(self.cst_recons, axis=0)
            self.cst_originals.clear()
            self.cst_recons.clear()

            n_features = all_original.shape[1]
            n_cols = min(4, n_features)
            n_rows = int(np.ceil(n_features / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
            axes = np.atleast_1d(axes).flatten()

            for feature_idx in range(n_features):
                ax = axes[feature_idx]
                orig = all_original[:, feature_idx]
                reco = all_recon[:, feature_idx]
                ax.scatter(orig, reco, alpha=0.3, s=1)
                ax.set_xlabel("Original")
                ax.set_ylabel("Reconstructed")
                ax.set_title(f"Feature {feature_idx}")
                ax.plot(
                    [orig.min(), orig.max()],
                    [orig.min(), orig.max()],
                    "r--",
                )

            for i in range(n_features, len(axes)):
                axes[i].axis("off")

            fig.tight_layout()
            trainer.logger.experiment.log(
                {"val/recon_features": wandb.Image(fig), "epoch": trainer.current_epoch}
            )
            plt.close(fig)

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
            plot_order = ["pt", "mass", "eta", "phi", "radial_dist", "truth_pt", "reco_pt"]
            labels = {
                "pt": "Jet pt residual (truth - reco)",
                "mass": "Jet mass residual (truth - reco)",
                "eta": "Jet eta residual (truth - reco)",
                "phi": "Jet phi residual (truth - reco)",
                "radial_dist": "Constituent radial distance",
                "truth_pt": "Truth jet pt [MeV]",
                "reco_pt": "Reconstructed jet pt [MeV]",
            }
            vars_to_plot = [key for key in plot_order if key in all_residuals]
            n_vars = len(vars_to_plot)
            n_cols = min(3, n_vars)
            n_rows = int(np.ceil(n_vars / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4.5 * n_rows))
            axes = np.atleast_1d(axes).flatten()

            for i, var in enumerate(vars_to_plot):
                ax = axes[i]
                label = labels.get(var, var)
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

            # Hide unused axes
            for i in range(n_vars, len(axes)):
                axes[i].axis("off")

            fig.tight_layout()
            trainer.logger.experiment.log(
                {"val/jet_residuals": wandb.Image(fig), "epoch": trainer.current_epoch}
            )
            plt.close(fig)

            truth_pt = all_residuals.get("truth_pt")
            pt_ratio = all_residuals.get("pt_ratio")
            if truth_pt is None or pt_ratio is None:
                return
            truth_pt = np.asarray(truth_pt).reshape(-1)
            pt_ratio = np.asarray(pt_ratio).reshape(-1)
            if truth_pt.size != pt_ratio.size:
                min_size = min(truth_pt.size, pt_ratio.size)
                log.warning(
                    "Mismatched truth_pt (%d) and pt_ratio (%d) lengths; truncating to %d.",
                    truth_pt.size,
                    pt_ratio.size,
                    min_size,
                )
                truth_pt = truth_pt[:min_size]
                pt_ratio = pt_ratio[:min_size]

            finite_mask = np.isfinite(truth_pt) & np.isfinite(pt_ratio) & (pt_ratio > 0)
            truth_pt = truth_pt[finite_mask]
            pt_ratio = pt_ratio[finite_mask]
            if truth_pt.size == 0:
                return

            pt_min = self.iqr_med_pt_min
            pt_max = self.iqr_med_pt_max
            n_bins = self.iqr_med_n_bins
            min_entries_per_bin = self.iqr_med_min_entries_per_bin

            pt_bin_edges = np.linspace(pt_min, pt_max, n_bins + 1)
            pt_bin_centers = 0.5 * (pt_bin_edges[:-1] + pt_bin_edges[1:])
            iqr_over_median = np.full(n_bins, np.nan)

            for i in range(n_bins):
                in_bin = truth_pt >= pt_bin_edges[i]
                if i == n_bins - 1:
                    in_bin &= truth_pt <= pt_bin_edges[i + 1]
                else:
                    in_bin &= truth_pt < pt_bin_edges[i + 1]

                ratios_in_bin = pt_ratio[in_bin]
                if ratios_in_bin.size < min_entries_per_bin:
                    continue

                median = np.median(ratios_in_bin)
                if not np.isfinite(median) or median <= 0:
                    continue

                q25, q75 = np.percentile(ratios_in_bin, [25, 75])
                iqr_over_median[i] = (q75 - q25) / median

            fig, ax = plt.subplots(figsize=(6, 6))
            valid = np.isfinite(iqr_over_median)
            if np.any(valid):
                ax.plot(pt_bin_centers[valid], iqr_over_median[valid], marker="o", linewidth=0)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No bins with sufficient entries",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                )

            ax.set_xlim(pt_min, pt_max)
            ax.set_ylim(0.0, np.nanmax(iqr_over_median) * 2)
            ax.set_xlabel("Truth jet $p_T$ [MeV]", fontsize=12)
            ax.set_ylabel("IQR(reco/truth) / median(reco/truth)", fontsize=12)
            fig.tight_layout()
            trainer.logger.experiment.log(
                {
                    "val/jet_pt_response_iqr_over_median_vs_truth_pt": wandb.Image(fig),
                    "epoch": trainer.current_epoch,
                }
            )
            plt.close(fig)
