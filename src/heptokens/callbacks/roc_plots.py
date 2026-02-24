# In src/heptokens/callbacks/roc_plots.py (new file)
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import torch as T
import wandb
from lightning import Callback
from sklearn.metrics import auc, roc_curve


class ROCPlotCallback(Callback):
    """Callback to generate and log HEP-style ROC curves during validation."""

    def __init__(self):
        super().__init__()
        self.validation_outputs = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Collect validation batch outputs."""
        if trainer.sanity_checking:
            return

        # Get predictions and labels
        with T.no_grad():
            logits = pl_module(batch)
            probs = T.softmax(logits, dim=1)

        self.validation_outputs.append(
            {"probs": probs.detach().cpu(), "labels": batch["labels"].detach().cpu()}
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate and log ROC curves at end of validation epoch."""
        if not self.validation_outputs or trainer.sanity_checking:
            return

        # Concatenate all batches
        all_probs = T.cat([x["probs"] for x in self.validation_outputs], dim=0).numpy()
        all_labels = T.cat([x["labels"] for x in self.validation_outputs], dim=0).numpy()

        # Clear stored outputs
        self.validation_outputs.clear()

        # Create ROC plots if we have a logger
        if trainer.logger is not None and hasattr(trainer.logger, "experiment"):
            # Set HEP style
            plt.style.use(hep.style.CMS)

            # Create one-vs-rest ROC curves for each class
            fig, ax = plt.subplots(figsize=(8, 8))

            for class_idx in range(pl_module.n_classes):
                # Binary classification: this class vs all others
                y_true_binary = (all_labels == class_idx).astype(int)
                y_score = all_probs[:, class_idx]

                fpr, tpr, _ = roc_curve(y_true_binary, y_score)
                roc_auc = auc(fpr, tpr)

                # HEP style: Signal efficiency (TPR) vs Background rejection (1/FPR)
                rejection = np.where(fpr > 0, 1.0 / fpr, 0)

                # Plot (limit rejection to reasonable range for visibility)
                valid_idx = (tpr > 0) & (rejection > 0) & (rejection < 1e4)
                ax.plot(
                    tpr[valid_idx],
                    rejection[valid_idx],
                    label=f"Class {class_idx} (AUC={roc_auc:.3f})",
                    linewidth=2,
                )

            ax.set_xlabel("Signal Efficiency", fontsize=14)
            ax.set_ylabel("Background Rejection (1/FPR)", fontsize=14)
            ax.set_yscale("log")
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([1, 1e4])
            ax.legend(loc="best", fontsize=10)
            ax.grid(True, alpha=0.3)

            # Log to wandb
            trainer.logger.experiment.log(
                {"valid/roc_curves": wandb.Image(fig), "epoch": trainer.current_epoch}
            )
            plt.close(fig)
