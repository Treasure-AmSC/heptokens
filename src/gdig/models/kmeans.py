"""PyTorch Lightning wrapper for MinibatchKMeans clustering."""

import logging
from typing import Any, Dict

import torch
from lightning import LightningModule
from torchpq.clustering import MinibatchKMeans

log = logging.getLogger(__name__)


class LitMinibatchKMeans(LightningModule):
    """Lightning Module wrapper for MinibatchKMeans clustering.

    This allows using MinibatchKMeans within the Lightning training framework,
    with logging, checkpointing, and other Lightning features.

    Args:
        n_clusters: Number of clusters
        distance: Distance metric ('euclidean', 'cosine', 'manhattan')
        init_mode: Initialization method ('random', 'kmeans++')
        learning_rate: Learning rate for optimizer (not used in kmeans, but required by Lightning)
        log_every_n_steps: How often to log metrics
        **kwargs: Additional arguments passed to MinibatchKMeans
    """

    def __init__(
        self,
        n_clusters: int,
        distance: str = "euclidean",
        init_mode: str = "random",
        learning_rate: float = 1e-3,  # Not used but convention for Lightning
        log_every_n_steps: int = 10,
        data_sample: torch.Tensor = None,  # Unused, for compatibility with train.py
        n_classes: int = None,  # Unused, for compatibility with train.py
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        # Enable manual optimization for K-means (no gradient-based training)
        self.automatic_optimization = False

        self.kmeans = MinibatchKMeans(
            n_clusters=n_clusters,
            distance=distance,
            init_mode=init_mode,
            **kwargs,
        )
        self.log_every_n_steps = log_every_n_steps

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict cluster labels for input data.

        Args:
            batch: Dictionary containing 'csts' tensor of shape [batch_size, n_csts, d_vector]
                and 'mask' tensor of shape [batch_size, n_csts]

        Returns:
            Cluster labels of shape [batch_size, n_csts]
            Masked positions will have label -1 or 0 (you can choose)
        """
        csts = batch["csts"]
        mask = batch["mask"]
        # csts: [batch_size, n_csts, d_vector]
        batch_size, n_csts, _ = csts.shape
        # Initialize output with -1 for masked positions
        labels = torch.full((batch_size, n_csts), -1, dtype=torch.long, device=csts.device)
        # Get valid constituents and their positions
        valid_csts = csts[mask].T  # [d_vector, n_valid]
        # Predict labels for valid constituents
        valid_labels = self.kmeans.predict(valid_csts)
        # Place predicted labels back into the output tensor
        labels[mask] = valid_labels
        return labels

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform one minibatch K-means update.

        Args:
            batch: Dictionary containing data tensors
            batch_idx: Index of current batch

        Returns:
            Dummy loss tensor (K-means doesn't have a traditional loss)
        """
        # Extract data from batch (adjust key based on your data structure)
        # Assume batch has 'data' or 'x' key
        csts = batch["csts"]
        mask = batch["mask"]

        # Fit minibatch
        labels = self.kmeans.fit_minibatch(csts[mask].T)

        # Log metrics
        if batch_idx % self.log_every_n_steps == 0:
            self.log("train/inertia", self.kmeans.inertia, prog_bar=True)
            self.log("train/error", self.kmeans.error, prog_bar=True)
            self.log("train/iteration", float(self.kmeans._iteration))

            # Log cluster distribution
            unique_labels, counts = labels.unique(return_counts=True)
            self.log("train/n_active_clusters", len(unique_labels))
            self.log("train/min_cluster_size", counts.min().item())
            self.log("train/max_cluster_size", counts.max().item())
            self.log("train/mean_cluster_size", counts.float().mean().item())

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Evaluate clustering on validation data without updating centroids."""
        csts = batch["csts"]
        mask = batch["mask"]
        # Apply mask and transpose
        valid_csts = csts[mask].T
        # Predict labels without updating centroids
        labels = self.kmeans.predict(valid_csts)
        # Compute inertia on validation set
        maxsims, _ = self.kmeans.get_labels(valid_csts, self.kmeans.centroids)
        val_inertia = self.kmeans.calculate_inertia(maxsims).item()
        self.log("val/inertia", val_inertia, prog_bar=True)
        # Log cluster distribution
        unique_labels, counts = labels.unique(return_counts=True)
        self.log("val/n_active_clusters", len(unique_labels))
        self.log("val/min_cluster_size", counts.min().item())
        self.log("val/max_cluster_size", counts.max().item())

        return {"val_inertia": val_inertia, "labels": labels}

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Predict cluster assignments for new data.

        Args:
            batch: Dictionary containing data tensors
            batch_idx: Index of current batch

        Returns:
            Cluster labels
        """
        return self(batch)

    def configure_optimizers(self):
        """K-means doesn't use gradient-based optimization.

        Returns a dummy optimizer to satisfy Lightning requirements.
        """
        # Return a dummy optimizer (not actually used)
        return None

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Save K-means specific state."""
        checkpoint["kmeans_centroids"] = self.kmeans.centroids
        checkpoint["kmeans_n_points"] = self.kmeans.n_points_in_clusters
        checkpoint["kmeans_iteration"] = self.kmeans._iteration

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Load K-means specific state."""
        if "kmeans_centroids" in checkpoint:
            self.kmeans.centroids = checkpoint["kmeans_centroids"]
        if "kmeans_n_points" in checkpoint:
            self.kmeans.n_points_in_clusters = checkpoint["kmeans_n_points"]
        if "kmeans_iteration" in checkpoint:
            self.kmeans._iteration = checkpoint["kmeans_iteration"]

    def topk(self, query: torch.Tensor, k: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
        """Get top-k closest clusters for query points.

        Args:
            query: Query tensor of shape [d_vector, n_query] or [n_query, d_vector]
            k: Number of closest clusters to return

        Returns:
            Tuple of (similarities, cluster_indices)
        """
        if query.dim() == 2 and query.shape[0] > query.shape[1]:
            query = query.T
        return self.kmeans.topk(query, k=k)
