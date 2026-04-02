"""For loading in jets from HDF files and creating a mappable dataset."""

import logging
from functools import partial

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from heptokens.data.atlas_mappable import BaseMapModule
from heptokens.data.collation import collate_and_transform
from heptokens.utils.plot_physics import JET_FEATURES, TRACK_FEATURES

log = logging.getLogger(__name__)


class IterMapDataset(IterableDataset):
    """Iterable dataset that loads jets on-demand without loading everything into memory."""

    def __init__(
        self,
        file_path: str,
        jet_features: list | None = None,
        cst_features: list | None = None,
        label_key: str = "HadronConeExclTruthLabelID",
        num_jets: int | None = None,
        num_csts: int | None = None,
        max_jet_pt: float | None = None,
        max_cst_pt: float | None = None,
        chunk_size: int = 1000,
    ) -> None:
        super().__init__()
        if jet_features is None:
            jet_features = JET_FEATURES
        if cst_features is None:
            cst_features = TRACK_FEATURES

        self.file_path = file_path
        self.jet_features = jet_features
        self.cst_features = cst_features
        self.label_key = label_key
        self.num_csts = num_csts
        self.chunk_size = chunk_size
        self.max_jet_pt = max_jet_pt
        self.max_cst_pt = max_cst_pt
        self.pt_idx = jet_features.index("pt") if "pt" in jet_features else None
        self.cst_pt_idx = cst_features.index("pt") if "pt" in cst_features else None

        # Determine total number of jets and build label map
        with h5py.File(file_path, mode="r") as handle:
            jets_ds = handle["jets"]
            total_jets = len(jets_ds)
            self.num_jets = min(total_jets, num_jets) if num_jets is not None else total_jets

            # Load labels to build label map (labels are typically small)
            labels = jets_ds[self.label_key][: self.num_jets]
            unique_labels = np.unique(labels)
            self.label_map = {label: i for i, label in enumerate(unique_labels)}

            # Get number of constituents
            tracks_ds = handle["tracks"]
            total_csts = tracks_ds.shape[1]
            if num_csts is None:
                self.num_csts = total_csts
            else:
                self.num_csts = min(total_csts, num_csts)

        log.info(
            f"IterMapDataset: {self.num_jets} jets with {self.num_csts}"
            f"constituents from {self.file_path}"
        )

    def __iter__(self):
        """Iterate through dataset in chunks to manage memory."""
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # Single-process data loading
            start_idx = 0
            end_idx = self.num_jets
        else:
            # Multi-process data loading: split workload among workers
            per_worker = int(np.ceil(self.num_jets / worker_info.num_workers))
            worker_id = worker_info.id
            start_idx = worker_id * per_worker
            end_idx = min(start_idx + per_worker, self.num_jets)

        # Open file once per worker
        with h5py.File(self.file_path, mode="r") as handle:
            jets_ds = handle["jets"]
            tracks_ds = handle["tracks"]

            # Process data in chunks
            for chunk_start in range(start_idx, end_idx, self.chunk_size):
                chunk_end = min(chunk_start + self.chunk_size, end_idx)

                # Load jet features for this chunk
                jets_chunk = np.empty(
                    (chunk_end - chunk_start, len(self.jet_features)), dtype=np.float32
                )
                for i, feat in enumerate(self.jet_features):
                    jets_chunk[:, i] = jets_ds[feat][chunk_start:chunk_end]

                # Load labels for this chunk
                labels_chunk = jets_ds[self.label_key][chunk_start:chunk_end]
                labels_mapped = np.array([self.label_map[label] for label in labels_chunk])

                # Load constituent features for this chunk
                tracks_chunk = tracks_ds[chunk_start:chunk_end, : self.num_csts]
                csts_chunk = np.empty(
                    (chunk_end - chunk_start, self.num_csts, len(self.cst_features)),
                    dtype=np.float32,
                )
                for i, key in enumerate(self.cst_features):
                    csts_chunk[:, :, i] = tracks_chunk[key]

                # Load validity mask for this chunk
                mask_chunk = tracks_chunk["valid"]

                # Apply jet pT cut
                keep = np.ones(chunk_end - chunk_start, dtype=bool)
                if self.max_jet_pt is not None and self.pt_idx is not None:
                    keep &= jets_chunk[:, self.pt_idx] <= self.max_jet_pt
                # Apply constituent pT cut
                if self.max_cst_pt is not None and self.cst_pt_idx is not None:
                    cst_pts = csts_chunk[:, :, self.cst_pt_idx]
                    max_per_jet = np.where(mask_chunk, cst_pts, 0).max(axis=1)
                    keep &= max_per_jet <= self.max_cst_pt

                # Yield individual samples from this chunk
                for i in range(chunk_end - chunk_start):
                    if not keep[i]:
                        continue
                    yield {
                        "jets": jets_chunk[i],
                        "csts": csts_chunk[i],
                        "mask": mask_chunk[i],
                        "labels": labels_mapped[i],
                    }

    def __len__(self) -> int:
        """Return total number of jets."""
        return self.num_jets


class PresplitIterModule(BaseMapModule):
    """DataModule for pre-split train/val/test datasets using iterable datasets."""

    def __init__(
        self,
        *,
        train_path: str,
        val_path: str,
        test_path: str,
        chunk_size: int = 1000,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.chunk_size = chunk_size

        # Extract data_config from kwargs for dataset creation
        data_config = {k: v for k, v in self.data_config.items()}
        data_config["chunk_size"] = chunk_size

        # Initialize validation set early to get a sample
        self.valid_set = IterMapDataset(self.val_path, **data_config)

    def setup(self, stage: str) -> None:
        """Sets up the relevant datasets."""
        data_config = {k: v for k, v in self.data_config.items()}
        data_config["chunk_size"] = self.chunk_size

        if stage in {"fit", "train"}:
            self.train_set = IterMapDataset(self.train_path, **data_config)
        if stage in {"predict", "test"}:
            self.test_set = IterMapDataset(self.test_path, **data_config)

    def _get_dataloader(
        self, dataset: IterableDataset, shuffle: bool, drop_last: bool
    ) -> DataLoader:
        """Override dataloader creation for iterable datasets."""
        collate_fn = None
        if self.transforms is not None:
            collate_fn = partial(collate_and_transform, transforms=self.transforms)

        dataloader_kwargs = {}
        if self.num_workers > 0:
            dataloader_kwargs["persistent_workers"] = self.persistent_workers
            if self.multiprocessing_context is not None:
                dataloader_kwargs["multiprocessing_context"] = self.multiprocessing_context

        # Note: shuffle parameter is ignored for IterableDataset
        # Shuffling would need to be implemented within the dataset's __iter__ method
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
            **dataloader_kwargs,
        )


class SingleFileIterModule(BaseMapModule):
    """DataModule that splits a single dataset file into train/val/test using iterable datasets."""

    def __init__(
        self,
        *,
        data_path: str,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
        seed: int = 42,
        chunk_size: int = 1000,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if not abs(train_frac + val_frac + test_frac - 1.0) < 1e-6:
            raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

        self.data_path = data_path
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.seed = seed
        self.chunk_size = chunk_size

        # Determine total size and split indices
        with h5py.File(data_path, mode="r") as handle:
            total_size = len(handle["jets"])

        # Calculate split sizes
        train_size = int(total_size * train_frac)
        val_size = int(total_size * val_frac)

        # Generate shuffled indices if seed is provided
        if seed is not None:
            rng = np.random.default_rng(seed)
            self.indices = rng.permutation(total_size)
        else:
            self.indices = np.arange(total_size)

        # Split indices into train/val/test
        self.train_indices = self.indices[:train_size]
        self.val_indices = self.indices[train_size : train_size + val_size]
        self.test_indices = self.indices[train_size + val_size :]

        # Create dataset configs
        data_config = {k: v for k, v in self.data_config.items()}
        data_config["chunk_size"] = chunk_size

        # Initialize splits as IterMapDataset with index filtering
        self.train_set = IndexedIterMapDataset(
            self.data_path, indices=self.train_indices, **data_config
        )
        self.valid_set = IndexedIterMapDataset(
            self.data_path, indices=self.val_indices, **data_config
        )
        self.test_set = IndexedIterMapDataset(
            self.data_path, indices=self.test_indices, **data_config
        )

    def setup(self, stage: str) -> None:
        """Datasets are already split in __init__."""
        pass

    def _get_dataloader(
        self, dataset: IterableDataset, shuffle: bool, drop_last: bool
    ) -> DataLoader:
        """Override dataloader creation for iterable datasets."""
        collate_fn = None
        if self.transforms is not None:
            collate_fn = partial(collate_and_transform, transforms=self.transforms)

        dataloader_kwargs = {}
        if self.num_workers > 0:
            dataloader_kwargs["persistent_workers"] = self.persistent_workers
            if self.multiprocessing_context is not None:
                dataloader_kwargs["multiprocessing_context"] = self.multiprocessing_context

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
            **dataloader_kwargs,
        )


class IndexedIterMapDataset(IterableDataset):
    """Iterable dataset that loads specific indices from an HDF5 file."""

    def __init__(
        self,
        file_path: str,
        indices: np.ndarray,
        jet_features: list | None = None,
        cst_features: list | None = None,
        label_key: str = "HadronConeExclTruthLabelID",
        num_csts: int | None = None,
        max_jet_pt: float | None = None,
        max_cst_pt: float | None = None,
        chunk_size: int = 1000,
    ) -> None:
        super().__init__()
        if jet_features is None:
            jet_features = JET_FEATURES
        if cst_features is None:
            cst_features = TRACK_FEATURES

        self.file_path = file_path
        self.indices = indices
        self.jet_features = jet_features
        self.cst_features = cst_features
        self.label_key = label_key
        self.chunk_size = chunk_size
        self.max_jet_pt = max_jet_pt
        self.max_cst_pt = max_cst_pt
        self.pt_idx = jet_features.index("pt") if "pt" in jet_features else None
        self.cst_pt_idx = cst_features.index("pt") if "pt" in cst_features else None

        # Build label map from all indices
        with h5py.File(file_path, mode="r") as handle:
            jets_ds = handle["jets"]

            # Load labels for these indices (still memory efficient for most use cases)
            labels = jets_ds[self.label_key][indices]
            unique_labels = np.unique(labels)
            self.label_map = {label: i for i, label in enumerate(unique_labels)}

            # Get number of constituents
            tracks_ds = handle["tracks"]
            total_csts = tracks_ds.shape[1]
            if num_csts is None:
                self.num_csts = total_csts
            else:
                self.num_csts = min(total_csts, num_csts)

        log.info(f"IndexedIterMapDataset: {len(indices)} jets with {self.num_csts} constituents")

    def __iter__(self):
        """Iterate through the specified indices in chunks."""
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # Single-process data loading
            worker_indices = self.indices
        else:
            # Multi-process data loading: split indices among workers
            per_worker = int(np.ceil(len(self.indices) / worker_info.num_workers))
            worker_id = worker_info.id
            start_idx = worker_id * per_worker
            end_idx = min(start_idx + per_worker, len(self.indices))
            worker_indices = self.indices[start_idx:end_idx]

        # Sort indices for more efficient HDF5 access
        sorted_worker_indices = np.sort(worker_indices)

        # Open file once per worker
        with h5py.File(self.file_path, mode="r") as handle:
            jets_ds = handle["jets"]
            tracks_ds = handle["tracks"]

            # Process in chunks
            for chunk_start in range(0, len(sorted_worker_indices), self.chunk_size):
                chunk_end = min(chunk_start + self.chunk_size, len(sorted_worker_indices))
                chunk_indices = sorted_worker_indices[chunk_start:chunk_end]

                # Load jet features for these indices
                jets_chunk = np.empty(
                    (len(chunk_indices), len(self.jet_features)), dtype=np.float32
                )
                for i, feat in enumerate(self.jet_features):
                    jets_chunk[:, i] = jets_ds[feat][chunk_indices]

                # Load labels
                labels_chunk = jets_ds[self.label_key][chunk_indices]
                labels_mapped = np.array([self.label_map[label] for label in labels_chunk])

                # Load constituent features
                tracks_chunk = tracks_ds[chunk_indices, : self.num_csts]
                csts_chunk = np.empty(
                    (len(chunk_indices), self.num_csts, len(self.cst_features)), dtype=np.float32
                )
                for i, key in enumerate(self.cst_features):
                    csts_chunk[:, :, i] = tracks_chunk[key]

                # Load validity mask
                mask_chunk = tracks_chunk["valid"]

                # Apply jet pT cut
                keep = np.ones(len(chunk_indices), dtype=bool)
                if self.max_jet_pt is not None and self.pt_idx is not None:
                    keep &= jets_chunk[:, self.pt_idx] <= self.max_jet_pt
                # Apply constituent pT cut
                if self.max_cst_pt is not None and self.cst_pt_idx is not None:
                    cst_pts = csts_chunk[:, :, self.cst_pt_idx]
                    max_per_jet = np.where(mask_chunk, cst_pts, 0).max(axis=1)
                    keep &= max_per_jet <= self.max_cst_pt

                # Yield individual samples
                for i in range(len(chunk_indices)):
                    if not keep[i]:
                        continue
                    yield {
                        "jets": jets_chunk[i],
                        "csts": csts_chunk[i],
                        "mask": mask_chunk[i],
                        "labels": labels_mapped[i],
                    }

    def __len__(self) -> int:
        """Return number of indices."""
        return len(self.indices)
