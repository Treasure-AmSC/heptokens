"""Memory-efficient iterable dataloaders for Z'->ttbar (tops) and QCD jets.

Mirrors the design of atlas_iterable.py but for the PFCands / jet_kinematics
HDF5 format.  Only one chunk of jets lives in RAM at a time, so these are
safe for the large QCD files (~1.7 GB / ~1.5 M jets each).

Classes
-------
ZPrimeIterDataset
    Streams a contiguous range of jets from one file in chunks.
IndexedZPrimeIterDataset
    Streams a specific (possibly shuffled) set of row indices from one file.
ZPrimeIterModule
    Lightning DataModule combining multiple tops + QCD files.
    Splits each file independently into train / val / test, then chains them,
    so all splits see a proportional mix of every file.
"""

import logging
from functools import partial
from pathlib import Path

import h5py
import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import ChainDataset, DataLoader, IterableDataset

try:
    from heptokens.data.collation import collate_and_transform
except ImportError:
    collate_and_transform = None  # type: ignore[assignment]

from heptokens.data.zprime_mappable import (
    DEFAULT_CST_FEATURES,
    DEFAULT_JET_FEATURES,
)

log = logging.getLogger(__name__)


# ── single-file streaming dataset ─────────────────────────────────────────────

class ZPrimeIterDataset(IterableDataset):
    """Streams jets from a contiguous range of a single ZPrime HDF5 file.

    Memory usage is bounded by ``chunk_size`` jets at a time regardless of
    how large the file is.

    Args:
        file_path: Path to the HDF5 file.
        label: Integer class label (0 = QCD, 1 = tops).
        num_jets: Cap on total jets to stream. None = all.
        num_csts: Max constituents per jet (≤ 150).
        cst_feature_indices: PFCands column indices to include (default 0-9).
        jet_feature_indices: jet_kinematics column indices (default 0-3).
        load_weights: If True, yield a ``weights`` key per sample (QCD only).
        chunk_size: Number of jets read from disk at once.
    """

    def __init__(
        self,
        file_path: str,
        label: int,
        num_jets: int | None = None,
        num_csts: int | None = None,
        cst_feature_indices: list[int] | None = None,
        jet_feature_indices: list[int] | None = None,
        load_weights: bool = False,
        chunk_size: int = 1000,
    ) -> None:
        super().__init__()
        if cst_feature_indices is None:
            cst_feature_indices = DEFAULT_CST_FEATURES
        if jet_feature_indices is None:
            jet_feature_indices = DEFAULT_JET_FEATURES

        self.file_path = file_path
        self.label = label
        self.cst_feature_indices = cst_feature_indices
        self.jet_feature_indices = jet_feature_indices
        self.load_weights = load_weights
        self.chunk_size = chunk_size

        with h5py.File(file_path, mode="r") as f:
            total = f["PFCands"].shape[0]
            self.num_jets = min(total, num_jets) if num_jets is not None else total
            max_csts = f["PFCands"].shape[1]
            self.num_csts = min(max_csts, num_csts) if num_csts is not None else max_csts
            self._has_weights = "weight" in f

        if load_weights and not self._has_weights:
            log.warning("load_weights=True but no 'weight' dataset in %s", file_path)

        log.info(
            "ZPrimeIterDataset: %d jets (label=%d) from %s",
            self.num_jets, label, Path(file_path).name,
        )

    def __len__(self) -> int:
        return self.num_jets

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            start, end = 0, self.num_jets
        else:
            per_worker = int(np.ceil(self.num_jets / worker_info.num_workers))
            start = worker_info.id * per_worker
            end = min(start + per_worker, self.num_jets)

        yield from self._stream(start, end)

    def _stream(self, start: int, end: int):
        """Read [start, end) from disk in chunks and yield individual samples."""
        with h5py.File(self.file_path, mode="r") as f:
            pf_ds = f["PFCands"]
            jk_ds = f["jet_kinematics"]
            wt_ds = f["weight"] if (self.load_weights and self._has_weights) else None

            for chunk_start in range(start, end, self.chunk_size):
                chunk_end = min(chunk_start + self.chunk_size, end)
                n = chunk_end - chunk_start

                raw = pf_ds[chunk_start:chunk_end, : self.num_csts, :]
                mask_chunk = raw[:, :, 10].astype(bool)
                csts_chunk = raw[:, :, self.cst_feature_indices].astype(np.float32)

                jk_chunk = jk_ds[chunk_start:chunk_end, :]
                jets_chunk = jk_chunk[:, self.jet_feature_indices].astype(np.float32)

                wt_chunk = wt_ds[chunk_start:chunk_end].astype(np.float32) if wt_ds is not None else None

                for i in range(n):
                    sample = {
                        "csts":   csts_chunk[i],
                        "mask":   mask_chunk[i],
                        "jets":   jets_chunk[i],
                        "labels": np.int64(self.label),
                    }
                    if wt_chunk is not None:
                        sample["weights"] = wt_chunk[i]
                    yield sample


# ── index-filtered streaming dataset ──────────────────────────────────────────

class IndexedZPrimeIterDataset(IterableDataset):
    """Streams a specific set of row indices from a single ZPrime HDF5 file.

    Indices are sorted before reading to maximise sequential HDF5 access and
    then split evenly across DataLoader workers.

    Args:
        file_path: Path to the HDF5 file.
        indices: 1-D array of row indices to stream.
        label: Integer class label (0 = QCD, 1 = tops).
        num_csts: Max constituents per jet (≤ 150).
        cst_feature_indices: PFCands column indices.
        jet_feature_indices: jet_kinematics column indices.
        load_weights: If True, yield a ``weights`` key (QCD only).
        chunk_size: Rows read from disk at once.
    """

    def __init__(
        self,
        file_path: str,
        indices: np.ndarray,
        label: int,
        num_csts: int | None = None,
        cst_feature_indices: list[int] | None = None,
        jet_feature_indices: list[int] | None = None,
        load_weights: bool = False,
        chunk_size: int = 1000,
    ) -> None:
        super().__init__()
        if cst_feature_indices is None:
            cst_feature_indices = DEFAULT_CST_FEATURES
        if jet_feature_indices is None:
            jet_feature_indices = DEFAULT_JET_FEATURES

        self.file_path = file_path
        self.indices = np.asarray(indices)
        self.label = label
        self.cst_feature_indices = cst_feature_indices
        self.jet_feature_indices = jet_feature_indices
        self.load_weights = load_weights
        self.chunk_size = chunk_size

        with h5py.File(file_path, mode="r") as f:
            max_csts = f["PFCands"].shape[1]
            self.num_csts = min(max_csts, num_csts) if num_csts is not None else max_csts
            self._has_weights = "weight" in f

        if load_weights and not self._has_weights:
            log.warning("load_weights=True but no 'weight' dataset in %s", file_path)

        log.info(
            "IndexedZPrimeIterDataset: %d jets (label=%d) from %s",
            len(self.indices), label, Path(file_path).name,
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_indices = self.indices
        else:
            per_worker = int(np.ceil(len(self.indices) / worker_info.num_workers))
            start = worker_info.id * per_worker
            end = min(start + per_worker, len(self.indices))
            worker_indices = self.indices[start:end]

        # Sort for efficient sequential HDF5 access
        sorted_indices = np.sort(worker_indices)
        yield from self._stream(sorted_indices)

    def _stream(self, sorted_indices: np.ndarray):
        with h5py.File(self.file_path, mode="r") as f:
            pf_ds = f["PFCands"]
            jk_ds = f["jet_kinematics"]
            wt_ds = f["weight"] if (self.load_weights and self._has_weights) else None

            for chunk_start in range(0, len(sorted_indices), self.chunk_size):
                chunk_end = min(chunk_start + self.chunk_size, len(sorted_indices))
                idx = sorted_indices[chunk_start:chunk_end]

                raw = pf_ds[idx, : self.num_csts, :]
                mask_chunk = raw[:, :, 10].astype(bool)
                csts_chunk = raw[:, :, self.cst_feature_indices].astype(np.float32)

                jk_chunk = jk_ds[idx, :]
                jets_chunk = jk_chunk[:, self.jet_feature_indices].astype(np.float32)

                wt_chunk = wt_ds[idx].astype(np.float32) if wt_ds is not None else None

                for i in range(len(idx)):
                    sample = {
                        "csts":   csts_chunk[i],
                        "mask":   mask_chunk[i],
                        "jets":   jets_chunk[i],
                        "labels": np.int64(self.label),
                    }
                    if wt_chunk is not None:
                        sample["weights"] = wt_chunk[i]
                    yield sample


# ── DataModule ─────────────────────────────────────────────────────────────────

class ZPrimeIterModule(LightningDataModule):
    """Iterable Lightning DataModule combining QCD (label=0) and tops (label=1).

    Each file is split independently into train / val / test fractions using
    shuffled indices, then all per-file split datasets are chained together.
    This guarantees every split sees a proportional mix from every file
    without ever loading more than ``chunk_size`` jets into RAM.

    Args:
        tops_paths: List of tops HDF5 file paths.
        qcd_paths: List of QCD HDF5 file paths.
        num_jets_per_file: Cap on jets used from each file.
        train_frac / val_frac / test_frac: Split ratios (must sum to 1).
        seed: RNG seed for reproducible index shuffling.
        cst_feature_indices: PFCands column indices (default 0-9).
        jet_feature_indices: jet_kinematics column indices (default 0-3).
        num_csts: Max constituents per jet (≤ 150).
        load_weights_qcd: Include per-jet ``weights`` key from QCD files.
        chunk_size: Jets read from disk per chunk.
        batch_size / num_workers / pin_memory: DataLoader settings.
        transforms: Optional collation transforms (same interface as ATLAS loaders).
    """

    def __init__(
        self,
        tops_paths: list[str],
        qcd_paths: list[str],
        num_jets_per_file: int | None = None,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
        seed: int = 42,
        cst_feature_indices: list[int] | None = None,
        jet_feature_indices: list[int] | None = None,
        num_csts: int | None = None,
        load_weights_qcd: bool = False,
        chunk_size: int = 1000,
        batch_size: int = 1000,
        num_workers: int = 4,
        pin_memory: bool = True,
        transforms: dict | None = None,
    ) -> None:
        super().__init__()
        if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
            raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.transforms = transforms

        dataset_kwargs = dict(
            num_csts=num_csts,
            cst_feature_indices=cst_feature_indices,
            jet_feature_indices=jet_feature_indices,
            chunk_size=chunk_size,
        )

        rng = np.random.default_rng(seed)

        train_ds, val_ds, test_ds = [], [], []

        for path, label, use_weights in (
            *((p, 1, False) for p in tops_paths),
            *((p, 0, load_weights_qcd) for p in qcd_paths),
        ):
            with h5py.File(path, mode="r") as f:
                n_total = f["PFCands"].shape[0]
            n = min(n_total, num_jets_per_file) if num_jets_per_file is not None else n_total

            indices = rng.permutation(n)
            t = int(n * train_frac)
            v = int(n * val_frac)

            kw = dict(label=label, load_weights=use_weights, **dataset_kwargs)
            train_ds.append(IndexedZPrimeIterDataset(path, indices[:t],          **kw))
            val_ds.append(  IndexedZPrimeIterDataset(path, indices[t : t + v],   **kw))
            test_ds.append( IndexedZPrimeIterDataset(path, indices[t + v :],     **kw))

        # ChainDataset iterates through each component dataset sequentially
        self.train_set = ChainDataset(train_ds)
        self.valid_set = ChainDataset(val_ds)
        self.test_set  = ChainDataset(test_ds)

        n_tops = sum(len(d) for d in train_ds if d.label == 1)
        n_qcd  = sum(len(d) for d in train_ds if d.label == 0)
        log.info(
            "ZPrimeIterModule: train=%d tops + %d QCD | val+test from %d files",
            n_tops, n_qcd, len(tops_paths) + len(qcd_paths),
        )

    # ── LightningDataModule interface ─────────────────────────────────────────

    def setup(self, stage: str) -> None:
        """Splits are pre-built in __init__; nothing to do here."""

    def _make_dataloader(self, dataset: IterableDataset) -> DataLoader:
        collate_fn = None
        if self.transforms is not None and collate_and_transform is not None:
            collate_fn = partial(collate_and_transform, transforms=self.transforms)

        kwargs: dict = {}
        if self.num_workers > 0:
            kwargs["persistent_workers"] = True

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
            **kwargs,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_dataloader(self.train_set)

    def val_dataloader(self) -> DataLoader:
        return self._make_dataloader(self.valid_set)

    def test_dataloader(self) -> DataLoader:
        return self._make_dataloader(self.test_set)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def get_n_classes(self) -> int:
        return 2

    def get_data_sample(self) -> dict:
        return next(iter(self.valid_set))
