"""COCOA truth-particle dataset and DataModule for ROOT files."""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, IterableDataset

from heptokens.data.cocoa_base import (
    EVENTNUMBER_BRANCH,
    load_root_arrays,
    root_files_from_dir,
)
from heptokens.data.collation import collate_and_transform

log = logging.getLogger(__name__)

# Continuous/position features fed to the tokenizer. Order matters: pt=0, eta=1,
# phi=2, e=3 so ReconstructionMonitor's default (pt/deta/dphi) indices apply.
TRUTHPART_FEATURES = ["particle_pt", "particle_eta", "particle_phi", "particle_e"]

# Extra branches needed to derive the 3-way particle class (not tokenized directly).
TRUTHPART_CLASS_BRANCHES = ["particle_pdgid", "particle_track_idx"]

# 0: charged, 1: neutral hadron, 2: photon (matches HEP4M truthpart_class).
N_TRUTHPART_CLASSES = 3


def particle_class(pdgid: np.ndarray, track_idx: np.ndarray) -> np.ndarray:
    """Derive the 3-way particle class per truth particle.

    0: charged (has an associated track), 1: neutral hadron, 2: photon.
    """
    cls = np.where(track_idx >= 0, 0, 1)
    cls = np.where(pdgid == 22, 2, cls)
    return cls


def build_truthpart_csts(
    arrays: dict[str, np.ndarray], max_csts: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build fixed-size continuous constituents, class indices, and a validity mask.

    Returns:
        csts: float32 [n_events, max_csts, len(TRUTHPART_FEATURES)] (continuous only)
        class_labels: int64 [n_events, max_csts] class index in {0,1,2} (0 on padding)
        mask: bool [n_events, max_csts]
    """
    pts = arrays[TRUTHPART_FEATURES[0]]
    n_events = len(pts)
    n_features = len(TRUTHPART_FEATURES)

    csts = np.zeros((n_events, max_csts, n_features), dtype=np.float32)
    class_labels = np.zeros((n_events, max_csts), dtype=np.int64)
    mask = np.zeros((n_events, max_csts), dtype=bool)

    for i in range(n_events):
        n = min(len(pts[i]), max_csts)
        if n == 0:
            continue
        mask[i, :n] = True
        for f_idx, feat in enumerate(TRUTHPART_FEATURES):
            csts[i, :n, f_idx] = arrays[feat][i][:n]
        class_labels[i, :n] = particle_class(
            arrays["particle_pdgid"][i][:n], arrays["particle_track_idx"][i][:n]
        )

    return csts, class_labels, mask


class COCOATruthPartDataset(Dataset):
    """In-memory dataset that loads COCOA truth particles from ROOT files."""

    def __init__(
        self,
        root_files: list[str | Path],
        max_csts: int = 16,
        num_events: int | None = None,
    ) -> None:
        super().__init__()
        self.max_csts = max_csts
        branches = TRUTHPART_FEATURES + TRUTHPART_CLASS_BRANCHES + [EVENTNUMBER_BRANCH]

        csts_parts = []
        class_parts = []
        mask_parts = []
        event_parts = []
        loaded = 0

        for fpath in root_files:
            remaining = None if num_events is None else num_events - loaded
            if remaining is not None and remaining <= 0:
                break

            arrays = load_root_arrays(fpath, branches, max_entries=remaining)
            csts, class_labels, mask = build_truthpart_csts(arrays, max_csts)
            csts_parts.append(csts)
            class_parts.append(class_labels)
            mask_parts.append(mask)
            event_parts.append(np.asarray(arrays[EVENTNUMBER_BRANCH], dtype=np.int64))
            loaded += len(csts)

        self.csts = np.concatenate(csts_parts, axis=0)
        self.class_labels = np.concatenate(class_parts, axis=0)
        self.mask = np.concatenate(mask_parts, axis=0)
        self.event_numbers = np.concatenate(event_parts, axis=0)

        if num_events is not None and len(self.csts) > num_events:
            self.csts = self.csts[:num_events]
            self.class_labels = self.class_labels[:num_events]
            self.mask = self.mask[:num_events]
            self.event_numbers = self.event_numbers[:num_events]

        log.info(
            f"Loaded {len(self.csts):,} events from {len(root_files)} files "
            f"({self.csts.shape[-1]} features, max_csts={max_csts})"
        )

    def __len__(self) -> int:
        return len(self.csts)

    def __getitem__(self, idx: int) -> dict:
        return {
            "csts": self.csts[idx],
            "class_labels": self.class_labels[idx],
            "mask": self.mask[idx],
            "eventNumber": self.event_numbers[idx],
            "labels": 0,
        }


class COCOATruthPartModule(LightningDataModule):
    """Lightning DataModule for COCOA truth-particle data from ROOT files."""

    def __init__(
        self,
        data_dir: str,
        train_dir: str = "train_topup2_20M_cells256",
        val_dir: str = "val_100K_cells256",
        test_dir: str = "test_1M_cells256_isInfFalse",
        features: list[str] | None = None,
        max_csts: int = 16,
        num_events: int | None = None,
        batch_size: int = 1024,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool | None = None,
        transforms: dict | None = None,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        # `features` is accepted for config compatibility; the tokenized feature
        # set (continuous + one-hot class) is fixed by build_truthpart_csts.
        self.features = features or TRUTHPART_FEATURES
        self.max_csts = max_csts
        self.num_events = num_events
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = (
            num_workers > 0 if persistent_workers is None else persistent_workers
        )
        self.transforms = transforms

    def _make_dataset(self, split_dir: str):
        files = root_files_from_dir(self.data_dir / split_dir)
        return COCOATruthPartDataset(files, self.max_csts, self.num_events)

    def setup(self, stage: str = "fit") -> None:
        if stage in {"fit", "train"}:
            self.train_set = self._make_dataset(self.train_dir)
            self.valid_set = self._make_dataset(self.val_dir)
        if stage in {"validate"}:
            self.valid_set = self._make_dataset(self.val_dir)
        if stage in {"test", "predict"}:
            self.test_set = self._make_dataset(self.test_dir)

    def _get_dataloader(self, dataset, shuffle: bool, drop_last: bool) -> DataLoader:
        collate_fn = None
        if self.transforms is not None:
            collate_fn = partial(collate_and_transform, transforms=self.transforms)

        kwargs = {}
        if self.num_workers > 0:
            kwargs["persistent_workers"] = self.persistent_workers

        is_iterable = isinstance(dataset, IterableDataset)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=shuffle if not is_iterable else False,
            drop_last=drop_last,
            collate_fn=collate_fn,
            **kwargs,
        )

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.train_set, shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.valid_set, shuffle=False, drop_last=False)

    def test_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.test_set, shuffle=False, drop_last=False)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def get_data_sample(self) -> dict:
        if not hasattr(self, "valid_set"):
            self.setup("validate")
        ds = self.valid_set
        if isinstance(ds, IterableDataset):
            return next(iter(ds))
        return ds[0]

    def get_n_classes(self) -> int:
        return 1
