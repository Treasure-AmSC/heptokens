"""COCOA topo cluster dataset and DataModule for ROOT files."""

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
    TOPO_FEATURES,
    load_root_arrays,
    pad_jagged_to_fixed,
    root_files_from_dir,
)
from heptokens.data.collation import collate_and_transform

log = logging.getLogger(__name__)


class COCOATopoDataset(Dataset):
    """In-memory dataset that loads COCOA topo clusters from ROOT files."""

    def __init__(
        self,
        root_files: list[str | Path],
        features: list[str] | None = None,
        max_csts: int = 50,
        num_events: int | None = None,
    ) -> None:
        super().__init__()
        if features is None:
            features = TOPO_FEATURES
        self.features = features
        self.max_csts = max_csts

        csts_parts = []
        mask_parts = []
        event_parts = []
        loaded = 0

        for fpath in root_files:
            remaining = None if num_events is None else num_events - loaded
            if remaining is not None and remaining <= 0:
                break

            arrays = load_root_arrays(fpath, features + [EVENTNUMBER_BRANCH], max_entries=remaining)
            csts, mask = pad_jagged_to_fixed(arrays, features, max_csts)
            csts_parts.append(csts)
            mask_parts.append(mask)
            event_parts.append(np.asarray(arrays[EVENTNUMBER_BRANCH], dtype=np.int64))
            loaded += len(csts)

        self.csts = np.concatenate(csts_parts, axis=0)
        self.mask = np.concatenate(mask_parts, axis=0)
        self.event_numbers = np.concatenate(event_parts, axis=0)

        if num_events is not None and len(self.csts) > num_events:
            self.csts = self.csts[:num_events]
            self.mask = self.mask[:num_events]
            self.event_numbers = self.event_numbers[:num_events]

        log.info(
            f"Loaded {len(self.csts):,} events from {len(root_files)} files "
            f"({len(features)} features, max_csts={max_csts})"
        )

    def __len__(self) -> int:
        return len(self.csts)

    def __getitem__(self, idx: int) -> dict:
        return {
            "csts": self.csts[idx],
            "mask": self.mask[idx],
            "eventNumber": self.event_numbers[idx],
            "labels": 0,
        }


class COCOATopoModule(LightningDataModule):
    """Lightning DataModule for COCOA topo cluster data from ROOT files."""

    def __init__(
        self,
        data_dir: str,
        train_dir: str = "train_topup2_20M_cells256",
        val_dir: str = "val_100K_cells256",
        test_dir: str = "test_1M_cells256_isInfFalse",
        features: list[str] | None = None,
        max_csts: int = 50,
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
        self.features = features or TOPO_FEATURES
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
        return COCOATopoDataset(files, self.features, self.max_csts, self.num_events)

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

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=shuffle,
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
        return self.valid_set[0]

    def get_n_classes(self) -> int:
        return 1
