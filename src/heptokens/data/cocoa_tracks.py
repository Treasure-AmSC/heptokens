"""COCOA track dataset and DataModule for ROOT files."""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, IterableDataset

from heptokens.data.cocoa_base import (
    TRACK_FEATURES,
    load_root_arrays,
    pad_jagged_to_fixed,
    root_files_from_dir,
)
from heptokens.data.collation import collate_and_transform

log = logging.getLogger(__name__)


class COCOATrackDataset(Dataset):
    """In-memory dataset that loads COCOA tracks from ROOT files.

    Each item is a single jet (event) with its track constituents
    padded/truncated to max_csts.
    """

    def __init__(
        self,
        root_files: list[str | Path],
        features: list[str] | None = None,
        max_csts: int = 15,
        num_events: int | None = None,
    ) -> None:
        super().__init__()
        if features is None:
            features = TRACK_FEATURES
        self.features = features
        self.max_csts = max_csts

        csts_parts = []
        mask_parts = []
        loaded = 0

        for fpath in root_files:
            remaining = None if num_events is None else num_events - loaded
            if remaining is not None and remaining <= 0:
                break

            arrays = load_root_arrays(fpath, features, max_entries=remaining)
            csts, mask = pad_jagged_to_fixed(arrays, features, max_csts)
            csts_parts.append(csts)
            mask_parts.append(mask)
            loaded += len(csts)

        self.csts = np.concatenate(csts_parts, axis=0)
        self.mask = np.concatenate(mask_parts, axis=0)

        if num_events is not None and len(self.csts) > num_events:
            self.csts = self.csts[:num_events]
            self.mask = self.mask[:num_events]

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
            "labels": 0,
        }


class COCOATrackIterableDataset(IterableDataset):
    """Streaming dataset that loads one ROOT file at a time."""

    def __init__(
        self,
        root_files: list[str | Path],
        features: list[str] | None = None,
        max_csts: int = 15,
        num_events: int | None = None,
    ) -> None:
        super().__init__()
        if features is None:
            features = TRACK_FEATURES
        self.root_files = [Path(f) for f in root_files]
        self.features = features
        self.max_csts = max_csts
        self.num_events = num_events

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        files = self.root_files

        if worker_info is not None:
            per_worker = len(files) // worker_info.num_workers
            start = worker_info.id * per_worker
            end = start + per_worker if worker_info.id < worker_info.num_workers - 1 else len(files)
            files = files[start:end]

        yielded = 0
        for fpath in files:
            if self.num_events is not None and yielded >= self.num_events:
                return

            remaining = None if self.num_events is None else self.num_events - yielded
            arrays = load_root_arrays(fpath, self.features, max_entries=remaining)
            csts, mask = pad_jagged_to_fixed(arrays, self.features, self.max_csts)

            for i in range(len(csts)):
                if self.num_events is not None and yielded >= self.num_events:
                    return
                yield {"csts": csts[i], "mask": mask[i], "labels": 0}
                yielded += 1


class COCOATrackModule(LightningDataModule):
    """Lightning DataModule for COCOA track data from ROOT files."""

    def __init__(
        self,
        data_dir: str,
        train_dir: str = "train_topup2_20M_cells256",
        val_dir: str = "val_100K_cells256",
        test_dir: str = "test_1M_cells256_isInfFalse",
        features: list[str] | None = None,
        max_csts: int = 15,
        num_events: int | None = None,
        batch_size: int = 1024,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool | None = None,
        transforms: dict | None = None,
        streaming: bool = False,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.features = features or TRACK_FEATURES
        self.max_csts = max_csts
        self.num_events = num_events
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = (
            num_workers > 0 if persistent_workers is None else persistent_workers
        )
        self.transforms = transforms
        self.streaming = streaming

    def _make_dataset(self, split_dir: str):
        files = root_files_from_dir(self.data_dir / split_dir)
        if self.streaming:
            return COCOATrackIterableDataset(
                files, self.features, self.max_csts, self.num_events
            )
        return COCOATrackDataset(files, self.features, self.max_csts, self.num_events)

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
