"""For loading events from event-level HDF files and creating a mappable dataset."""

import logging

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, random_split

from heptokens.data.atlas_mappable import BaseMapModule

log = logging.getLogger(__name__)

EVENT_PARTICLE_FEATURES: list[str] = ["pt", "eta", "phi"]


class EventMapDataset(Dataset):
    """Loads event-level data from HDF5 files produced by convert_xaod_to_h5.py.

    The HDF5 file must contain:
    - "events" dataset: structured array with at least an "n_particles" field
    - "particles" dataset: structured array of shape [n_events, max_particles]
      with a "valid" field and feature fields (pt, eta, phi, etc.)
    """

    def __init__(
        self,
        file_path: str,
        particle_features: list | None = None,
        num_events: int | None = None,
        num_particles: int | None = None,
    ) -> None:
        super().__init__()
        if particle_features is None:
            particle_features = list(EVENT_PARTICLE_FEATURES)
        self.particle_features = particle_features

        self.data_dict = {}
        with h5py.File(file_path, mode="r") as handle:
            events_ds = handle["events"]
            particles_ds = handle["particles"]

            total_events = events_ds.shape[0]
            n_events = min(total_events, num_events) if num_events else total_events

            total_particles = particles_ds.shape[1]
            n_particles = min(total_particles, num_particles) if num_particles else total_particles

            # Load particles slice (structured array)
            particles_slice = particles_ds[:n_events, :n_particles]

            # Build csts array: [n_events, n_particles, n_features]
            num_features = len(particle_features)
            self.data_dict["csts"] = np.empty(
                (n_events, n_particles, num_features), dtype=np.float32
            )
            for i, key in enumerate(particle_features):
                self.data_dict["csts"][:, :, i] = particles_slice[key].astype(np.float32)

            # Validity mask
            self.data_dict["mask"] = particles_slice["valid"]

            # Event-level features (analogous to jet features)
            events_slice = events_ds[:n_events]
            self.data_dict["jets"] = events_slice["n_particles"].astype(np.float32).reshape(-1, 1)

            # Placeholder labels (no event-level classification target yet)
            self.data_dict["labels"] = np.zeros(n_events, dtype=np.int64)

        self.num_events = self.data_dict["csts"].shape[0]
        self.num_particles = self.data_dict["csts"].shape[1]
        log.info(
            f"Loaded {self.num_events} events with up to {self.num_particles} particles "
            f"({num_features} features) from {file_path}"
        )

    def __len__(self) -> int:
        return self.num_events

    def __getitem__(self, idx: int) -> dict:
        return {k: v[idx] for k, v in self.data_dict.items()}


class EventSingleFileMapModule(BaseMapModule):
    """DataModule that loads an event-level HDF5 file and splits into train/val/test."""

    def __init__(
        self,
        *,
        data_path: str,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
        seed: int = 42,
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

        full_dataset = EventMapDataset(self.data_path, **self.data_config)
        total_size = len(full_dataset)
        train_size = int(total_size * train_frac)
        val_size = int(total_size * val_frac)
        test_size = total_size - train_size - val_size

        generator = torch.Generator().manual_seed(seed)
        self.train_set, self.valid_set, self.test_set = random_split(
            full_dataset, [train_size, val_size, test_size], generator=generator
        )

    def setup(self, stage: str) -> None:
        """Datasets are already split in __init__."""
