"""Dataloaders for Z'->ttbar (tops) and QCD jet files from CERNBox.

File format (both tops and QCD):
    PFCands        : (n_jets, 150, 11) float32
    jet_kinematics : (n_jets, 4)       float32
    jet_tagging    : (n_jets, 13)      float32
    event_info     : (n_jets, 3)       int64
    weight         : (n_jets,)         float64   [QCD only – pt reweighting]

PFCands feature indices (PFCANDS_FEATURES):
    0  px       [GeV]
    1  py       [GeV]
    2  pz       [GeV]
    3  E        [GeV]
    4  d0val    [cm]  (-1 for neutrals)
    5  d0err    [cm]  (-1 for neutrals)
    6  dzval    [cm]  (-1 for neutrals)
    7  dzerr    [cm]  (-1 for neutrals)
    8  charge   {-1, 0, 1}
    9  pdgId    {-211,-13,-11,1,11,13,22,130,211}
    10 valid    {0, 1}  (0 = padding slot)

jet_kinematics feature indices (JET_KIN_FEATURES):
    0  pt   [GeV]
    1  eta
    2  phi
    3  mass [GeV]
"""

import logging
from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path

import h5py
import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler, random_split

try:
    from heptokens.data.collation import collate_and_transform
except ImportError:  # allow import without full project deps installed
    collate_and_transform = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

# ── feature name constants ────────────────────────────────────────────────────

PFCANDS_FEATURES = [
    "px", "py", "pz", "E",
    "d0val", "d0err", "dzval", "dzerr",
    "charge", "pdgId", "valid",
]

JET_KIN_FEATURES = ["pt", "eta", "phi", "mass"]

# Default: use all kinematic + IP features, but exclude the "valid" flag
# (validity is captured by the mask instead)
DEFAULT_CST_FEATURES = list(range(10))   # indices 0-9
DEFAULT_JET_FEATURES = list(range(4))    # indices 0-3


# ── single-file dataset ───────────────────────────────────────────────────────

class ZPrimeDataset(Dataset):
    """Mappable dataset for a single tops or QCD HDF5 file.

    Each sample is a dict:
        csts   : float32 (num_csts, n_cst_features)
        mask   : bool    (num_csts,)              True = real particle
        jets   : float32 (n_jet_features,)
        labels : int64   scalar
        weights: float32 scalar                   [only if load_weights=True]

    Args:
        file_path: Path to the HDF5 file.
        label: Integer class label assigned to every jet in this file.
            Typically 0 = QCD, 1 = tops.
        num_jets: Maximum number of jets to load. None loads all.
        num_csts: Maximum number of constituents per jet. None keeps all 150.
        cst_feature_indices: Which PFCands features to include (default 0-9).
        jet_feature_indices: Which jet_kinematics features to include (default 0-3).
        load_weights: If True, try to load the ``weight`` dataset for pt
            reweighting (present in QCD files). Silently skipped if absent.
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
    ) -> None:
        super().__init__()

        if cst_feature_indices is None:
            cst_feature_indices = DEFAULT_CST_FEATURES
        if jet_feature_indices is None:
            jet_feature_indices = DEFAULT_JET_FEATURES

        self.label = label
        self.cst_feature_indices = cst_feature_indices
        self.jet_feature_indices = jet_feature_indices

        with h5py.File(file_path, mode="r") as f:
            n_total = f["PFCands"].shape[0]
            n_jets = min(n_total, num_jets) if num_jets is not None else n_total
            max_csts = f["PFCands"].shape[1]
            if num_csts is not None:
                max_csts = min(max_csts, num_csts)

            # ── constituents ──────────────────────────────────────────────────
            raw = f["PFCands"][:n_jets, :max_csts, :]  # (N, C, 11)

            # validity mask: dedicated "valid" flag (index 10)
            self.mask = raw[:, :, 10].astype(bool)  # (N, C)

            # requested constituent features
            self.csts = raw[:, :, cst_feature_indices].astype(np.float32)  # (N, C, F)

            # ── jet-level features ────────────────────────────────────────────
            jk = f["jet_kinematics"][:n_jets, :]          # (N, 4)
            self.jets = jk[:, jet_feature_indices].astype(np.float32)  # (N, Fj)

            # ── labels ────────────────────────────────────────────────────────
            self.labels = np.full(n_jets, label, dtype=np.int64)

            # ── optional weights ──────────────────────────────────────────────
            self.weights: np.ndarray | None = None
            if load_weights:
                if "weight" in f:
                    self.weights = f["weight"][:n_jets].astype(np.float32)
                else:
                    log.warning(
                        "load_weights=True but no 'weight' dataset found in %s", file_path
                    )

        self.n_jets = n_jets
        log.info(
            "Loaded %d jets (label=%d) from %s", n_jets, label, Path(file_path).name
        )

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self.n_jets

    def __getitem__(self, idx: int) -> dict:
        item = {
            "csts": self.csts[idx],
            "mask": self.mask[idx],
            "jets": self.jets[idx],
            "labels": self.labels[idx],
        }
        if self.weights is not None:
            item["weights"] = self.weights[idx]
        return item


# ── multi-file convenience wrapper ────────────────────────────────────────────

class MultiFileDataset(Dataset):
    """Concatenates multiple :class:`ZPrimeDataset` instances into one.

    Useful for combining several tops batches or QCD pt-bins.

    Args:
        file_paths: List of HDF5 file paths.
        label: Class label applied to all files.
        num_jets_per_file: Optional cap on jets loaded from each file.
        **dataset_kwargs: Forwarded to :class:`ZPrimeDataset`.
    """

    def __init__(
        self,
        file_paths: list[str],
        label: int,
        num_jets_per_file: int | None = None,
        **dataset_kwargs,
    ) -> None:
        super().__init__()
        datasets = [
            ZPrimeDataset(p, label=label, num_jets=num_jets_per_file, **dataset_kwargs)
            for p in file_paths
        ]
        self._cat = ConcatDataset(datasets)
        # Aggregate weights if any file has them
        weight_arrays = [d.weights for d in datasets]
        if all(w is not None for w in weight_arrays):
            self.weights = np.concatenate(weight_arrays)
        else:
            self.weights = None

    def __len__(self) -> int:
        return len(self._cat)

    def __getitem__(self, idx: int) -> dict:
        return self._cat[idx]


# ── DataModule ────────────────────────────────────────────────────────────────

class ZPrimeMapModule(LightningDataModule):
    """Lightning DataModule combining QCD (label=0) and tops (label=1) jets.

    Loads one or more HDF5 files for each class, concatenates them, then
    performs a random train / val / test split.

    Args:
        tops_paths: List of tops HDF5 file paths (Z'->ttbar jets).
        qcd_paths: List of QCD HDF5 file paths.
        num_jets_per_file: Cap on jets loaded from each individual file.
        train_frac: Fraction of combined dataset used for training.
        val_frac: Fraction used for validation.
        test_frac: Fraction used for testing.
        seed: Random seed for the split.
        cst_feature_indices: PFCands feature indices to expose.
        jet_feature_indices: jet_kinematics feature indices to expose.
        num_csts: Maximum constituents per jet (≤150).
        use_weights_qcd: If True load per-jet pt weights from QCD files.
        weighted_sampling: If True use :class:`WeightedRandomSampler` for the
            training DataLoader to reweight the QCD pt spectrum.
        batch_size: DataLoader batch size.
        num_workers: DataLoader worker processes.
        pin_memory: DataLoader pin_memory flag.
        transforms: Optional dict of transform configs forwarded to the
            collation function (same interface as existing dataloaders).
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
        use_weights_qcd: bool = False,
        weighted_sampling: bool = False,
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
        self.weighted_sampling = weighted_sampling

        dataset_kwargs = dict(
            num_csts=num_csts,
            cst_feature_indices=cst_feature_indices,
            jet_feature_indices=jet_feature_indices,
        )

        tops_ds = MultiFileDataset(
            tops_paths, label=1, num_jets_per_file=num_jets_per_file, **dataset_kwargs
        )
        qcd_ds = MultiFileDataset(
            qcd_paths, label=0, num_jets_per_file=num_jets_per_file,
            load_weights=use_weights_qcd, **dataset_kwargs
        )

        full_dataset = ConcatDataset([qcd_ds, tops_ds])
        total = len(full_dataset)
        train_size = int(total * train_frac)
        val_size = int(total * val_frac)
        test_size = total - train_size - val_size

        generator = torch.Generator().manual_seed(seed)
        self.train_set, self.valid_set, self.test_set = random_split(
            full_dataset, [train_size, val_size, test_size], generator=generator
        )

        # Build sample weights for WeightedRandomSampler (training only)
        self._train_sample_weights: torch.Tensor | None = None
        if weighted_sampling:
            self._train_sample_weights = self._build_sample_weights(
                full_dataset, qcd_ds, self.train_set.indices
            )

        log.info(
            "ZPrimeMapModule: %d train / %d val / %d test jets (%d tops + %d QCD total)",
            train_size, val_size, test_size, len(tops_ds), len(qcd_ds),
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_sample_weights(
        full_dataset: ConcatDataset,
        qcd_ds: MultiFileDataset,
        train_indices: list[int],
    ) -> torch.Tensor:
        """Create per-sample weights for the training set.

        Jets from QCD files that carry HDF5 weights are reweighted by those
        values.  Tops jets and unweighted QCD jets receive weight 1.
        """
        n_total = len(full_dataset)
        all_weights = np.ones(n_total, dtype=np.float32)

        # QCD lives at the start of full_dataset (ConcatDataset([qcd, tops]))
        if qcd_ds.weights is not None:
            all_weights[: len(qcd_ds)] = qcd_ds.weights

        train_w = all_weights[train_indices]
        return torch.from_numpy(train_w)

    def _make_collate(self):
        if self.transforms is not None:
            return partial(collate_and_transform, transforms=self.transforms)
        return None

    def _make_dataloader(self, dataset, *, shuffle: bool, sampler=None) -> DataLoader:
        kwargs: dict = {}
        if self.num_workers > 0:
            kwargs["persistent_workers"] = True
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=shuffle,
            collate_fn=self._make_collate(),
            **kwargs,
        )

    # ── LightningDataModule interface ─────────────────────────────────────────

    def setup(self, stage: str) -> None:
        """Splits are pre-built in __init__; nothing to do here."""

    def train_dataloader(self) -> DataLoader:
        sampler = None
        if self.weighted_sampling and self._train_sample_weights is not None:
            sampler = WeightedRandomSampler(
                self._train_sample_weights,
                num_samples=len(self.train_set),
                replacement=True,
            )
        return self._make_dataloader(self.train_set, shuffle=sampler is None, sampler=sampler)

    def val_dataloader(self) -> DataLoader:
        return self._make_dataloader(self.valid_set, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_dataloader(self.test_set, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def get_n_classes(self) -> int:
        return 2

    def get_data_sample(self) -> dict:
        return next(iter(self.valid_set))
