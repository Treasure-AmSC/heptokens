"""Generic map and iterable datasets for two-group HDF5 structured arrays.

Covers any HDF5 file that stores event-level data in one group and
set-level data in a second group, with named fields accessed as
``ds["fieldname"][:]``.  The canonical example is ATLAS jets/tracks, but the
classes are fully configurable and work for any similar schema.

Typical HDF5 layout::

    <obj_group>/
        <obj_features[0]>: float [N]
        <obj_features[1]>: float [N]
        ...
        <label_key>:       int   [N]       # optional
    <set_group>/
        <set_features[0]>: float [N, max_elements]
        <set_features[1]>: float [N, max_elements]
        ...
        <mask_key>:        bool  [N, max_elements]
"""

import logging
from collections.abc import Callable

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, Subset

from heptokens.data.base import BaseMapModule

log = logging.getLogger(__name__)


class StructuredArrayHDF5Dataset(Dataset):
    """Map-style dataset for two-group HDF5 structured arrays.

    Loads the full dataset into memory on construction.

    Args:
        file_path: Path to the HDF5 file.
        obj_group: Name of the object-level group (e.g. ``"jets"``).
        set_group: Name of the set-level group (e.g. ``"tracks"``).
        obj_features: Field names to load from ``obj_group``.
        set_features: Field names to load from ``set_group``.
        label_key: Field name in ``obj_group`` used as classification label.
            Pass ``None`` to omit labels from the output dict.
        mask_key: Field name in ``set_group`` used as the validity mask.
        output_obj_key: Output dict key for object-level features.
        output_cst_key: Output dict key for set-level features.
        output_mask_key: Output dict key for the mask.
        output_labels_key: Output dict key for labels (only used when
            ``label_key`` is not ``None``).
        num_samples: If set, only load the first ``num_samples`` events.
        num_elements: If set, only load the first ``num_elements`` set members.
    """

    def __init__(
        self,
        file_path: str,
        obj_group: str,
        set_group: str,
        obj_features: list[str],
        set_features: list[str],
        label_key: str | None = None,
        mask_key: str = "valid",
        output_obj_key: str = "jets",
        output_cst_key: str = "csts",
        output_mask_key: str = "mask",
        output_labels_key: str = "labels",
        num_samples: int | None = None,
        num_elements: int | None = None,
        filter_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] | None = None,
    ) -> None:
        super().__init__()
        self.output_obj_key = output_obj_key
        self.output_cst_key = output_cst_key
        self.output_mask_key = output_mask_key
        self.output_labels_key = output_labels_key
        self.label_key = label_key

        with h5py.File(file_path, mode="r") as f:
            obj_node = f[obj_group]
            set_node = f[set_group]

            # Compound datasets: single slice read (avoids re-decompressing per field)
            # Groups: per-field read (only option)
            if isinstance(obj_node, h5py.Dataset):
                n_total = len(obj_node)
                n = min(n_total, num_samples) if num_samples is not None else n_total
                obj_raw = obj_node[:n]
                obj_array = np.column_stack(
                    [obj_raw[feat].astype(np.float32) for feat in obj_features]
                )
                raw_labels = obj_raw[label_key] if label_key is not None else None
            else:
                n_total = len(obj_node[obj_features[0]])
                n = min(n_total, num_samples) if num_samples is not None else n_total
                obj_array = np.empty((n, len(obj_features)), dtype=np.float32)
                for i, feat in enumerate(obj_features):
                    obj_array[:, i] = obj_node[feat][:n]
                raw_labels = obj_node[label_key][:n] if label_key is not None else None

            labels_array = None
            if raw_labels is not None:
                unique = np.unique(raw_labels)
                label_map = {lbl: idx for idx, lbl in enumerate(unique)}
                labels_array = np.array([label_map[lbl] for lbl in raw_labels], dtype=np.int64)

            if isinstance(set_node, h5py.Dataset):
                set_raw = set_node[:n]
                first_field = set_raw[set_features[0]]
                total_elements = first_field.shape[1] if first_field.ndim > 1 else 1
                nc = (
                    min(total_elements, num_elements)
                    if num_elements is not None
                    else total_elements
                )
                set_array = np.empty((n, nc, len(set_features)), dtype=np.float32)
                for i, feat in enumerate(set_features):
                    set_array[:, :, i] = set_raw[feat][:, :nc]
                mask_array = set_raw[mask_key][:, :nc]
            else:
                total_elements = set_node[set_features[0]].shape[1]
                nc = (
                    min(total_elements, num_elements)
                    if num_elements is not None
                    else total_elements
                )
                set_array = np.empty((n, nc, len(set_features)), dtype=np.float32)
                for i, feat in enumerate(set_features):
                    set_array[:, :, i] = set_node[feat][:n, :nc]
                mask_array = set_node[mask_key][:n, :nc]

        if filter_fn is not None:
            keep = filter_fn(obj_array, set_array, mask_array)
            n_before = n
            obj_array = obj_array[keep]
            set_array = set_array[keep]
            mask_array = mask_array[keep]
            if labels_array is not None:
                labels_array = labels_array[keep]
            n = int(keep.sum())
            log.info("Filtered %d → %d events", n_before, n)

        self.obj = torch.from_numpy(obj_array)
        self.set_data = torch.from_numpy(set_array)
        self.mask = torch.from_numpy(mask_array)
        self.labels = torch.from_numpy(labels_array) if labels_array is not None else None

        log.info(
            "Loaded %d events from %s (elements=%d, obj_feats=%d, set_feats=%d)",
            n,
            file_path,
            nc,
            len(obj_features),
            len(set_features),
        )

    def __len__(self) -> int:
        return len(self.obj)

    def __getitem__(self, idx: int) -> dict:
        sample = {
            self.output_obj_key: self.obj[idx],
            self.output_cst_key: self.set_data[idx],
            self.output_mask_key: self.mask[idx],
        }
        if self.labels is not None:
            sample[self.output_labels_key] = self.labels[idx]
        return sample


class StructuredArrayIterableDataset(IterableDataset):
    """Iterable dataset for two-group HDF5 structured arrays.

    Streams data in chunks without pre-loading. Supports multi-worker
    DataLoaders via ``get_worker_info`` splitting, and uses contiguous-slice
    HDF5 access when possible (orders of magnitude faster than fancy indexing).

    An optional ``indices`` array restricts the dataset to a subset of events,
    enabling train/val/test splits from a single file without duplication.

    Args:
        file_path: Path to the HDF5 file.
        obj_group: Name of the object-level group (e.g. ``"jets"``).
        set_group: Name of the set-level group (e.g. ``"tracks"``).
        obj_features: Field names to load from ``obj_group``.
        set_features: Field names to load from ``set_group``.
        label_key: Field name in ``obj_group`` used as label. ``None`` to omit.
        mask_key: Field name in ``set_group`` used as the validity mask.
        output_obj_key: Output dict key for object-level features.
        output_cst_key: Output dict key for set-level features.
        output_mask_key: Output dict key for the mask.
        output_labels_key: Output dict key for labels.
        num_samples: Cap total events (applied before ``indices``).
        num_elements: Cap number of set members loaded per event.
        indices: If provided, only iterate over these row indices. Indices are
            sorted for efficient HDF5 access.
        chunk_size: Number of events loaded per HDF5 read call.
    """

    def __init__(
        self,
        file_path: str,
        obj_group: str,
        set_group: str,
        obj_features: list[str],
        set_features: list[str],
        label_key: str | None = None,
        mask_key: str = "valid",
        output_obj_key: str = "jets",
        output_cst_key: str = "csts",
        output_mask_key: str = "mask",
        output_labels_key: str = "labels",
        num_samples: int | None = None,
        num_elements: int | None = None,
        indices: np.ndarray | None = None,
        chunk_size: int = 1000,
        filter_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], bool] | None = None,
    ) -> None:
        super().__init__()
        self.file_path = file_path
        self.obj_group = obj_group
        self.set_group = set_group
        self.obj_features = obj_features
        self.set_features = set_features
        self.label_key = label_key
        self.mask_key = mask_key
        self.output_obj_key = output_obj_key
        self.output_cst_key = output_cst_key
        self.output_mask_key = output_mask_key
        self.output_labels_key = output_labels_key
        self.chunk_size = chunk_size
        self.filter_fn = filter_fn

        with h5py.File(file_path, mode="r") as f:
            obj_node = f[obj_group]
            set_node = f[set_group]
            self._obj_is_compound = isinstance(obj_node, h5py.Dataset)
            self._set_is_compound = isinstance(set_node, h5py.Dataset)

            n_total = len(obj_node) if self._obj_is_compound else len(obj_node[obj_features[0]])
            if num_samples is not None:
                n_total = min(n_total, num_samples)

            if indices is None:
                self.indices = np.arange(n_total)
            else:
                self.indices = indices[indices < n_total]

            if self._set_is_compound:
                sample_row = set_node[0]
                first_field = sample_row[set_features[0]]
                total_elements = len(first_field) if hasattr(first_field, "__len__") else 1
            else:
                total_elements = set_node[set_features[0]].shape[1]
            self.num_elements = (
                min(total_elements, num_elements) if num_elements is not None else total_elements
            )

            # Build label map up front (labels are typically small)
            self.label_map: dict | None = None
            if label_key is not None:
                if self._obj_is_compound:
                    labels = obj_node[label_key][self.indices]
                else:
                    labels = obj_node[label_key][self.indices]
                unique = np.unique(labels)
                self.label_map = {lbl: idx for idx, lbl in enumerate(unique)}

        log.info(
            "StructuredArrayIterableDataset: %d events from %s",
            len(self.indices),
            file_path,
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            worker_indices = self.indices
            initial_chunk = self.chunk_size
        else:
            n_workers = worker_info.num_workers
            worker_id = worker_info.id
            per_worker = int(np.ceil(len(self.indices) / n_workers))
            start = worker_id * per_worker
            end = min(start + per_worker, len(self.indices))
            worker_indices = self.indices[start:end]
            # Stagger first chunk so workers don't all hit the file simultaneously
            initial_chunk = max(
                1,
                self.chunk_size - worker_id * (self.chunk_size // max(n_workers, 1)),
            )

        sorted_indices = np.sort(worker_indices)

        with h5py.File(self.file_path, mode="r") as f:
            obj_node = f[self.obj_group]
            set_node = f[self.set_group]

            pos = 0
            first = True
            while pos < len(sorted_indices):
                cs = initial_chunk if first else self.chunk_size
                first = False
                chunk_end = min(pos + cs, len(sorted_indices))
                chunk_idx = sorted_indices[pos:chunk_end]
                pos = chunk_end

                # Use slice when indices are contiguous (much faster for HDF5)
                idx_start, idx_end = int(chunk_idx[0]), int(chunk_idx[-1]) + 1
                sel = (
                    slice(idx_start, idx_end)
                    if idx_end - idx_start == len(chunk_idx)
                    else chunk_idx
                )

                # Single read per node for compound datasets
                if self._obj_is_compound:
                    obj_raw = obj_node[sel]
                    obj_chunk = np.column_stack(
                        [obj_raw[feat].astype(np.float32) for feat in self.obj_features]
                    )
                    labels_chunk = None
                    if self.label_key is not None and self.label_map is not None:
                        labels_chunk = np.array(
                            [self.label_map[lbl] for lbl in obj_raw[self.label_key]], dtype=np.int64
                        )
                else:
                    obj_chunk = np.empty((len(chunk_idx), len(self.obj_features)), dtype=np.float32)
                    for i, feat in enumerate(self.obj_features):
                        obj_chunk[:, i] = obj_node[feat][sel]
                    labels_chunk = None
                    if self.label_key is not None and self.label_map is not None:
                        raw = obj_node[self.label_key][sel]
                        labels_chunk = np.array(
                            [self.label_map[lbl] for lbl in raw], dtype=np.int64
                        )

                if self._set_is_compound:
                    set_raw = set_node[sel]
                    set_chunk = np.empty(
                        (len(chunk_idx), self.num_elements, len(self.set_features)),
                        dtype=np.float32,
                    )
                    for i, feat in enumerate(self.set_features):
                        set_chunk[:, :, i] = set_raw[feat][:, : self.num_elements]
                    mask_chunk = set_raw[self.mask_key][:, : self.num_elements]
                else:
                    set_chunk = np.empty(
                        (len(chunk_idx), self.num_elements, len(self.set_features)),
                        dtype=np.float32,
                    )
                    for i, feat in enumerate(self.set_features):
                        set_chunk[:, :, i] = set_node[feat][sel, : self.num_elements]
                    mask_chunk = set_node[self.mask_key][sel, : self.num_elements]

                for i in range(len(chunk_idx)):
                    if self.filter_fn is not None and not self.filter_fn(
                        obj_chunk[i], set_chunk[i], mask_chunk[i]
                    ):
                        continue
                    sample = {
                        self.output_obj_key: obj_chunk[i],
                        self.output_cst_key: set_chunk[i],
                        self.output_mask_key: mask_chunk[i],
                    }
                    if labels_chunk is not None:
                        sample[self.output_labels_key] = labels_chunk[i]
                    yield sample


class StructuredArrayModule(BaseMapModule):
    """Lightning DataModule wrapping a map-style dataset with fraction-based splits.

    Accepts a dataset factory (Hydra partial) so dataset params live entirely in
    config — no module code changes needed when dataset kwargs evolve.

    Example Hydra config::

        _target_: heptokens.data.structured_array.StructuredArrayModule
        dataset:
          _target_: heptokens.data.structured_array.StructuredArrayHDF5Dataset
          _partial_: true
          file_path: /path/to/data.h5
          obj_group: jets
          set_group: tracks
          obj_features: [pt, eta, phi]
          set_features: [pt, deta, dphi]
          num_samples: 100000
          num_elements: 40
        batch_size: 1024

    Args:
        dataset: Callable that returns a map-style Dataset (use ``_partial_: true``
            in Hydra config to pass a partially-applied dataset constructor).
        train_frac: Fraction of events for training.
        val_frac: Fraction of events for validation.
        test_frac: Fraction of events for testing.
        seed: Random seed for index shuffling before split.
        **kwargs: Passed to BaseMapModule (batch_size, num_workers, etc.).
    """

    def __init__(
        self,
        *,
        dataset: Callable[..., Dataset],
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        seed: int = 42,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._make_dataset = dataset
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.seed = seed

    def setup(self, stage: str | None = None) -> None:
        full_ds = self._make_dataset()
        n = len(full_ds)
        num_elements = full_ds.set_data.shape[1]
        rng = np.random.default_rng(self.seed)
        indices = rng.permutation(n)

        n_train = int(n * self.train_frac)
        n_val = int(n * self.val_frac)

        self.train_set = Subset(full_ds, indices[:n_train])
        self.valid_set = Subset(full_ds, indices[n_train : n_train + n_val])
        self.test_set = Subset(full_ds, indices[n_train + n_val :])
        self.test_set.num_elements = num_elements

        if full_ds.labels is not None and self.n_classes is None:
            self.n_classes = int(full_ds.labels.max()) + 1

    def get_data_sample(self) -> torch.Tensor:
        if not hasattr(self, "train_set"):
            self.setup()
        return self.train_set[0]["csts"].clone()


class StructuredArrayIterableModule(BaseMapModule):
    """Lightning DataModule wrapping an iterable dataset with fraction-based splits.

    Streams data from HDF5 in chunks without loading the full file into memory.
    Accepts a dataset factory (Hydra partial) targeting StructuredArrayIterableDataset.

    Example Hydra config::

        _target_: heptokens.data.structured_array.StructuredArrayIterableModule
        dataset:
          _target_: heptokens.data.structured_array.StructuredArrayIterableDataset
          _partial_: true
          file_path: /path/to/data.h5
          obj_group: jets
          set_group: tracks
          obj_features: [pt, eta, phi]
          set_features: [pt, deta, dphi]
          num_elements: 40
          chunk_size: 4096
        batch_size: 1024

    Args:
        dataset: Callable that returns an IterableDataset. Called with ``indices=``
            kwarg to create per-split datasets.
        train_frac: Fraction of events for training.
        val_frac: Fraction of events for validation.
        test_frac: Fraction of events for testing.
        seed: Random seed for index shuffling before split.
        **kwargs: Passed to BaseMapModule (batch_size, num_workers, etc.).
    """

    def __init__(
        self,
        *,
        dataset: Callable[..., IterableDataset],
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        seed: int = 42,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._make_dataset = dataset
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.seed = seed

    def setup(self, stage: str | None = None) -> None:
        probe = self._make_dataset()
        n = len(probe)
        num_elements = probe.num_elements
        rng = np.random.default_rng(self.seed)
        indices = rng.permutation(n)

        n_train = int(n * self.train_frac)
        n_val = int(n * self.val_frac)

        self.train_set = self._make_dataset(indices=indices[:n_train])
        self.valid_set = self._make_dataset(indices=indices[n_train : n_train + n_val])
        self.test_set = self._make_dataset(indices=indices[n_train + n_val :])
        self.test_set.num_elements = num_elements

    def train_dataloader(self):
        return self._get_dataloader(self.train_set, shuffle=False, drop_last=True)

    def get_data_sample(self) -> torch.Tensor:
        if not hasattr(self, "train_set"):
            self.setup()
        return next(iter(self.train_set))["csts"]
