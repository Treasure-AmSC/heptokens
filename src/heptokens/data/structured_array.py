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
            obj_ds = f[obj_group]
            set_ds = f[set_group]

            n_total = len(obj_ds[obj_features[0]])
            n = min(n_total, num_samples) if num_samples is not None else n_total

            # Load object features
            obj_array = np.empty((n, len(obj_features)), dtype=np.float32)
            for i, feat in enumerate(obj_features):
                obj_array[:, i] = obj_ds[feat][:n]

            # Load labels
            labels_array = None
            if label_key is not None:
                raw = obj_ds[label_key][:n]
                unique = np.unique(raw)
                label_map = {lbl: idx for idx, lbl in enumerate(unique)}
                labels_array = np.array([label_map[lbl] for lbl in raw], dtype=np.int64)

            # Load set-level features
            total_elements = set_ds[set_features[0]].shape[1]
            nc = min(total_elements, num_elements) if num_elements is not None else total_elements

            set_array = np.empty((n, nc, len(set_features)), dtype=np.float32)
            for i, feat in enumerate(set_features):
                set_array[:, :, i] = set_ds[feat][:n, :nc]

            mask_array = set_ds[mask_key][:n, :nc]

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
            obj_ds = f[obj_group]
            set_ds = f[set_group]

            n_total = len(obj_ds[obj_features[0]])
            if num_samples is not None:
                n_total = min(n_total, num_samples)

            if indices is None:
                self.indices = np.arange(n_total)
            else:
                self.indices = indices[indices < n_total]

            total_elements = set_ds[set_features[0]].shape[1]
            self.num_elements = (
                min(total_elements, num_elements) if num_elements is not None else total_elements
            )

            # Build label map up front (labels are typically small)
            self.label_map: dict | None = None
            if label_key is not None:
                labels = obj_ds[label_key][self.indices]
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
            obj_ds = f[self.obj_group]
            set_ds = f[self.set_group]

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

                # Load object features
                obj_chunk = np.empty((len(chunk_idx), len(self.obj_features)), dtype=np.float32)
                for i, feat in enumerate(self.obj_features):
                    obj_chunk[:, i] = obj_ds[feat][sel]

                # Load labels
                labels_chunk = None
                if self.label_key is not None and self.label_map is not None:
                    raw = obj_ds[self.label_key][sel]
                    labels_chunk = np.array([self.label_map[lbl] for lbl in raw], dtype=np.int64)

                # Load set-level features
                set_chunk = np.empty(
                    (len(chunk_idx), self.num_elements, len(self.set_features)), dtype=np.float32
                )
                for i, feat in enumerate(self.set_features):
                    set_chunk[:, :, i] = set_ds[feat][sel, : self.num_elements]

                mask_chunk = set_ds[self.mask_key][sel, : self.num_elements]

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
    """Lightning DataModule wrapping StructuredArrayHDF5Dataset with fraction-based splits.

    Loads a single HDF5 file and splits events into train/val/test by index.

    Args:
        data_path: Path to the HDF5 file.
        obj_group: Name of the object-level group.
        set_group: Name of the set-level group.
        obj_features: Field names to load from obj_group.
        set_features: Field names to load from set_group.
        label_key: Label field in obj_group (None to omit).
        mask_key: Mask field in set_group.
        num_elements: Cap on set members per event.
        filter_fn: Optional function (obj_array, set_array, mask_array) → bool mask.
            Applied after loading to remove events. Hydra-instantiable.
        train_frac: Fraction of events for training.
        val_frac: Fraction of events for validation.
        test_frac: Fraction of events for testing.
        seed: Random seed for index shuffling before split.
        **kwargs: Passed to BaseMapModule (batch_size, num_workers, etc.).
    """

    def __init__(
        self,
        *,
        data_path: str,
        obj_group: str,
        set_group: str,
        obj_features: list[str],
        set_features: list[str],
        label_key: str | None = None,
        mask_key: str = "valid",
        num_elements: int | None = None,
        filter_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] | None = None,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        seed: int = 42,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.data_path = data_path
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.seed = seed
        self._ds_kwargs = dict(
            file_path=data_path,
            obj_group=obj_group,
            set_group=set_group,
            obj_features=obj_features,
            set_features=set_features,
            label_key=label_key,
            mask_key=mask_key,
            num_elements=num_elements,
            filter_fn=filter_fn,
        )

    def setup(self, stage: str | None = None) -> None:
        full_ds = StructuredArrayHDF5Dataset(**self._ds_kwargs)
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

    def get_data_sample(self) -> torch.Tensor:
        if not hasattr(self, "train_set"):
            self.setup()
        return self.train_set[0]["csts"].clone()
