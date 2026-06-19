"""Generic map-style dataset for flat HDF5 array files.

Covers any HDF5 file where each key is a plain float/bool array of shape
``[N, ...]``.  The canonical examples are the HEP4M particle-set modalities
(``x_cont``, ``x_mask``, ``positions``) and the calo point-cloud voxel data.

The class accepts a ``key_map`` dict that maps HDF5 dataset keys to output dict
keys, and an optional ``dtypes`` dict for per-key dtype coercion.

Example usage (HEP4M particle-set, no key renaming)::

    ds = FlatArrayHDF5Dataset(
        "data.h5",
        key_map={"x_cont": "x_cont", "x_mask": "x_mask", "positions": "positions"},
        dtypes={"x_cont": torch.float32, "x_mask": torch.bool, "positions": torch.float32},
    )

Example usage (ATLAS jets mapped to generic keys)::

    ds = FlatArrayHDF5Dataset(
        "data.h5",
        key_map={"jets": "jets", "csts": "csts", "mask": "mask"},
    )
"""

import logging
from collections.abc import Callable

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from heptokens.data.base import BaseMapModule

log = logging.getLogger(__name__)

_DEFAULT_DTYPE: dict[str, torch.dtype] = {}


class FlatArrayHDF5Dataset(Dataset):
    """Map-style dataset for flat HDF5 array files.

    Loads the full dataset into memory on construction.

    Args:
        file_path: Path to the HDF5 file.
        key_map: ``{hdf5_key: output_key}`` mapping. All listed keys are loaded;
            the output dict uses the mapped names.
        dtypes: ``{output_key: torch.dtype}`` for explicit dtype coercion.
            Keys not listed here are loaded with their HDF5 native dtype coerced
            to ``float32`` for floating-point arrays and ``bool`` for boolean
            arrays.
        num_samples: If set, only load the first ``num_samples`` events.
    """

    def __init__(
        self,
        file_path: str,
        key_map: dict[str, str],
        dtypes: dict[str, torch.dtype] | None = None,
        num_samples: int | None = None,
    ) -> None:
        super().__init__()
        dtypes = dtypes or {}
        self._data: dict[str, torch.Tensor] = {}

        with h5py.File(file_path, "r") as f:
            for hdf5_key, out_key in key_map.items():
                arr = f[hdf5_key][:num_samples]
                if out_key in dtypes:
                    tensor = torch.tensor(arr, dtype=dtypes[out_key])
                elif arr.dtype == bool or str(arr.dtype) == "bool":
                    tensor = torch.tensor(arr, dtype=torch.bool)
                else:
                    tensor = torch.tensor(arr, dtype=torch.float32)
                self._data[out_key] = tensor

        n = len(next(iter(self._data.values())))
        log.info("Loaded %d events from %s (keys: %s)", n, file_path, list(self._data))

    def __len__(self) -> int:
        return len(next(iter(self._data.values())))

    def __getitem__(self, idx: int) -> dict:
        return {key: tensor[idx] for key, tensor in self._data.items()}


class FlatArrayModule(BaseMapModule):
    """Lightning DataModule wrapping a flat-array dataset with fraction-based splits.

    Accepts a dataset factory (Hydra partial) so dataset params live entirely in
    config — no module code changes needed when dataset kwargs evolve.

    Example Hydra config::

        _target_: heptokens.data.flat_array.FlatArrayModule
        dataset:
          _target_: heptokens.data.flat_array.FlatArrayHDF5Dataset
          _partial_: true
          file_path: /path/to/data.h5
          key_map: {csts: csts, mask: mask}
          num_samples: 10000
        input_key: csts
        batch_size: 512

    Args:
        dataset: Callable that returns a map-style Dataset (use ``_partial_: true``
            in Hydra config).
        input_key: Which output key to return from ``get_data_sample()``.
        train_frac: Fraction of events for training.
        val_frac: Fraction of events for validation.
        test_frac: Fraction of events for testing.
        seed: Random seed for index shuffling before split.
        **kwargs: Passed to BaseMapModule (batch_size, num_workers, etc.).
    """

    def __init__(
        self,
        *,
        dataset: "Callable[..., Dataset]",
        input_key: str = "csts",
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        seed: int = 42,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._make_dataset = dataset
        self.input_key = input_key
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.seed = seed

    def setup(self, stage: str | None = None) -> None:
        full_ds = self._make_dataset()
        n = len(full_ds)
        rng = np.random.default_rng(self.seed)
        indices = rng.permutation(n)

        n_train = int(n * self.train_frac)
        n_val = int(n * self.val_frac)

        self.train_set = Subset(full_ds, indices[:n_train])
        self.valid_set = Subset(full_ds, indices[n_train : n_train + n_val])
        self.test_set = Subset(full_ds, indices[n_train + n_val :])

    def get_data_sample(self) -> torch.Tensor:
        if not hasattr(self, "train_set"):
            self.setup()
        return self.train_set[0][self.input_key]
