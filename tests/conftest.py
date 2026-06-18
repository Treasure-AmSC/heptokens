import h5py
import numpy as np
import pytest


@pytest.fixture
def hdf5_file(tmp_path):
    """Synthetic two-group HDF5: jets (obj-level) and tracks (cst-level)."""
    n_jets = 100
    n_csts = 20
    path = tmp_path / "jets.h5"
    rng = np.random.default_rng(42)

    with h5py.File(path, "w") as f:
        jets = f.create_group("jets")
        jets.create_dataset("pt", data=rng.uniform(20, 200, size=n_jets).astype(np.float32))
        jets.create_dataset("eta", data=rng.uniform(-2.5, 2.5, size=n_jets).astype(np.float32))
        jets.create_dataset("phi", data=rng.uniform(-3.14, 3.14, size=n_jets).astype(np.float32))
        jets.create_dataset("label", data=rng.choice([0, 1, 2], size=n_jets).astype(np.int64))

        tracks = f.create_group("tracks")
        tracks.create_dataset(
            "pt", data=rng.uniform(0, 50, size=(n_jets, n_csts)).astype(np.float32)
        )
        tracks.create_dataset(
            "eta", data=rng.uniform(-2.5, 2.5, size=(n_jets, n_csts)).astype(np.float32)
        )
        tracks.create_dataset(
            "phi", data=rng.uniform(-3.14, 3.14, size=(n_jets, n_csts)).astype(np.float32)
        )
        valid = np.zeros((n_jets, n_csts), dtype=bool)
        for i in range(n_jets):
            n_valid = rng.integers(3, n_csts + 1)
            valid[i, :n_valid] = True
        tracks.create_dataset("valid", data=valid)

    return path
