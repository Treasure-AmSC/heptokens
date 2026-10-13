"""Positions-out building blocks: PositionTokenizer, phi_to_cossin, slice_transformer and
``position_features`` in the structured-array datasets."""

import h5py
import numpy as np
import pytest
import torch
from sklearn.preprocessing import QuantileTransformer

from heptokens.data.positions import phi_to_cossin
from heptokens.data.structured_array import (
    StructuredArrayHDF5Dataset,
    StructuredArrayIterableDataset,
)
from heptokens.data.transforms import create_preprocessing_transformer, slice_transformer
from heptokens.models.pos_tokenizer import PositionTokenizer


def test_tokenizer_bins_clamps_and_decodes():
    tok = PositionTokenizer(ranges=[(-1.0, 1.0), (0.0, 4.0)], n_bins=[4, 2])
    x = torch.tensor([[-1.0, 0.0], [-0.26, 1.99], [0.99, 2.0], [5.0, -3.0], [-5.0, 9.0]])
    idx = tok(x)
    assert idx.dtype == torch.long
    assert idx.tolist() == [[0, 0], [1, 0], [3, 1], [3, 0], [0, 1]]
    assert torch.equal(tok.decode(idx[:3]), torch.tensor([[-0.75, 1.0], [-0.25, 1.0], [0.75, 3.0]]))


def test_tokenizer_roundtrip_within_half_bin():
    tok = PositionTokenizer()
    x = torch.stack(
        [
            torch.empty(1000).uniform_(-3, 3),
            torch.empty(1000).uniform_(-1, 1),
            torch.empty(1000).uniform_(-1, 1),
        ],
        dim=-1,
    )
    assert torch.all((tok.decode(tok(x)) - x).abs() <= tok.widths / 2 + 1e-6)


def test_tokenizer_rejects_wrong_shapes():
    with pytest.raises(ValueError):
        PositionTokenizer(ranges=[(0, 1), (0, 1)], n_bins=[4])
    with pytest.raises(ValueError):
        PositionTokenizer()(torch.zeros(5, 2))


def test_phi_to_cossin():
    pos = np.array([[[0.5, 0.3], [1.0, -2.0]]], dtype=np.float32)
    mask = np.array([[True, False]])
    out = phi_to_cossin(pos, mask)
    cos, sin = np.cos(np.float32(0.3)), np.sin(np.float32(0.3))
    assert out.shape == (1, 2, 3)
    assert np.array_equal(out[0, 0], [0.5, cos, sin])
    assert np.array_equal(out[0, 1], [0.0, 0.0, 0.0])
    assert np.array_equal(phi_to_cossin(pos[..., ::-1], phi_index=0)[0, 0], [cos, sin, 0.5])


def test_slice_transformer_matches_full_on_kept_columns():
    rng = np.random.default_rng(0)
    x = rng.lognormal(size=(500, 5)).astype(np.float32)
    full = create_preprocessing_transformer(
        "log_standard", log_feature_indices=[0, 3], n_features=5
    )
    full.fit(x)
    keep = [0, 3, 4]
    sliced = slice_transformer(full, keep)
    assert sliced.n_features_in_ == 3
    assert np.allclose(sliced.transform(x[:, keep]), full.transform(x)[:, keep], rtol=0, atol=1e-6)
    z = full.transform(x)
    assert np.allclose(
        sliced.inverse_transform(z[:, keep]), full.inverse_transform(z)[:, keep], rtol=1e-6
    )


def test_slice_transformer_rejects_quantile():
    with pytest.raises(TypeError):
        slice_transformer(QuantileTransformer(), [0])


def _compound_file(group_path, tmp_path):
    """Same content as the ``hdf5_file`` fixture, stored as compound datasets."""
    path = tmp_path / "compound.h5"
    with h5py.File(group_path) as src, h5py.File(path, "w") as dst:
        for grp in ("jets", "tracks"):
            names = list(src[grp])
            arr = np.empty(
                len(src[grp][names[0]]),
                dtype=[(k, src[grp][k].dtype, src[grp][k].shape[1:]) for k in names],
            )
            for k in names:
                arr[k] = src[grp][k][:]
            dst.create_dataset(grp, data=arr)
    return path


@pytest.mark.parametrize("layout", ["group", "compound"])
def test_map_dataset_position_features(hdf5_file, tmp_path, layout):
    path = hdf5_file if layout == "group" else _compound_file(hdf5_file, tmp_path)
    kw = dict(file_path=str(path), obj_group="jets", set_group="tracks", obj_features=["pt"])
    full = StructuredArrayHDF5Dataset(set_features=["pt", "eta", "phi"], **kw)
    out = StructuredArrayHDF5Dataset(set_features=["pt"], position_features=["eta", "phi"], **kw)
    assert "positions" not in full[0]
    assert torch.equal(out.set_data, full.set_data[..., :1])
    assert torch.equal(out.positions, full.set_data[..., 1:])
    assert torch.equal(out[3]["positions"], full[3]["csts"][:, 1:])


def test_map_dataset_positions_follow_filter(hdf5_file):
    kw = dict(file_path=str(hdf5_file), obj_group="jets", set_group="tracks", obj_features=["pt"])
    keep_high_pt = lambda obj, csts, mask: obj[:, 0] > 100  # noqa: E731
    full = StructuredArrayHDF5Dataset(
        set_features=["pt", "eta", "phi"], filter_fn=keep_high_pt, **kw
    )
    out = StructuredArrayHDF5Dataset(
        set_features=["pt"], position_features=["eta", "phi"], filter_fn=keep_high_pt, **kw
    )
    assert len(out) == len(full) < 100
    assert torch.equal(out.positions, full.set_data[..., 1:])


@pytest.mark.parametrize("layout", ["group", "compound"])
def test_iterable_dataset_position_features(hdf5_file, tmp_path, layout):
    path = hdf5_file if layout == "group" else _compound_file(hdf5_file, tmp_path)
    kw = dict(
        file_path=str(path), obj_group="jets", set_group="tracks", obj_features=["pt"], chunk_size=7
    )
    full = list(StructuredArrayIterableDataset(set_features=["pt", "eta", "phi"], **kw))
    out = list(
        StructuredArrayIterableDataset(set_features=["pt"], position_features=["eta", "phi"], **kw)
    )
    assert len(out) == len(full) == 100
    assert "positions" not in full[0]
    for a, b in zip(full, out):
        assert np.array_equal(b["csts"], a["csts"][:, :1])
        assert np.array_equal(b["positions"], a["csts"][:, 1:])
