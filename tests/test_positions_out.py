"""Checks for the positions-out code path (run: python tests/test_positions_out.py).

Uses real COCOA / JetSet data. Also compares against the heptokens-cocoa (Location B)
implementation, which must give identical positions, bins and jet pT.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import torch as T

from heptokens.callbacks.recon import ReconstructionMonitor
from heptokens.data.atlas_iterable import SingleFileIterModule
from heptokens.data.atlas_mappable import MapDataset
from heptokens.data.cocoa_base import root_files_from_dir
from heptokens.data.cocoa_topos import COCOATopoDataset
from heptokens.data.cocoa_tracks import COCOATrackDataset
from heptokens.models.pos_tokenizer import PositionTokenizer

B_SRC = Path("/sdf/data/atlas/u/jkrupa/heptokens-new/heptokens-cocoa/src")
RES = Path("/sdf/data/atlas/u/jkrupa/heptokens/resources")
COCOA_VAL = Path("/fs/ddn/sdf/group/atlas/d/jkrupa/treasure/4m/data_new/COCOA_samples_singlejet/val_100K_cells256")
JETSET_SMALL = "/sdf/scratch/users/j/jkrupa/data/atlas/mc-flavtag-ttbar-small.h5"
JETSET_CST = [
    "pt", "deta", "dphi", "d0", "d0RelativeToBeamspot", "d0Uncertainty",
    "d0RelativeToBeamspotUncertainty", "z0RelativeToBeamspot", "z0RelativeToBeamspotUncertainty",
    "z0SinTheta", "z0SinThetaUncertainty", "lifetimeSignedD0", "lifetimeSignedD0Significance",
    "lifetimeSignedZ0SinTheta", "lifetimeSignedZ0SinThetaSignificance", "theta", "thetaUncertainty",
    "qOverP", "qOverPUncertainty", "ptfrac",
]
TRACKS = ["track_pt", "track_eta", "track_phi", "track_d0", "track_z0"]
TOPOS = ["topo_eta", "topo_phi", "topo_e", "topo_rho", "topo_sigma_eta", "topo_sigma_phi", "topo_ecal_e", "topo_hcal_e"]
N_EVENTS = 2048
FILES = root_files_from_dir(COCOA_VAL)[:1]


def _drop(features, pos):
    return [f for f in features if f not in pos]


def _cocoa_pair(cls, features, pos):
    full = cls(FILES, features, num_events=N_EVENTS)
    out = cls(FILES, _drop(features, pos), num_events=N_EVENTS, position_features=pos)
    return full, out


def _jetset_pair():
    kw = dict(jet_features=["pt", "mass", "eta", "phi"], num_jets=N_EVENTS, num_csts=40,
              max_jet_pt=7.0e6, max_cst_pt=1.0e6)
    full = MapDataset(JETSET_SMALL, cst_features=JETSET_CST, **kw)
    out = MapDataset(JETSET_SMALL, cst_features=_drop(JETSET_CST, ["deta", "dphi"]),
                     position_features=["deta", "dphi"], **kw)
    return full.data_dict, out.data_dict


def test_sliced_scalers():
    """full_scaler(full_features)[:, kept] == sliced_scaler(kept_features), on real constituents."""
    tr_full, _ = _cocoa_pair(COCOATrackDataset, TRACKS, ["track_eta", "track_phi"])
    to_full, _ = _cocoa_pair(COCOATopoDataset, TOPOS, ["topo_eta", "topo_phi"])
    js_full, _ = _jetset_pair()
    cases = [
        ("JetSet", "cst_quantiles", js_full["csts"][js_full["mask"]], [1, 2]),
        ("COCOA tracks", "cocoa_cst_log_standard", tr_full.csts[tr_full.mask], [1, 2]),
        ("COCOA topos", "cocoa_topo_log_standard", to_full.csts[to_full.mask], [0, 1]),
    ]
    for name, stem, x, drop in cases:
        full = joblib.load(RES / f"{stem}.joblib")
        sliced = joblib.load(RES / f"{stem}_posout.joblib")
        keep = [i for i in range(x.shape[1]) if i not in drop]
        fwd = np.abs(full.transform(x)[:, keep] - sliced.transform(x[:, keep])).max()
        z = full.transform(x)
        inv = np.abs(full.inverse_transform(z)[:, keep] - sliced.inverse_transform(z[:, keep])).max()
        print(f"  sliced scaler {name:13s} rows={len(x):6d} kept={len(keep):2d}  max|fwd diff|={fwd:.3e}  max|inv diff|={inv:.3e}")
        assert fwd <= 1e-6 and inv <= 1e-6 * max(1.0, np.abs(x).max())


def test_dataset_split():
    """positions-out datasets: csts == positions-in csts minus position columns; positions = (eta, cos, sin) or (deta, dphi)."""
    for cls, feats, pos in [(COCOATrackDataset, TRACKS, ["track_eta", "track_phi"]),
                            (COCOATopoDataset, TOPOS, ["topo_eta", "topo_phi"])]:
        full, out = _cocoa_pair(cls, feats, pos)
        keep = [feats.index(f) for f in _drop(feats, pos)]
        i_eta, i_phi = feats.index(pos[0]), feats.index(pos[1])
        m = full.mask
        exp = np.stack([full.csts[..., i_eta], np.cos(full.csts[..., i_phi]), np.sin(full.csts[..., i_phi])], -1)
        exp[~m] = 0
        assert np.array_equal(out.mask, m) and np.array_equal(out.csts, full.csts[..., keep])
        assert np.array_equal(out.positions, exp) and "positions" in out[0] and "positions" not in full[0]
        print(f"  {cls.__name__:17s} csts/mask/positions bitwise equal to positions-in split ({m.sum()} csts)")
    full, out = _jetset_pair()
    keep = [JETSET_CST.index(f) for f in _drop(JETSET_CST, ["deta", "dphi"])]
    # padded (invalid) JetSet track slots hold NaN in the file
    assert np.array_equal(out["mask"], full["mask"])
    assert np.array_equal(out["csts"], full["csts"][..., keep], equal_nan=True)
    assert np.array_equal(out["positions"], full["csts"][..., 1:3], equal_nan=True)
    print(f"  MapDataset (JetSet) csts/mask/positions bitwise equal ({full['mask'].sum()} csts)")
    dm = SingleFileIterModule(
        data_path=JETSET_SMALL, n_classes=4, train_frac=0.001, val_frac=0.001, test_frac=0.998, num_workers=0,
        jet_features=["pt", "mass", "eta", "phi"], cst_features=_drop(JETSET_CST, ["deta", "dphi"]),
        position_features=["deta", "dphi"], num_csts=40,
    )
    s = next(iter(dm.test_set))
    assert s["positions"].shape == (40, 2) and s["csts"].shape == (40, 18)
    print("  IndexedIterMapDataset (JetSet test path) yields positions (40, 2) and csts (40, 18)")


class _Identity:
    """Stand-in LightningModule: 'reconstruction' = input, so only the monitor logic is exercised."""

    def encode(self, batch):
        return (batch["csts"], None, None)

    def decode(self, z, batch):
        return z

    def log(self, *a, **k):
        pass


def test_monitor_guards():
    scaler = joblib.load(RES / "cocoa_cst_log_standard_posout.joblib")
    _, out = _cocoa_pair(COCOATrackDataset, TRACKS, ["track_eta", "track_phi"])
    batch = {"csts": T.from_numpy(scaler.transform(out.csts.reshape(-1, 3)).reshape(out.csts.shape)).float(),
             "mask": T.from_numpy(out.mask), "positions": T.from_numpy(out.positions)}
    no_pos = {k: v for k, v in batch.items() if k != "positions"}
    tok = PositionTokenizer()
    for b, t, msg in [(batch, None, "no pos_tokenizer"), (no_pos, tok, "no 'positions'")]:
        try:
            ReconstructionMonitor(scaler, max_batches=1, pos_tokenizer=t)._on_batch_end(None, _Identity(), b, 0)
        except ValueError as e:
            assert msg in str(e), e
            print(f"  guard raised as expected: {e}")
        else:
            raise AssertionError(f"guard did not raise ({msg})")
    mon = ReconstructionMonitor(scaler, max_batches=1, pos_tokenizer=tok)
    mon._on_batch_end(None, _Identity(), batch, 0)
    r = mon.jet_residuals[0]["pt_ratio"]
    r = r[np.isfinite(r)]
    q = np.percentile(r, [25, 50, 75])
    print(f"  identity 'reconstruction' through monitor: jet pT IQR/median = {(q[2] - q[0]) / q[1]:.3e} (binning only)")


def test_a_vs_b():
    """A vs B: positions, bins and decoded centres bitwise; jet pT from the same reco constituents."""
    sys.path.insert(0, str(B_SRC))
    from heptokens_cocoa.callbacks.topo_recon import TopoReconstructionMonitor
    from heptokens_cocoa.data.cocoa_topos import COCOATopoDataset as BTopo
    from heptokens_cocoa.models.pos_tokenizer import PositionTokenizer as BTok

    content = _drop(TOPOS, ["topo_eta", "topo_phi"])
    a = COCOATopoDataset(FILES, content, num_events=N_EVENTS, position_features=["topo_eta", "topo_phi"])
    b = BTopo(FILES, content_features=content, position_features=["topo_eta", "topo_phi"], num_events=N_EVENTS)
    assert np.array_equal(a.mask, b.mask) and np.array_equal(a.csts, b.csts)
    print(f"  topo positions A vs B: bitwise equal = {np.array_equal(a.positions, b.positions)}, "
          f"max|diff| = {np.abs(a.positions - b.positions).max():.1e}")

    pos = T.from_numpy(b.positions)
    np.savez("/tmp/design_cfg/ab_topo_batch.npz", positions=b.positions, csts=b.csts, mask=b.mask)
    ta, tb = PositionTokenizer(), BTok(ranges=[[-3.0, 3.0], [-1.0, 1.0], [-1.0, 1.0]], n_bins=1024)
    ia, ib = ta(pos), tb(pos)
    da, db = ta.decode(ia), tb.decode(ib)
    print(f"  PositionTokenizer A vs B: indices bitwise equal = {T.equal(ia, ib)}, "
          f"decoded bitwise equal = {T.equal(da, db)} ({ia.numel()} values)")
    assert T.equal(ia, ib) and T.equal(da, db)

    rng = np.random.default_rng(0)
    reco = b.csts.copy()
    reco[..., 0] *= rng.lognormal(0.0, 0.05, size=reco[..., 0].shape).astype(np.float32)
    np.savez("/tmp/design_cfg/ab_topo_reco.npz", reco=reco)
    reco_t, mask_t = T.from_numpy(reco), T.from_numpy(b.mask)
    eta, cosp, sinp = (db[..., i].numpy() for i in range(3))
    jb = TopoReconstructionMonitor(cst_fn=None, energy_idx=0, content_features=content, pos_tokenizer=tb)
    ja = ReconstructionMonitor(cst_fn=None, energy_idx=0, pos_tokenizer=ta)
    pb = jb._compute_jet_from_clusters(reco_t, eta, cosp, sinp, mask_t)["pt"]
    dir_a = ja._directions_from_positions(da)
    pa = ja._compute_jet_from_positions(reco_t, *dir_a, mask_t)["pt"]
    rel = np.abs(pa - pb) / np.maximum(np.abs(pb), 1e-12)
    print(f"  jet pT A vs B on same reco constituents: bitwise equal = {np.array_equal(pa, pb)}, "
          f"max|diff| = {np.abs(pa - pb).max():.3e} GeV, max rel diff = {rel.max():.3e}")
    assert np.allclose(pa, pb, rtol=1e-6, atol=0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            print(name)
            fn()
    print("ALL PASSED")
