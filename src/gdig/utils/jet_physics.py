# Add to imports at top
import awkward as ak
import numpy as np


# Add JetHelper class to the file or import it
class JetHelper:
    def __init__(self, mode="vec_sum"):
        assert mode in ["vec_sum"], (
            f"Unsupported mode {mode}. Only 'vec_sum' is currently supported."
        )
        self.mode = mode

    def compute_jets_PtEtaPhiE(self, pts, etas, phis, es):
        pxs = pts * np.cos(phis)
        pys = pts * np.sin(phis)
        pzs = pts * np.sinh(etas)

        jet_es = ak.sum(es, axis=-1)
        jet_pxs = ak.sum(pxs, axis=-1)
        jet_pys = ak.sum(pys, axis=-1)
        jet_pzs = ak.sum(pzs, axis=-1)

        jet_pts = np.sqrt(jet_pxs**2 + jet_pys**2)
        jet_pts = ak.where(jet_pts == 0, 1e-10, jet_pts)
        jet_etas = np.arcsinh(jet_pzs / jet_pts)
        jet_phis = np.arctan2(jet_pys, jet_pxs)

        return jet_pts, jet_etas, jet_phis, jet_es
