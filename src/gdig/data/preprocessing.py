from collections.abc import Iterable

import numpy as np
import torch as T
from sklearn.base import BaseEstimator
from torch import nn
from torch.utils.data import default_collate

from gdig.models.vq_vae import LitVqVae


def collate_and_transform(
    batch: Iterable[dict],
    do_default_collate: bool = True,
    transforms: list[callable] | None = None,
) -> dict:
    """Collate the batch and apply the transforms.

    Why this not lightning's on_before_batch_transfer?
    This still runs inside the pytorch multiprocessing pool for data loading.
    Thus it runs asynchonously for each batch being prepared.
    """
    if do_default_collate:
        batch = default_collate(batch)
    if transforms is not None:
        for transform in transforms:
            batch = transform(batch)
    return batch


def tokenize_batch(
    jet_dict: dict[T.Tensor],
    token_fn: nn.Module,
) -> dict:
    """Add token versions of the constituents to the jet_dict."""
    csts = jet_dict["csts"]
    mask = jet_dict["mask"]
    out = token_fn.predict(csts[mask].T.contiguous()).long()
    jet_dict["tokens"] = T.zeros(mask.shape, dtype=T.long)
    jet_dict["tokens"][mask] = out
    return jet_dict


class VqvaeTokenizer:
    """Add VQ-VAE token IDs to the batch using a trained checkpoint."""

    def __init__(self, ckpt_path: str) -> None:
        self.ckpt_path = ckpt_path
        self._model: LitVqVae | None = None

    def _get_model(self) -> LitVqVae:
        if self._model is None:
            device = T.device("cuda" if T.cuda.is_available() else "cpu")
            self._model = LitVqVae.load_from_checkpoint(self.ckpt_path, map_location=device)
            self._model.to(device)
            self._model.eval()
        return self._model

    def __call__(self, jet_dict: dict[T.Tensor]) -> dict:
        model = self._get_model()
        device = next(model.parameters()).device
        with T.no_grad():
            batch = {
                k: (v.to(device) if isinstance(v, T.Tensor) else v) for k, v in jet_dict.items()
            }
            _, indices, _ = model.encode(batch)
            # indices = indices.to("cpu")  # NOTE: this is needed if running in preprocessing.
        jet_dict["tokens"] = indices
        return jet_dict


def preprocess_batch(
    jet_dict: dict[T.Tensor],
    cst_fn: BaseEstimator,
    jet_fn: BaseEstimator,
) -> dict:
    """Preprocess a batch of jets already stored as pytorch tensors."""
    csts = jet_dict["csts"]
    mask = jet_dict["mask"]
    jets = jet_dict["jets"]

    # Pad the csts with zeros to match what the cst_fn expects
    if (feat_diff := cst_fn.n_features_in_ - csts.shape[-1]) > 0:
        zeros = np.zeros((csts.shape[:-1] + (feat_diff,)), dtype=csts.dtype)
        csts = np.concatenate((csts, zeros), axis=-1)
    csts[mask] = T.from_numpy(cst_fn.transform(csts[mask])).float()
    if feat_diff > 0:
        csts = csts[:, :-feat_diff]  # Remove the padding
    jet_dict["csts"] = csts

    jets = T.from_numpy(jet_fn.transform(jets)).float()
    jet_dict["jets"] = jets

    return jet_dict


def inverse_preprocess_batch(
    jet_dict: dict[T.Tensor],
    cst_fn: BaseEstimator,
    jet_fn: BaseEstimator,
) -> dict:
    """Apply inverse preprocessing to a batch of jets.

    Args:
        jet_dict: Dictionary containing 'csts', 'jets', and 'mask' tensors
        cst_fn: Fitted QuantileTransformer for constituents
        jet_fn: Fitted QuantileTransformer for jets

    Returns:
        Dictionary with inverse-transformed constituents and jets
    """
    csts = jet_dict["csts"].clone()  # Clone to avoid modifying original
    mask = jet_dict["mask"]
    jets = jet_dict["jets"].clone()

    # Inverse transform constituents
    # Only transform valid (masked) constituents
    if mask.any():
        valid_csts = csts[mask].cpu().numpy()
        inverse_csts = cst_fn.inverse_transform(valid_csts)
        csts[mask] = T.from_numpy(inverse_csts).float()

    # Inverse transform jets
    inverse_jets = jet_fn.inverse_transform(jets.cpu().numpy())
    jets = T.from_numpy(inverse_jets).float()

    # Update dictionary
    jet_dict_inverse = jet_dict.copy()
    jet_dict_inverse["csts"] = csts
    jet_dict_inverse["jets"] = jets

    return jet_dict_inverse


def mask_batch(
    jet_dict: dict,
    mask_fraction: float = 0.4,
    key: str = "null_mask",
) -> dict:
    """Applies a masking function of a batch of jets.

    Will add a new key to the jet_dict with the locations of the new mask.
    """
    null_mask = T.stack([mask_jet(mask, mask_fraction=mask_fraction) for mask in jet_dict["mask"]])
    jet_dict[key] = null_mask
    return jet_dict


def mask_jet(
    mask: T.Tensor,
    mask_fraction: float = 0.5,
    seed: int | None = None,
) -> np.ndarray:
    """Randomly drop a fraction of the jet based on the total number of constituents."""
    if seed is not None:
        T.manual_seed(seed)
    max_drop = int(T.floor(mask_fraction * mask.sum()))
    null_mask = T.zeros_like(mask, dtype=T.bool)

    # Exit now if we are not dropping any nodes
    if max_drop == 0:
        return null_mask

    # Generate a random score per node, the lowest frac will be killed
    rand = T.rand(len(mask))
    rand[~mask] = 9999
    drop_idx = T.argsort(rand)[:max_drop]
    null_mask[drop_idx] = True

    return null_mask
