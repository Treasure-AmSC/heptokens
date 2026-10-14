"""Helpers for positions-out data (positions carried separately from ``csts``)."""

import numpy as np


def phi_to_cossin(
    positions: np.ndarray, mask: np.ndarray | None = None, phi_index: int = 1
) -> np.ndarray:
    """Replace the angle column ``phi_index`` of ``positions`` [..., P] by (cos, sin).

    Returns [..., P + 1]; entries where ``mask`` is False are set to zero.
    """
    phi = positions[..., phi_index]
    out = np.concatenate(
        [
            positions[..., :phi_index],
            np.cos(phi)[..., None],
            np.sin(phi)[..., None],
            positions[..., phi_index + 1 :],
        ],
        axis=-1,
    )
    if mask is not None:
        out[~mask] = 0.0
    return out
