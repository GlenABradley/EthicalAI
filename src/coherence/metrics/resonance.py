from __future__ import annotations

"""Resonance metric.

Compute magnitude-sensitive axis utilities and aggregate them into a scalar
resonance score using either linear weights or a Choquet capacity.

No cosine similarity is used.
"""

from typing import Optional, Sequence

import numpy as np

from coherence.axis.pack import AxisPack
from coherence.axis.choquet import choquet_integral


def project(X: np.ndarray, pack: AxisPack) -> np.ndarray:
    """Project vectors onto axis columns Q.

    Args
    - X: (d,) or (n, d)
    - pack: AxisPack with Q of shape (d, k)

    Returns
    - Y: (k,) or (n, k) of raw axis coordinates (no normalization)
    """
    if X.ndim == 1:
        return np.asarray(pack.Q.T @ X, dtype=np.float32)
    elif X.ndim == 2:
        return np.asarray(X @ pack.Q, dtype=np.float32)
    else:
        raise ValueError("X must be 1D or 2D array")


def utilities(coords: np.ndarray, pack: AxisPack) -> np.ndarray:
    """Apply per-axis affine transform: lambda * coord + beta.

    Shapes preserved: (k,) -> (k,), (n,k) -> (n,k)
    """
    lam = pack.lambda_.astype(np.float32)
    beta = pack.beta.astype(np.float32)
    if coords.ndim == 1:
        return lam * coords + beta
    elif coords.ndim == 2:
        return coords * lam.reshape(1, -1) + beta.reshape(1, -1)
    else:
        raise ValueError("coords must be 1D or 2D array")


def aggregate(u: np.ndarray, pack: AxisPack) -> np.ndarray:
    """Aggregate per-axis utilities to a scalar per sample.

    If pack.mu is non-empty, use discrete Choquet integral.
    Otherwise use linear weights dot-product.
    """
    if u.ndim == 1:
        if pack.mu:
            return np.array(choquet_integral(u.tolist(), pack.mu), dtype=np.float32)
        return np.array(np.dot(u, pack.weights), dtype=np.float32)
    elif u.ndim == 2:
        out = []
        if pack.mu:
            for row in u:
                out.append(choquet_integral(row.tolist(), pack.mu))
            return np.asarray(out, dtype=np.float32)
        return np.asarray(u @ pack.weights.reshape(-1, 1), dtype=np.float32).reshape(-1)
    else:
        raise ValueError("u must be 1D or 2D array")


def resonance(X: np.ndarray, pack: AxisPack) -> np.ndarray:
    """End-to-end resonance: aggregate( utilities( project(X, Q) ) ).

    Returns a scalar for 1D input or a (n,) array for 2D.
    """
    coords = project(X, pack)
    u = utilities(coords, pack)
    return aggregate(u, pack)

"""Resonance metrics (Milestone 2).

# TODO: @builder implement projection->utility->aggregate (no cosine)
"""
