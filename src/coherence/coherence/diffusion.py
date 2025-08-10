from __future__ import annotations

"""Heat-kernel diffusion approximation (Milestone 4).

Implements spectral heat-kernel smoothing on a graph Laplacian L:
    y = exp(-tau * L) x
Supports single or multiple tau values and batched signals.
"""

from typing import Iterable, List, Sequence, Tuple, Union

import numpy as np


def _eigh(L: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Eigen-decompose symmetric Laplacian."""
    vals, vecs = np.linalg.eigh(L)
    # Ensure float32 for determinism and consistency
    return np.asarray(vals, dtype=np.float32), np.asarray(vecs, dtype=np.float32)


def _apply_heat_kernel(U: np.ndarray, lam: np.ndarray, X: np.ndarray, tau: float) -> np.ndarray:
    """Apply exp(-tau L) to X using spectral decomposition L = U diag(lam) U^T.

    Handles X shape (n,) or (n,m).
    """
    # Compute coefficients in eigenbasis
    if X.ndim == 1:
        c = U.T @ X
        g = np.exp(-tau * lam)
        y = U @ (g * c)
        return np.asarray(y, dtype=np.float32)
    elif X.ndim == 2:
        c = U.T @ X  # (n,n) @ (n,m) -> (n,m)
        g = np.exp(-tau * lam).reshape(-1, 1)
        Y = U @ (g * c)
        return np.asarray(Y, dtype=np.float32)
    else:
        raise ValueError("X must be 1D or 2D")


def diffuse(L: np.ndarray, X: np.ndarray, tau: Union[float, Sequence[float]]) -> np.ndarray:
    """Diffuse signals X over graph Laplacian L with heat kernel exp(-tau L).

    Args
    - L: (n,n) symmetric Laplacian
    - X: (n,) or (n,m) signals
    - tau: float or sequence of floats. If sequence, returns stacked outputs.

    Returns
    - If tau is float: same shape as X
    - If tau is sequence: (len(tau), ...) stacked along first axis
    """
    lam, U = _eigh(L)
    if isinstance(tau, (list, tuple)):
        outs: List[np.ndarray] = []
        for t in tau:
            outs.append(_apply_heat_kernel(U, lam, X, float(t)))
        return np.stack(outs, axis=0)
    else:
        return _apply_heat_kernel(U, lam, X, float(tau))
