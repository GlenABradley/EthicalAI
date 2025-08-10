from __future__ import annotations

"""Graph substrate and Laplacian (Milestone 4).

Build a simple skip graph over token indices with edges for |i-j|<=max_skip.
Uniform weights by default.
"""

from typing import Literal, Tuple

import numpy as np


def build_skip_adjacency(n: int, max_skip: int, weight: Literal["uniform", "inv_dist"] = "uniform") -> np.ndarray:
    """Build symmetric adjacency W for a skip graph.

    - uniform: weight 1.0 for 1 <= |i-j| <= max_skip
    - inv_dist: weight 1/|i-j| for 1 <= |i-j| <= max_skip
    """
    W = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for d in range(1, max_skip + 1):
            j = i + d
            if j >= n:
                break
            w = 1.0 if weight == "uniform" else 1.0 / float(d)
            W[i, j] = W[j, i] = w
    return W


def laplacian(W: np.ndarray) -> np.ndarray:
    """Compute combinatorial Laplacian L = D - W for symmetric nonnegative W."""
    deg = np.sum(W, axis=1)
    D = np.diag(deg.astype(np.float32))
    L = D - W
    # Symmetrize numerically
    return 0.5 * (L + L.T)


def build_graph(n: int, max_skip: int, weight: Literal["uniform", "inv_dist"] = "uniform") -> Tuple[np.ndarray, np.ndarray]:
    """Return (W, L) for the skip graph."""
    W = build_skip_adjacency(n, max_skip, weight)
    L = laplacian(W)
    return W, L
