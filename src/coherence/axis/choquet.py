from __future__ import annotations

"""Choquet integral utilities.

Implements a discrete Choquet integral for finite index sets {0..k-1}.
Capacity `mu` is a mapping from frozenset[int] -> float with mu(set()) = 0 and
monotonicity assumed by caller. For small k (axes count), this is efficient.
"""

from typing import Dict, FrozenSet, Iterable, List, Sequence

import numpy as np


def choquet_integral(x: Sequence[float], mu: Dict[FrozenSet[int], float]) -> float:
    """Compute the discrete Choquet integral of x w.r.t. capacity mu.

    Args
    - x: length-k sequence of utilities
    - mu: mapping from subset of indices to capacity value

    Returns
    - scalar Choquet integral value

    Notes
    - Uses standard ascending-sort formula: sum_i (x_(i) - x_(i-1)) * mu(A_i),
      where x_(0) = 0 and A_i = {j | x_j >= x_(i)} for i=1..k.
    """
    x_arr = np.asarray(x, dtype=np.float32)
    k = x_arr.shape[0]
    if k == 0:
        return 0.0
    # Sort ascending
    order = np.argsort(x_arr)
    x_sorted = x_arr[order]
    prev = 0.0
    total = 0.0
    for i in range(k):
        xi = float(x_sorted[i])
        # Indices with value >= xi
        mask = x_arr >= xi - 1e-12
        A_i = frozenset(int(j) for j in np.nonzero(mask)[0].tolist())
        mu_val = float(mu.get(A_i, 0.0))
        total += (xi - prev) * mu_val
        prev = xi
    return float(total)

"""Choquet integral for axis aggregation (Milestone 2).

# TODO: @builder implement in Milestone 2
"""
