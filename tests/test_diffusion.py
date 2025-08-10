import pytest
import numpy as np

from coherence.coherence.substrate import build_graph
from coherence.coherence.diffusion import diffuse

def dirichlet_energy(L: np.ndarray, x: np.ndarray) -> float:
    if x.ndim == 1:
        return float(x.T @ (L @ x))
    else:
        # sum over columns
        return float(np.sum((x.T @ L) * x.T))


def test_energy_decreases_with_tau():
    n = 10
    max_skip = 2
    W, L = build_graph(n, max_skip, weight="uniform")
    rng = np.random.default_rng(42)
    x = rng.standard_normal(n).astype(np.float32)

    taus = [0.0, 0.05, 0.1, 0.2]
    Y = diffuse(L, x, taus)  # shape (len(taus), n)
    energies = [dirichlet_energy(L, y) for y in Y]
    # Non-increasing energy with tau
    for a, b in zip(energies, energies[1:]):
        assert b <= a + 1e-6
