import pytest
import numpy as np

from coherence.axis.pack import AxisPack
from coherence.metrics.resonance import project, utilities, aggregate, resonance

def make_pack_linear(k=2, d=4):
    Q = np.zeros((d, k), dtype=np.float32)
    Q[0, 0] = 1.0
    Q[1, 1] = 1.0
    names = [f"a{i}" for i in range(k)]
    lambda_ = np.array([2.0, 0.5], dtype=np.float32)  # different scales
    beta = np.array([0.1, -0.2], dtype=np.float32)
    weights = np.array([0.6, 0.4], dtype=np.float32)
    return AxisPack(names=names, Q=Q, lambda_=lambda_, beta=beta, weights=weights, mu={}, meta={})


def test_resonance_linear_weights_deterministic():
    pack = make_pack_linear()
    # Single vector emphasizes first axis
    x = np.array([3.0, 1.0, 0.0, 0.0], dtype=np.float32)
    coords = project(x, pack)  # [3,1]
    assert np.allclose(coords, np.array([3.0, 1.0], dtype=np.float32))
    u = utilities(coords, pack)  # [2*3+0.1, 0.5*1-0.2] = [6.1, 0.3]
    assert np.allclose(u, np.array([6.1, 0.3], dtype=np.float32))
    agg = aggregate(u, pack)  # 0.6*6.1 + 0.4*0.3 = 3.66 + 0.12 = 3.78
    assert np.isclose(agg, 3.78, atol=1e-6)
    # Batch
    X = np.array([[3.0, 1.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]], dtype=np.float32)
    r = resonance(X, pack)
    # For second row: coords=[0,2], u=[0.1, 0.5*2-0.2]=[0.1,0.8]; agg=0.6*0.1+0.4*0.8=0.06+0.32=0.38
    expected = np.array([3.78, 0.38], dtype=np.float32)
    assert np.allclose(r, expected, atol=1e-6)


def test_resonance_choquet_two_axes():
    # Same Q, lambda, beta, but use a simple capacity favoring axis 1 when both are large
    d, k = 4, 2
    Q = np.zeros((d, k), dtype=np.float32)
    Q[0, 0] = 1.0
    Q[1, 1] = 1.0
    names = ["a0", "a1"]
    lambda_ = np.array([1.0, 1.0], dtype=np.float32)
    beta = np.array([0.0, 0.0], dtype=np.float32)
    weights = np.array([0.5, 0.5], dtype=np.float32)
    mu = {
        frozenset(): 0.0,  # optional
        frozenset({0}): 0.3,
        frozenset({1}): 0.7,
        frozenset({0, 1}): 1.0,
    }
    pack = AxisPack(names=names, Q=Q, lambda_=lambda_, beta=beta, weights=weights, mu=mu, meta={})

    # Utilities equal to coords here
    x = np.array([2.0, 1.0, 0.0, 0.0], dtype=np.float32)
    u = utilities(project(x, pack), pack)  # [2,1]
    # Choquet ascending: sorted u = [1,2]
    # i=0: (1-0)*mu({0,1}) = 1*1.0 = 1.0
    # i=1: (2-1)*mu({0}) where elements >= 2 is only index0 -> mu({0})=0.3
    # total = 1.0 + 1*0.3 = 1.3
    val = aggregate(u, pack)
    assert np.isclose(val, 1.3, atol=1e-6)
