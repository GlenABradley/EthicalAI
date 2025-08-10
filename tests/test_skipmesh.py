import pytest
import numpy as np

from coherence.axis.pack import AxisPack
from coherence.coherence.skipmesh import build_skip_pairs, span_skip_pairs, span_coherence
from coherence.metrics.resonance import resonance

def make_pack_simple(d=4, k=2):
    Q = np.zeros((d, k), dtype=np.float32)
    Q[0, 0] = 1.0
    Q[1, 1] = 1.0
    names = ["a0", "a1"]
    lambda_ = np.ones((k,), dtype=np.float32)
    beta = np.zeros((k,), dtype=np.float32)
    weights = np.array([0.5, 0.5], dtype=np.float32)
    return AxisPack(names=names, Q=Q, lambda_=lambda_, beta=beta, weights=weights, mu={}, meta={})


def test_build_and_span_pairs_basic():
    pairs = build_skip_pairs(5, max_skip=2)
    # Expected pairs: (0,1),(0,2),(1,2),(1,3),(2,3),(2,4),(3,4)
    assert (0, 1) in pairs and (0, 2) in pairs and (3, 4) in pairs
    assert (0, 3) not in pairs and (2, 0) not in pairs

    sp = span_skip_pairs(1, 4, max_skip=2)
    # span indices 1..3 -> relative pairs (0,1),(0,2),(1,2) mapped -> (1,2),(1,3),(2,3)
    assert sp == [(1, 2), (1, 3), (2, 3)]


def test_span_coherence_monotonic_with_signal():
    pack = make_pack_simple()
    # token vectors (n=5,d=4), build a ridge along e0
    X = np.zeros((5, 4), dtype=np.float32)
    X[:, 0] = np.array([0.0, 0.5, 1.0, 0.5, 0.0], dtype=np.float32)
    # Span covering the peak should have higher coherence than edges
    c_mid = span_coherence(X, pack, 1, 4, max_skip=2)
    c_edge = span_coherence(X, pack, 0, 2, max_skip=2)
    assert c_mid > c_edge
