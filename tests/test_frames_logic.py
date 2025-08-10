import pytest
import numpy as np

from coherence.axis.pack import AxisPack
from coherence.frames.srl_lite import build_frames
from coherence.frames.vectorize import frame_embedding
from coherence.frames.logic.eval import Truth, lnot, land, lor

def make_pack_simple(d=4, k=2):
    Q = np.zeros((d, k), dtype=np.float32)
    Q[0, 0] = 1.0
    Q[1, 1] = 1.0
    names = ["a0", "a1"]
    lambda_ = np.ones((k,), dtype=np.float32)
    beta = np.zeros((k,), dtype=np.float32)
    weights = np.array([0.5, 0.5], dtype=np.float32)
    return AxisPack(names=names, Q=Q, lambda_=lambda_, beta=beta, weights=weights, mu={}, meta={})


def test_srl_lite_and_vectorize_deterministic():
    pack = make_pack_simple()
    # Create token vectors with a clear saliency peak at positions 2 and 5 along e0
    n, d = 8, 4
    X = np.zeros((n, d), dtype=np.float32)
    X[2, 0] = 1.0
    X[5, 0] = 0.8

    frames = build_frames(X, pack, saliency_thresh=0.1, arg_band=0.5, max_arg_len=2)
    assert len(frames) >= 1
    # Check first frame structure
    fr = frames[0]
    assert fr.predicate[0] == fr.predicate[1] - 1  # single-token predicate
    # Embedding length should be 3*d (predicate, arg_left, arg_right)
    emb = frame_embedding(X, fr)
    assert emb.shape[0] == 3 * d
    # Predicate segment equals token vector at predicate index
    pred_vec = emb[:d]
    idx = fr.predicate[0]
    np.testing.assert_allclose(pred_vec, X[idx], atol=1e-6)


def test_three_valued_logic_tables():
    assert lnot(Truth.TRUE) == Truth.FALSE
    assert lnot(Truth.FALSE) == Truth.TRUE
    assert lnot(Truth.UNKNOWN) == Truth.UNKNOWN

    assert land(Truth.TRUE, Truth.TRUE) == Truth.TRUE
    assert land(Truth.FALSE, Truth.TRUE) == Truth.FALSE
    assert land(Truth.UNKNOWN, Truth.TRUE) == Truth.UNKNOWN

    assert lor(Truth.FALSE, Truth.FALSE) == Truth.FALSE
    assert lor(Truth.TRUE, Truth.FALSE) == Truth.TRUE
    assert lor(Truth.UNKNOWN, Truth.FALSE) == Truth.UNKNOWN
