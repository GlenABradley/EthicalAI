import numpy as np

from coherence.axis.pack import AxisPack
from coherence.pipeline.orchestrator import run_pipeline_from_vectors, OrchestratorParams


def make_pack_simple(d=6, k=3):
    Q = np.zeros((d, k), dtype=np.float32)
    for i in range(k):
        Q[i, i] = 1.0
    names = [f"a{i}" for i in range(k)]
    lambda_ = np.ones((k,), dtype=np.float32)
    beta = np.zeros((k,), dtype=np.float32)
    weights = np.ones((k,), dtype=np.float32) / k
    return AxisPack(names=names, Q=Q, lambda_=lambda_, beta=beta, weights=weights, mu={}, meta={})


def test_orchestrator_shapes():
    n, d = 12, 6
    X = np.zeros((n, d), dtype=np.float32)
    # create two peaks on first 3 dimensions
    X[3, 0] = 1.0
    X[8, 1] = 0.8
    X[6, 2] = 0.6
    pack = make_pack_simple(d=d, k=3)

    params = OrchestratorParams(max_span_len=4, max_skip=2, diffusion_tau=0.1)
    out = run_pipeline_from_vectors(X, pack, params)

    tokens = out["tokens"]
    assert tokens["vectors"].shape == (n, d)
    assert tokens["saliency"].shape == (n,)
    assert tokens["signal"].shape == (n,)

    spans = out["spans"]
    # number of spans with max_len 4
    expected_spans = sum(min(n - i, 4) for i in range(n))
    assert len(spans["spans"]) == expected_spans
    assert spans["coherence"].shape == (expected_spans,)

    frames = out["frames"]
    frame_vecs = out["frame_vectors"]
    d3 = 3 * d
    assert frame_vecs.ndim == 2
    assert frame_vecs.shape[1] == d3
