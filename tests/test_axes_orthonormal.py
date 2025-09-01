import numpy as np
from ethicalai.axes.utils import gram_schmidt

def test_gram_schmidt_orthonormal():
    rng = np.random.default_rng(0)
    vecs = [rng.normal(size=8).astype(np.float32) for _ in range(4)]
    basis = gram_schmidt(vecs)
    # norms ~ 1
    for b in basis:
        assert abs(np.linalg.norm(b) - 1.0) < 1e-5
    # pairwise dot ~ 0
    for i in range(len(basis)):
        for j in range(i+1, len(basis)):
            assert abs(float(basis[i] @ basis[j])) < 1e-5
