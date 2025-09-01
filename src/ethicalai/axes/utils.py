import numpy as np
from typing import List

def gram_schmidt(vecs: List[np.ndarray]) -> List[np.ndarray]:
    basis: List[np.ndarray] = []
    for v in vecs:
        w = v.astype(np.float64).copy()
        for b in basis:
            w -= (w @ b) * b
        n = np.linalg.norm(w)
        if n > 1e-12:
            basis.append((w / n).astype(np.float32))
    return basis
