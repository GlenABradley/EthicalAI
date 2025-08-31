import numpy as np
from typing import List
def gram_schmidt(vecs: List[np.ndarray]) -> List[np.ndarray]:
    basis = []
    for v in vecs:
        w = v.copy().astype(float)
        for b in basis:
            w -= (w @ b) * b
        n = np.linalg.norm(w)
        if n > 1e-12: basis.append(w/n)
    return basis
