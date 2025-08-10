from __future__ import annotations

"""SkipMesh pair generation and span coherence (Milestone 3).

Provides utilities to generate skip-connected token pairs and compute a
span-level coherence as the mean resonance across its skip pairs.
"""

from typing import List, Sequence, Tuple

import numpy as np

from coherence.axis.pack import AxisPack
from coherence.metrics.resonance import resonance


def build_skip_pairs(n_tokens: int, max_skip: int) -> List[Tuple[int, int]]:
    """Generate all (i,j) with 0 <= i < j < n_tokens and 1 <= j-i <= max_skip."""
    pairs: List[Tuple[int, int]] = []
    for i in range(n_tokens):
        for j in range(i + 1, min(n_tokens, i + 1 + max_skip)):
            pairs.append((i, j))
    return pairs


def span_skip_pairs(start: int, end: int, max_skip: int) -> List[Tuple[int, int]]:
    """Skip pairs restricted to span [start, end)."""
    if end <= start:
        return []
    span_len = end - start
    rel = build_skip_pairs(span_len, max_skip)
    return [(start + i, start + j) for (i, j) in rel]


def span_coherence(token_vectors: np.ndarray, pack: AxisPack, start: int, end: int, max_skip: int) -> float:
    """Compute span coherence as mean resonance over skip pairs' mean vectors.

    Args
    - token_vectors: (n, d)
    - pack: AxisPack
    - start, end: span bounds [start, end)
    - max_skip: maximum allowed skip size

    Returns
    - scalar coherence; 0.0 if no pairs
    """
    pairs = span_skip_pairs(start, end, max_skip)
    if not pairs:
        return 0.0
    vals = []
    for i, j in pairs:
        m = 0.5 * (token_vectors[i] + token_vectors[j])
        vals.append(float(resonance(m, pack)))
    return float(np.mean(np.asarray(vals, dtype=np.float32)))
