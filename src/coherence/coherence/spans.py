from __future__ import annotations

"""Span scoring and minimal span extraction (Milestone 3).

Implements simple, deterministic span utilities:
- token_saliency: per-token resonance
- span_scores: all spans up to max_len scored by mean token saliency
- best_span: highest-scoring span within max_len
"""

from typing import List, Tuple

import numpy as np

from coherence.axis.pack import AxisPack
from coherence.metrics.resonance import resonance


def token_saliency(token_vectors: np.ndarray, pack: AxisPack) -> np.ndarray:
    """Compute per-token resonance saliency."""
    scores = resonance(token_vectors, pack)
    return np.asarray(scores, dtype=np.float32)


def span_scores(token_vectors: np.ndarray, pack: AxisPack, max_len: int) -> List[Tuple[int, int, float]]:
    """Score all spans [i,j) with 1 <= j-i <= max_len by mean token saliency."""
    n = token_vectors.shape[0]
    sal = token_saliency(token_vectors, pack)
    out: List[Tuple[int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, min(n, i + max_len) + 1):
            s = float(np.mean(sal[i:j]))
            out.append((i, j, s))
    return out


def best_span(token_vectors: np.ndarray, pack: AxisPack, max_len: int) -> Tuple[int, int, float]:
    """Return the highest-scoring span under the given max_len constraint."""
    all_spans = span_scores(token_vectors, pack, max_len)
    if not all_spans:
        return (0, 0, 0.0)
    # max by score; stable by smallest start then shortest length
    all_spans.sort(key=lambda t: (-t[2], t[1] - t[0], t[0]))
    return all_spans[0]
