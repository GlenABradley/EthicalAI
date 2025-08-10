"""Scoring utilities (Milestone 6).

Compute token-, span-, and frame-level vectors and scores from inputs.
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np

from coherence.axis.pack import AxisPack
from coherence.coherence.spans import token_saliency, span_scores
from coherence.coherence.skipmesh import span_coherence
from coherence.frames.schema import Frame
from coherence.frames.vectorize import frame_embedding


def compute_token_vectors(
    token_vectors: np.ndarray,
    pack: AxisPack,
    *,
    external_signal: np.ndarray = None,
) -> Dict[str, np.ndarray]:
    """Return token-level fields: vectors, saliency, and optional signal."""
    X = np.asarray(token_vectors, dtype=np.float32)
    sal = token_saliency(X, pack)
    sig = external_signal if external_signal is not None else sal
    sig = np.asarray(sig, dtype=np.float32)
    return {
        "vectors": X,
        "saliency": sal,
        "signal": sig,
    }


def compute_span_vectors(
    token_vectors: np.ndarray,
    pack: AxisPack,
    *,
    max_len: int,
    max_skip: int,
) -> Dict[str, object]:
    """Enumerate spans and compute coherence per span using SkipMesh.

    Returns dict with:
    - spans: List[(i,j)]
    - coherence: np.ndarray shape (num_spans,)
    """
    X = np.asarray(token_vectors, dtype=np.float32)
    n = X.shape[0]
    spans: List[Tuple[int, int]] = []
    coh: List[float] = []
    for i in range(n):
        for j in range(i + 1, min(n, i + max_len) + 1):
            spans.append((i, j))
            c = span_coherence(X, pack, i, j, max_skip)
            coh.append(float(c))
    return {
        "spans": spans,
        "coherence": np.asarray(coh, dtype=np.float32),
    }


def compute_frame_vectors(token_vectors: np.ndarray, frames: Sequence[Frame]) -> np.ndarray:
    """Stack role-aware frame embeddings into an array of shape (m, 3d)."""
    X = np.asarray(token_vectors, dtype=np.float32)
    d = X.shape[1]
    if not frames:
        return np.zeros((0, 3 * d), dtype=np.float32)
    embs = [frame_embedding(X, fr) for fr in frames]
    return np.stack(embs, axis=0).astype(np.float32)
