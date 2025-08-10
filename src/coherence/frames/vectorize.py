from __future__ import annotations

"""Role-aware frame embeddings (Milestone 5).

Compute simple role-aware embeddings by averaging token vectors over
role spans and concatenating in a fixed role order.
"""

from typing import Iterable, List, Sequence

import numpy as np

from coherence.frames.schema import Frame, Span


def span_mean(token_vectors: np.ndarray, span: Span) -> np.ndarray:
    """Mean of vectors in [start, end). Returns zeros if empty span."""
    s, e = span
    if e <= s:
        return np.zeros((token_vectors.shape[1],), dtype=np.float32)
    return np.asarray(token_vectors[s:e].mean(axis=0), dtype=np.float32)


def frame_embedding(
    token_vectors: np.ndarray,
    frame: Frame,
    roles_order: Sequence[str] = ("predicate", "arg_left", "arg_right"),
) -> np.ndarray:
    """Concatenate mean vectors of roles in roles_order.

    Missing roles are zero vectors of dim d.
    """
    d = token_vectors.shape[1]
    parts: List[np.ndarray] = []
    for role in roles_order:
        if role == "predicate":
            v = span_mean(token_vectors, frame.predicate)
        else:
            span = frame.roles.get(role, (0, 0))
            v = span_mean(token_vectors, span)
        parts.append(v.reshape(-1))
    return np.concatenate(parts, axis=0).astype(np.float32)
