from __future__ import annotations

"""Axis pack builder.

Builds an `AxisPack` from seed phrases or precomputed seed vectors.

Algorithm per-axis
- Encode positive and negative seed sets.
- Compute mean difference v = mean(pos) - mean(neg).
- Stack axis vectors for all axes into matrix A = [v1, v2, ...] (d x k).
- Orthonormalize columns via QR decomposition: Q, _ = np.linalg.qr(A).
- Return AxisPack with Q and default parameters.

No cosine similarity is used; magnitudes are respected before QR.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from coherence.axis.pack import AxisPack


def _mean(arrs: Sequence[np.ndarray]) -> np.ndarray:
    if len(arrs) == 0:
        raise ValueError("Empty seed set")
    # Stack safely in float32
    X = np.asarray(np.stack(arrs, axis=0), dtype=np.float32)
    return np.mean(X, axis=0)


def _qr_orthonormalize(A: np.ndarray) -> np.ndarray:
    # A: (d, k)
    Q, _ = np.linalg.qr(A)
    return np.asarray(Q, dtype=np.float32)


def _diff_of_means(pos_vecs: Sequence[np.ndarray], neg_vecs: Sequence[np.ndarray]) -> np.ndarray:
    vp = _mean(pos_vecs)
    vn = _mean(neg_vecs)
    v = vp - vn
    return np.asarray(v, dtype=np.float32)


def build_axis_pack_from_vectors(
    seeds_vecs: Mapping[str, Tuple[Sequence[np.ndarray], Sequence[np.ndarray]]],
    *,
    lambda_init: float = 1.0,
    beta_init: float = 0.0,
    weights_init: Optional[Sequence[float]] = None,
    meta: Optional[Dict[str, object]] = None,
) -> AxisPack:
    """Build AxisPack from precomputed positive/negative seed vectors.

    seeds_vecs maps axis name -> (pos_vectors, neg_vectors). Each vector must have same dimension d.
    """
    names: List[str] = list(seeds_vecs.keys())
    if len(names) == 0:
        raise ValueError("No axes provided")
    # Compute axis directions
    vs: List[np.ndarray] = []
    for name in names:
        pos, neg = seeds_vecs[name]
        v = _diff_of_means(pos, neg)
        vs.append(v)
    # Stack and orthonormalize
    A = np.stack(vs, axis=1)  # (d, k)
    Q = _qr_orthonormalize(A)
    k = len(names)
    d = Q.shape[0]
    lambda_arr = np.full((k,), float(lambda_init), dtype=np.float32)
    beta_arr = np.full((k,), float(beta_init), dtype=np.float32)
    if weights_init is None:
        weights_arr = np.full((k,), 1.0 / float(k), dtype=np.float32)
    else:
        w = np.asarray(list(weights_init), dtype=np.float32)
        if w.shape != (k,):
            raise ValueError("weights_init must have shape (k,)")
        weights_arr = w
    return AxisPack(
        names=names,
        Q=Q.astype(np.float32),
        lambda_=lambda_arr,
        beta=beta_arr,
        weights=weights_arr,
        mu={},
        meta=meta or {},
    )


def build_axis_pack_from_seeds(
    seeds: Mapping[str, Dict[str, List[str]]],
    *,
    encode_fn,
    lambda_init: float = 1.0,
    beta_init: float = 0.0,
    weights_init: Optional[Sequence[float]] = None,
    meta: Optional[Dict[str, object]] = None,
) -> AxisPack:
    """Build AxisPack from text seeds using an encoding function.

    seeds maps axis name -> {"positive": [...], "negative": [...]}.
    encode_fn(texts: List[str]) -> np.ndarray returns (N, d).
    """
    seeds_vecs: Dict[str, Tuple[Sequence[np.ndarray], Sequence[np.ndarray]]] = {}
    for name, groups in seeds.items():
        pos = groups.get("positive", [])
        neg = groups.get("negative", [])
        if not pos or not neg:
            raise ValueError(f"Axis '{name}' must have both positive and negative seeds")
        pos_vecs = [v for v in np.asarray(encode_fn(pos), dtype=np.float32)]
        neg_vecs = [v for v in np.asarray(encode_fn(neg), dtype=np.float32)]
        seeds_vecs[name] = (pos_vecs, neg_vecs)
    return build_axis_pack_from_vectors(
        seeds_vecs,
        lambda_init=lambda_init,
        beta_init=beta_init,
        weights_init=weights_init,
        meta=meta,
    )

"""Axis builder (Milestone 1).

Will build AxisPack from seeds using diff-of-means with orthonormalization.

# TODO: @builder implement in Milestone 1
"""
