from __future__ import annotations

"""Pipeline orchestrator (Milestone 6).

Run end-to-end scoring starting from token vectors and an AxisPack.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from coherence.axis.pack import AxisPack
from coherence.coherence.spans import token_saliency
from coherence.coherence.skipmesh import span_coherence
from coherence.coherence.substrate import build_graph
from coherence.coherence.diffusion import diffuse
from coherence.frames.srl_lite import build_frames
from coherence.frames.vectorize import frame_embedding
from coherence.frames.schema import Frame
from coherence.pipeline.scoring import (
    compute_token_vectors,
    compute_span_vectors,
    compute_frame_vectors,
)


@dataclass
class OrchestratorParams:
    max_span_len: int = 5
    max_skip: int = 2
    diffusion_tau: Optional[float] = None


def run_pipeline_from_vectors(
    token_vectors: np.ndarray,
    pack: AxisPack,
    params: OrchestratorParams = OrchestratorParams(),
) -> Dict[str, object]:
    """Run pipeline using precomputed token vectors.

    Returns a dictionary with token, span, and frame-level outputs.
    """
    X = np.asarray(token_vectors, dtype=np.float32)

    # Optional diffusion over token dimension (scalar signal per token)
    token_signal = token_saliency(X, pack)
    if params.diffusion_tau is not None and X.shape[0] >= 2:
        _, L = build_graph(X.shape[0], max_skip=params.max_skip, weight="uniform")
        token_signal = diffuse(L, token_signal, params.diffusion_tau)

    tokens_out = compute_token_vectors(X, pack, external_signal=token_signal)
    spans_out = compute_span_vectors(X, pack, max_len=params.max_span_len, max_skip=params.max_skip)

    frames: List[Frame] = build_frames(X, pack, saliency_thresh=0.0, arg_band=0.5, max_arg_len=2)
    frame_vecs = compute_frame_vectors(X, frames)

    return {
        "tokens": tokens_out,  # dict of arrays
        "spans": spans_out,    # dict: list and array views
        "frames": frames,      # list[Frame]
        "frame_vectors": frame_vecs,  # (m, 3d)
    }
