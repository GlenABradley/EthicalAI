from __future__ import annotations

from typing import List, Dict
import numpy as np
from fastapi import APIRouter, HTTPException

from coherence.api.models import (
    AnalyzeText,
    AnalyzeResponse,
    TokenVectors,
    SpanOutput,
    FrameOutput,
    AxialVectorsModel,
)
from coherence.axis.pack import AxisPack
from coherence.encoders.text_sbert import get_default_encoder
from coherence.metrics.resonance import project, utilities, aggregate
from coherence.pipeline.orchestrator import run_pipeline_from_vectors, OrchestratorParams

router = APIRouter()


def _tokenize(text: str) -> List[str]:
    # Simple deterministic splitter; align with indexing default
    return text.split()


@router.post("", response_model=AnalyzeResponse)
def analyze(req: AnalyzeText) -> AnalyzeResponse:
    texts = req.texts if req.texts else ([req.text] if req.text else [])
    if not texts:
        raise HTTPException(status_code=400, detail="No texts provided")

    if not req.axis_pack_id:
        # Frontend tests allow 501 when axis selection isn't implemented
        raise HTTPException(status_code=501, detail="axis_pack_id is required for analyze")

    axis_pack_id = req.axis_pack_id
    pack_path = f"data/axes/{axis_pack_id}.json"
    try:
        pack = AxisPack.load(pack_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Axis pack not found")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid axis pack: {e}")

    text = texts[0]
    tokens = _tokenize(text)
    if not tokens:
        # Return empty shapes
        return AnalyzeResponse(
            axes={"id": axis_pack_id, "names": pack.names, "k": pack.k},
            tokens=TokenVectors(alpha=[], u=[], r=[], U=[]),
            spans=[],
            frames=[],
            frame_spans=[],
            tau_used=[0.0],
        )

    enc = get_default_encoder()
    X = enc.encode(tokens).astype(np.float32)  # (n,d) or (d,)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[1] != pack.Q.shape[0]:
        raise HTTPException(status_code=409, detail=f"Encoder dimension {X.shape[1]} does not match axis pack dimension {pack.Q.shape[0]}")

    # Run core orchestrator for spans/frames topology
    params = OrchestratorParams(max_span_len=5, max_skip=2, diffusion_tau=None)
    out = run_pipeline_from_vectors(X, pack, params)

    # Token axial vectors
    alpha_toks: List[List[float]] = []
    u_toks: List[List[float]] = []
    r_toks: List[List[float]] = []
    U_toks: List[float] = []
    for i in range(X.shape[0]):
        a = project(X[i], pack)
        u = utilities(a, pack)
        U = float(aggregate(u, pack))
        alpha_toks.append(a.astype(np.float32).tolist())
        u_toks.append(u.astype(np.float32).tolist())
        r_toks.append(u.astype(np.float32).tolist())  # TODO: gating t, r
        U_toks.append(U)

    tokens_out = TokenVectors(alpha=alpha_toks, u=u_toks, r=r_toks, U=U_toks)

    # Spans axial vectors using mean(X[i:j])
    spans_out: List[SpanOutput] = []
    spans = out.get("spans", {}).get("spans", [])
    cohesion = out.get("spans", {}).get("coherence", np.zeros((0,), dtype=np.float32))
    for (i, j), C in zip(spans, cohesion):
        x = X[i:j].mean(axis=0)
        a = project(x, pack)
        u = utilities(a, pack)
        U = float(aggregate(u, pack))
        vec = AxialVectorsModel(
            alpha=a.astype(np.float32).tolist(),
            u=u.astype(np.float32).tolist(),
            r=u.astype(np.float32).tolist(),
            U=U,
            C=float(C),
            t=1.0,
            tau=0.0,
        )
        spans_out.append(SpanOutput(start=i, end=j, vectors=vec))

    # Frames: compute per-frame mean embedding over predicate + args, then vectors
    frames_out: List[FrameOutput] = []
    frame_spans_out: List[SpanOutput] = []
    frames = out.get("frames", [])
    for fr in frames:
        # Collect indices
        idxs: List[int] = list(range(fr.predicate[0], fr.predicate[1]))
        for _, (s, e) in fr.roles.items():
            idxs.extend(range(s, e))
        idxs = [ix for ix in idxs if 0 <= ix < X.shape[0]]
        if not idxs:
            continue
        x = X[idxs].mean(axis=0)
        a = project(x, pack)
        u = utilities(a, pack)
        U = float(aggregate(u, pack))
        fv = AxialVectorsModel(
            alpha=a.astype(np.float32).tolist(),
            u=u.astype(np.float32).tolist(),
            r=u.astype(np.float32).tolist(),
            U=U,
            t=1.0,
            tau=0.0,
        )
        frames_out.append(FrameOutput(id=str(fr.id), vectors=fv))
        # Also include predicate span vectors for convenience
        px = X[fr.predicate[0]:fr.predicate[1]].mean(axis=0)
        pa = project(px, pack)
        pu = utilities(pa, pack)
        pU = float(aggregate(pu, pack))
        pvec = AxialVectorsModel(
            alpha=pa.astype(np.float32).tolist(),
            u=pu.astype(np.float32).tolist(),
            r=pu.astype(np.float32).tolist(),
            U=pU,
            C=None,
            t=1.0,
            tau=0.0,
        )
        frame_spans_out.append(SpanOutput(start=int(fr.predicate[0]), end=int(fr.predicate[1]), vectors=pvec))

    return AnalyzeResponse(
        axes={"id": axis_pack_id, "names": pack.names, "k": pack.k},
        tokens=tokens_out,
        spans=spans_out,
        frames=frames_out,
        frame_spans=frame_spans_out,
        tau_used=[0.0],
    )
