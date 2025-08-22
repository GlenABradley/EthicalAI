from __future__ import annotations

from typing import Dict, List
import numpy as np
from fastapi import APIRouter, HTTPException

from coherence.api.models import SearchRequest, SearchResponse, SearchHit, AxialVectorsModel
from coherence.agent.query_map import u_from_nl
from coherence.axis.pack import AxisPack
from coherence.index.ann import has_index, query as ann_query, get_payloads

router = APIRouter()


def _score_candidate(u_q: np.ndarray, cand: dict, names: List[str], hyper: Dict[str, float], w: np.ndarray) -> float:
    """Compute rerank score according to spec.

    Align = sum w_i min(u_xi, u_qi) / (sum w_i u_qi + 1e-6)
    Prox  = 1 - sum w_i |u_xi - u_qi| / (sum w_i + 1e-6)
    GateU = sum w_i r_xi  (here r=u)
    Sx = gamma * Align + (1-gamma) * Prox
    Rank = beta*C_x + (1-beta)*(alpha*GateU + (1-alpha)*Sx)
    """
    u_x = np.asarray(cand["u"], dtype=np.float32)
    r_x = np.asarray(cand.get("r", cand["u"]), dtype=np.float32)
    Cx = float(cand.get("C", 0.0))
    beta = float(hyper.get("beta", 0.3))
    alpha = float(hyper.get("alpha", 0.5))
    gamma = float(hyper.get("gamma", 0.6))

    denom_align = float((w * u_q).sum() + 1e-6)
    denom_prox = float((w).sum() + 1e-6)
    align = float((w * np.minimum(u_x, u_q)).sum() / denom_align)
    prox = float(1.0 - (w * np.abs(u_x - u_q)).sum() / denom_prox)
    gateU = float((w * r_x).sum())
    sx = gamma * align + (1.0 - gamma) * prox
    rank = beta * Cx + (1.0 - beta) * (alpha * gateU + (1.0 - alpha) * sx)
    return rank


@router.post("", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    """Search API: ANN recall on u, rerank with non-cosine scoring.

    Returns top_k hits with vectors and span metadata.
    """
    axis_pack_id = req.axis_pack_id
    if not has_index(axis_pack_id):
        raise HTTPException(status_code=400, detail="Index not built for axis_pack_id; call /index first.")

    pack = AxisPack.load(f"data/axes/{axis_pack_id}.json")
    # Use pack weights if available; fallback to uniform
    w = getattr(pack, "weights", None)
    w_vec = np.asarray(w, dtype=np.float32) if w is not None else np.ones((pack.k,), dtype=np.float32)

    # Map query to u_q
    if req.query.type == "nl":
        u_q = u_from_nl(req.query.text or "", pack)
    elif req.query.type == "weights" and req.query.u is not None:
        u_q = np.asarray(req.query.u, dtype=np.float32)
        if u_q.shape[0] != pack.k:
            raise HTTPException(status_code=400, detail="Query u length must equal k of axis pack")
    else:
        u_q = u_from_nl(str(req.query), pack)

    # ANN recall
    recall_k = int(req.top_k) * 4
    idxs, _ = ann_query(axis_pack_id, u_q, recall_k)
    payloads = get_payloads(axis_pack_id)
    cands = [payloads[i] for i in idxs if i < len(payloads)]

    # Filters
    minC = float(req.filters.minC)
    thr = req.filters.thresholds or {}
    # Map thresholds by axis index if provided by name
    thr_idx = np.full((pack.k,), -np.inf, dtype=np.float32)
    for name, val in thr.items():
        if name in pack.names:
            thr_idx[pack.names.index(name)] = float(val)
    filtered = []
    for c in cands:
        if float(c.get("C", 0.0)) < minC:
            continue
        uvec = np.asarray(c["u"], dtype=np.float32)
        if np.any(uvec < thr_idx):
            continue
        filtered.append(c)

    # If empty, relax thresholds by 20%
    if not filtered and thr:
        thr_idx *= 0.8
        for c in cands:
            if float(c.get("C", 0.0)) < minC:
                continue
            uvec = np.asarray(c["u"], dtype=np.float32)
            if np.any(uvec < thr_idx):
                continue
            filtered.append(c)

    # Rerank
    scored = [(_score_candidate(u_q, c, pack.names, req.hyper.model_dump(), w_vec), c) for c in filtered]
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [c for _, c in scored[: req.top_k]]

    # Preload frames grouped by doc_id for quick lookup (lazy import to avoid pyarrow at import time)
    frames_by_doc: Dict[str, List[dict]] = {}
    try:
        from coherence.index.store import iterate_frames  # type: ignore
        for fr in iterate_frames(axis_pack_id):
            frames_by_doc.setdefault(str(fr.get("doc_id")), []).append(fr)
    except Exception:
        # If storage backend unavailable, proceed without frames
        frames_by_doc = {}

    hits: List[SearchHit] = []
    for c in top:
        vectors = AxialVectorsModel(
            alpha=c["alpha"], u=c["u"], r=c.get("r", c["u"]), U=float(c.get("U", 0.0)), C=float(c.get("C", 0.0)), t=float(c.get("t", 1.0)), tau=float(c.get("tau", 0.0))
        )
        span = {"start": int(c["start"]), "end": int(c["end"]), "text": c.get("text", "")}
        # Attach frames whose predicate overlaps the span
        doc_frames = frames_by_doc.get(str(c.get("doc_id")), [])
        start_s, end_s = int(c["start"]), int(c["end"])
        related_frames: List[Dict[str, object]] = []
        for fr in doc_frames:
            ps, pe = int(fr.get("pred_start", -1)), int(fr.get("pred_end", -1))
            if ps < 0 or pe < 0:
                continue
            # overlap check: [ps,pe) intersects [start_s,end_s)
            if not (pe <= start_s or ps >= end_s):
                related_frames.append({
                    "id": fr.get("frame_id"),
                    "pred_start": ps,
                    "pred_end": pe,
                    "roles": fr.get("roles", {}),
                    "U": float(fr.get("U", 0.0)),
                })
        hits.append(
            SearchHit(
                doc_id=str(c["doc_id"]),
                span=span,
                vectors=vectors,
                frames=related_frames,
                score=_score_candidate(u_q, c, pack.names, req.hyper.model_dump(), w_vec),
            )
        )

    return SearchResponse(hits=hits)
