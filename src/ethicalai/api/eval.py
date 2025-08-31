from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
from ..types import DecisionProof, AxisPack
from ..eval.spans import project_scores
from ..eval.minspan import minimal_veto_spans
from ..encoders import get_encoder
from .axes import ACTIVE

router = APIRouter(prefix="/v1/eval", tags=["eval"])

class EvalRequest(BaseModel):
    text: str
    window: int = 32
    stride: int = 16

class EvalResponse(BaseModel):
    proof: Dict
    spans: List[Dict]
    per_axis: Optional[Dict[str, List[float]]] = None

@router.post("/text", response_model=EvalResponse)
def eval_text(req: EvalRequest):
    pack: Optional[AxisPack] = ACTIVE["pack"]
    if not pack or not getattr(pack, "axes", None):
        raise HTTPException(409, "No active axis pack. Build or activate one via /v1/axes/*")
    enc = get_encoder()
    X = enc.encode_text(req.text)  # [T,D]
    if X.shape[1] != pack.dim:
        raise HTTPException(500, f"Encoder dim {X.shape[1]} != pack dim {pack.dim}")
    spans = project_scores(X, pack, req.window, req.stride)
    veto = minimal_veto_spans(spans)
    proof: DecisionProof = {
        "objective":"Maximize human autonomy based on objective empirical truth",
        "pack_id": pack.id,
        "spans": veto,
        "aggregation":{"type":"OR"},
        "final":{"action":"allow" if not veto else "refuse", "rationale":"veto if any axis crosses τ"}
    }
    return EvalResponse(proof=proof, spans=veto, per_axis=None)
