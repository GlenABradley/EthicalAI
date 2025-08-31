from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict
import numpy as np
from coherence.encoders.text_sbert import get_default_encoder
from ..types import DecisionProof, AxisPack
from ..axes.build import build_axis_pack
from ..axes.calibrate import pick_thresholds
from ..api.axes import ACTIVE
from ..eval.spans import project_scores
from ..eval.minspan import minimal_veto_spans

router = APIRouter(prefix="/v1/eval", tags=["eval"])

class EvalRequest(BaseModel):
    text: str
    window: int = 32
    stride: int = 16

class EvalResponse(BaseModel):
    proof: Dict
    spans: List[Dict]
    per_axis: Dict[str, List[float]] | None = None

def _embed(text:str) -> np.ndarray:
    enc = get_default_encoder()
    tokens = text.split()
    return enc.encode_tokens(tokens)

@router.post("/text", response_model=EvalResponse)
def eval_text(req: EvalRequest):
    if ACTIVE["pack"] is None:
        return EvalResponse(proof={"error": "No active axis pack"}, spans=[], per_axis=None)
    X = _embed(req.text)
    spans = project_scores(X, ACTIVE["pack"], req.window, req.stride)
    veto = minimal_veto_spans(spans)
    proof: DecisionProof = {
        "objective":"Maximize human autonomy based on objective empirical truth",
        "pack_id": ACTIVE["pack"].id,
        "spans": veto,
        "aggregation":{"type":"OR"},
        "final":{"action":"allow" if not veto else "refuse", "rationale":"veto if any axis crosses τ"}
    }
    return EvalResponse(proof=proof, spans=veto, per_axis=None)
