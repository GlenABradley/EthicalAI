from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict
from ..types import AxisPack
from .axes import ACTIVE
from ..constitution.decoding import rank_candidates

router = APIRouter(prefix="/v1/interaction", tags=["interaction"])

class AskReq(BaseModel):
    prompt: str

@router.post("/respond")
def respond(req: AskReq):
    # Placeholder: produce 3 naive candidates. Replace with real LLM calls.
    cands = [{"text": f"{req.prompt} (option {i})", "logprob": -0.1*i} for i in range(3)]
    pack: AxisPack = ACTIVE["pack"]
    ranked = rank_candidates(cands, pack)
    return {"final": ranked["choice"]["text"], "proof": ranked["proof"], "alternatives": cands}
