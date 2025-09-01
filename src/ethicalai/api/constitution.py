from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple
from ..types import AxisPack
from ..constitution.decoding import rank_candidates
from .axes import ACTIVE

router = APIRouter(prefix="/v1/constitution", tags=["constitution"])

class RankReq(BaseModel):
    candidates: List[Dict]  # [{"text": str, "logprob": float}, ...]
    pref_axes: Optional[Tuple[str,str]] = ("autonomy","truthfulness")

@router.post("/rank")
def rank(req: RankReq):
    pack: Optional[AxisPack] = ACTIVE["pack"]
    if not pack:
        raise HTTPException(409, "No active axis pack. Build or activate one via /v1/axes/*")
    return rank_candidates(req.candidates, pack, pref_axes=req.pref_axes or ("autonomy","truthfulness"))
