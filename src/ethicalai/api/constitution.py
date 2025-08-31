from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict
from ..types import AxisPack
from ..constitution.decoding import rank_candidates
from .axes import ACTIVE

router = APIRouter(prefix="/v1/constitution", tags=["constitution"])

class RankReq(BaseModel):
    candidates: List[Dict]  # [{"text":..., "logprob":...}, ...]

@router.post("/rank")
def rank(req: RankReq):
    pack: AxisPack = ACTIVE["pack"]
    return rank_candidates(req.candidates, pack)
