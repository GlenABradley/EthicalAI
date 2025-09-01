from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from ..types import AxisPack
from .axes import ACTIVE
from ..constitution.decoding import rank_candidates
from ..interaction.policy import load_policy
from ..interaction.generate import GEN_REGISTRY
from ..interaction.explain import refusal_message, suggest_alternatives

router = APIRouter(prefix="/v1/interaction", tags=["interaction"])

class AskReq(BaseModel):
    prompt: str
    n: int = 3
    generator: str = "naive"  # hook for later

class AskResp(BaseModel):
    final: str
    proof: Dict
    alternatives: List[str]
    chosen: Dict
    candidates: List[Dict]
    policy: Dict

@router.post("/respond", response_model=AskResp)
def respond(req: AskReq):
    pack: Optional[AxisPack] = ACTIVE.get("pack")
    if pack is None or not getattr(pack, "axes", None):
        raise HTTPException(400, "No active axis pack. Build & /v1/axes/activate first.")

    policy = load_policy()
    gen = GEN_REGISTRY.get(req.generator)
    if gen is None:
        raise HTTPException(400, f"Unknown generator: {req.generator}")

    cands = gen(req.prompt, req.n)

    # NOTE: ranker currently uses pack thresholds directly; hook policy.multiplier here if desired
    try:
        result = rank_candidates(cands, pack)
        proof = result.get("proof", {})
        final_action = proof.get("final", {}).get("action", "allow")

        if final_action == "allow":
            final_text = result["choice"]["text"]
            alts = []
        else:
            # refusal path with transparent proof + safer alternatives
            veto_spans = proof.get("spans", [])
            final_text = refusal_message(veto_spans)
            alts = suggest_alternatives(req.prompt)

        return AskResp(
            final=final_text,
            proof=proof,
            alternatives=alts,
            chosen=result.get("choice", {}),
            candidates=cands,
            policy={
                "strictness": policy.strictness,
                "thresholds_multiplier": policy.thresholds_multiplier,
                "weights": policy.weights,
                "forms": policy.forms,
            },
        )
    except Exception as e:
        # Never 500; return transparent refusal with reason
        proof = {
            "final": {"action": "refuse", "reason": str(e)},
            "spans": [],
            "pack_id": getattr(pack, "id", None),
        }
        return AskResp(
            final=refusal_message([]),
            proof=proof,
            alternatives=suggest_alternatives(req.prompt),
            chosen={},
            candidates=cands,
            policy={
                "strictness": policy.strictness,
                "thresholds_multiplier": policy.thresholds_multiplier,
                "weights": policy.weights,
                "forms": policy.forms,
            },
        )
