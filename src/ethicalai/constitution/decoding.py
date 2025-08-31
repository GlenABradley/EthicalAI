from __future__ import annotations
from typing import List, Dict
from ..types import DecisionProof, AxisPack
from ..eval.spans import project_scores
from ..eval.minspan import minimal_veto_spans
import numpy as np
from coherence.encoders.text_sbert import get_default_encoder

def rank_candidates(cands: List[Dict], pack: AxisPack) -> Dict:
    # cand = {"text": str, "logprob": float}
    enc = get_default_encoder()
    best = None
    for c in cands:
        X = enc.encode_tokens(c["text"].split())
        spans = project_scores(X, pack)
        veto = minimal_veto_spans(spans)
        if veto: 
            continue
        autonomy_scores = [span["score"] for span in spans if span["axis"] == "autonomy"]
        truth_scores = [span["score"] for span in spans if span["axis"] == "truthfulness"]
        mean_autonomy = np.mean(autonomy_scores) if autonomy_scores else 0
        mean_truth = np.mean(truth_scores) if truth_scores else 0
        composite = (mean_autonomy + mean_truth) / 2 + c.get("logprob", 0.0) * 0.01
        if (best is None) or (composite > best["score"]):
            best = {"choice": c, "score": composite, "proof": {"spans": spans, "final":{"action":"allow"}}}
    return best or {"choice": cands[0], "score": -1e9, "proof":{"final":{"action":"refuse"}}} 
