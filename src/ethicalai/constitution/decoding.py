from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import numpy as np
from ..types import AxisPack, DecisionProof
from ..eval.spans import project_scores
from ..eval.minspan import minimal_veto_spans
from ..encoders import get_encoder, align_dim

def _axis_composite(spans: List[Dict], axes: Tuple[str, str]) -> float:
    """Average window scores for the named axes. If axis missing, treat as 0."""
    collected = {a: [] for a in axes}
    for s in spans:
        if s["axis"] in collected:
            collected[s["axis"]].append(float(s["score"]))
    vals = [(np.mean(collected[a]) if collected[a] else 0.0) for a in axes]
    return float(np.mean(vals)) if vals else 0.0

def _encode_tokens(text: str, dim: int, enc=None) -> np.ndarray:
    enc = enc or get_encoder()
    X = enc.encode_text(text)  # [T,D] or [D]
    if X.ndim == 1:
        return align_dim(X, dim)[None, :]
    if X.shape[1] != dim:
        X = np.stack([align_dim(t, dim) for t in X], axis=0)
    return X.astype(np.float32)

def _score_candidate(text: str, logprob: float, pack: AxisPack, enc=None,
                     window: int = 32, stride: int = 16,
                     pref_axes: Tuple[str, str] = ("autonomy", "truthfulness")):
    X = _encode_tokens(text, pack.dim, enc=enc)
    spans = project_scores(X, pack, window, stride)
    veto = minimal_veto_spans(spans)
    composite = _axis_composite(spans, pref_axes)
    # Lexicographic key: (no_veto_flag, composite, logprob)
    key = (0 if veto else 1, composite, float(logprob))
    proof: DecisionProof = {
        "objective":"Maximize human autonomy based on objective empirical truth",
        "pack_id": pack.id,
        "spans": veto if veto else spans,  # return the spans that mattered
        "aggregation":{"type":"OR"},
        "final":{"action":"refuse" if veto else "allow",
                 "rationale":"veto if any axis crosses τ"}
    }
    return key, proof

def rank_candidates(cands: List[Dict], pack: AxisPack, enc=None,
                    pref_axes: Tuple[str,str]=("autonomy","truthfulness")) -> Dict:
    """Rerank by (no-veto) > (autonomy+truthfulness composite) > (logprob)."""
    assert cands, "No candidates provided"
    best = None
    best_key = None
    best_proof = None
    for c in cands:
        text = c.get("text","")
        logp = float(c.get("logprob", 0.0))
        key, proof = _score_candidate(text, logp, pack, enc=enc, pref_axes=pref_axes)
        if (best_key is None) or (key > best_key):
            best_key, best_proof, best = key, proof, c
    # Ensure a deterministic structure:
    return {"choice": best, "key": list(best_key), "proof": best_proof}
