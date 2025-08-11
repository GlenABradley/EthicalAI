from __future__ import annotations

"""Rule-based SRL (Milestone 5).

Axis-only, deterministic predicate/argument extraction using resonance
saliency. No keyword triggers.
"""

from typing import Dict, List, Tuple, Optional

import numpy as np

from coherence.axis.pack import AxisPack
from coherence.coherence.spans import token_saliency
from coherence.frames.schema import Frame, Span
from coherence.frames.detectors.evidence import detect_evidence_spans
from coherence.frames.detectors.conditional import detect_condition_spans


def _local_maxima(x: np.ndarray) -> List[int]:
    idx: List[int] = []
    n = x.shape[0]
    for i in range(n):
        left = x[i - 1] if i - 1 >= 0 else -np.inf
        right = x[i + 1] if i + 1 < n else -np.inf
        if x[i] >= left and x[i] >= right:
            idx.append(i)
    return idx


def detect_predicates(token_vectors: np.ndarray, pack: AxisPack, saliency_thresh: float = 0.0) -> List[int]:
    s = token_saliency(token_vectors, pack)
    peaks = _local_maxima(s)
    return [i for i in peaks if s[i] >= saliency_thresh]


def _expand_arg(s: np.ndarray, center: int, max_len: int, thresh: float) -> Span:
    n = s.shape[0]
    start = center
    end = center + 1
    # Expand while saliency above threshold and within max_len
    while start - 1 >= 0 and (center - (start - 1)) < max_len and s[start - 1] >= thresh:
        start -= 1
    while end < n and ((end - 1) - center) < max_len and s[end] >= thresh:
        end += 1
    return (start, end)


def build_frames(
    token_vectors: np.ndarray,
    pack: AxisPack,
    *,
    saliency_thresh: float = 0.0,
    arg_band: float = 0.5,
    max_arg_len: int = 3,
    debug: bool = False,
    role_mode: str = "lr",
    detect_evidence: bool = False,
    detect_condition: bool = False,
    token_texts: Optional[List[str]] = None,
) -> List[Frame]:
    """Construct frames around saliency peaks.

    - predicate: single-token span at peak index
    - arguments: left/right spans expanded where saliency >= arg_band * pred_sal
    """
    s = token_saliency(token_vectors, pack)
    preds = detect_predicates(token_vectors, pack, saliency_thresh)
    frames: List[Frame] = []
    for p in preds:
        pred_span: Span = (p, p + 1)
        pred_sal = float(s[p])
        arg_thresh = arg_band * pred_sal
        left_center = max(0, p - 1)
        right_center = min(token_vectors.shape[0] - 1, p + 1)
        left = _expand_arg(s, left_center, max_arg_len, arg_thresh)
        right = _expand_arg(s, right_center, max_arg_len, arg_thresh)
        roles: Dict[str, Span] = {}
        if role_mode == "agent_patient":
            # Simple heuristic: higher-mean-saliency side -> agent; other -> patient
            l_mean = float(s[left[0]:left[1]].mean()) if left[1] > left[0] else -np.inf
            r_mean = float(s[right[0]:right[1]].mean()) if right[1] > right[0] else -np.inf
            if left[1] - left[0] > 0 or right[1] - right[0] > 0:
                if l_mean >= r_mean:
                    if left[1] - left[0] > 0:
                        roles["agent"] = left
                    if right[1] - right[0] > 0:
                        roles["patient"] = right
                else:
                    if right[1] - right[0] > 0:
                        roles["agent"] = right
                    if left[1] - left[0] > 0:
                        roles["patient"] = left
        else:
            if left[1] - left[0] > 0:
                roles["arg_left"] = left
            if right[1] - right[0] > 0:
                roles["arg_right"] = right

        # Optional detectors (cue-based) if token_texts provided
        det_meta: Dict[str, List[Span]] = {}
        if token_texts is not None:
            if detect_evidence:
                ev = detect_evidence_spans(token_texts, pivot=p)
                if ev:
                    # take the first short span near predicate
                    roles["evidence"] = ev[0]
                    det_meta.setdefault("evidence", []).extend(ev)
            if detect_condition:
                cd = detect_condition_spans(token_texts, pivot=p)
                if cd:
                    roles["condition"] = cd[0]
                    det_meta.setdefault("condition", []).extend(cd)
        fid = f"f{p}"
        meta: Dict[str, object] = {}
        if debug:
            # Collect lightweight debug info without affecting behavior
            meta["debug"] = {
                "pred_index": int(p),
                "pred_saliency": float(pred_sal),
                "pred_candidates": [int(i) for i in preds],
                "arg_band": float(arg_band),
                "max_arg_len": int(max_arg_len),
                "arg_left_expansion": [int(left[0]), int(left[1])],
                "arg_right_expansion": [int(right[0]), int(right[1])],
            }
            meta["debug"]["role_mode_applied"] = role_mode
            if det_meta:
                meta["debug"]["detectors"] = {
                    k: [[int(a), int(b)] for (a, b) in v] for k, v in det_meta.items()
                }
        frames.append(Frame(id=fid, predicate=pred_span, roles=roles, score=pred_sal, meta=meta))
    return frames
