from __future__ import annotations
from typing import List, Dict
from ..types import SpanScore

def minimal_veto_spans(spans: List[SpanScore]) -> List[SpanScore]:
    """Greedy filter: keep first breach per axis and expand minimally.
    Placeholder (improve in Phase 2/3)."""
    veto: List[SpanScore] = []
    seen = set()
    for s in spans:
        if s["breached"] and s["axis"] not in seen:
            veto.append(s)
            seen.add(s["axis"])
    return veto
