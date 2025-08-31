from __future__ import annotations
from typing import List, Dict
from ..types import SpanScore
def minimal_veto_spans(spans: List[SpanScore]) -> List[SpanScore]:
    # Greedy shrink within each axis/window family (placeholder; refine)
    veto = [s for s in spans if s["breached"]]
    return veto
