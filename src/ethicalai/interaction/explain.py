from __future__ import annotations
from typing import List, Dict

def refusal_message(veto_spans: List[Dict]) -> str:
    if not veto_spans:
        return "No policy breach detected."
    axes = sorted({s["axis"] for s in veto_spans})
    bullet = "; ".join(f"{a}" for a in axes)
    return (
        "I can’t provide that as-is because it would breach the policy axes: "
        f"{bullet}. I’m happy to help in a safer direction."
    )

def suggest_alternatives(prompt: str) -> List[str]:
    # Very simple heuristics; replace with prompt-editing later
    return [
        f"Provide high-level guidance about: {prompt}",
        f"Explain safety best-practices related to: {prompt}",
        f"Reframe as an educational/ethical analysis of: {prompt}",
    ]
