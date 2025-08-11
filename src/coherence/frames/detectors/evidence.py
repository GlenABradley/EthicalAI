"""Evidence detector (F1 additive).

Simple cue-based span finder. No heavy NLP.
"""

from __future__ import annotations

from typing import List, Tuple

Span = Tuple[int, int]

# Tunable cues
EVIDENCE_CUES = {"because", "since", "as", "per", "according", "evidence", "docs"}


def detect_evidence_spans(tokens: List[str], *, pivot: int, window: int = 4) -> List[Span]:
    """Return short spans (<=2 tokens) near pivot that match evidence cues.

    Args
    - tokens: list of token strings (lowercased recommended)
    - pivot: predicate token index
    - window: max absolute distance from pivot to consider
    """
    n = len(tokens)
    out: List[Span] = []
    for i, tok in enumerate(tokens):
        if abs(i - pivot) > window:
            continue
        t = tok.lower()
        if t in EVIDENCE_CUES:
            # Prefer token itself; allow including next token if it's a preposition/article sized word
            end = min(i + 1, n)
            out.append((i, end))
    return out
