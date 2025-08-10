from __future__ import annotations

"""Frame, Edge, FrameGraph schema (Milestone 5).

Lightweight data structures for semantic frames.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


Span = Tuple[int, int]  # [start, end)


@dataclass
class Frame:
    """A predicate-argument frame with role spans over tokens.

    Roles are arbitrary strings (e.g., 'predicate', 'arg1', 'arg2').
    """

    id: str
    predicate: Span
    roles: Dict[str, Span] = field(default_factory=dict)
    score: float = 0.0
    meta: Dict[str, object] = field(default_factory=dict)


@dataclass
class Edge:
    src: str
    dst: str
    label: str


@dataclass
class FrameGraph:
    frames: List[Frame]
    edges: List[Edge] = field(default_factory=list)
