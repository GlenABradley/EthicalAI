from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypedDict, List, Dict
import numpy as np

class Encoder(Protocol):
    def encode_text(self, text: str) -> np.ndarray: ...
    def encode_tokens(self, tokens: List[str]) -> np.ndarray: ...

@dataclass
class Axis:
    name: str
    vector: np.ndarray  # unit
    threshold: float
    provenance: Dict

@dataclass
class AxisPack:
    id: str
    axes: List[Axis]
    dim: int
    meta: Dict

class SpanScore(TypedDict):
    i: int
    j: int
    axis: str
    score: float
    threshold: float
    breached: bool

class DecisionProof(TypedDict):
    objective: str
    pack_id: str
    spans: List[SpanScore]
    aggregation: Dict
    final: Dict
