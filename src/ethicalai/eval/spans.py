from __future__ import annotations
import numpy as np
from typing import List, Tuple
from ..types import AxisPack, SpanScore

def pooled(x: np.ndarray) -> np.ndarray:
    # x: [T, D] -> [D] mean-pool
    return x.mean(axis=0)

def sliding_windows(T:int, window:int, stride:int) -> List[Tuple[int,int]]:
    out: List[Tuple[int,int]] = []
    i = 0
    while i < T:
        j = min(T, i+window)
        out.append((i,j))
        i += stride
    return out

def project_scores(X: np.ndarray, pack: AxisPack, window:int=32, stride:int=16) -> List[SpanScore]:
    # X: [T, D]
    spans: List[SpanScore] = []
    if X.ndim != 2 or X.shape[1] != pack.dim:
        raise ValueError("Embedding dim mismatch")
    for i,j in sliding_windows(X.shape[0], window, stride):
        v = pooled(X[i:j])
        for ax in pack.axes:
            s = float(v @ ax.vector)
            spans.append({"i":i,"j":j,"axis":ax.name,"score":s,"threshold":ax.threshold,"breached": s>ax.threshold})
    return spans
