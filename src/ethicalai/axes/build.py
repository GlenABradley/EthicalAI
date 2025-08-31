from __future__ import annotations
import json, uuid
import numpy as np
from typing import List, Dict
from .utils import gram_schmidt
from ..types import Axis, AxisPack

def build_axis_pack(seed_vectors: List[np.ndarray], names: List[str], meta: Dict) -> AxisPack:
    ortho = gram_schmidt(seed_vectors)
    axes = []
    for name, v in zip(names, ortho):
        unit = v / (np.linalg.norm(v) + 1e-12)
        axes.append(Axis(name=name, vector=unit, threshold=0.0, provenance={"seed":"embed"}))
    return AxisPack(id=str(uuid.uuid4()), axes=axes, dim=len(axes[0].vector), meta=meta)
