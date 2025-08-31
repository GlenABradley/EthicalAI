from __future__ import annotations
import uuid, numpy as np
from typing import List, Dict
from .utils import gram_schmidt
from ..types import Axis, AxisPack

def build_axis_pack(seed_vectors: List[np.ndarray], names: List[str], meta: Dict) -> AxisPack:
    assert len(seed_vectors) == len(names) and len(seed_vectors) > 0
    ortho = gram_schmidt(seed_vectors)
    axes = []
    for name, v in zip(names, ortho):
        v = v / (np.linalg.norm(v) + 1e-12)
        axes.append(Axis(name=name, vector=v.astype(np.float32), threshold=0.0, provenance={"seed":"phrases"}))
    dim = axes[0].vector.shape[0]
    return AxisPack(id=str(uuid.uuid4()), axes=axes, dim=dim, meta=meta or {})
