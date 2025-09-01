from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

def phi_hinge(score, tau, margin=0.0):
    # penalty if score violates (for positive-aligned axes, we discourage > tau when axis is a *risk*);
    # you'll flip sign/roles by axis category in Phase 3 policy.
    return max(0.0, float(score - tau - margin))

def regularizer_numpy(span_scores: Dict[str, List[Tuple[float, float]]],
                      weights: Dict[str, float], lam: float = 0.1) -> float:
    """span_scores: {axis: [(score, tau), ...]} → penalty scalar.
       weights: per-axis weight. lam: global multiplier.
    """
    total = 0.0
    for axis, items in span_scores.items():
        w = float(weights.get(axis, 1.0))
        for s, t in items:
            total += w * phi_hinge(s, t)
    return lam * total

def regularizer_torch(span_scores: Dict[str, List[Tuple["torch.Tensor","torch.Tensor"]]],
                      weights: Dict[str, float], lam: float = 0.1) -> "torch.Tensor":
    """Torch variant for training loops. Only used if torch is available."""
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch not available")
    total = torch.zeros((), dtype=torch.float32)
    for axis, items in span_scores.items():
        w = float(weights.get(axis, 1.0))
        for s, t in items:
            total = total + w * torch.clamp(s - t, min=0.0)
    return lam * total
