from __future__ import annotations
from typing import List
import os
import numpy as np

_CACHED = None

def _is_test_mode() -> bool:
    return os.getenv("COHERENCE_TEST_MODE", "").lower() in ("1","true","yes")

def align_dim(v: np.ndarray, d: int) -> np.ndarray:
    """
    Ensure 1D vector v has exactly length d by truncation or zero-padding.
    """
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    if v.shape[0] == d:
        return v
    if v.shape[0] > d:
        return v[:d]
    out = np.zeros((d,), dtype=np.float32)
    out[: v.shape[0]] = v
    return out

class _HashEncoder:
    """Deterministic, dependency-free encoder.
    Uses SHA-256 based mapping for stability across runs.
    """
    def __init__(self, dim: int = 384):
        self.dim = dim
    def _tok_vec(self, tok: str) -> np.ndarray:
        import hashlib
        b = hashlib.sha256(tok.encode("utf-8")).digest()
        # Build a length-d float vector from bytes deterministically
        reps = (self.dim + len(b) - 1) // len(b)
        arr = np.frombuffer((b * reps)[: self.dim], dtype=np.uint8).astype(np.float32)
        arr = (arr - arr.mean()) / (arr.std() + 1e-6)
        n = np.linalg.norm(arr) + 1e-12
        return (arr / n).astype(np.float32)
    def encode_tokens(self, tokens: List[str]) -> np.ndarray:
        if not tokens:
            return np.zeros((1, self.dim), dtype=np.float32)
        return np.stack([self._tok_vec(t) for t in tokens], axis=0)
    def encode_text(self, text: str) -> np.ndarray:
        return self.encode_tokens(text.split())

def get_encoder():
    """
    Best-available encoder:
      - COHERENCE_TEST_MODE=true -> 16-D deterministic hash encoder
      - Else try coherence.models.get_encoder()
      - Else fallback to 384-D deterministic hash encoder
    """
    global _CACHED
    if _CACHED is not None:
        return _CACHED
    if _is_test_mode():
        _CACHED = _HashEncoder(dim=16)
        return _CACHED
    try:
        from coherence.models import get_encoder as _coh_get  # type: ignore
        _CACHED = _coh_get()
        return _CACHED
    except Exception:
        _CACHED = _HashEncoder(dim=384)
        return _CACHED
