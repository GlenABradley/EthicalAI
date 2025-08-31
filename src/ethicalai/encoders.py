from __future__ import annotations
from typing import List
import numpy as np

class _FallbackEncoder:
    """Deterministic, dependency-free encoder for CI.
    Maps tokens to vectors via stable hashing; mean-pools for text."""
    def __init__(self, dim: int = 384):
        self.dim = dim
    def _tok_vec(self, tok: str) -> np.ndarray:
        h = abs(hash(tok)) % (10**9)
        rng = np.random.default_rng(h)
        v = rng.standard_normal(self.dim).astype(np.float32)
        v /= (np.linalg.norm(v) + 1e-12)
        return v
    def encode_tokens(self, tokens: List[str]) -> np.ndarray:
        if not tokens:
            return np.zeros((1, self.dim), dtype=np.float32)
        return np.stack([self._tok_vec(t) for t in tokens], axis=0)
    def encode_text(self, text: str) -> np.ndarray:
        tokens = text.split()
        X = self.encode_tokens(tokens)
        return X

def get_encoder():
    """Try to return the real coherence encoder; else fallback."""
    # Adapt this import to your repo if you already expose an encoder
    # e.g., from coherence.models import get_encoder
    try:
        from coherence.models import get_encoder as _coh_get  # type: ignore
        enc = _coh_get()
        # Must expose encode_text(str)->[T,D] and encode_tokens(List[str])->[T,D]
        return enc
    except Exception:
        return _FallbackEncoder(dim=384)
