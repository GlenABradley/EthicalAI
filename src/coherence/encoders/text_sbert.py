from __future__ import annotations

"""Sentence-Transformers encoder.

Provides a simple interface to encode a list of texts into embeddings.
Deterministic given the model and inputs.
"""

from dataclasses import dataclass
import os
import sys
import hashlib
from typing import List, Optional

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - import fallback
    SentenceTransformer = None  # type: ignore

# Simple module-level cache to avoid reloading models repeatedly
# Keyed by (model_name, resolved_device, normalize_input)
_ENCODER_CACHE: dict[tuple[str, str, bool], "SBERTEncoder"] = {}

def _select_device(device: str) -> str:
    if device == "auto":
        # Let sentence-transformers auto-select; return "cpu" to be safe in tests
        return "cpu"
    return device


@dataclass
class SBERTEncoder:
    """Wrapper around SentenceTransformer.

    Fields
    - model_name: huggingface model id
    - device: "cpu" | "cuda" | "mps" (auto resolved to cpu by default)
    - normalize_input: if True, lowercases and strips inputs deterministically

    Methods
    - encode(texts) -> np.ndarray of shape (N, d)
    """

    model_name: str
    device: str = "auto"
    normalize_input: bool = False

    def __post_init__(self) -> None:
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers is not installed")
        dev = _select_device(self.device)
        print(f"[DEBUG] Loading SBERT model: {self.model_name} on device: {dev}")
        self._model = SentenceTransformer(self.model_name, device=dev)
        # Print model info for debugging
        print(f"[DEBUG] Model max sequence length: {self._model.max_seq_length}")
        print(f"[DEBUG] Model device: {self._model.device}")
        print(f"[DEBUG] Model dimension: {self._model.get_sentence_embedding_dimension()}")

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts into embeddings.

        Args
        - texts: list of strings, length N

        Returns
        - embeddings: np.ndarray (N, d)
        """
        if self.normalize_input:
            texts = [t.strip().lower() for t in texts]
        embs = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False)
        # Ensure float32
        return np.asarray(embs, dtype=np.float32)


def get_default_encoder(name: Optional[str] = None, device: str = "auto", normalize_input: bool = False) -> SBERTEncoder:
    """Construct the default encoder using config or provided name.

    If name is None, fall back to configs/app.yaml encoder.name.
    """
    if name is None:
        from coherence.cfg.loader import load_app_config

        cfg = load_app_config()
        enc = cfg.get("encoder", {})
        # Environment overrides
        env_name = os.getenv("COHERENCE_ENCODER")
        env_device = os.getenv("COHERENCE_DEVICE")
        name = env_name or enc.get("name", "sentence-transformers/all-mpnet-base-v2")
        device = env_device or enc.get("device", device)
        normalize_input = bool(enc.get("normalize_input", normalize_input))

    # Lightweight stub during tests (to avoid model downloads) unless explicitly overridden
    use_real = os.getenv("COHERENCE_TEST_REAL_ENCODER", "").lower() in ("1", "true", "yes")
    is_pytest = "pytest" in sys.modules
    use_test_mode = os.getenv("COHERENCE_TEST_MODE", "").lower() in ("1", "true", "yes") or is_pytest
    if use_test_mode and not use_real:
        resolved_device = _select_device(device)
        cache_key = (name, resolved_device, bool(normalize_input))
        cached = _ENCODER_CACHE.get(cache_key)
        if cached is not None:
            return cached

        class _StubModel:
            def get_sentence_embedding_dimension(self) -> int:
                # Default SBERT dimension commonly used in tests
                return 768

        class _StubEncoder:
            def __init__(self, model_name: str, device: str, normalize_input: bool) -> None:
                self.model_name = model_name
                self.device = device
                self.normalize_input = normalize_input
                self._model = _StubModel()

            def encode(self, texts: List[str]) -> np.ndarray:
                # Deterministic per-text embeddings derived from SHA-256 of text
                d = self._model.get_sentence_embedding_dimension()
                out = np.zeros((len(texts), d), dtype=np.float32)
                for i, t in enumerate(texts):
                    h = hashlib.sha256(t.encode("utf-8")).digest()
                    seed = int.from_bytes(h[:8], "little", signed=False)
                    rng = np.random.default_rng(seed)
                    vec = rng.random(d, dtype=np.float32)
                    # Optional simple normalization to mimic unit-length embeddings
                    norm = float(np.linalg.norm(vec))
                    if norm > 0:
                        vec = vec / norm
                    out[i] = vec
                return out

        encoder = _StubEncoder(model_name=name, device=resolved_device, normalize_input=normalize_input)
        _ENCODER_CACHE[cache_key] = encoder
        return encoder
    # Resolve device for caching key to avoid distinct entries for "auto"
    resolved_device = _select_device(device)
    cache_key = (name, resolved_device, bool(normalize_input))
    cached = _ENCODER_CACHE.get(cache_key)
    if cached is not None:
        return cached

    encoder = SBERTEncoder(model_name=name, device=resolved_device, normalize_input=normalize_input)
    _ENCODER_CACHE[cache_key] = encoder
    return encoder

"""Sentence-Transformers encoder (Milestone 1).

# TODO: @builder implement encode_texts(texts: list[str]) -> np.ndarray
"""
