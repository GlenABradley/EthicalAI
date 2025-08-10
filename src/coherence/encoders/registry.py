from __future__ import annotations

"""Encoder registry.

Provides factory helpers to create encoders by name.
"""

from typing import Any

from coherence.encoders.text_sbert import SBERTEncoder, get_default_encoder


def get_encoder(name: str = "sbert", **kwargs: Any) -> SBERTEncoder:
    """Return an encoder instance by registry name.

    Currently supported:
    - "sbert": Sentence-Transformers encoder. Accepts kwargs for SBERTEncoder.
    - "default": same as sbert with config-driven defaults.
    """
    key = (name or "sbert").lower()
    if key in ("default", "sbert"):
        if key == "default" and not kwargs:
            return get_default_encoder()
        return SBERTEncoder(**kwargs)  # type: ignore[arg-type]
    raise ValueError(f"Unknown encoder: {name}")

"""Encoder registry (Milestone 1).

# TODO: @builder implement registry for pluggable encoders
"""
