from __future__ import annotations

"""Query mapping utilities for agent usage.

Functions here map natural language queries to per-axis utility vectors u (k,).
"""
from typing import List
import numpy as np

from coherence.axis.pack import AxisPack
from coherence.cfg.loader import load_app_config
from coherence.encoders.registry import get_encoder
from coherence.metrics.resonance import project, utilities


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def u_from_nl(query_text: str, pack: AxisPack) -> np.ndarray:
    """Map natural language query to utility vector u (k,).

    Steps:
    - encode text with default encoder -> x_q (d,)
    - alpha = project(x_q, pack)
    - u = utilities(alpha, pack)
    - optional squash via sigmoid if search.squash
    """
    cfg = load_app_config()
    squash = bool(cfg.get("search", {}).get("squash", True))

    enc = get_encoder()
    xq = enc.encode([query_text])[0]  # (d,)
    alpha = project(xq, pack)  # (k,)
    u = utilities(alpha, pack)  # (k,)
    if squash:
        u = _sigmoid(u)
    return u.astype(np.float32)
