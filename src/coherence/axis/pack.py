"""AxisPack dataclass and JSON IO for semantic axis operations.

This module defines the core AxisPack dataclass which represents a multi-dimensional
semantic space for ethical evaluation and text analysis. An AxisPack contains:

- Orthonormal projection matrix Q for mapping text embeddings to axis space
- Axis names and metadata for interpretability
- Calibration parameters (lambda, beta) for scoring
- Optional Choquet integral weights for non-linear aggregation
- Optional veto thresholds for hard constraints

Shapes:
- Q: (d, k) orthonormal columns where Q.T @ Q = I_k
- lambda_/beta/weights: (k,) calibration and weighting vectors
- mu: mapping from frozenset[int] -> float for Choquet capacity (optional)

Determinism:
- All operations are pure functions with deterministic I/O
- Reproducible results given the same inputs and random seeds
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Union

import json
import logging
import numpy as np


def _mu_to_json(mu: Optional[Dict[FrozenSet[int], float]]) -> Dict[str, float]:
    """Convert Choquet capacity from frozenset keys to JSON-serializable format.
    
    Args:
        mu: Choquet capacity mapping from axis subsets to weights.
        
    Returns:
        JSON-serializable dict with comma-separated string keys.
    """
    if not mu:
        return {}
    out: Dict[str, float] = {}
    for kset, v in mu.items():
        key = ",".join(str(i) for i in sorted(kset))
        out[key] = float(v)
    return out


def _mu_from_json(d: Optional[Dict[str, float]]) -> Dict[FrozenSet[int], float]:
    """Convert JSON Choquet capacity to frozenset-keyed format.
    
    Args:
        d: JSON dict with comma-separated string keys.
        
    Returns:
        Choquet capacity mapping with frozenset keys.
    """
    if not d:
        return {}
    out: dict[frozenset[int], float] = {}
    for key, v in d.items():
        if key.strip() == "":
            continue
        idx = frozenset(int(x) for x in key.split(","))
        out[idx] = float(v)
    return out


@dataclass
class AxisPack:
    """Container for semantic axis pack data and operations.
    
    An AxisPack represents a learned semantic space with multiple ethical
    or conceptual dimensions. It provides methods for:
    - Projecting text embeddings onto semantic axes
    - Computing axis scores with calibration
    - Aggregating scores across dimensions
    - Detecting veto conditions

    Attributes
    - names: list of axis names (len k)
    - Q: (d, k) float32 array with orthonormal columns
    - lambda_: (k,) scaling for utilities
    - beta: (k,) bias for utilities
    - weights: (k,) aggregation weights (linear); ignored if mu provided
    - mu: Choquet capacity mapping; empty -> linear aggregation
    - meta: free-form metadata
    """

    names: List[str]
    Q: np.ndarray
    lambda_: np.ndarray
    beta: np.ndarray
    weights: np.ndarray
    mu: Dict[FrozenSet[int], float]
    meta: Dict[str, object]

    @property
    def k(self) -> int:
        return len(self.names)

    @property
    def d(self) -> int:
        return int(self.Q.shape[0])

    def validate(self) -> None:
        k = self.k
        logger = logging.getLogger(__name__)
        logger.debug(f"Validating AxisPack with k={k}")
        
        # Check Q dimensions
        logger.debug(f"Q shape: {self.Q.shape}, ndim: {self.Q.ndim}")
        assert self.Q.ndim == 2, f"Q must be 2D, got shape {self.Q.shape}"
        assert self.Q.shape[1] == k, f"Q second dim must equal k, got {self.Q.shape[1]} vs k={k}"
        
        # Check other parameter shapes
        for arr, name in ((self.lambda_, "lambda_"), (self.beta, "beta"), (self.weights, "weights")):
            logger.debug(f"{name} shape: {arr.shape}, expected: ({k},)")
            assert arr.shape == (k,), f"{name} must be shape ({k},), got {arr.shape}"
        
        # Check orthonormality (allow small numerical error)
        qtq = self.Q.T @ self.Q
        logger.debug(f"Q.T @ Q shape: {qtq.shape}, expected: ({k}, {k})")
        logger.debug(f"Q.T @ Q diagonal: {np.diag(qtq)}")
        logger.debug(f"Q.T @ Q off-diagonal max: {np.max(np.abs(qtq - np.diag(np.diag(qtq))))}")
        
        if not np.allclose(qtq, np.eye(k), atol=1e-5):
            logger.error("Q columns not orthonormal within tolerance")
            logger.error(f"Q.T @ Q:\n{qtq}")
            logger.error(f"Expected identity:\n{np.eye(k)}")
            raise ValueError("Q columns not orthonormal within tolerance")

    def to_json_obj(self) -> dict:
        self.validate()
        return {
            "names": list(self.names),
            "Q": self.Q.astype(np.float32).tolist(),
            "lambda": self.lambda_.astype(float).tolist(),
            "beta": self.beta.astype(float).tolist(),
            "weights": self.weights.astype(float).tolist(),
            "mu": _mu_to_json(self.mu),
            "meta": self.meta or {},
        }

    @staticmethod
    def from_json_obj(obj: dict) -> "AxisPack":
        names = list(obj["names"])
        Q = np.array(obj["Q"], dtype=np.float32)
        lambda_ = np.array(obj.get("lambda", [1.0] * len(names)), dtype=np.float32)
        beta = np.array(obj.get("beta", [0.0] * len(names)), dtype=np.float32)
        weights = np.array(obj.get("weights", [1.0 / max(1, len(names))] * len(names)), dtype=np.float32)
        mu = _mu_from_json(obj.get("mu", {}))
        meta = obj.get("meta", {})
        pack = AxisPack(names=names, Q=Q, lambda_=lambda_, beta=beta, weights=weights, mu=mu, meta=meta)
        pack.validate()
        return pack

    def save(self, path: Union[Path, str]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(self.to_json_obj(), f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: Union[Path, str]) -> "AxisPack":
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return AxisPack.from_json_obj(obj)

"""AxisPack dataclass and IO (Milestone 1).

# TODO: @builder implement in Milestone 1
"""
