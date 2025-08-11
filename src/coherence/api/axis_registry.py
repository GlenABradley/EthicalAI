from __future__ import annotations

import json
import os
import threading
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Optional, TypedDict

import numpy as np

DEFAULT_ARTIFACTS_DIR = os.getenv("COHERENCE_ARTIFACTS_DIR", "artifacts")
ACTIVE_FILE = "active.json"
SUPPORTED_SCHEMA_VERSIONS = {"axis-pack/1.1"}


class LoadedPack(TypedDict):
    pack_id: str
    Q: np.ndarray
    lambda_: np.ndarray
    beta: np.ndarray
    weights: np.ndarray
    names: list[str]
    meta: dict
    hash: str
    D: int
    k: int


class AxisRegistry:
    def __init__(self, artifacts_dir: Path, encoder_dim: int):
        self.root = Path(artifacts_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.encoder_dim = int(encoder_dim)
        self._lock = threading.RLock()
        self._active: Optional[str] = None
        self._cache: dict[str, LoadedPack] = {}
        self._active_path = self.root / ACTIVE_FILE
        # Restore active if present
        if self._active_path.exists():
            try:
                data = json.loads(self._active_path.read_text(encoding="utf-8"))
                self._active = data.get("pack_id")
            except Exception:
                pass

    def _npz_path(self, pack_id: str) -> Path:
        return self.root / f"axis_pack:{pack_id}.npz"

    def _meta_path(self, pack_id: str) -> Path:
        return self.root / f"axis_pack:{pack_id}.meta.json"

    def _hash_file(self, p: Path) -> str:
        b = p.read_bytes()
        return sha256(b).hexdigest()

    def _validate_Q(self, Q: np.ndarray, names: list[str]) -> None:
        if not isinstance(Q, np.ndarray):
            raise ValueError("Q must be a numpy array")
        if Q.ndim != 2:
            raise ValueError(f"Q must be 2D, got {Q.ndim}D")
        D, k = Q.shape
        if D != self.encoder_dim:
            raise ValueError(f"Axis pack dim {D} != encoder {self.encoder_dim}")
        if k != len(names):
            raise ValueError(f"k={k} != len(names)={len(names)}")
        qtq = Q.T @ Q
        if not np.allclose(qtq, np.eye(k), atol=1e-4):
            raise ValueError("Q columns not orthonormal within tolerance")

    def _validate_meta(self, meta: dict) -> None:
        schema = meta.get("schema_version")
        if schema not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(f"Unsupported schema_version: {schema}")
        # Prefer nested builder_params.encoder_dim if present; else fallback to top-level
        bp = meta.get("builder_params") or {}
        enc_dim = bp.get("encoder_dim", meta.get("encoder_dim"))
        if enc_dim is None:
            raise ValueError("Missing encoder_dim in meta")
        if int(enc_dim) != int(self.encoder_dim):
            raise ValueError(f"Encoder dim mismatch: meta {enc_dim} != registry {self.encoder_dim}")

    def load(self, pack_id: str) -> LoadedPack:
        with self._lock:
            if pack_id in self._cache:
                return self._cache[pack_id]
            return self._load_from_disk(pack_id)

    def _load_from_disk(self, pack_id: str) -> LoadedPack:
        """Always load the pack from disk and refresh cache, validating meta and Q.

        Used by activate() to avoid serving stale cached content when on-disk
        artifacts have changed (e.g., during tests that tamper meta for negative cases).
        """
        npz_p = self._npz_path(pack_id)
        meta_p = self._meta_path(pack_id)
        if not npz_p.exists() or not meta_p.exists():
            raise FileNotFoundError(f"Axis pack {pack_id} not found")
        pack_hash = self._hash_file(npz_p)
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        npz = np.load(npz_p)
        Q = np.asarray(npz["Q"], dtype=np.float32)
        lambda_ = np.asarray(npz.get("lambda_"), dtype=np.float32) if "lambda_" in npz else np.full((Q.shape[1],), 1.0, dtype=np.float32)
        beta = np.asarray(npz.get("beta"), dtype=np.float32) if "beta" in npz else np.zeros((Q.shape[1],), dtype=np.float32)
        weights = np.asarray(npz.get("weights"), dtype=np.float32) if "weights" in npz else np.full((Q.shape[1],), 1.0/float(Q.shape[1]), dtype=np.float32)
        names = list(meta.get("names", []))
        # Validate meta and shapes
        self._validate_meta(meta)
        self._validate_Q(Q, names)
        D, k = Q.shape
        lp: LoadedPack = {
            "pack_id": pack_id,
            "Q": Q,
            "lambda_": lambda_,
            "beta": beta,
            "weights": weights,
            "names": names,
            "meta": meta,
            "hash": pack_hash,
            "D": int(D),
            "k": int(k),
        }
        # Refresh cache (simple LRU of size 4)
        if len(self._cache) >= 4:
            self._cache.pop(next(iter(self._cache)))
        self._cache[pack_id] = lp
        return lp

    def activate(self, pack_id: str) -> LoadedPack:
        # Force a fresh load from disk to validate artifacts and meta
        lp = self._load_from_disk(pack_id)
        with self._lock:
            self._active = pack_id
            self._active_path.write_text(json.dumps({"pack_id": pack_id, "hash": lp["hash"]}), encoding="utf-8")
        return lp

    def get_active(self) -> Optional[LoadedPack]:
        with self._lock:
            if self._active is None:
                return None
            try:
                return self.load(self._active)
            except Exception:
                return None


# Global singleton handle populated at app startup
REGISTRY: Optional[AxisRegistry] = None


def init_registry(encoder_dim: int, artifacts_dir: Optional[str] = None) -> AxisRegistry:
    global REGISTRY
    if REGISTRY is None:
        REGISTRY = AxisRegistry(Path(artifacts_dir or DEFAULT_ARTIFACTS_DIR), encoder_dim)
    return REGISTRY
