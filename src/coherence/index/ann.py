from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
from pathlib import Path
from coherence.cfg.loader import load_app_config

# In-memory registry
_indices: Dict[str, object] = {}
_payloads: Dict[str, List[dict]] = {}
_dims: Dict[str, int] = {}
_backends: Dict[str, str] = {}


def has_index(axis_pack_id: str) -> bool:
    return axis_pack_id in _indices


def get_payloads(axis_pack_id: str) -> List[dict]:
    return _payloads.get(axis_pack_id, [])


def init_index(axis_pack_id: str, k: int, backend: str | None = None) -> None:
    cfg = load_app_config()
    be = backend or cfg.get("ann", {}).get("backend", "numpy")
    _dims[axis_pack_id] = int(k)
    _backends[axis_pack_id] = be
    if be == "hnsw":
        try:
            import hnswlib  # type: ignore
        except Exception:  # pragma: no cover
            # Fallback silently to numpy if hnsw not available
            be = "numpy"
            _backends[axis_pack_id] = be
    if be == "hnsw":
        import hnswlib  # type: ignore
        space = cfg.get("ann", {}).get("space", "l2")
        M = int(cfg.get("ann", {}).get("M", 32))
        efc = int(cfg.get("ann", {}).get("ef_construction", 200))
        index = hnswlib.Index(space=space, dim=k)
        index.init_index(max_elements=100000, M=M, ef_construction=efc)
        _indices[axis_pack_id] = index
    else:
        # numpy backend stores just a matrix and grows dynamically
        _indices[axis_pack_id] = np.empty((0, k), dtype=np.float32)
    _payloads.setdefault(axis_pack_id, [])


def add(axis_pack_id: str, items: np.ndarray, ids: List[str], payloads: List[dict]) -> None:
    be = _backends.get(axis_pack_id, "numpy")
    if be == "hnsw":
        index = _indices[axis_pack_id]
        index.add_items(items, np.arange(index.get_current_count(), index.get_current_count() + items.shape[0]))
    else:
        mat = _indices[axis_pack_id]
        _indices[axis_pack_id] = np.vstack([mat, items.astype(np.float32)])
    _payloads[axis_pack_id].extend(payloads)


def query(axis_pack_id: str, vec: np.ndarray, top_k: int) -> Tuple[List[int], np.ndarray]:
    be = _backends.get(axis_pack_id, "numpy")
    if be == "hnsw":
        index = _indices[axis_pack_id]
        labels, distances = index.knn_query(vec.astype(np.float32), k=top_k)
        return labels[0].tolist(), distances[0]
    else:
        mat = _indices.get(axis_pack_id)
        if mat is None or mat.shape[0] == 0:
            return [], np.array([])
        # L2 distance
        diff = mat - vec.astype(np.float32)[None, :]
        dists = np.sqrt((diff * diff).sum(axis=1))
        idx = np.argsort(dists)[:top_k]
        return idx.tolist(), dists[idx]
