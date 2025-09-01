"""Frames API (v1)"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import sqlite3

import logging
import math
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

import coherence.api.axis_registry as axis_registry
from coherence.axis.pack import AxisPack
from coherence.memory.store import create_store, FrameStore

router = APIRouter()

_STORE: Optional[FrameStore] = None
_STORE_DB_PATH: Optional[str] = None


def get_store() -> FrameStore:
    global _STORE, _STORE_DB_PATH
    db_path = Path(os.environ.get("COHERENCE_ARTIFACTS_DIR", "artifacts")) / "frames.sqlite"
    if _STORE is None or _STORE_DB_PATH != str(db_path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _STORE = create_store(db_path)
        _STORE_DB_PATH = str(db_path)
    return _STORE  # type: ignore[return-value]


# ======== Models ========
class FrameItem(BaseModel):
    id: str
    predicate: Optional[List[int]] = None
    roles: Optional[Dict[str, List[int]]] = None
    coords: Optional[List[float]] = None
    role_coords: Optional[Dict[str, List[float]]] = None
    meta: Optional[Dict[str, Any]] = None


class IndexRequest(BaseModel):
    doc_id: str = Field(...)
    pack_id: Optional[str] = Field(None, description="Axis pack id; if provided, use this pack to resolve k,d")
    d: Optional[int] = Field(None, description="Embedding dim; required if deriving without pack_id")
    frames: List[FrameItem]
    frame_vectors: Optional[List[List[float]]] = None


class IndexResponse(BaseModel):
    ingested: int
    k: int


class SearchResponseItem(BaseModel):
    frame_id: str
    doc_id: str
    axis_idx: int
    coord: float
    predicate: List[int]
    pack_id: str
    pack_hash: str


class SearchResponse(BaseModel):
    items: List[SearchResponseItem]


class TraceResponseItem(BaseModel):
    frame_id: str
    doc_id: str
    predicate: List[int]
    pack_id: str
    pack_hash: str


class TraceResponse(BaseModel):
    items: List[TraceResponseItem]


# ======== Helpers ========

def _validate_coords(name: str, arr, k: int) -> None:
    if arr is None:
        return
    if not isinstance(arr, (list, tuple)):
        raise HTTPException(status_code=422, detail=f"{name} must be a list of numbers (got {type(arr).__name__})")
    if len(arr) != k:
        raise HTTPException(status_code=422, detail=f"{name} length must equal k ({k}); got {len(arr)}")
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in arr):
        raise HTTPException(status_code=422, detail=f"{name} must contain only finite numbers")

def _load_pack(req_pack_id: Optional[str]) -> Dict[str, Any]:
    reg = axis_registry.REGISTRY
    if reg is None:
        # Initialize registry if not present
        try:
            from coherence.encoders.text_sbert import get_default_encoder
            enc = get_default_encoder()
            artifacts_dir = os.environ.get("COHERENCE_ARTIFACTS_DIR", "artifacts")
            axis_registry.REGISTRY = axis_registry.init_registry(encoder_dim=enc._model.get_sentence_embedding_dimension(), artifacts_dir=artifacts_dir)
            reg = axis_registry.REGISTRY
        except Exception:
            raise HTTPException(status_code=500, detail="Registry not initialized")
    try:
        if req_pack_id:
            return reg.load(req_pack_id)
        active = reg.get_active()
        if not active:
            raise HTTPException(status_code=400, detail="No axis pack provided and no active pack")
        return reg.load(active["pack_id"])  # type: ignore[index]
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Pack not found")
    except ValueError as e:
        # dimension or orthonormality mismatch
        raise HTTPException(status_code=409, detail=str(e))


def _axis_to_index(axis: Union[str, int], names: List[str]) -> int:
    if isinstance(axis, int):
        return axis
    try:
        return int(axis)
    except (TypeError, ValueError):
        pass
    if axis in names:
        return names.index(axis)
    raise HTTPException(status_code=422, detail=f"Axis not found: {axis}")


# ======== Routes ========

@router.post("/index", response_model=IndexResponse)
def index_frames(req: IndexRequest) -> IndexResponse:
    log = logging.getLogger("coherence.api")
    # Resolve k,d with priority: pack_id > active pack > derived from coords > none
    k: int = 0
    d: int = 0
    pack_id: str = ""
    pack_hash: str = ""
    source = "none"

    reg = axis_registry.REGISTRY
    if req.pack_id:
        if reg is None:
            # Initialize registry if not present
            try:
                from coherence.encoders.text_sbert import get_default_encoder
                enc = get_default_encoder()
                artifacts_dir = os.environ.get("COHERENCE_ARTIFACTS_DIR", "artifacts")
                reg = axis_registry.init_registry(encoder_dim=enc._model.get_sentence_embedding_dimension(), artifacts_dir=artifacts_dir)
            except Exception:
                raise HTTPException(status_code=500, detail="Registry not initialized")
        try:
            lp = reg.load(req.pack_id)
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail="pack_id not found")
        k = int(lp["Q"].shape[1])
        d = int(lp["Q"].shape[0])
        pack_id = lp.get("pack_id", req.pack_id)
        pack_hash = lp.get("hash", "")
        source = "pack"
    elif reg is not None:
        active = reg.get_active()
        if active:
            k = int(active["Q"].shape[1])
            d = int(active["Q"].shape[0])
            pack_id = active.get("pack_id", "")
            pack_hash = active.get("hash", "")
            source = "active"
        elif req.frames:
            first = req.frames[0]
            if first.coords is not None:
                k = int(len(first.coords))
                if req.d is None:
                    raise HTTPException(status_code=422, detail="d required when deriving k from coords without pack")
                d = int(req.d)
                source = "derived"
    elif req.frames:
        first = req.frames[0]
        if first.coords is not None:
            k = int(len(first.coords))
            if req.d is None:
                raise HTTPException(status_code=422, detail="d required when deriving k from coords without pack")
            d = int(req.d)
            source = "derived"

    # If we still cannot resolve k, return 422 (caller must provide pack_id or coords+d)
    if k == 0:
        raise HTTPException(status_code=422, detail="Cannot resolve k; provide pack_id or coords with d")

    # Validate and build frames to ingest: allow missing coords (ingest frame row only)
    frames_payload: List[Dict[str, Any]] = []
    for f in req.frames:
        # Validate coords if present; hard-fail on mismatch
        _validate_coords("coords", f.coords, k)
        # Validate role_coords if present; hard-fail on mismatch
        role_coords_out: Dict[str, List[float]] = {}
        if f.role_coords:
            for rname, arr in f.role_coords.items():
                _validate_coords(f"role_coords[{rname}]", arr, k)
                role_coords_out[rname] = arr  # safe: validated
        frames_payload.append(
            {
                "id": f.id,
                "predicate": f.predicate or [0, 0],
                "roles": f.roles or {},
                "coords": f.coords,  # may be None; store will skip frame_axis writes
                "role_coords": role_coords_out,
                "meta": f.meta or {},
            }
        )

    log.info(
        "[v1_frames.index] pack_source=%s resolved k=%s d=%s frames_in=%d ingested=%d",
        source,
        k,
        d,
        len(req.frames),
        len(frames_payload),
    )

    # Ingest only if we actually have valid frames
    ing = 0
    if frames_payload:
        try:
            store = get_store()
            ing = store.put(
                doc_id=req.doc_id,
                frames=frames_payload,
                frame_vectors=req.frame_vectors,
                pack_id=pack_id,
                pack_hash=pack_hash,
                k=int(k),  # type: ignore[arg-type]
                d=int(d),  # type: ignore[arg-type]
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

    return IndexResponse(ingested=ing, k=int(k))  # type: ignore[arg-type]


@router.get("/search", response_model=SearchResponse)
def search_frames(
    axis: str = Query(..., description="Axis name or index"),
    min: float = Query(...),
    max: float = Query(...),
    limit: int = Query(100, ge=1, le=1000),
    pack_id: Optional[str] = None,
) -> SearchResponse:
    lp = _load_pack(pack_id)
    names: List[str] = lp["names"]
    idx = _axis_to_index(axis, names)
    store = get_store()
    items = store.search(axis_idx=idx, min_val=min, max_val=max, limit=limit)
    return SearchResponse(items=[SearchResponseItem(**it) for it in items])


@router.get("/trace/{entity}", response_model=TraceResponse)
def trace_entity(entity: str, limit: int = 100, pack_id: Optional[str] = None) -> TraceResponse:
    _ = _load_pack(pack_id)  # ensure pack exists; not used for MVP match
    store = get_store()
    items = store.trace(entity_str=entity, limit=limit)
    return TraceResponse(items=[TraceResponseItem(**it) for it in items])


@router.get("/stats")
def stats() -> Dict[str, Any]:
    """Return database stats and active pack summary."""
    store = get_store()
    db_path = str(store.db_path)
    size = os.path.getsize(db_path) if os.path.exists(db_path) else 0
    counts = {"frames": 0, "frame_axis": 0, "frame_vectors": 0}
    last_ts = 0
    try:
        cur = store.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM frames")
        counts["frames"] = int(cur.fetchone()[0])
        cur.execute("SELECT COUNT(*) FROM frame_axis")
        counts["frame_axis"] = int(cur.fetchone()[0])
        cur.execute("SELECT COUNT(*) FROM frame_vectors")
        counts["frame_vectors"] = int(cur.fetchone()[0])
        cur.execute("SELECT MAX(created_at) FROM frames")
        row = cur.fetchone()
        last_ts = int(row[0]) if row and row[0] is not None else 0
    except Exception:
        pass

    active_pack = None
    reg = axis_registry.REGISTRY
    if reg is not None:
        lp = reg.get_active()
        if lp is not None:
            active_pack = {
                "pack_id": lp["pack_id"],
                "k": int(lp["k"]),
                "schema_version": lp["meta"].get("schema_version"),
            }

    return {
        "db_path": db_path,
        "db_size_bytes": size,
        "counts": counts,
        "last_ingest_ts": last_ts,
        "active_pack": active_pack,
    }
