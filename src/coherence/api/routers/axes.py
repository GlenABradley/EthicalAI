from __future__ import annotations

from typing import Dict, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
import time
import numpy as np

from coherence.api.models import CreateAxisPack
from coherence.axis.builder import build_axis_pack_from_seeds
from coherence.axis.pack import AxisPack
from coherence.encoders.registry import get_encoder


DATA_AXES_DIR = Path("data/axes")
DATA_AXES_DIR.mkdir(parents=True, exist_ok=True)

router = APIRouter()


class CreateAxisPackResponse(BaseModel):
    """Response after creating an axis pack."""

    axis_pack_id: str
    k: int
    names: List[str]


@router.get("/list")
def list_packs() -> Dict[str, List[Dict[str, object]]]:
    """List available axis packs found under data/axes.

    Returns a list of summaries with id, names, and k.
    """
    items: List[Dict[str, object]] = []
    for p in sorted(DATA_AXES_DIR.glob("*.json")):
        try:
            pack = AxisPack.load(p)
            items.append({"id": p.stem, "names": pack.names, "k": pack.k})
        except Exception:
            # skip unreadable packs
            continue
    return {"items": items}


@router.get("/{axis_pack_id}")
def get_pack(axis_pack_id: str) -> Dict[str, object]:
    """Get a single axis pack summary by ID."""
    f = DATA_AXES_DIR / f"{axis_pack_id}.json"
    if not f.exists():
        raise HTTPException(status_code=404, detail="Axis pack not found")
    pack = AxisPack.load(f)
    return {
        "id": axis_pack_id,
        "names": pack.names,
        "k": pack.k,
        "d": pack.d,
        "meta": pack.meta,
    }


@router.post("/create", response_model=CreateAxisPackResponse)
def create_pack(payload: CreateAxisPack) -> CreateAxisPackResponse:
    """Create an axis pack from seed phrases and persist it.

    Uses the default encoder and diff-of-means builder.
    Saves to data/axes/{axis_pack_id}.json
    """
    if not payload.axes:
        raise HTTPException(status_code=400, detail="No axes provided")

    # Prepare seeds mapping expected by builder
    seeds = {a.name: {"positive": a.positives, "negative": a.negatives} for a in payload.axes}

    enc = get_encoder()  # default from config
    pack = build_axis_pack_from_seeds(
        seeds,
        encode_fn=enc.encode,
        lambda_init=1.0 if payload.lambda_ is None else None,
        beta_init=0.0 if payload.beta is None else None,
        weights_init=None if payload.weights is None else payload.weights,
        meta={"built_from": "api.create", "ts": time.time()},
    )

    # Apply overrides if provided
    if payload.lambda_ is not None:
        if len(payload.lambda_) != pack.k:
            raise HTTPException(status_code=400, detail="lambda_ length must equal k")
        pack.lambda_ = np.asarray(payload.lambda_, dtype=np.float32)
    if payload.beta is not None:
        if len(payload.beta) != pack.k:
            raise HTTPException(status_code=400, detail="beta length must equal k")
        pack.beta = np.asarray(payload.beta, dtype=np.float32)
    if payload.weights is not None:
        if len(payload.weights) != pack.k:
            raise HTTPException(status_code=400, detail="weights length must equal k")
        pack.weights = np.asarray(payload.weights, dtype=np.float32)

    # Simple deterministic id from names and time bucket
    ts = int(time.time())
    base = "ap_" + str(ts)
    out_path = DATA_AXES_DIR / f"{base}.json"
    pack.save(out_path)
    return CreateAxisPackResponse(axis_pack_id=base, k=pack.k, names=pack.names)
