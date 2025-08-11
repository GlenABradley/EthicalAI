from __future__ import annotations

import json
import os
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from coherence.api.axis_registry import REGISTRY, init_registry
from coherence.axis.advanced_builder import build_advanced_axis_pack
from coherence.encoders.text_sbert import get_default_encoder

router = APIRouter()

SCHEMA_VERSION = "axis-pack/1.1"
DEFAULT_ARTIFACTS_DIR = os.getenv("COHERENCE_ARTIFACTS_DIR", "artifacts")


class BuildRequest(BaseModel):
    json_paths: Optional[List[str]] = Field(None, description="Paths to axis JSON configs")
    override: Optional[Dict] = Field(None, description="Builder overrides")
    pack_id: Optional[str] = Field(None, description="Optional custom pack id")


class BuildResponse(BaseModel):
    pack_id: str
    dim: int
    k: int
    names: List[str]
    pack_hash: str


@router.post("/build", response_model=BuildResponse, status_code=status.HTTP_201_CREATED)
def build_axis_pack(req: BuildRequest) -> BuildResponse:
    # Ensure registry
    reg = REGISTRY
    if reg is None:
        try:
            enc = get_default_encoder()
            reg = init_registry(encoder_dim=enc._model.get_sentence_embedding_dimension())
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Registry init failed: {e}")

    # Encoder for building
    enc = get_default_encoder()
    encode_fn = enc.encode

    json_paths = req.json_paths or []
    if not json_paths:
        raise HTTPException(status_code=400, detail="json_paths are required for build")

    try:
        builder_kwargs = req.override or {}
        axis_pack = build_advanced_axis_pack(json_paths, encode_fn, **builder_kwargs)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Axis build failed: {e}")

    # Save artifacts
    artifacts_dir = Path(DEFAULT_ARTIFACTS_DIR)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Temporary npz path to compute hash first
    temp_npz = artifacts_dir / "axis_pack:temp_build.npz"
    np.savez_compressed(
        temp_npz,
        Q=axis_pack.Q,
        lambda_=axis_pack.lambda_,
        beta=axis_pack.beta,
        weights=axis_pack.weights,
    )
    npz_bytes = temp_npz.read_bytes()
    pack_hash = sha256(npz_bytes).hexdigest()

    # Determine pack_id
    if req.pack_id:
        pack_id = req.pack_id
    else:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        pack_id = f"ap_{ts}_{pack_hash[:8]}"

    # Final paths
    npz_path = artifacts_dir / f"axis_pack:{pack_id}.npz"
    meta_path = artifacts_dir / f"axis_pack:{pack_id}.meta.json"

    # Move temp to final
    npz_path.write_bytes(npz_bytes)
    try:
        temp_npz.unlink(missing_ok=True)  # type: ignore[arg-type]
    except Exception:
        pass

    names = list(axis_pack.names)
    D, k = int(axis_pack.Q.shape[0]), len(names)

    # Compute json_embeddings_hash from normalized JSON contents
    try:
        normalized_jsons: List[str] = []
        for p in json_paths:
            obj = json.loads(Path(p).read_text(encoding="utf-8"))
            normalized = json.dumps(obj, sort_keys=True, separators=(",", ":"))
            normalized_jsons.append(normalized)
        json_embeddings_hash = sha256("".join(normalized_jsons).encode("utf-8")).hexdigest()
    except Exception:
        json_embeddings_hash = ""

    builder_params = {
        "whitening_method": (axis_pack.meta or {}).get("whitening_method", "cov"),
        "use_lda": bool((axis_pack.meta or {}).get("use_lda", True)),
        "orthogonalize": bool((axis_pack.meta or {}).get("orthogonalize", True)),
        "margin_alpha": float((axis_pack.meta or {}).get("margin_alpha", 0.0)),
        "encoder_model": enc.model_name,
        "encoder_dim": enc._model.get_sentence_embedding_dimension(),
    }

    meta = {
        "schema_version": SCHEMA_VERSION,
        "encoder_model": enc.model_name,
        "encoder_dim": enc._model.get_sentence_embedding_dimension(),
        "names": names,
        "modes": axis_pack.meta.get("modes", {}) if axis_pack.meta else {},
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "builder_version": "advanced-builder-b",
        "pack_hash": pack_hash,
        "json_embeddings_hash": json_embeddings_hash,
        "builder_params": builder_params,
        "notes": axis_pack.meta.get("notes", "") if axis_pack.meta else "",
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # Activate the newly built pack so downstream endpoints use it as active
    try:
        lp = reg.activate(pack_id)
    except FileNotFoundError:
        # Should not happen since we just wrote artifacts; fallback to non-activated return
        lp = {"pack_id": pack_id, "D": D, "k": k, "names": names, "hash": pack_hash}

    return BuildResponse(
        pack_id=lp.get("pack_id", pack_id),
        dim=int(lp.get("D", D)),
        k=int(lp.get("k", k)),
        names=lp.get("names", names),
        pack_hash=lp.get("hash", pack_hash),
    )


class ActivateResponse(BaseModel):
    active: Dict[str, object]


@router.post("/{pack_id}/activate", response_model=ActivateResponse)
def activate_pack(pack_id: str) -> ActivateResponse:
    reg = REGISTRY
    if reg is None:
        try:
            enc = get_default_encoder()
            reg = init_registry(encoder_dim=enc._model.get_sentence_embedding_dimension())
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Registry init failed: {e}")
    try:
        lp = reg.activate(pack_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Pack not found")
    except ValueError as e:
        # Dimension or orthonormality mismatch
        raise HTTPException(status_code=409, detail=str(e))
    return ActivateResponse(active={"pack_id": lp["pack_id"], "dim": lp["D"], "k": lp["k"], "pack_hash": lp["hash"]})


class GetResponse(BaseModel):
    pack_id: str
    dim: int
    k: int
    names: List[str]
    meta: Dict[str, object]
    pack_hash: str


@router.get("/{pack_id}", response_model=GetResponse)
def get_pack(pack_id: str) -> GetResponse:
    reg = REGISTRY
    if reg is None:
        try:
            enc = get_default_encoder()
            reg = init_registry(encoder_dim=enc._model.get_sentence_embedding_dimension())
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Registry init failed: {e}")
    try:
        lp = reg.load(pack_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Pack not found")
    return GetResponse(
        pack_id=lp["pack_id"],
        dim=lp["D"],
        k=lp["k"],
        names=lp["names"],
        meta=lp["meta"],
        pack_hash=lp["hash"],
    )


class ExportResponse(BaseModel):
    pack_id: str
    names: List[str]
    Q: List[List[float]]
    lambda_: List[float]
    beta: List[float]
    weights: List[float]


@router.get("/{pack_id}/export", response_model=ExportResponse)
def export_pack(pack_id: str) -> ExportResponse:
    """Export full axis pack vectors as JSON for inline testing.

    Note: Large payloads; intended for dev/test only.
    """
    reg = REGISTRY
    if reg is None:
        try:
            enc = get_default_encoder()
            reg = init_registry(encoder_dim=enc._model.get_sentence_embedding_dimension())
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Registry init failed: {e}")
    try:
        lp = reg.load(pack_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Pack not found")
    return ExportResponse(
        pack_id=lp["pack_id"],
        names=lp["names"],
        Q=lp["Q"].tolist(),
        lambda_=lp["lambda_"].tolist(),
        beta=lp["beta"].tolist(),
        weights=lp["weights"].tolist(),
    )
