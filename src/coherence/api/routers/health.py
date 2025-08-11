from __future__ import annotations

from fastapi import APIRouter
from typing import Dict, Any
import os
from pathlib import Path

import coherence.api.axis_registry as axis_registry
from coherence.cfg.loader import load_app_config
from coherence.encoders.text_sbert import get_default_encoder

router = APIRouter()


@router.get("/ready")
def ready() -> Dict[str, Any]:
    """Readiness probe with encoder and active pack info."""
    encoder_model = os.getenv("COHERENCE_ENCODER")
    if not encoder_model:
        cfg = load_app_config()
        encoder_model = cfg.get("encoder", {}).get("name", "sentence-transformers/all-mpnet-base-v2")

    # Ensure registry exists so it can restore active from artifacts
    reg = getattr(axis_registry, "REGISTRY", None)
    if reg is None:
        try:
            enc = get_default_encoder()
            reg = axis_registry.init_registry(encoder_dim=enc._model.get_sentence_embedding_dimension())
        except Exception:
            reg = None

    encoder_dim = None
    active = None
    if reg is not None:
        try:
            encoder_dim = int(reg.encoder_dim)
        except Exception:
            encoder_dim = None
        lp = reg.get_active()
        if lp is not None:
            active = {
                "pack_id": lp["pack_id"],
                "k": lp["k"],
                "pack_hash": lp["hash"],
                "schema_version": lp["meta"].get("schema_version"),
            }

    resp = {
        "status": "ok",
        "encoder_model": encoder_model,
        "encoder_dim": encoder_dim,
        "active_pack": active,
    }
    # Frames DB info
    artifacts = Path(os.environ.get("COHERENCE_ARTIFACTS_DIR", "artifacts"))
    frames_db = artifacts / "frames.sqlite"
    resp["frames_db_present"] = frames_db.exists()
    resp["frames_db_size_bytes"] = frames_db.stat().st_size if frames_db.exists() else 0
    return resp
