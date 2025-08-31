from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np, json, pathlib, os
from ..types import AxisPack, Axis
from ..axes.build import build_axis_pack
from ..axes.calibrate import pick_thresholds
from ..encoders import get_encoder

router = APIRouter(prefix="/v1/axes", tags=["axes"])
ART_DIR = pathlib.Path(os.getenv("COHERENCE_ARTIFACTS_DIR", "artifacts")); ART_DIR.mkdir(exist_ok=True)
ACTIVE: Dict[str, Optional[AxisPack]] = {"pack": None}

class BuildRequest(BaseModel):
    names: List[str]
    # Option A: phrases per axis (preferred)
    seed_phrases: Optional[List[List[str]]] = None
    # Option B: raw vectors (must match encoder dim)
    seed_vectors: Optional[List[List[float]]] = None
    meta: Dict = {}
    # Optional labeled scores for thresholds: {axis: [(score,label),...]}
    thresholds_data: Optional[Dict[str, List[List[float]]]] = None
    fpr_max: float = 0.05

@router.post("/build")
def build(req: BuildRequest):
    enc = get_encoder()
    dim = getattr(enc, "dim", 384)
    # Build seed vectors
    if req.seed_vectors:
        vecs = [np.array(v, dtype=np.float32) for v in req.seed_vectors]
    elif req.seed_phrases:
        vecs = []
        for phrases in req.seed_phrases:
            if not phrases:
                raise HTTPException(400, "seed_phrases entries must be non-empty")
            # mean pool over phrases → each phrase mean-pooled over tokens
            phrase_vecs = []
            for p in phrases:
                X = enc.encode_text(p)  # [T,D]
                phrase_vecs.append(X.mean(axis=0))
            v = np.stack(phrase_vecs, axis=0).mean(axis=0)
            vecs.append(v.astype(np.float32))
    else:
        raise HTTPException(400, "Provide seed_phrases or seed_vectors")

    pack = build_axis_pack(vecs, req.names, req.meta)

    # Optional threshold calibration
    if req.thresholds_data:
        scores = {k: [(float(s), int(l)) for s,l in v] for k,v in req.thresholds_data.items()}
        pack = pick_thresholds(pack, scores, fpr_max=req.fpr_max)

    ACTIVE["pack"] = pack

    # Persist
    np.savez_compressed(ART_DIR / f"axis_pack:{pack.id}.npz", **{a.name:a.vector for a in pack.axes})
    (ART_DIR / f"axis_pack:{pack.id}.meta.json").write_text(json.dumps({
        "meta": pack.meta,
        "names": [a.name for a in pack.axes],
        "dim": pack.dim,
        "thresholds": {a.name:a.threshold for a in pack.axes}
    }, indent=2))

    return {"pack_id": pack.id, "axes":[a.name for a in pack.axes], "dim": pack.dim}

@router.post("/activate")
def activate(pack_id: str):
    meta_path = ART_DIR / f"axis_pack:{pack_id}.meta.json"
    npz_path  = ART_DIR / f"axis_pack:{pack_id}.npz"
    if not (meta_path.exists() and npz_path.exists()):
        raise HTTPException(404, "Axis pack not found")
    arrs = np.load(npz_path)
    meta = json.loads(meta_path.read_text())
    thresholds = meta.get("thresholds", {})
    axes = [
        Axis(name=k, vector=arrs[k], threshold=float(thresholds.get(k, 0.0)), provenance=meta.get("meta", {}))
        for k in arrs.files
    ]
    ACTIVE["pack"] = AxisPack(
        id=pack_id,
        axes=axes,
        dim=axes[0].vector.shape[0],
        meta=meta.get("meta",{})
    )
    return {"ok": True}

@router.get("/active")
def active():
    p = ACTIVE["pack"]
    return {"pack_id": getattr(p, "id", None), "axes": [a.name for a in getattr(p, "axes", [])], "dim": getattr(p, "dim", None)}
