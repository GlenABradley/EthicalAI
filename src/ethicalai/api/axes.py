from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict
import numpy as np, json, pathlib
from coherence.encoders.text_sbert import get_default_encoder
from ..types import AxisPack, Axis
from ..axes.build import build_axis_pack
from ..axes.calibrate import pick_thresholds

router = APIRouter(prefix="/v1/axes", tags=["axes"])
ART_DIR = pathlib.Path("artifacts"); ART_DIR.mkdir(exist_ok=True)
ACTIVE = {"pack": None}

class BuildRequest(BaseModel):
    names: List[str]
    meta: Dict = {}

@router.post("/build")
def build(req: BuildRequest):
    enc = get_default_encoder()
    seed_vectors = [enc.encode_text(name) for name in req.names]
    pack = build_axis_pack(seed_vectors, req.names, req.meta)
    ACTIVE["pack"] = pack
    # persist
    np.savez_compressed(ART_DIR / f"axis_pack:{pack.id}.npz", **{a.name:a.vector for a in pack.axes})
    (ART_DIR / f"axis_pack:{pack.id}.meta.json").write_text(json.dumps({"meta":pack.meta}))
    return {"pack_id": pack.id, "axes":[a.name for a in pack.axes]}

@router.post("/activate")
def activate(pack_id: str):
    meta_path = ART_DIR / f"axis_pack:{pack_id}.meta.json"
    npz_path  = ART_DIR / f"axis_pack:{pack_id}.npz"
    arrs = np.load(npz_path)
    axes = [Axis(name=k, vector=arrs[k], threshold=0.0, provenance={}) for k in arrs.files]
    ACTIVE["pack"] = AxisPack(id=pack_id, axes=axes, dim=axes[0].vector.shape[0], meta=json.loads(meta_path.read_text()).get("meta",{}))
    return {"ok": True}

@router.get("/active")
def active():
    p = ACTIVE["pack"]
    return {"pack_id": getattr(p, "id", None), "axes": [a.name for a in getattr(p, "axes", [])]}
