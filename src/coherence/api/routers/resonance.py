from __future__ import annotations

from typing import List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from coherence.axis.pack import AxisPack
from coherence.metrics.resonance import resonance as resonance_fn, utilities as utilities_fn, project as project_fn
from coherence.encoders.text_sbert import get_default_encoder

router = APIRouter()


class AxisPackModel(BaseModel):
    names: List[str]
    Q: List[List[float]]
    lambda_: Optional[List[float]] = Field(None, alias="lambda")
    beta: Optional[List[float]] = None
    weights: Optional[List[float]] = None
    mu: Optional[dict] = None
    meta: Optional[dict] = None

    def to_axis_pack(self) -> AxisPack:
        obj = {
            "names": self.names,
            "Q": self.Q,
            "lambda": self.lambda_ if self.lambda_ is not None else [1.0] * len(self.names),
            "beta": self.beta if self.beta is not None else [0.0] * len(self.names),
            "weights": self.weights if self.weights is not None else [1.0 / max(1, len(self.names))] * len(self.names),
            "mu": self.mu or {},
            "meta": self.meta or {},
        }
        return AxisPack.from_json_obj(obj)


class ResonanceRequest(BaseModel):
    vectors: Optional[List[List[float]]] = Field(None, description="(n,d) vectors; if omitted, texts must be provided")
    texts: Optional[List[str]] = Field(None, description="Texts to auto-embed; used if vectors not provided")
    axis_pack: AxisPackModel
    return_intermediate: bool = False
    encoder_name: Optional[str] = None
    device: Optional[str] = None
    normalize_input: Optional[bool] = None


class ResonanceResponse(BaseModel):
    scores: List[float]
    coords: Optional[List[List[float]]] = None
    utilities: Optional[List[List[float]]] = None


@router.post("/resonance", response_model=ResonanceResponse)
def resonance(req: ResonanceRequest) -> ResonanceResponse:
    pack = req.axis_pack.to_axis_pack()

    X: np.ndarray
    if req.vectors is not None:
        X = np.asarray(req.vectors, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
    elif req.texts is not None:
        try:
            enc = get_default_encoder(
                name=req.encoder_name,
                device=req.device or "auto",
                normalize_input=bool(req.normalize_input) if req.normalize_input is not None else False,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load encoder: {e}")
        try:
            X = enc.encode(req.texts)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Encoding failed: {e}")
    else:
        raise HTTPException(status_code=400, detail="Provide either vectors or texts")

    try:
        scores = resonance_fn(X, pack)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Resonance failed: {e}")

    coords_out = None
    utils_out = None
    if req.return_intermediate:
        try:
            coords = project_fn(X, pack)
            utils = utilities_fn(coords, pack)
            coords_out = np.asarray(coords, dtype=np.float32).tolist()
            utils_out = np.asarray(utils, dtype=np.float32).tolist()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Intermediate computation failed: {e}")

    return ResonanceResponse(
        scores=np.asarray(scores, dtype=np.float32).reshape(-1).tolist(),
        coords=coords_out,
        utilities=utils_out,
    )
