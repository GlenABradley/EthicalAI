from __future__ import annotations

from typing import List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from coherence.axis.pack import AxisPack
import coherence.api.axis_registry as axis_registry
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
    axis_pack: Optional[AxisPackModel] = None
    pack_id: Optional[str] = Field(None, description="Optional axis pack id to load from server registry")
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
    # Resolve AxisPack: pack_id > inline axis_pack
    if req.pack_id:
        reg = getattr(axis_registry, "REGISTRY", None)
        if reg is None:
            try:
                enc0 = get_default_encoder()
                reg = axis_registry.init_registry(encoder_dim=enc0._model.get_sentence_embedding_dimension())
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Registry init failed: {e}")
        try:
            lp = reg.load(req.pack_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Pack not found")
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        pack = AxisPack(
            names=lp["names"],
            Q=lp["Q"],
            lambda_=lp["lambda_"],
            beta=lp["beta"],
            weights=lp["weights"],
            mu={},
            meta=lp["meta"],
        )
    else:
        if req.axis_pack is not None:
            pack = req.axis_pack.to_axis_pack()
        else:
            # Try active
            reg = getattr(axis_registry, "REGISTRY", None)
            if reg is None:
                try:
                    enc0 = get_default_encoder()
                    reg = axis_registry.init_registry(encoder_dim=enc0._model.get_sentence_embedding_dimension())
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"No axis pack provided and no registry available: {e}")
            lp = reg.get_active()
            if lp is None:
                raise HTTPException(status_code=400, detail="No axis pack provided and no active pack")
            pack = AxisPack(
                names=lp["names"],
                Q=lp["Q"],
                lambda_=lp["lambda_"],
                beta=lp["beta"],
                weights=lp["weights"],
                mu={},
                meta=lp["meta"],
            )

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
        # Dimension check before scoring
        if X.shape[1] != pack.Q.shape[0]:
            raise HTTPException(status_code=422, detail=f"Embedding dim {X.shape[1]} != axis pack dim {pack.Q.shape[0]}")
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
