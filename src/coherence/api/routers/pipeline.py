from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from coherence.axis.pack import AxisPack
from coherence.encoders.text_sbert import get_default_encoder
from coherence.pipeline.orchestrator import OrchestratorParams, run_pipeline_from_vectors

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


class PipelineParams(BaseModel):
    max_span_len: int = 5
    max_skip: int = 2
    diffusion_tau: Optional[float] = None


class AnalyzeRequest(BaseModel):
    vectors: Optional[List[List[float]]] = Field(None, description="(n,d) token vectors; if omitted, texts must be provided")
    texts: Optional[List[str]] = Field(None, description="Texts to auto-embed; used if vectors not provided")
    axis_pack: AxisPackModel
    params: PipelineParams = PipelineParams()
    encoder_name: Optional[str] = None
    device: Optional[str] = None
    normalize_input: Optional[bool] = None


class AnalyzeResponse(BaseModel):
    tokens: Dict[str, Any]
    spans: Dict[str, Any]
    frames: List[Dict[str, Any]]
    frame_vectors: List[List[float]]


def _frame_to_dict(f) -> Dict[str, Any]:
    return {
        "id": f.id,
        "predicate": list(f.predicate),
        "roles": {k: list(v) for k, v in f.roles.items()},
        "score": float(f.score),
        "meta": f.meta or {},
    }


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    pack = req.axis_pack.to_axis_pack()

    X: np.ndarray
    if req.vectors is not None:
        X = np.asarray(req.vectors, dtype=np.float32)
        if X.ndim != 2:
            raise HTTPException(status_code=400, detail="vectors must be 2D (n,d)")
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
            # Each text becomes one token for now; callers should pass token vectors for finer granularity
            X = enc.encode(req.texts)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Encoding failed: {e}")
    else:
        raise HTTPException(status_code=400, detail="Provide either vectors or texts")

    try:
        params = OrchestratorParams(
            max_span_len=req.params.max_span_len,
            max_skip=req.params.max_skip,
            diffusion_tau=req.params.diffusion_tau,
        )
        out = run_pipeline_from_vectors(X, pack, params)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}")

    frames = [ _frame_to_dict(f) for f in out["frames"] ]
    tokens_dict = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in out["tokens"].items()}
    spans_dict = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in out["spans"].items()}
    frame_vecs = out["frame_vectors"]
    if hasattr(frame_vecs, "tolist"):
        frame_vecs = frame_vecs.tolist()

    return AnalyzeResponse(
        tokens=tokens_dict,
        spans=spans_dict,
        frames=frames,
        frame_vectors=frame_vecs,
    )
