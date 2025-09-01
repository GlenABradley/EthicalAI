from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from coherence.axis.pack import AxisPack
from coherence.cfg.loader import load_app_config
import coherence.api.axis_registry as axis_registry
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
    # Debug/inspection flags (router-only additions; optional and additive)
    debug_frames: bool = False
    return_role_projections: bool = False
    # F1 additive flags (default off)
    role_mode: str = "lr"  # "lr" | "agent_patient"
    detect_evidence: bool = False
    detect_condition: bool = False


class AnalyzeRequest(BaseModel):
    vectors: Optional[List[List[float]]] = Field(None, description="(n,d) token vectors; if omitted, texts must be provided")
    texts: Optional[List[str]] = Field(None, description="Texts to auto-embed; used if vectors not provided")
    axis_pack: Optional[AxisPackModel] = None
    pack_id: Optional[str] = Field(None, description="Optional axis pack id to load from server registry")
    params: PipelineParams = PipelineParams()
    encoder_name: Optional[str] = None
    device: Optional[str] = None
    normalize_input: Optional[bool] = None


class AnalyzeResponse(BaseModel):
    tokens: Dict[str, Any]
    spans: Dict[str, Any]
    frames: List[Dict[str, Any]]
    frame_vectors: List[List[float]]
    # Additive, optional fields present only when requested
    frame_role_coords: Optional[List[Dict[str, Any]]] = None
    frame_coords: Optional[List[Dict[str, Any]]] = None


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
    # Check payload size first before other validations
    if req.texts is not None:
        try:
            cfg = load_app_config()
            max_chars = int(cfg.get("api", {}).get("max_doc_chars", 100000))
        except Exception:
            max_chars = 100000
        total_chars = sum(len(t) for t in req.texts)
        if total_chars > max_chars:
            raise HTTPException(status_code=413, detail="max_doc_chars exceeded")
    
    # Resolve AxisPack: pack_id > inline axis_pack > active
    if req.pack_id:
        if axis_registry.REGISTRY is None:
            raise HTTPException(status_code=500, detail="Registry not initialized")
        try:
            lp = axis_registry.REGISTRY.load(req.pack_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Pack not found")
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        pack = AxisPack(names=lp["names"], Q=lp["Q"], lambda_=lp["lambda_"], beta=lp["beta"], weights=lp["weights"], mu={}, meta=lp["meta"])
    elif req.axis_pack is not None:
        pack = req.axis_pack.to_axis_pack()
    else:
        if axis_registry.REGISTRY is None:
            raise HTTPException(status_code=400, detail="No axis pack provided and no registry available")
        lp = axis_registry.REGISTRY.get_active()
        if lp is None:
            raise HTTPException(status_code=400, detail="No axis pack provided and no active pack")
        pack = AxisPack(names=lp["names"], Q=lp["Q"], lambda_=lp["lambda_"], beta=lp["beta"], weights=lp["weights"], mu={}, meta=lp["meta"]) 

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
        token_texts = list(req.texts)
    else:
        raise HTTPException(status_code=400, detail="Provide either vectors or texts")

    try:
        op = OrchestratorParams(
            max_span_len=int(req.params.max_span_len),
            max_skip=int(req.params.max_skip),
            diffusion_tau=req.params.diffusion_tau,
            debug_frames=bool(req.params.debug_frames),
            return_role_projections=bool(req.params.return_role_projections),
            role_mode=str(req.params.role_mode),
            detect_evidence=bool(req.params.detect_evidence),
            detect_condition=bool(req.params.detect_condition),
        )
        if X.shape[1] != pack.Q.shape[0]:
            raise HTTPException(status_code=422, detail=f"Embedding dim {X.shape[1]} != axis pack dim {pack.Q.shape[0]}")
        out = run_pipeline_from_vectors(X, pack, op, token_texts=locals().get("token_texts"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}")

    frames = [ _frame_to_dict(f) for f in out["frames"] ]
    tokens_dict = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in out["tokens"].items()}
    spans_dict = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in out["spans"].items()}
    frame_vecs = out["frame_vectors"]
    if hasattr(frame_vecs, "tolist"):
        frame_vecs = frame_vecs.tolist()

    resp_kwargs: Dict[str, Any] = dict(
        tokens=tokens_dict,
        spans=spans_dict,
        frames=frames,
        frame_vectors=frame_vecs,
    )
    # Add optional fields if present
    if req.params.return_role_projections:
        if "frame_role_coords" in out:
            resp_kwargs["frame_role_coords"] = out["frame_role_coords"]
        if "frame_coords" in out:
            resp_kwargs["frame_coords"] = out["frame_coords"]
    return AnalyzeResponse(**resp_kwargs)
