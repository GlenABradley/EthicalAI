from __future__ import annotations

from typing import List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from coherence.encoders.text_sbert import get_default_encoder

router = APIRouter()


class EmbedRequest(BaseModel):
    texts: List[str] = Field(..., description="List of input texts")
    encoder_name: Optional[str] = Field(None, description="HF model id override")
    device: Optional[str] = Field(None, description='"cpu" | "cuda" | "mps" | "auto"')
    normalize_input: Optional[bool] = Field(None, description="Lowercase/strip inputs deterministically")


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    shape: List[int]
    model_name: str
    device: str


@router.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest) -> EmbedResponse:
    # Validate input
    if not req.texts:
        raise HTTPException(status_code=422, detail="texts cannot be empty")
    
    try:
        enc = get_default_encoder(
            name=req.encoder_name,
            device=req.device or "auto",
            normalize_input=bool(req.normalize_input) if req.normalize_input is not None else False,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load encoder: {e}")

    try:
        arr: np.ndarray = enc.encode(req.texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Encoding failed: {e}")

    return EmbedResponse(
        embeddings=arr.astype(np.float32).tolist(),
        shape=[int(x) for x in arr.shape],
        model_name=enc.model_name,
        device=enc.device,
    )
