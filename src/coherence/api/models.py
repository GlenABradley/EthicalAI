from __future__ import annotations

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class AxisSeed(BaseModel):
    """Axis seed specification.

    Fields
    - name: axis name
    - positives/negatives: example strings
    """

    name: str
    positives: List[str] = Field(default_factory=list)
    negatives: List[str] = Field(default_factory=list)


class CreateAxisPack(BaseModel):
    """Create an axis pack request.

    - axes: list of `AxisSeed`
    - method: "diffmean" | "cca" | "lda"
    - choquet_capacity: optional capacity mapping (keys like "0,2,3")
    - lambda_/beta/weights: optional per-axis parameters
    """

    axes: List[AxisSeed]
    method: Literal["diffmean", "cca", "lda"] = "diffmean"
    choquet_capacity: Optional[Dict[str, float]] = None
    lambda_: Optional[List[float]] = None
    beta: Optional[List[float]] = None
    weights: Optional[List[float]] = None


class AnalyzeText(BaseModel):
    """Analyze text request payload."""

    axis_pack_id: str
    texts: List[str]
    options: Dict[str, object] = {}


class AxialVectorsModel(BaseModel):
    """Per-axis vectors for a unit (token/span/frame).

    Shapes
    - alpha/u/r: length-k
    - U: scalar
    - C: optional scalar (spans only)
    - t: gating scalar in [0,1]
    - tau: diffusion scale used
    """

    alpha: List[float]
    u: List[float]
    r: List[float]
    U: float
    C: Optional[float] = None
    t: float = 1.0
    tau: float = 0.0


class TokenVectors(BaseModel):
    """Token-level vectors.

    Shapes
    - alpha/u/r: (N,k) lists of lists
    - U: (N,) list of scalars
    """

    alpha: List[List[float]]
    u: List[List[float]]
    r: List[List[float]]
    U: List[float]


class SpanOutput(BaseModel):
    start: int
    end: int
    vectors: AxialVectorsModel


class FrameOutput(BaseModel):
    id: str
    vectors: AxialVectorsModel


class AnalyzeResponse(BaseModel):
    """API response for analysis.

    Contains axes metadata, vectors at multiple granularities, and taus used.
    """

    axes: Dict[str, object]
    tokens: TokenVectors
    spans: List[SpanOutput]
    frames: List[FrameOutput]
    frame_spans: List[SpanOutput]
    tau_used: List[float]
