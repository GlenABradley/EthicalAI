# Coherence API Models Reference

Exact Pydantic schemas used by the API. Source: `src/coherence/api/models.py`.

---

## Axis Seeds and Pack Creation

- AxisSeed
  - name: string
  - positives: List<string> = []
  - negatives: List<string> = []

- CreateAxisPack
  - axes: List<AxisSeed>
  - method: "diffmean" | "cca" | "lda" = "diffmean"
  - choquet_capacity?: Dict<string, float>
  - lambda_?: List<float>
  - beta?: List<float>
  - weights?: List<float>

---

## Analyze (Legacy)

- AnalyzeText
  - axis_pack_id: string
  - texts: List<string>
  - options: Dict<string, object> = {}

- AxialVectorsModel
  - alpha: List<float>
  - u: List<float>
  - r: List<float>
  - U: float
  - C?: float
  - t: float = 1.0
  - tau: float = 0.0

- TokenVectors
  - alpha: List<List<float>>
  - u: List<List<float>>
  - r: List<List<float>>
  - U: List<float>

- SpanOutput
  - start: int
  - end: int
  - vectors: AxialVectorsModel

- FrameOutput
  - id: string
  - vectors: AxialVectorsModel

- AnalyzeResponse
  - axes: Dict<string, object>
  - tokens: TokenVectors
  - spans: List<SpanOutput>
  - frames: List<FrameOutput>
  - frame_spans: List<SpanOutput>
  - tau_used: List<float>

---

## Index/Search (Legacy)

- IndexDoc
  - doc_id: string
  - text: string

- IndexRequest
  - axis_pack_id: string
  - texts: List<IndexDoc>
  - options: Dict<string, object> = {}

- IndexResponse
  - indexed: List<string>
  - anns_built: bool
  - tau_used: List<float>

- QuerySpec
  - type: "nl" | "weights" | "expr" = "nl"
  - text?: string
  - u?: List<float>
  - expr?: string

- SearchFilters
  - tau: float = 0.0
  - minC: float = 0.0
  - thresholds: Dict<string, float> = {}

- SearchHyper
  - beta: float = 0.3
  - alpha: float = 0.5
  - gamma: float = 0.6

- SearchRequest
  - axis_pack_id: string
  - query: QuerySpec
  - filters: SearchFilters = {}
  - hyper: SearchHyper = {}
  - top_k: int = 10

- SearchHit
  - doc_id: string
  - span: Dict<string, object>
  - vectors: AxialVectorsModel
  - frames: List<Dict<string, object>> = []
  - score: float

- SearchResponse
  - hits: List<SearchHit>

---

## What-If (Stub)

- EditSpec
  - type: "remove_text" | "replace_text"
  - start: int
  - end: int
  - value?: string

- WhatIfRequest
  - axis_pack_id: string
  - doc_id: string
  - edits: List<EditSpec>

- WhatIfDelta
  - span_id: string
  - dU: float
  - dC: float
  - du: List<float>

- WhatIfResponse
  - deltas: List<WhatIfDelta>
