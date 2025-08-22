# Coherence API — Comprehensive Documentation

This document enumerates all endpoints exposed by the FastAPI app in `src/coherence/api/main.py`, with exact request/response schemas, behavior, error conditions, and example calls.

References:
- App wiring: `src/coherence/api/main.py`
- Routers: `src/coherence/api/routers/`
- Shared models: `src/coherence/api/models.py`

Environment/config:
- `COHERENCE_ARTIFACTS_DIR` (default `artifacts`) stores frames DB and v1 axis pack artifacts.
- `COHERENCE_ENCODER` can override default encoder in some endpoints.
- App initializes an AxisRegistry on startup if encoder loads.

---

## Health

- Path: `/health/ready`
- Method: GET
- Router: `src/coherence/api/routers/health.py`
- Request: none
- Response: JSON
  - `status`: "ok"
  - `encoder_model`: string
  - `encoder_dim`: int | null
  - `active_pack`: { `pack_id`: string, `k`: int, `pack_hash`: string, `schema_version`?: string } | null
  - `frames_db_present`: boolean
  - `frames_db_size_bytes`: int

Errors:
- None expected (best-effort), may fall back to nulls.

Example:
```bash
curl -s http://localhost:8000/health/ready
```

---

## Embedding

- Path: `/embed`
- Method: POST
- Router: `src/coherence/api/routers/embed.py`
- Request model: `EmbedRequest`
  - `texts`: List[str] (required)
  - `encoder_name`?: string
  - `device`?: "cpu" | "cuda" | "mps" | "auto"
  - `normalize_input`?: boolean
- Response model: `EmbedResponse`
  - `embeddings`: List[List[float]]  // shape (n,d)
  - `shape`: [n, d]
  - `model_name`: string
  - `device`: string

Errors:
- 500 Failed to load encoder / Encoding failed

Example:
```bash
curl -sX POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["hello world", "coherence api"],
    "device": "auto",
    "normalize_input": true
  }'
```

---

## Resonance

- Path: `/resonance`
- Method: POST
- Router: `src/coherence/api/routers/resonance.py`
- Request model: `ResonanceRequest`
  - `vectors`?: List[List[float]]  // (n,d), mutually exclusive with texts
  - `texts`?: List[str]
  - `axis_pack`?: Inline AxisPackModel
    - `names`: List[str]
    - `Q`: List[List[float]]  // (d,k)
    - `lambda_`?: List[float] (alias "lambda")
    - `beta`?: List[float]
    - `weights`?: List[float]
    - `mu`?: dict
    - `meta`?: dict
  - `pack_id`?: string
  - `return_intermediate`: boolean (default false)
  - `encoder_name`?: string
  - `device`?: string
  - `normalize_input`?: boolean
- Response model: `ResonanceResponse`
  - `scores`: List[float]
  - `coords`?: List[List[float]]  // when `return_intermediate=true`
  - `utilities`?: List[List[float]]  // when `return_intermediate=true`

Behavior:
- Axis selection priority: pack_id > inline axis_pack > active registry pack.
- Validates `X.shape[1] == pack.Q.shape[0]`.

Errors:
- 400 Provide either vectors or texts; no pack available
- 404 Pack not found
- 422 Embedding dim mismatch; bad registry state
- 500 Encoder/registry/compute failure

Example:
```bash
curl -sX POST http://localhost:8000/resonance \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["A fast green car."],
    "pack_id": "ap_20240620_abcdef12",
    "return_intermediate": true
  }'
```

---

## Pipeline Analyze (Detailed Frames/Spans)

- Path: `/pipeline/analyze`
- Method: POST
- Router: `src/coherence/api/routers/pipeline.py`
- Request model: `AnalyzeRequest`
  - `vectors`?: List[List[float]]  // (n,d)
  - `texts`?: List[str]
  - `axis_pack`?: Inline AxisPackModel (see above)
  - `pack_id`?: string
  - `params`: PipelineParams
    - `max_span_len`: int = 5
    - `max_skip`: int = 2
    - `diffusion_tau`?: float
    - `debug_frames`: bool = false
    - `return_role_projections`: bool = false
    - `role_mode`: "lr" | "agent_patient" = "lr"
    - `detect_evidence`: bool = false
    - `detect_condition`: bool = false
  - `encoder_name`?: string
  - `device`?: string
  - `normalize_input`?: boolean
- Response model: `AnalyzeResponse` (pipeline)
  - `tokens`: Dict[str, Any]
  - `spans`: Dict[str, Any]
  - `frames`: List[Dict[str, Any]]
  - `frame_vectors`: List[List[float]]
  - `frame_role_coords`?: List[Dict[str, Any]]
  - `frame_coords`?: List[Dict[str, Any]]

Behavior:
- Axis selection: `pack_id` | `axis_pack` | active
- Limits total text chars by `api.max_doc_chars` (default 100000).
- Validates `d` vs pack `Q.shape[0]`.

Errors:
- 400 Missing inputs; payload too large; no pack
- 404 Pack not found
- 422 Dim mismatch
- 500 Encoder/pipeline failure

Example:
```bash
curl -sX POST http://localhost:8000/pipeline/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Alice met Bob in Paris yesterday."],
    "pack_id": "ap_20240620_abcdef12",
    "params": {
      "max_span_len": 5,
      "return_role_projections": true
    }
  }'
```

---

## v1 Axes (Artifact-backed)

- Prefix: `/v1/axes`
- Router: `src/coherence/api/routers/v1_axes.py`

### Build Axis Pack

- Path: `/v1/axes/build`
- Method: POST
- Request model: `BuildRequest`
  - `json_paths`?: List[str]  // required non-empty
  - `override`?: Dict  // advanced builder kwargs
  - `pack_id`?: string
- Response model: `BuildResponse`
  - `pack_id`: string
  - `dim`: int
  - `k`: int
  - `names`: List[str]
  - `pack_hash`: string

Behavior:
- Uses `build_advanced_axis_pack` with default encoder.
- Saves:
  - `{ARTIFACTS}/axis_pack:{pack_id}.npz`
  - `{ARTIFACTS}/axis_pack:{pack_id}.meta.json`
- Default `pack_id` = `ap_{UTC_YYYYMMDD_HHMMSS}_{hash8}`
- Activates new pack.

Errors:
- 400 Missing json_paths; build failed
- 500 Registry init failed

Example:
```bash
curl -sX POST http://localhost:8000/v1/axes/build \
  -H "Content-Type: application/json" \
  -d '{
    "json_paths": ["configs/axis_packs/sample.json"],
    "override": {"orthogonalize": true}
  }'
```

### Activate Axis Pack

- Path: `/v1/axes/{pack_id}/activate`
- Method: POST
- Response model: `ActivateResponse`
  - `active`: { `pack_id`: string, `dim`: int, `k`: int, `pack_hash`: string }

Errors:
- 404 Pack not found
- 409 Dimension/orthonormality mismatch
- 500 Registry init failed

Example:
```bash
curl -sX POST http://localhost:8000/v1/axes/ap_20240620_abcdef12/activate
```

### Get Axis Pack Summary

- Path: `/v1/axes/{pack_id}`
- Method: GET
- Response model: `GetResponse`
  - `pack_id`: string
  - `dim`: int
  - `k`: int
  - `names`: List[str]
  - `meta`: Dict[str, Any]
  - `pack_hash`: string

Errors:
- 404 Pack not found
- 500 Registry init failed

Example:
```bash
curl -s http://localhost:8000/v1/axes/ap_20240620_abcdef12
```

### Export Full Axis Pack

- Path: `/v1/axes/{pack_id}/export`
- Method: GET
- Response model: `ExportResponse`
  - `pack_id`: string
  - `names`: List[str]
  - `Q`: List[List[float]]
  - `lambda_`: List[float]
  - `beta`: List[float]
  - `weights`: List[float]

Notes:
- Large payloads intended for dev/test.

Errors:
- 404 Pack not found
- 500 Registry init failed

Example:
```bash
curl -s http://localhost:8000/v1/axes/ap_20240620_abcdef12/export
```

---

## v1 Frames (SQLite-backed)

- Prefix: `/v1/frames`
- Router: `src/coherence/api/routers/v1_frames.py`

### Index Frames

- Path: `/v1/frames/index`
- Method: POST
- Request model: `IndexRequest`
  - `doc_id`: string
  - `pack_id`?: string  // preferred to derive k,d
  - `d`?: int           // required when deriving k from coords without pack
  - `frames`: List[FrameItem]
    - `FrameItem`:
      - `id`: string
      - `predicate`?: List[int]  // [start, end]
      - `roles`?: Dict[str, List[int]]  // { roleName: [start,end] }
      - `coords`?: List[float]    // length k
      - `role_coords`?: Dict[str, List[float]] // per-role length k
      - `meta`?: Dict[str, Any]
  - `frame_vectors`?: List[List[float]]
- Response model: `IndexResponse`
  - `ingested`: int
  - `k`: int

Behavior:
- Resolves k,d in priority:
  - pack_id -> active registry -> derive from coords (needs d provided)
- Validates coords and role_coords lengths and finiteness.

Errors:
- 400 pack_id not found
- 422 cannot resolve k; missing d when deriving; length mismatches; invalid numbers

Example:
```bash
curl -sX POST http://localhost:8000/v1/frames/index \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "doc-001",
    "pack_id": "ap_20240620_abcdef12",
    "frames": [
      {
        "id": "f1",
        "predicate": [0, 2],
        "roles": {"agent": [0,1], "patient": [2,3]},
        "coords": [0.2, 0.8, 0.1],
        "role_coords": {"agent":[0.1,0.6,0.0], "patient":[0.2,0.7,0.1]},
        "meta": {"note":"example"}
      }
    ]
  }'
```

### Search Frames

- Path: `/v1/frames/search`
- Method: GET
- Query params:
  - `axis`: string  // axis name or index
  - `min`: float
  - `max`: float
  - `limit`: int (1..1000, default 100)
  - `pack_id`?: string
- Response model: `SearchResponse`
  - `items`: List<{ `frame_id`: string, `doc_id`: string, `axis_idx`: int, `coord`: float, `predicate`: List[int], `pack_id`: string, `pack_hash`: string }>

Example:
```bash
curl -s "http://localhost:8000/v1/frames/search?axis=Power&min=0.5&max=1.0&limit=50"
```

### Trace Entity

- Path: `/v1/frames/trace/{entity}`
- Method: GET
- Query:
  - `limit`?: int = 100
  - `pack_id`?: string
- Response model: `TraceResponse`
  - `items`: List<{ `frame_id`, `doc_id`, `predicate`: List[int], `pack_id`, `pack_hash` }>

Example:
```bash
curl -s "http://localhost:8000/v1/frames/trace/Alice?limit=50"
```

### Stats

- Path: `/v1/frames/stats`
- Method: GET
- Response:
  - `db_path`: string
  - `db_size_bytes`: int
  - `counts`: { `frames`: int, `frame_axis`: int, `frame_vectors`: int }
  - `last_ingest_ts`: int
  - `active_pack`?: { `pack_id`: string, `k`: int, `schema_version`?: string }

Example:
```bash
curl -s http://localhost:8000/v1/frames/stats
```

---

## Legacy Axes (File-based)

- Prefix: `/axes`
- Router: `src/coherence/api/routers/axes.py`

### List Axis Packs
- Path: `/axes/list`
- Method: GET
- Response: `{ items: [ { id: string, names: List[str], k: int } ] }`

Example:
```bash
curl -s http://localhost:8000/axes/list
```

### Get Axis Pack Summary
- Path: `/axes/{axis_pack_id}`
- Method: GET
- Response: `{ id, names, k, d, meta }`

Example:
```bash
curl -s http://localhost:8000/axes/my_pack_id
```

### Create Axis Pack from Seeds
- Path: `/axes/create`
- Method: POST
- Request model: `CreateAxisPack`
  - `axes`: List<AxisSeed>
    - `AxisSeed`: { `name`: string, `positives`: List[str], `negatives`: List[str] }
  - `method`: "diffmean" | "cca" | "lda" = "diffmean"
  - `choquet_capacity`?: Dict<string, float>
  - `lambda_`?: List[float]
  - `beta`?: List[float]
  - `weights`?: List[float]
- Response model: `CreateAxisPackResponse`
  - `axis_pack_id`: string
  - `k`: int
  - `names`: List[str]

Example:
```bash
curl -sX POST http://localhost:8000/axes/create \
  -H "Content-Type: application/json" \
  -d '{
    "axes": [
      {"name":"Power","positives":["dominate","lead"],"negatives":["submit","obey"]},
      {"name":"Care","positives":["help","nurture"],"negatives":["harm","neglect"]}
    ],
    "method": "diffmean"
  }'
```

---

## Index (Document ANN Indexing)

- Path: `/index`
- Method: POST
- Router: `src/coherence/api/routers/index.py`
- Request model: `IndexRequest` (from `src/coherence/api/models.py`)
  - `axis_pack_id`: string  // pack must exist under `data/axes/`
  - `texts`: List<IndexDoc>
    - `IndexDoc`: { `doc_id`: string, `text`: string }
  - `options`: Dict<string, object> (backend options)
- Response model: `IndexResponse`
  - `indexed`: List<string>  // doc_ids indexed
  - `anns_built`: bool
  - `tau_used`: List<float>

Errors:
- 400 on invalid values/pack issues (propagated from pipeline)

Example:
```bash
curl -sX POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{
    "axis_pack_id": "ap_20240620_abcdef12",
    "texts": [
      {"doc_id": "doc-1", "text": "Alice met Bob in Paris."},
      {"doc_id": "doc-2", "text": "Charlie visited Berlin."}
    ],
    "options": { "tokenizer": "simple", "taus": [0.0] }
  }'
```

---

## Search (ANN + Rerank)

- Path: `/search`
- Method: POST
- Router: `src/coherence/api/routers/search.py`
- Request model: `SearchRequest`
  - `axis_pack_id`: string
  - `query`: `QuerySpec`
    - `type`: "nl" | "weights" | "expr" = "nl"
    - `text`?: string
    - `u`?: List[float]
    - `expr`?: string
  - `filters`: `SearchFilters` (default)
    - `tau`: float = 0.0
    - `minC`: float = 0.0
    - `thresholds`: Dict<string, float> = {}
  - `hyper`: `SearchHyper` (default)
    - `beta`: float = 0.3
    - `alpha`: float = 0.5
    - `gamma`: float = 0.6
  - `top_k`: int = 10
- Response model: `SearchResponse`
  - `hits`: List<`SearchHit`>
    - `doc_id`: string
    - `span`: Dict<string, object>  // {start,end,text}
    - `vectors`: `AxialVectorsModel`
    - `frames`: List<Dict<string, object>>  // related frames
    - `score`: float

Behavior:
- Requires ANN index exists for `axis_pack_id`; else 400.
- For "nl" maps text to u via `u_from_nl`.
- Recall top_k*4 then rerank; filters by `minC` and thresholds by axis.

Errors:
- 400 Missing index; vector length mismatch; etc.

Example (natural language query):
```bash
curl -sX POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "axis_pack_id": "ap_20240620_abcdef12",
    "query": { "type": "nl", "text": "powerful leader helps team" },
    "filters": { "minC": 0.2, "thresholds": {"Power": 0.3} },
    "hyper": { "beta": 0.3, "alpha": 0.5, "gamma": 0.6 },
    "top_k": 5
  }'
```

Example (explicit weights):
```bash
curl -sX POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "axis_pack_id": "ap_20240620_abcdef12",
    "query": { "type": "weights", "u": [0.8, 0.1, 0.4] },
    "top_k": 5
  }'
```

---

## What-If (Stub)

- Path: `/whatif`
- Method: POST
- Router: `src/coherence/api/routers/whatif.py`
- Request model: `WhatIfRequest`
  - `axis_pack_id`: string
  - `doc_id`: string
  - `edits`: List<EditSpec>
    - `EditSpec`:
      - `type`: "remove_text" | "replace_text"
      - `start`: int
      - `end`: int
      - `value`?: string
- Response model: `WhatIfResponse`
  - `deltas`: List<WhatIfDelta> (currently empty)
    - `WhatIfDelta`: `span_id`: string, `dU`: float, `dC`: float, `du`: List[float]

Behavior:
- Currently returns `deltas: []`.

Example:
```bash
curl -sX POST http://localhost:8000/whatif \
  -H "Content-Type: application/json" \
  -d '{
    "axis_pack_id": "ap_20240620_abcdef12",
    "doc_id": "doc-1",
    "edits": [{"type":"remove_text","start":0,"end":5}]
  }'
```

---

## Analyze (Legacy, File-based Pack)

- Path: `/analyze`
- Method: POST
- Router: `src/coherence/api/routers/analyze.py`
- Request model: `AnalyzeText`
  - `axis_pack_id`: string
  - `texts`: List[str]
  - `options`: Dict<string, object> = {}
- Response model: `AnalyzeResponse` (legacy)
  - `axes`: { `id`: string, `names`: List[str], `k`: int }
  - `tokens`: `TokenVectors`
    - `alpha`: List[List[float]]
    - `u`: List[List[float]]
    - `r`: List[List[float]]
    - `U`: List[float]
  - `spans`: List<`SpanOutput`>
  - `frames`: List<`FrameOutput`>
  - `frame_spans`: List<`SpanOutput`>
  - `tau_used`: List[float]

Errors:
- 400 No texts; axis pack not found in `data/axes/{axis_pack_id}.json`

Example:
```bash
curl -sX POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "axis_pack_id": "my_pack_id",
    "texts": ["Alice met Bob in Paris."]
  }'
```

---

# Shared Models (from `src/coherence/api/models.py`)

- `AxisSeed`: { `name`: string, `positives`: List<string>=[], `negatives`: List<string>=[] }
- `CreateAxisPack`:
  - `axes`: List<AxisSeed>
  - `method`: "diffmean" | "cca" | "lda" = "diffmean"
  - `choquet_capacity`?: Dict<string, float>
  - `lambda_`?: List<float>
  - `beta`?: List<float>
  - `weights`?: List<float>

- `AnalyzeText`:
  - `axis_pack_id`: string
  - `texts`: List<string>
  - `options`: Dict<string, object> = {}

- `AxialVectorsModel`:
  - `alpha`: List<float>
  - `u`: List<float>
  - `r`: List<float>
  - `U`: float
  - `C`?: float
  - `t`: float = 1.0
  - `tau`: float = 0.0

- `TokenVectors`:
  - `alpha`: List<List<float>>
  - `u`: List<List<float>>
  - `r`: List<List[float]>
  - `U`: List<float>

- `SpanOutput`: { `start`: int, `end`: int, `vectors`: AxialVectorsModel }
- `FrameOutput`: { `id`: string, `vectors`: AxialVectorsModel }

- `AnalyzeResponse`:
  - `axes`: Dict<string, object>
  - `tokens`: TokenVectors
  - `spans`: List<SpanOutput>
  - `frames`: List<FrameOutput>
  - `frame_spans`: List<SpanOutput>
  - `tau_used`: List<float>

- `IndexDoc`: { `doc_id`: string, `text`: string }
- `IndexRequest`:
  - `axis_pack_id`: string
  - `texts`: List<IndexDoc>
  - `options`: Dict<string, object> = {}
- `IndexResponse`:
  - `indexed`: List<string>
  - `anns_built`: bool
  - `tau_used`: List<float>

- `QuerySpec`: { `type`: "nl"|"weights"|"expr"="nl", `text`?: string, `u`?: List<float>, `expr`?: string }
- `SearchFilters`: { `tau`: float=0.0, `minC`: float=0.0, `thresholds`: Dict<string,float>={} }
- `SearchHyper`: { `beta`: float=0.3, `alpha`: float=0.5, `gamma`: float=0.6 }
- `SearchRequest`:
  - `axis_pack_id`: string
  - `query`: QuerySpec
  - `filters`: SearchFilters = {}
  - `hyper`: SearchHyper = {}
  - `top_k`: int = 10
- `SearchHit`:
  - `doc_id`: string
  - `span`: Dict<string, object>
  - `vectors`: AxialVectorsModel
  - `frames`: List<Dict<string, object>> = []
  - `score`: float
- `SearchResponse`: { `hits`: List<SearchHit> }

- `EditSpec`: { `type`: "remove_text"|"replace_text", `start`: int, `end`: int, `value`?: string }
- `WhatIfRequest`: { `axis_pack_id`: string, `doc_id`: string, `edits`: List<EditSpec> }
- `WhatIfDelta`: { `span_id`: string, `dU`: float, `dC`: float, `du`: List<float> }
- `WhatIfResponse`: { `deltas`: List<WhatIfDelta> }

---

# Notes and Tips

- v1 endpoints (`/v1/axes`, `/v1/frames`) integrate with the AxisRegistry and artifact store. Legacy endpoints (`/axes`, `/analyze`, `/index`, `/search`) use `data/axes/` packs and an ANN store.
- For `resonance` and `pipeline/analyze`, if you pass `texts`, ensure the encoder dimension matches the axis pack’s `Q.shape[0]` or you’ll get 422.
- For `/v1/frames/index`, if you don’t pass `pack_id`, either ensure an active pack exists or include `coords` and `d` so k and d can be derived.

---

# Appendix

- App prefixes are defined in `src/coherence/api/main.py`:
  - `/health`, `/embed`, `/resonance`, `/pipeline`, `/v1/axes`, `/v1/frames`, `/axes`, `/index`, `/search`, `/whatif`, `/analyze`
- Artifacts directory: `COHERENCE_ARTIFACTS_DIR` (default `artifacts/`)
- Frames DB path: `{ARTIFACTS}/frames.sqlite`
