# EthicalAI API Documentation

Comprehensive documentation for the EthicalAI REST API, which provides semantic resonance analysis, ethical evaluation with veto span detection, axis pack management, and frame-based semantic analysis. The API is built on FastAPI and integrates the Coherence semantic resonance engine with EthicalAI's ethical evaluation layer.

## Architecture Overview

The API consists of two main layers:

1. **Coherence Layer**: Core semantic resonance engine providing embeddings, resonance analysis, pipeline processing, and frame management
2. **EthicalAI Layer**: Ethical evaluation endpoints with veto span detection based on orthonormalized semantic axes

## Base URL

All API endpoints are relative to the base URL of your deployment:

- Local development: `http://localhost:8080`  
- Production: Configure via `COHERENCE_API_HOST` and `COHERENCE_API_PORT`

## Authentication

Currently no authentication required for local development. Production deployments should implement API key authentication via middleware.

## Response Format

Standard JSON response format:

- **Success (2xx)**: JSON with requested data
- **Client Error (4xx)**: JSON with `detail` field describing the error
- **Server Error (5xx)**: JSON with `detail` field and stack trace in debug mode

### Architecture Notes

## Environment Variables

- `COHERENCE_ENCODER_MODEL`: Embedding model (default: `all-MiniLM-L6-v2`)
- `COHERENCE_ENCODER_DIM`: Embedding dimensions (default: 384)
- `COHERENCE_ARTIFACTS_DIR`: Storage for axis packs (default: `artifacts/`)
- `COHERENCE_API_CORS_ORIGINS`: CORS allowed origins (default: `["*"]`)
- `COHERENCE_USE_TEST_ENCODER`: Use test encoder (default: `false`)

---

## Core Endpoints

### Health Check

**GET** `/health/ready`

Returns the current status of the API and its components.

**Response:**

```json
{
  "status": "ok",
  "encoder_model": "all-MiniLM-L6-v2",
  "encoder_dim": 384,
  "active_pack": {
    "pack_id": "ap_20241231_abc123",
    "k": 5,
    "pack_hash": "a1b2c3d4",
    "schema_version": "1.0.0"
  },
  "frames_db_present": true,
  "frames_db_size_bytes": 1048576,
  "version": "1.0.0",
  "uptime_seconds": 12345
}
```

**Status Codes:**

- `200`: Service is healthy
- `503`: One or more components are not ready

### Root Info

**GET** `/`

Returns basic API information.

**Response:**

```json
{
  "name": "Coherence API",
  "version": "1.0.0",
  "docs": "/docs",
  "openapi": "/openapi.json"
}
```

---

## Embedding Endpoints

### Generate Embeddings

**POST** `/embed`

Converts text into dense vector representations using SentenceTransformer models.

**Request Body:**

```json
{
  "texts": ["Text to embed"],
  "encoder_name": "all-MiniLM-L6-v2",
  "device": "auto",
  "normalize_input": true
}
```

**Parameters:**

- `texts` (required): Array of text strings to embed
- `encoder_name` (optional): Model name (default: configured encoder)
- `device` (optional): Device for computation ("cpu", "cuda", "mps", "auto")
- `normalize_input` (optional): Normalize text before encoding (default: true)

**Response:**

```json
{
  "embeddings": [[0.1, 0.2, ...]],
  "shape": [1, 384],
  "model_name": "all-MiniLM-L6-v2",
  "device": "cpu"
}
```

**Status Codes:**

- `200`: Success
- `400`: Invalid input
- `422`: Validation error
- `500`: Encoder failure

---

## Semantic Resonance Analysis

### Analyze Resonance

**POST** `/resonance`

Analyzes text or vectors against semantic axes to compute resonance scores.

**Request Body:**

```json
{
  "texts": ["Sample text to analyze"],
  "axis_pack": {
    "names": ["autonomy", "fairness"],
    "Q": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
    "lambda_": [1.0, 0.8],
    "beta": [0.0, 0.0, 0.0],
    "weights": [0.33, 0.33, 0.34],
    "meta": {
      "description": "Sample ethical dimensions"
    }
  },
  "return_intermediate": true
}
```

**Parameters:**

- `texts` (required): List of text strings to analyze
- `vectors` (optional, array of arrays of numbers): Pre-computed embedding vectors (mutually exclusive with `texts`)
- `axis_pack` (optional, object): Inline axis pack definition
  - `names` (required, array of strings): Names of the ethical dimensions
  - `Q` (required, matrix of numbers): Orthonormal basis vectors for the axes
  - `lambda_` (optional, array of numbers): Eigenvalues for each axis
  - `beta` (optional, array of numbers): Bias terms for each axis
  - `weights` (optional, array of numbers): Weights for combining axes
  - `mu` (optional, object): Mean vector for centering
  - `meta` (optional, object): Additional metadata about the axis pack
- `pack_id` (optional, string): ID of a pre-defined axis pack to use
- `return_intermediate` (optional, boolean): Whether to include intermediate calculations in the response

**Response:**

```json
{
  "results": [{
    "text": "Text to analyze",
    "embedding": [0.1, 0.2, ...],
    "aligned_embedding": [0.15, 0.25, ...],
    "projections": {
      "autonomy": 0.78,
      "fairness": 0.65
    }
  }],
  "axis_pack_id": "ap_1756646151",
  "encoder_model": "all-MiniLM-L6-v2"
}
```

**Response Fields:**

- `resonance_scores`: Array of score arrays, one per input text
- `texts`: The input texts that were analyzed
- `axis_names`: Names of the ethical dimensions
- `intermediate` (if `return_intermediate=true`): Detailed calculation results
  - `embeddings`: Raw embedding vectors
  - `projections`: Projections onto each axis
  - `scores`: Raw scores before normalization
- `model_name`: Name of the embedding model used
- `axis_pack_id`: ID of the axis pack used for analysis

**Status Codes:**

- `200`: Success
- `400`: Invalid input
- `422`: Validation error
- `500`: Processing error

### Pipeline Analysis

**POST** `/pipeline`

Executes a full semantic analysis pipeline on text, including encoding, alignment, and projection.

**Request Body:**

```json
{
  "texts": ["Text to analyze"],
  "axis_pack_id": "ap_1756646151",
  "return_embeddings": true
}
```

---

## Advanced Pipeline Analysis

### Analyze with Frames and Spans

**POST** `/pipeline/analyze`

Performs detailed semantic analysis including span detection and frame extraction.

**Request Body:**

```json
{
  "texts": ["Alice met Bob in Paris"],
  "pack_id": "ap_1756646151",
  "params": {
    "max_span_len": 5,
    "max_skip": 2,
    "return_role_projections": true
  }
}
```

**Parameters:**

- `texts` or `vectors`: Input text or pre-computed embeddings
- `pack_id` or `axis_pack`: Axis pack specification
- `params`:
  - `max_span_len`: Maximum span length (default: 5)
  - `max_skip`: Maximum word skip (default: 2)
  - `diffusion_tau`: Diffusion parameter
  - `debug_frames`: Include debug info (default: false)
  - `return_role_projections`: Return role projections (default: false)
  - `role_mode`: "lr" or "agent_patient" (default: "lr")
  - `detect_evidence`: Detect evidence spans (default: false)
  - `detect_condition`: Detect conditional spans (default: false)

**Response:**

```json
{
  "tokens": {
    "text": "Alice met Bob in Paris",
    "tokens": ["Alice", "met", "Bob", "in", "Paris"]
  },
  "spans": {
    "detected": [
      {"start": 0, "end": 2, "text": "Alice met", "score": 0.85}
    ]
  },
  "frames": [
    {
      "type": "meeting",
      "participants": ["Alice", "Bob"],
      "location": "Paris"
    }
  ],
  "frame_vectors": [[0.1, 0.2, ...]],
  "frame_role_coords": {
    "agent": [0.3, 0.4, ...],
    "patient": [0.5, 0.6, ...]
  }
}
```

---

## Axis Pack Management

### Build Axis Pack

**POST** `/v1/axes/build`

Builds a new axis pack from JSON configuration files.

**Request Body:**

```json
{
  "json_paths": ["configs/axis_packs/sample.json"],
  "override": {"orthogonalize": true},
  "pack_id": "custom_pack_id"
}
```

**Response:**

```json
{
  "pack_id": "ap_1756646151",
  "dim": 384,
  "k": 5,
  "names": ["autonomy", "fairness", "non_aggression", "transparency", "beneficence"],
  "pack_hash": "a1b2c3d4"
}
```

### Activate Axis Pack

**POST** `/v1/axes/{pack_id}/activate`

Activates a specific axis pack for use in resonance analysis.

**Response:**

```json
{
  "active": {
    "pack_id": "ap_1756646151",
    "dim": 384,
    "k": 5
  },
  "pack_hash": "a1b2c3d4"
}
```

### Get Axis Pack Info

**GET** `/v1/axes/{pack_id}`

Retrieves information about a specific axis pack.

**Response:**

```json
{
  "pack_id": "ap_1756646151",
  "dim": 384,
  "k": 5,
  "names": ["autonomy", "fairness", "non_aggression", "transparency", "beneficence"],
  "meta": {
    "description": "Ethical evaluation axes",
    "created": "2024-12-31T00:00:00Z"
  },
  "pack_hash": "a1b2c3d4"
}
```

### List Axis Packs

**GET** `/v1/axes`

Lists all available axis packs.

**Response:**

```json
{
  "packs": [
    {
      "pack_id": "ap_1756646151",
      "dim": 384,
      "k": 5,
      "names": ["autonomy", "fairness", "non_aggression", "transparency", "beneficence"],
      "active": true
    }
  ],
  "total": 1
}
```

### Export Axis Pack

**GET** `/v1/axes/{pack_id}/export`

Exports the full axis pack data including vectors.

**Response:**

```json
{
  "pack_id": "ap_1756646151",
  "Q": [[0.1, 0.2, ...], ...],
  "lambda_": [1.0, 0.9, ...],
  "beta": [0.0, 0.0, ...],
  "mu": [0.1, 0.2, ...],
  "names": ["autonomy", "fairness", ...],
  "meta": {...}
}
```

---

## EthicalAI Evaluation Endpoints

### Evaluate Text

**POST** `/v1/eval/text`

Performs ethical evaluation on text using the active axis pack with veto span detection.

**Request Body:**

```json
{
  "text": "AI systems should respect human autonomy and dignity",
  "threshold": 0.7,
  "detect_veto_spans": true
}
```

**Parameters:**

- `text` (required): Text to evaluate
- `threshold` (optional): Veto threshold for concerning content (default: 0.7)
- `detect_veto_spans` (optional): Enable veto span detection (default: true)

**Response:**

```json
{
  "decision": "approved",
  "scores": {
    "autonomy": 0.92,
    "fairness": 0.78,
    "non_aggression": 0.85,
    "transparency": 0.71,
    "beneficence": 0.82
  },
  "average_score": 0.816,
  "veto_spans": [],
  "decision_proof": {
    "axis_pack_id": "ap_1756646151",
    "veto_threshold": 0.7,
    "veto_rationale": "No concerning spans detected",
    "passed_axes": ["autonomy", "fairness", "non_aggression", "beneficence"],
    "borderline_axes": ["transparency"]
  }
}
```

**Status Codes:**

- `200`: Evaluation complete
- `400`: Invalid input
- `422`: Validation error
- `500`: Evaluation failure

### Active Axis Pack

**GET** `/v1/eval/active`

Returns the currently active axis pack for ethical evaluation.

**Response:**

```json
{
  "pack_id": "ap_1756646151",
  "names": ["autonomy", "fairness", "non_aggression", "transparency", "beneficence"],
  "k": 5,
  "dim": 384,
  "pack_hash": "a1b2c3d4"
}
```

### Set Active Axis Pack

**POST** `/v1/eval/active/{pack_id}`

Sets the active axis pack for ethical evaluation.

**Response:**

```json
{
  "message": "Axis pack activated",
  "pack_id": "ap_1756646151"
}
```

---

## Frame Management

### Frame Indexing (SQLite-backed)

**POST** `/v1/frames/index`

Indexes semantic frames into SQLite database for fast retrieval.

**Request Body:**

```json
{
  "doc_id": "doc-001",
  "pack_id": "ap_1756646151",
  "frames": [
    {
      "id": "frame-1",
      "predicate": [0, 2],
      "roles": {
        "agent": [0, 1],
        "patient": [2, 3]
      },
      "coords": [0.2, 0.8, 0.1, 0.5, 0.3],
      "role_coords": {
        "agent": [0.1, 0.6, 0.0, 0.4, 0.2],
        "patient": [0.2, 0.7, 0.1, 0.6, 0.4]
      },
      "meta": {"confidence": 0.95}
    }
  ]
}
```

**Parameters:**

- `doc_id` (required): Document identifier
- `pack_id` (optional): Axis pack ID to derive k and d
- `d` (optional): Dimension when deriving k from coords without pack
- `frames` (required): List of frame items:
  - `id`: Frame identifier
  - `predicate`: Token indices [start, end]
  - `roles`: Role name to token indices mapping
  - `coords`: Frame coordinates (length k)
  - `role_coords`: Per-role coordinates
  - `meta`: Additional metadata

**Response:**

```json
{
  "ingested": 1,
  "k": 5
}
```

**Status Codes:**

- `200`: Frames indexed successfully
- `400`: Pack ID not found
- `422`: Cannot resolve k; missing d when deriving; length mismatches

### Search Frames

**GET** `/v1/frames/search`

Searches for frames based on axis coordinates.

**Query Parameters:**

- `axis`: Axis name or index
- `min`: Minimum coordinate value
- `max`: Maximum coordinate value
- `limit`: Result limit (1-1000, default: 100)
- `pack_id` (optional): Specific axis pack ID

**Response:**

```json
{
  "items": [
    {
      "frame_id": "frame-1",
      "doc_id": "doc-001",
      "axis_idx": 0,
      "coord": 0.85,
      "predicate": [0, 2],
      "pack_id": "ap_1756646151",
      "pack_hash": "a1b2c3d4"
    }
  ]
}
```

### Trace Entity

**GET** `/v1/frames/trace/{entity}`

Traces frames containing a specific entity.

**Query Parameters:**

- `limit` (optional): Result limit (default: 100)
- `pack_id` (optional): Specific axis pack ID

**Response:**

```json
{
  "items": [
    {
      "frame_id": "frame-1",
      "doc_id": "doc-001",
      "predicate": [0, 2],
      "pack_id": "ap_1756646151",
      "pack_hash": "a1b2c3d4"
    }
  ]
}
```

### Frame Statistics

**GET** `/v1/frames/stats`

Returns database statistics and information.

**Response:**

```json
{
  "db_path": "/artifacts/frames.db",
  "db_size_bytes": 1048576,
  "counts": {
    "frames": 1250,
    "frame_axis": 6250,
    "frame_vectors": 1250
  },
  "last_ingest_ts": 1704089400,
  "active_pack": {
    "pack_id": "ap_1756646151",
    "k": 5,
    "schema_version": "1.0.0"
  }
}
```

---

## Constitution Endpoints

### Get Constitution

**GET** `/v1/constitution`

Returns the current ethical constitution configuration.

**Response:**

```json
{
  "principles": [
    {
      "name": "autonomy",
      "weight": 1.0,
      "description": "Respect for human autonomy and self-determination"
    },
    {
      "name": "fairness",
      "weight": 1.0,
      "description": "Ensure fair and unbiased treatment"
    }
  ],
  "veto_thresholds": {
    "default": 0.7,
    "critical": 0.9
  },
  "version": "1.0.0"
}
```

### Update Constitution

**POST** `/v1/constitution`

Updates the ethical constitution configuration.

**Request Body:**

```json
{
  "principles": [...],
  "veto_thresholds": {...}
}
```

---

## Interaction Endpoints

### Analyze Span Interaction

**POST** `/v1/interaction/span`

Analyzes ethical implications of text spans.

**Request Body:**

```json
{
  "text": "AI should maximize human benefit",
  "spans": [
    {"start": 0, "end": 2, "label": "agent"},
    {"start": 10, "end": 23, "label": "goal"}
  ]
}
```

**Response:**

```json
{
  "analysis": {
    "ethical_scores": {...},
    "span_interactions": [...],
    "recommendations": [...]
  }
}
```

---

## Legacy Endpoints

### List Axis Packs (Legacy)

**GET** `/axes/list`

Lists available axis packs (legacy file-based system).

**Response:**

```json
{
  "items": [
    {
      "id": "pack_id",
      "names": ["axis1", "axis2"],
      "k": 5
    }
  ]
}
```

### Get Axis Pack Summary (Legacy)

**GET** `/axes/{axis_pack_id}`

Returns axis pack details.

**Response:**

```json
{
  "id": "pack_id",
  "names": ["axis1", "axis2"],
  "k": 5,
  "d": 384,
  "meta": {...}
}
```

### Create Axis Pack from Seeds (Legacy)

**POST** `/axes/create`

Creates an axis pack from seed words.

**Request Body:**

```json
{
  "axes": [
    {
      "name": "autonomy",
      "positives": ["freedom", "choice", "independence"],
      "negatives": ["coercion", "control", "restriction"]
    }
  ],
  "method": "diffmean",
  "choquet_capacity": {},
  "lambda_": [1.0, 0.9, 0.8],
  "beta": [0.0, 0.0, 0.0],
  "weights": [1.0, 1.0, 1.0]
}
```

**Response:**

```json
{
  "axis_pack_id": "ap_generated",
  "k": 3,
  "names": ["autonomy"]
}
```

---

## Document Indexing and Search

### Index Documents

**POST** `/index`

Indexes documents for semantic search with ANN (Approximate Nearest Neighbor) indexing.

**Request Body:**

```json
{
  "axis_pack_id": "ap_1756646151",
  "texts": [
    {
      "doc_id": "doc-1",
      "text": "Alice met Bob in Paris."
    },
    {
      "doc_id": "doc-2",
      "text": "Charlie visited Berlin."
    }
  ],
  "options": {
    "tokenizer": "simple",
    "taus": [0.0]
  }
}
```

**Response:**

```json
{
  "indexed": ["doc-1", "doc-2"],
  "anns_built": true,
  "tau_used": [0.0]
}
```

### Search Documents

**POST** `/search`

Performs semantic search with ANN and reranking.

**Request Body:**

```json
{
  "axis_pack_id": "ap_1756646151",
  "query": {
    "type": "nl",
    "text": "powerful leader helps team"
  },
  "filters": {
    "tau": 0.0,
    "minC": 0.2,
    "thresholds": {"Power": 0.3}
  },
  "hyper": {
    "beta": 0.3,
    "alpha": 0.5,
    "gamma": 0.6
  },
  "top_k": 5
}
```

**Query Types:**

- `nl`: Natural language query
- `weights`: Explicit weight vector
- `expr`: Expression-based query

**Response:**

```json
{
  "hits": [
    {
      "doc_id": "doc-1",
      "span": {
        "start": 0,
        "end": 10,
        "text": "Alice met Bob"
      },
      "vectors": {...},
      "frames": [...],
      "score": 0.85
    }
  ]
}
```

---

## What-If Analysis

### What-If Scenario

**POST** `/whatif`

Analyzes the impact of text edits (stub implementation).

**Request Body:**

```json
{
  "axis_pack_id": "ap_1756646151",
  "doc_id": "doc-1",
  "edits": [
    {
      "type": "remove_text",
      "start": 0,
      "end": 5
    }
  ]
}
```

**Response:**

```json
{
  "deltas": []
}
```

**Note:** Currently returns empty deltas array.

---

## Analysis Endpoints

### Analyze Text (Legacy)

**POST** `/analyze`

Performs legacy analysis on text using file-based axis packs.

**Request Body:**

```json
{
  "axis_pack_id": "my_pack_id",
  "texts": ["Alice met Bob in Paris."],
  "options": {}
}
```

**Response:**

```json
{
  "axes": {
    "id": "my_pack_id",
    "names": ["axis1", "axis2"],
    "k": 3
  },
  "tokens": {
    "alpha": [[0.1, 0.2, ...], ...],
    "u": [[0.3, 0.4, ...], ...],
    "r": [[0.5, 0.6, ...], ...],
    "U": [0.7, 0.8, ...]
  },
  "spans": [...],
  "frames": [...],
  "frame_spans": [...],
  "tau_used": [0.0]
}
```

---

## Shared Models

Data models used across the API (from `src/coherence/api/models.py`):

### AxisSeed

```json
{
  "name": "string",
  "positives": ["string"],
  "negatives": ["string"]
}
```

### CreateAxisPack

```json
{
  "axes": [AxisSeed],
  "method": "diffmean",
  "choquet_capacity": {},
  "lambda_": [1.0],
  "beta": [0.0],
  "weights": [1.0]
}
```

### AnalyzeText

```json
{
  "axis_pack_id": "string",
  "texts": ["string"],
  "options": {}
}
```

### AxialVectorsModel

```json
{
  "alpha": [0.0],
  "u": [0.0],
  "r": [0.0],
  "U": 0.0,
  "C": 0.0,
  "t": 1.0,
  "tau": 0.0
}
```

### TokenVectors

```json
{
  "alpha": [[0.0]],
  "u": [[0.0]],
  "r": [[0.0]],
  "U": [0.0]
}
```

### SpanOutput & FrameOutput

```json
// SpanOutput
{
  "start": 0,
  "end": 10,
  "vectors": AxialVectorsModel
}

// FrameOutput
{
  "id": "string",
  "vectors": AxialVectorsModel
}
```

### AnalyzeResponse

```json
{
  "axes": {},
  "tokens": TokenVectors,
  "spans": [SpanOutput],
  "frames": [FrameOutput],
  "frame_spans": [SpanOutput],
  "tau_used": [0.0]
}
```

### Index Models

```json
// IndexDoc
{
  "doc_id": "string",
  "text": "string"
}

// IndexRequest
{
  "axis_pack_id": "string",
  "texts": [IndexDoc],
  "options": {}
}

// IndexResponse
{
  "indexed": ["string"],
  "anns_built": true,
  "tau_used": [0.0]
}
```

### Search Models

```json
// QuerySpec
{
  "type": "nl",
  "text": "string",
  "u": [0.0],
  "expr": "string"
}

// SearchFilters
{
  "tau": 0.0,
  "minC": 0.0,
  "thresholds": {}
}

// SearchHyper
{
  "beta": 0.3,
  "alpha": 0.5,
  "gamma": 0.6
}

// SearchRequest
{
  "axis_pack_id": "string",
  "query": QuerySpec,
  "filters": SearchFilters,
  "hyper": SearchHyper,
  "top_k": 10
}

// SearchHit
{
  "doc_id": "string",
  "span": {},
  "vectors": AxialVectorsModel,
  "frames": [],
  "score": 0.0
}

// SearchResponse
{
  "hits": [SearchHit]
}
```

### What-If Models

```json
// EditSpec
{
  "type": "remove_text",
  "start": 0,
  "end": 5,
  "value": "string"
}

// WhatIfRequest
{
  "axis_pack_id": "string",
  "doc_id": "string",
  "edits": [EditSpec]
}

// WhatIfResponse
{
  "deltas": []
}
```

---

## Notes and Tips

- v1 endpoints (`/v1/axes`, `/v1/frames`) integrate with the AxisRegistry and artifact store. Legacy endpoints (`/axes`, `/analyze`, `/index`, `/search`) use `data/axes/` packs and an ANN store.
- For `resonance` and `pipeline/analyze`, if you pass `texts`, ensure the encoder dimension matches the axis pack’s `Q.shape[0]` or you’ll get 422.
- For `/v1/frames/index`, if you don’t pass `pack_id`, either ensure an active pack exists or include `coords` and `d` so k and d can be derived.

---

## Appendix

- App prefixes are defined in `src/coherence/api/main.py`:
  - `/health`, `/embed`, `/resonance`, `/pipeline`, `/v1/axes`, `/v1/frames`, `/axes`, `/index`, `/search`, `/whatif`, `/analyze`
- Frames DB path: `{ARTIFACTS}/frames.sqlite`
