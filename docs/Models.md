# EthicalAI Models Reference

## Overview

This document provides a comprehensive reference for all data models, schemas, and structures used in the EthicalAI system. The models are implemented using Pydantic for type safety and validation.

## Core Models

### Embedding Model

The system uses **SentenceTransformer** for text embedding:

```python
# Model: all-MiniLM-L6-v2
# Dimensions: 384
# Max sequence length: 256 tokens
# Performance: ~2800 sentences/sec on CPU
```

### Model Configuration

```python
class EncoderConfig:
    model_name: str = "all-MiniLM-L6-v2"
    device: str = "cpu"  # or "cuda" if available
    normalize_embeddings: bool = True
    batch_size: int = 32
```

## Axis Models

### AxisSeed

**Purpose**: Defines seed examples for learning an axis direction.

```python
class AxisSeed(BaseModel):
    name: str                    # Axis name (e.g., "truthfulness")
    positives: List[str] = []    # Positive examples
    negatives: List[str] = []    # Negative examples
```

### AxisPack

**Purpose**: Represents a learned set of ethical axes.

```python
class AxisPack:
    id: str                      # Unique identifier (e.g., "ap_1756646151")
    k: int                       # Number of axes (typically 7)
    d: int                       # Embedding dimension (384)
    Q: np.ndarray               # Projection matrix (d × k)
    names: List[str]            # Axis names
    thresholds: Dict[str, float] # Veto thresholds per axis
    metadata: Dict              # Additional metadata
```

### CreateAxisPack

**Purpose**: Request model for creating a new axis pack.

```python
class CreateAxisPack(BaseModel):
    axes: List[AxisSeed]        # Axis definitions
    method: Literal["diffmean", "cca", "lda"] = "diffmean"
    choquet_capacity: Optional[Dict[str, float]] = None
    lambda_: Optional[List[float]] = None  # Regularization
    beta: Optional[List[float]] = None     # Smoothing
    weights: Optional[List[float]] = None  # Axis weights
```

## Analysis Models

### AnalyzeText

**Purpose**: Request model for text analysis.

```python
class AnalyzeText(BaseModel):
    axis_pack_id: Optional[str] = None    # Axis pack to use
    texts: Optional[List[str]] = None     # Multiple texts
    text: Optional[str] = None             # Single text
    options: Dict[str, object] = {}        # Analysis options
```

### AxialVectorsModel

**Purpose**: Represents projected vectors for a text unit.

```python
class AxialVectorsModel(BaseModel):
    alpha: List[float]          # Raw projection scores (k-dimensional)
    u: List[float]              # Unit-normalized scores
    r: List[float]              # Rectified scores (positive only)
    U: float                    # Magnitude (L2 norm)
    C: Optional[float] = None   # Choquet integral (spans only)
    t: float = 1.0              # Gating parameter [0,1]
    tau: float = 0.0            # Diffusion scale used
```

### TokenVectors

**Purpose**: Token-level vector representations.

```python
class TokenVectors(BaseModel):
    alpha: List[List[float]]    # Shape: (N_tokens, k_axes)
    u: List[List[float]]        # Normalized per token
    r: List[List[float]]        # Rectified per token
    U: List[float]              # Magnitudes per token
```

### SpanOutput

**Purpose**: Represents analysis results for a text span.

```python
class SpanOutput(BaseModel):
    start: int                  # Start token index
    end: int                    # End token index
    vectors: AxialVectorsModel  # Span's axial projections
```

### FrameOutput

**Purpose**: Semantic frame representation.

```python
class FrameOutput(BaseModel):
    id: str                     # Frame identifier
    vectors: AxialVectorsModel  # Frame's axial projections
```

### AnalyzeResponse

**Purpose**: Complete analysis response.

```python
class AnalyzeResponse(BaseModel):
    axes: Dict[str, object]     # Axis metadata
    tokens: TokenVectors        # Token-level analysis
    spans: List[SpanOutput]     # Span-level analysis
    frames: List[FrameOutput]   # Frame-level analysis
    frame_spans: List[SpanOutput] # Frame boundary spans
    tau_used: List[float]       # Diffusion parameters used
```

## Ethical Evaluation Models

### EvaluationRequest

**Purpose**: Request ethical evaluation of text.

```python
class EvaluationRequest(BaseModel):
    text: str                   # Text to evaluate
    pack_id: Optional[str]      # Specific axis pack to use
```

### VetoSpan

**Purpose**: Identifies problematic text segments.

```python
class VetoSpan(BaseModel):
    text: str                   # Problematic text segment
    axis: str                   # Axis triggering veto
    score: float                # Violation score
    threshold: float            # Threshold exceeded
    start: int                  # Start position
    end: int                    # End position
```

### DecisionProof

**Purpose**: Explainable ethical decision.

```python
class DecisionProof(BaseModel):
    action: Literal["allow", "refuse"]  # Decision
    veto_spans: List[VetoSpan]         # Violations found
    scores: Dict[str, float]            # All axis scores
    thresholds: Dict[str, float]       # Applied thresholds
    rationale: str                      # Human-readable explanation
    confidence: float                   # Decision confidence
```

### EvaluationResponse

**Purpose**: Complete ethical evaluation result.

```python
class EvaluationResponse(BaseModel):
    decision: DecisionProof     # Ethical decision
    axis_pack_id: str          # Axis pack used
    processing_time: float      # Time in seconds
```

## Frame Management Models

### Frame

**Purpose**: Semantic memory frame.

```python
class Frame(BaseModel):
    id: str                     # Unique identifier
    text: str                   # Original text
    embedding: List[float]      # 384-dim embedding
    projections: Dict[str, float] # Axis projections
    metadata: Dict              # Additional metadata
    created_at: datetime        # Creation timestamp
```

### FrameStats

**Purpose**: Frame database statistics.

```python
class FrameStats(BaseModel):
    total_frames: int           # Total frame count
    axis_pack_id: str          # Active axis pack
    axis_names: List[str]      # Available axes
    avg_scores: Dict[str, float] # Average per axis
    std_scores: Dict[str, float] # Std dev per axis
```

## Search and Indexing Models

### IndexDoc

**Purpose**: Document for indexing.

```python
class IndexDoc(BaseModel):
    doc_id: str                 # Document identifier
    text: str                   # Document content
```

### IndexRequest

**Purpose**: Batch indexing request.

```python
class IndexRequest(BaseModel):
    axis_pack_id: str           # Axis pack for indexing
    texts: List[IndexDoc]       # Documents to index
    options: Dict[str, object] = {}  # Backend options
```

### QuerySpec

**Purpose**: Search query specification.

```python
class QuerySpec(BaseModel):
    type: Literal["nl", "weights", "expr"] = "nl"
    text: Optional[str] = None  # Natural language query
    u: Optional[List[float]] = None  # Direct weight vector
    expr: Optional[str] = None  # Expression query
```

### SearchFilters

**Purpose**: Search filtering options.

```python
class SearchFilters(BaseModel):
    tau: float = 0.0            # Diffusion parameter
    minC: float = 0.0           # Minimum Choquet score
    thresholds: Dict[str, float] = {}  # Per-axis filters
```

### SearchRequest

**Purpose**: Complete search request.

```python
class SearchRequest(BaseModel):
    axis_pack_id: str           # Axis pack to use
    query: QuerySpec            # Query specification
    filters: SearchFilters      # Filtering options
    top_k: int = 10            # Results limit
```

### SearchHit

**Purpose**: Single search result.

```python
class SearchHit(BaseModel):
    doc_id: str                 # Document identifier
    span: Dict[str, object]     # Matching span info
    vectors: AxialVectorsModel  # Span projections
    frames: List[Dict] = []     # Related frames
    score: float                # Relevance score
```

## Constitution Models

### ConstitutionPrinciple

**Purpose**: Ethical principle definition.

```python
class ConstitutionPrinciple(BaseModel):
    id: str                     # Principle identifier
    name: str                   # Principle name
    description: str            # Detailed description
    axis_weights: Dict[str, float]  # Axis importance
    veto_thresholds: Dict[str, float]  # Veto levels
```

### ConstitutionUpdate

**Purpose**: Constitution modification request.

```python
class ConstitutionUpdate(BaseModel):
    principles: List[ConstitutionPrinciple]
    thresholds: Dict[str, float]
    metadata: Dict
```

## Performance Models

### PerformanceMetrics

**Purpose**: System performance tracking.

```python
class PerformanceMetrics(BaseModel):
    encoding_time: float        # Embedding generation time
    projection_time: float      # Axis projection time
    evaluation_time: float      # Total evaluation time
    tokens_processed: int       # Number of tokens
    throughput: float          # Tokens/second
```

## Error Models

### ErrorResponse

**Purpose**: Standardized error response.

```python
class ErrorResponse(BaseModel):
    error: str                  # Error type
    message: str                # Error description
    details: Optional[Dict]     # Additional context
    traceback: Optional[str]    # Debug information
```

## Model Serialization

### NPZ Format

Axis packs are stored in NumPy NPZ format:

```python
# File structure: data/axes/{pack_id}.npz
{
    "Q": ndarray,              # Projection matrix (d × k)
    "names": list,             # Axis names
    "thresholds": dict,        # Veto thresholds
    "metadata": dict           # Additional info
}
```

### JSON Metadata

Accompanying metadata in JSON:

```python
# File structure: data/axes/{pack_id}.json
{
    "id": "ap_1756646151",
    "created_at": "2024-01-01T00:00:00Z",
    "method": "diffmean",
    "k": 7,
    "d": 384,
    "axes": [...],
    "calibration": {...}
}
```

## Model Validation

### Constraints

- Embedding dimension: Must be 384
- Axis count: Typically 7, max 20
- Text length: Max 512 tokens
- Batch size: Max 100 documents
- Score range: [0, 1] after normalization
- Thresholds: Must be positive

### Type Safety

All models use Pydantic for:

- Automatic validation
- Type coercion
- JSON serialization
- OpenAPI schema generation
- Field documentation
