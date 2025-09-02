# EthicalAI Architecture

## System Overview

EthicalAI is a comprehensive framework for semantic analysis and ethical evaluation of text content using multi-dimensional axis-based representation. The system combines dense embeddings from sentence transformers with ethical reasoning through axial projection and veto span detection to provide nuanced ethical assessments and decision support.

## Core Components

### 1. Coherence Engine (`src/coherence/`)

The foundation layer responsible for semantic analysis and axial processing:

- **Encoder** (`encoder/`): Uses SentenceTransformer models (default: `all-MiniLM-L6-v2`, 384 dimensions) for dense embeddings
- **Axis Module** (`axis/`): Core axial projection system with `AxisPack`, `AxisArtifact`, and registry management
- **Pipeline** (`pipeline/`): Comprehensive text analysis pipeline with tokenization, embedding, and axial projection
- **Agent** (`agent/`): Autonomous processing capabilities and decision-making logic
- **Memory** (`memory/`): Frame-based semantic memory with SQLite backend
- **API** (`api/`): FastAPI application with modular routers for all endpoints

### 2. EthicalAI Layer (`src/ethicalai/`)

Implements ethical reasoning and evaluation capabilities:

- **Axes** (`axes/`): Manages ethical axis packs with calibration and evaluation logic
- **Constitution** (`constitution/`): Defines ethical principles and veto thresholds
- **Evaluator**: Performs ethical scoring, veto span detection, and decision proof generation
- **API** (`api/`): EthicalAI-specific endpoints for evaluation, active pack management, and interaction analysis

## Data Flow

### 1. Initialization

- **Encoder Loading**: SentenceTransformer model loaded once at startup
- **Axis Registry**: Initialize and load available axis packs from artifacts
- **Frame Database**: Initialize SQLite database for semantic memory
- **Configuration**: Load environment variables and configuration files

### 2. Text Processing Pipeline

```text
Input Text → Tokenization → Embedding → Axial Projection → Aggregation
```

- Text tokenized into overlapping spans (configurable window/stride)
- Each span embedded using sentence transformer
- Embeddings projected onto axis vectors using Q matrix
- Scores aggregated across spans with various aggregation strategies

### 3. Ethical Evaluation

- **Alignment**: Map embeddings to ethical dimensions via axis pack
- **Projection**: Calculate α (raw scores), u (normalized), r (rectified)
- **Veto Detection**: Identify spans violating ethical thresholds
- **Decision Proof**: Generate explanations with supporting evidence

### 4. Response Generation

- **Structured Output**: Scores, veto spans, decision proofs
- **Frame Integration**: Semantic frames provide context
- **Interaction Analysis**: Detailed span-level ethical analysis

## API Architecture

### Endpoint Organization

The API is organized into modular routers mounted on the main FastAPI app:

```python
# Main app structure (api/main.py)
app.include_router(health_router, prefix="/health")
app.include_router(embed_router, prefix="/embed")
app.include_router(resonance_router, prefix="/resonance")
app.include_router(pipeline_router, prefix="/pipeline")
app.include_router(v1_axes_router, prefix="/v1/axes")
app.include_router(v1_frames_router, prefix="/v1/frames")
app.include_router(ethicalai_eval_router, prefix="/v1/eval")
app.include_router(constitution_router, prefix="/v1/constitution")
app.include_router(interaction_router, prefix="/v1/interaction")
# Legacy endpoints
app.include_router(axes_router, prefix="/axes")
app.include_router(index_router, prefix="")
app.include_router(search_router, prefix="")
app.include_router(whatif_router, prefix="")
app.include_router(analyze_router, prefix="")
```

### Key Endpoints

- **Health**: `/health/ready`, `/health/live`
- **Embeddings**: `/embed/text`, `/embed/batch`
- **Axes**: `/v1/axes/build`, `/v1/axes/{pack_id}/activate`, `/v1/axes/list`
- **Evaluation**: `/v1/eval/text`, `/v1/eval/active`
- **Frames**: `/v1/frames/index`, `/v1/frames/search`, `/v1/frames/stats`
- **Constitution**: `/v1/constitution`, `/v1/constitution/update`

## Data Storage

### Artifacts Directory Structure

```text
artifacts/
├── axis_packs/
│   ├── ap_<timestamp>_<hash>.npz     # Axis pack data (Q matrix, metadata)
│   └── ap_<timestamp>_<hash>.json    # Axis pack metadata
├── frames.sqlite                      # Frame database
└── registry.json                      # Axis pack registry
```

### Axis Pack Format

**NPZ File Contents**:
- `Q`: Projection matrix (d × k) for axis vectors
- `names`: Axis names array
- `metadata`: Additional configuration

**Metadata JSON**:

- `id`: Unique pack identifier
- `k`: Number of axes
- `d`: Embedding dimension (384)
- `schema_version`: Format version
- `created_at`: Timestamp
- `calibration`: Optional calibration data

### Frame Database Schema

```sql
CREATE TABLE frames (
    id TEXT PRIMARY KEY,
    doc_id TEXT,
    text TEXT,
    pack_id TEXT,
    coords TEXT,  -- JSON array of coordinates
    metadata TEXT -- JSON metadata
);
```

## Configuration

### Environment Variables

```bash
# Core Configuration
COHERENCE_ENCODER_MODEL=all-MiniLM-L6-v2  # Sentence transformer model
COHERENCE_ENCODER_DIM=384                 # Embedding dimensions
COHERENCE_ARTIFACTS_DIR=artifacts/        # Storage directory
COHERENCE_API_CORS_ORIGINS=["*"]          # CORS configuration

# Testing
COHERENCE_TEST_REAL_ENCODER=1             # Use real encoder in tests
COHERENCE_USE_TEST_ENCODER=false          # Use test encoder (for CI)

# Logging
COHERENCE_LOG_LEVEL=INFO                  # Log verbosity
COHERENCE_LOG_FORMAT=json                 # Log format (json/text)
```

### Configuration Files

```yaml
# configs/app.yaml
api:
  host: 0.0.0.0
  port: 8000
  workers: 4
  
encoder:
  model: all-MiniLM-L6-v2
  cache_dir: ~/.cache/sentence_transformers
  
processing:
  batch_size: 32
  max_length: 512
```

**Axis Pack Configurations** (`configs/axis_packs/`):

- `deontology.json`: Duty-based ethical axes
- `consequentialism.json`: Outcome-based ethical axes
- `virtue_ethics.json`: Character-based ethical axes
- `intent_bad_inclusive.json`: Intent analysis axes

## Performance Optimization

### Model Management

- **Singleton Pattern**: Encoder loaded once and shared across requests
- **Precomputation**: Axis packs precomputed and cached in memory
- **Batch Processing**: Vectorized operations for multiple texts/spans

### Processing Strategies

- **Sliding Window**: Configurable window size and stride for long texts
- **Aggregation Options**: max, mean, weighted aggregation strategies
- **Parallel Processing**: Thread pool for independent computations

### Memory Efficiency

- **Streaming**: Process large texts in chunks
- **Garbage Collection**: Explicit cleanup of large tensors
- **Resource Limits**: Configurable limits on batch sizes and text length

### Caching

- **Embedding Cache**: LRU cache for frequently processed texts
- **Axis Registry**: In-memory cache of loaded axis packs
- **Frame Index**: SQLite indices for fast frame retrieval

## Security & Reliability

### Input Validation

- **Schema Validation**: Pydantic models for all request/response types
- **Size Limits**: Maximum text length and batch size enforcement
- **Content Filtering**: Sanitization of user inputs

### Error Handling

```python
# Structured error responses
{
    "detail": "Error description",
    "error_code": "AXIS_NOT_FOUND",
    "context": {"pack_id": "invalid_id"}
}
```

### Rate Limiting

- Request throttling per IP/API key
- Concurrent request limits
- Resource consumption monitoring

## Monitoring & Observability

### Health Checks

```json
// GET /health/ready
{
    "status": "ready",
    "encoder_loaded": true,
    "axis_packs_available": 5,
    "frame_db_connected": true
}
```

### Structured Logging

```json
{
    "timestamp": "2024-01-01T12:00:00Z",
    "level": "INFO",
    "message": "Request processed",
    "request_id": "uuid",
    "endpoint": "/v1/eval/text",
    "duration_ms": 125,
    "axis_pack": "ap_1756646151"
}
```

### Metrics

- Request latency histograms
- Embedding cache hit rates
- Axis pack usage statistics
- Memory and CPU utilization

## Testing Architecture

### Test Organization

```text
tests/
├── api/                    # API endpoint tests
├── memory/                 # Frame storage tests
├── test_*.py              # Component tests
└── conftest.py            # Shared fixtures
```

### Testing Principles

- **Real Encoder**: All tests use actual SentenceTransformer (no mocks)
- **Fixture-based**: Shared `api_client_real_encoder` fixture
- **Comprehensive**: Unit, integration, and end-to-end coverage
- **Performance**: Benchmark tests for critical paths

### Key Test Categories

1. **Unit Tests**: Component isolation with real dependencies
2. **Integration Tests**: API endpoint verification
3. **Frame Tests**: SQLite operations and indexing
4. **Axis Tests**: Pack creation, loading, and projection
5. **Performance Tests**: Latency and throughput benchmarks

## Deployment

### System Requirements

- **Python**: 3.8+ (3.10+ recommended)
- **Memory**: Minimum 2GB RAM (4GB+ recommended)
- **Storage**: 1GB for models + artifact storage
- **CPU**: Multi-core for parallel processing

### Development Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Download models
python download_model.py

# Run development server
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Deployment

```bash
# Using Gunicorn with Uvicorn workers
gunicorn api.main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 120 \
    --access-logfile - \
    --error-logfile -
```

### Docker Deployment

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["gunicorn", "api.main:app", "-c", "gunicorn.conf.py"]
```

## Scaling Architecture

### Horizontal Scaling

- **Stateless API**: Each request independent, enabling load balancing
- **Shared Storage**: Artifacts on NFS/S3 for multi-instance access
- **Database**: PostgreSQL/MySQL for distributed frame storage

### Vertical Scaling

- **GPU Acceleration**: CUDA support for faster embeddings
- **Memory Optimization**: Larger caches for better performance
- **Batch Processing**: Increased batch sizes with more RAM

### Distributed Architecture

```text
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│Load Balancer│────▶│ API Server 1│────▶│Shared Store │
└─────────────┘     └─────────────┘     └─────────────┘
                    ┌─────────────┐            │
                    │ API Server 2│────────────┘
                    └─────────────┘            │
                    ┌─────────────┐            │
                    │ API Server N│────────────┘
                    └─────────────┘
```

## Troubleshooting Guide

### Common Issues

#### 1. Model Loading Errors

```bash
# Error: Can't download model
Solution: Check internet connection and proxy settings
export HTTP_PROXY=http://proxy.example.com:8080

# Error: Insufficient memory
Solution: Use smaller model or increase system RAM
export COHERENCE_ENCODER_MODEL=all-MiniLM-L6-v2  # Smaller model
```

#### 2. Axis Pack Issues

```python
# Error: Axis pack not found
Solution: Verify pack exists in artifacts/
ls artifacts/axis_packs/

# Error: Dimension mismatch
Solution: Ensure encoder dimension matches pack dimension (384)
```

#### 3. Performance Problems

```bash
# Slow response times
Solutions:
- Enable embedding cache
- Reduce batch size
- Use GPU acceleration if available
- Profile with: python -m cProfile api.main
```

#### 4. Database Errors

```sql
-- Frame database locked
Solution: Check for concurrent writes, use WAL mode
PRAGMA journal_mode=WAL;
```

### Debug Mode

```bash
# Enable debug logging
export COHERENCE_LOG_LEVEL=DEBUG
export PYTHONUNBUFFERED=1

# Run with verbose output
python -m uvicorn api.main:app --log-level debug
