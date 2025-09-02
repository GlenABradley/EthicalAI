# EthicalAI

**EthicalAI** is a production-ready semantic analysis and ethical evaluation framework that provides empirical, measurable assessments of text content through orthonormalized semantic axes. Built on the Coherence engine for semantic resonance analysis, EthicalAI implements a sophisticated pipeline for multi-level text analysis including token, span, and frame-level semantic projections with configurable ethical dimensions.

## Core Principles

EthicalAI operates on three fundamental principles:

1. **Empirical Foundation**: All evaluations are grounded in measurable semantic vectors derived from SentenceTransformer embeddings (384-dimensional by default), projected onto orthonormal axis bases
2. **Orthogonal Axes**: Semantic and ethical dimensions are mathematically orthonormalized using Gram-Schmidt process to prevent conflation and ensure independent measurement
3. **Transparent Decisions**: Every evaluation produces an auditable proof chain with specific veto spans, per-axis scores, and aggregation rationale

## 🛠 Key Components

### Ethical Axes

EthicalAI evaluates content across seven core dimensions:

1. **Virtue Ethics** - Character and moral excellence
2. **Deontology** - Rule-based ethical reasoning
3. **Consequentialism** - Outcomes and consequences
4. **Autonomy** - Respect for individual agency
5. **Truthfulness** - Factual accuracy and honesty
6. **Non-Aggression** - Prevention of harm
7. **Fairness** - Impartial and just treatment

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ethicalai.git
cd ethicalai

# Install dependencies (use requirements-windows.txt on Windows)
pip install -r requirements.txt

# The SentenceTransformer model (all-MiniLM-L6-v2) will be downloaded automatically on first use
# Or manually download:
python download_model.py
```

### Starting the API Server

```bash
# Development mode with Coherence engine
uvicorn src.coherence.api.main:app --reload --port 8000

# Production mode
uvicorn src.coherence.api.main:app --host 0.0.0.0 --port 8000 --workers 4

# Alternative: Using the simplified API wrapper
uvicorn api.main:app --reload --port 8001
```

### Basic Usage

```python
import requests

# First, create and activate an axis pack
response = requests.post(
    "http://localhost:8000/api/v1/axes/create",
    json={
        "config": {
            "names": ["autonomy", "fairness", "non_aggression"],
            "seed_phrases": [
                ["freedom", "liberty", "independence", "self-determination"],
                ["equality", "justice", "fairness", "impartiality"],
                ["peace", "non-violence", "harmony", "cooperation"]
            ]
        }
    }
)
pack_id = response.json()["pack_id"]

# Activate the pack
requests.post(f"http://localhost:8000/api/v1/axes/{pack_id}/activate")

# Evaluate text content using EthicalAI layer
response = requests.post(
    "http://localhost:8000/ethicalai/v1/eval/text",
    json={
        "text": "This is the content to evaluate",
        "window": 32,
        "stride": 16
    }
)

result = response.json()
print(f"Action: {result['proof']['final']['action']}")
print(f"Veto Spans: {result['spans']}")

# Or use the analyze endpoint for detailed semantic analysis
response = requests.post(
    "http://localhost:8000/api/v1/analyze",
    json={
        "text": "This is the content to analyze",
        "axis_pack_id": pack_id
    }
)
```

## 📚 Documentation

- [API Reference](docs/API.md) - Comprehensive API documentation
- [Architecture](docs/ARCHITECTURE.md) - System design and components
- [Testing Guide](TESTING_GUIDE.md) - Running and writing tests
- [Ethical Framework](docs/ETHICS.md) - Principles and calibration
- [Model Details](docs/Models.md) - Underlying model information

## 🧪 Testing

Run tests with:

```bash
# All tests
pytest

# Specific test file
pytest tests/test_integration.py -v

# With real encoder (recommended)
COHERENCE_TEST_REAL_ENCODER=1 pytest
```

## 🏗 Project Structure

```text
EthicalAI/
├── api/                    # Simplified FastAPI wrapper
│   ├── main.py            # Main app with versioning and health checks
│   └── routes.py          # Router configuration
├── src/
│   ├── coherence/         # Core semantic resonance engine
│   │   ├── api/          # API layer with routers
│   │   │   ├── main.py   # Full Coherence API application
│   │   │   └── routers/  # Endpoint implementations
│   │   │       ├── v1_axes.py    # Axis pack management
│   │   │       ├── analyze.py    # Text analysis
│   │   │       ├── pipeline.py   # Orchestration pipeline
│   │   │       └── [other routers]
│   │   ├── axis/         # Axis pack data structures
│   │   │   └── pack.py   # AxisPack class implementation
│   │   ├── encoders/     # Text encoding utilities
│   │   │   └── text_sbert.py  # SentenceTransformer wrapper
│   │   ├── metrics/      # Resonance metrics
│   │   │   └── resonance.py   # Projection and utility functions
│   │   └── pipeline/     # Analysis orchestrator
│   │       └── orchestrator.py # Multi-level analysis pipeline
│   └── ethicalai/        # Ethical evaluation layer
│       ├── api/          # EthicalAI-specific endpoints
│       │   ├── eval.py   # Ethical evaluation with veto spans
│       │   └── axes.py   # Axis management utilities
│       ├── eval/         # Evaluation logic
│       │   ├── spans.py  # Score projection
│       │   └── minspan.py # Minimal veto span detection
│       └── encoders.py   # Encoding utilities
├── configs/              # Configuration files
│   ├── axis_packs/      # Pre-configured axis pack JSONs
│   ├── app.yaml         # Application configuration
│   └── logging.yaml     # Logging configuration
├── data/
│   ├── axes/            # Generated axis pack artifacts (.json, .npz)
│   └── calibration/     # JSONL calibration datasets
├── tests/               # Comprehensive test suite
│   ├── api/            # API endpoint tests
│   ├── conftest.py     # Test configuration (COHERENCE_TEST_REAL_ENCODER=1)
│   └── test_*.py       # Unit and integration tests
└── docs/               # Documentation
    ├── API.md          # API endpoint reference
    ├── ARCHITECTURE.md # System design documentation
    ├── ETHICS.md       # Ethical framework details
    └── Models.md       # Model specifications
```

## 🤝 Contributing

See [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/name`
3. Commit changes: `git commit -m 'Add feature'`
4. Push: `git push origin feature/name`
5. Open a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE).

## 🙏 Acknowledgments

- Built with FastAPI and PyTorch
- Inspired by ethical AI research

## 📖 Technical Documentation

### Core Components

- **Semantic Axes**: Define and build axis packs
- **Text Analysis**: Embeddings, resonance, and detailed analysis
- **Multi-scale Representations**: Tokens, spans, and frames
- **Frame Store**: SQLite-backed storage and search

### API Endpoints

- `/health/ready` - System status
- `/embed` - Text to vector encoding
- `/resonance` - Axis pack analysis
- `/pipeline/analyze` - Detailed text analysis
- `/v1/axes/*` - Axis pack management
- `/v1/frames/*` - Frame operations

### Key Files

- `src/coherence/api/main.py` - FastAPI application
- `src/coherence/api/routers/` - API endpoints
- `src/coherence/api/models.py` - Data models
- Legacy: `/axes`, `/index`, `/search`, `/analyze`, `/whatif`.

See `docs/API.md` for exhaustive endpoint specs and examples.

## Configuration

- `COHERENCE_ARTIFACTS_DIR` — where artifacts (axis packs, frames DB) are stored. Default: `artifacts/`.
- `COHERENCE_ENCODER` — optional encoder override for components that accept it.
- App config file: `configs/app.yaml` (log level, limits, etc.)
- Logging config: `configs/logging.yaml`

## Artifacts and Data

- Axis packs (v1): `{ARTIFACTS}/axis_pack:<pack_id>.npz` + `.meta.json`
- Frames DB: `{ARTIFACTS}/frames.sqlite`
- Legacy axis packs (file-based): `data/axes/<axis_pack_id>.json`

## Scripts

- Seed example axes:

  ```bash
  python scripts/seed_axes.py --config configs/axis_packs/sample.json
  ```

- Export OpenAPI schema (writes to `docs/openapi.json`):

  ```bash
  python scripts/export_openapi.py --out docs/openapi.json
  ```

- Generate client collections via Make (also regenerates OpenAPI):

  ```bash
  # default BASE_URL is http://localhost:8080
  make postman  # writes docs/postman_collection.json
  make thunder  # writes docs/thunder-collection_Coherence_API.json
  make collections  # openapi + both collections

  # override base URL
  make collections BASE_URL=http://127.0.0.1:8080
  ```

- PowerShell helper (Windows):

  ```powershell
  # Regenerate OpenAPI + Postman + Thunder into docs/
  .\tools\export-openapi.ps1 -BaseUrl "http://localhost:8080" -OutDir "docs"
  ```

## Testing

- Run tests:

  ```bash
  pytest -q
  ```

- Selected integration tests exercise API endpoints under `tests/`.

## Development Tips

- Use `uvicorn` with `--reload` for rapid iteration.
- Check `/health/ready` to verify active axis pack and frames DB presence.
- Prefer v1 endpoints (`/v1/axes`, `/v1/frames`) for production flows.

## Performance Characteristics

- **Encoding**: ~1-5ms per sentence with cached SentenceTransformer model
- **Projection**: <1ms for projecting 100 tokens onto 10-axis pack
- **Pipeline**: ~10-50ms for complete analysis (tokens + spans + frames)
- **Memory**: ~500MB for model + minimal overhead per axis pack
- **Throughput**: Handles 100+ requests/second on modern hardware

## Advanced Features

### Axis Pack Building

- Orthonormalization via Gram-Schmidt process
- Support for eigenvalue scaling and bias terms
- Override parameters for fine-tuning
- Multiple aggregation strategies (weighted mean/sum)

### Calibration System

- JSONL dataset support for empirical score distribution
- Automatic threshold determination based on percentiles
- Per-axis calibration with statistical validation

### Frame Detection

- Predicate identification with configurable thresholds
- Role assignment (agent/patient or left/right)
- Evidence and condition detection
- Frame-level semantic projections

## Contributing

We welcome contributions! Key areas:

- Additional axis pack configurations
- Enhanced frame detection algorithms
- Performance optimizations
- Extended language support

## Dependencies

**Core**:

- `fastapi>=0.104.0`: API framework
- `sentence-transformers>=2.2.2`: Text embeddings
- `numpy>=1.24.0`: Vector operations
- `pydantic>=2.0.0`: Data validation

**Development**:

- `pytest>=7.4.0`: Testing framework
- `ruff>=0.1.0`: Linting and formatting
- `mypy>=1.0.0`: Type checking

See `requirements.txt` for complete dependency list.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Built on **sentence-transformers** for robust text embeddings
- **FastAPI** for production-ready API infrastructure with automatic OpenAPI docs
- **NumPy** for efficient vector operations and linear algebra
- Inspired by research in semantic spaces and ethical AI alignment
