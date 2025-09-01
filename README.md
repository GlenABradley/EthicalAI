# EthicalAI

EthicalAI is a robust framework for evaluating and moderating AI-generated content through ethical principles. It analyzes text across multiple ethical dimensions to ensure responsible AI behavior.

## 🌟 Core Principles

1. **Empirical Truth**: Grounding decisions in verifiable facts
2. **Human Autonomy**: Respecting and preserving human agency
3. **Non-Aggression**: Preventing harmful or violent content
4. **Fairness**: Ensuring equitable treatment across diverse groups

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

### Technical Features

- **Real-time Analysis**: Low-latency content evaluation
- **Threshold-based Moderation**: Configurable sensitivity per dimension
- **Explainable Decisions**: Transparent reasoning for moderation
- **Custom Calibration**: Fine-tune with domain-specific data

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip (Python package manager)
- Git

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/GlenABradley/EthicalAI.git
   cd EthicalAI
   ```

2. Set up the environment:

   ```bash
   # Create and activate virtual environment
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # Unix/macOS
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. Download required models:

   ```bash
   python download_model.py
   ```

### Running the API Server

Start the FastAPI development server:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8080
```

Access the API at `http://localhost:8080` with interactive docs at `http://localhost:8080/docs`.

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
.
├── api/                  # API server
│   ├── main.py          # FastAPI app
│   └── routers/         # API routes
├── configs/             # Configs
│   └── axis_packs/      # Ethical dimensions
├── data/                # Data
│   ├── axes/            # Axis packs
│   └── calibration/     # Calibration data
├── docs/                # Documentation
├── scripts/             # Utilities
├── src/                 # Source
│   ├── coherence/       # Core logic
│   └── ethicalai/       # EthicalAI impl
├── tests/               # Tests
│   ├── api/             # API tests
│   └── integration/     # Integration tests
└── ui/                  # Web UI
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
