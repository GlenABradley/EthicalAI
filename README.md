# EthicalAI

EthicalAI is a comprehensive framework for analyzing and evaluating text through the lens of ethical principles and frameworks. It provides tools for embedding, analyzing, and visualizing text based on customizable ethical dimensions.

## 🌟 Key Features

- **Multi-dimensional Ethical Analysis**: Evaluate text against multiple ethical frameworks including consequentialism, deontology, and virtue ethics.
- **Custom Axis Packs**: Define and apply custom ethical dimensions for specialized analysis.
- **Real-time Processing**: Fast, efficient processing suitable for both batch and real-time applications.
- **Comprehensive API**: RESTful API with detailed documentation for easy integration.
- **Visualization Tools**: Built-in visualization for analyzing ethical dimensions in text.
- **Extensible Architecture**: Modular design allowing for custom components and extensions.

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

2. Create and activate a virtual environment:
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate
   
   # Unix/macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the API Server

Start the FastAPI development server:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8080
```

The API will be available at `http://localhost:8080` with interactive documentation at `http://localhost:8080/docs`.

## 🛠 Core Components

### API Endpoints

- **Text Analysis**: `/analyze` - Analyze text against ethical dimensions
- **Embeddings**: `/embed` - Generate text embeddings
- **Health Check**: `/health/ready` - System status and readiness
- **Axis Management**: `/v1/axes/*` - Manage ethical dimensions

### Key Modules

- **API Layer**: FastAPI-based REST interface
- **Processing Pipeline**: Text processing and ethical analysis
- **Vector Store**: Efficient storage and retrieval of embeddings
- **Visualization**: Tools for ethical analysis visualization

## 📚 Documentation

For detailed documentation, please refer to:

- [API Reference](docs/API.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Testing Guide](TESTING_GUIDE.md)
- [Ethical Guidelines](docs/ETHICS.md)
- [Model Details](docs/Models.md)

## 🧪 Running Tests

Run the full test suite:

```bash
pytest
```

For a specific test file:

```bash
pytest tests/test_end_to_end.py -v
```

## 🏗 Project Structure

```text
.
├── api/                  # API server implementation
│   ├── main.py          # FastAPI application
│   └── routers/         # API route definitions
├── configs/             # Configuration files
│   └── axis_packs/      # Predefined ethical dimension configurations
├── data/                # Data storage
│   ├── axes/            # Axis pack data
│   └── calibration/     # Calibration data
├── docs/                # Documentation
├── scripts/             # Utility scripts
├── src/                 # Source code
│   ├── coherence/       # Core analysis logic
│   └── ethicalai/       # EthicalAI-specific implementations
├── tests/               # Test suite
│   ├── api/             # API tests
│   └── integration/     # Integration tests
└── ui/                  # Web interface
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on how to contribute to this project.

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Open a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with FastAPI, PyTorch, and other amazing open-source projects
- Inspired by research in ethical AI and natural language processing

## Repository Layout

See the repository tree printed by the build steps. The Python source lives under `src/coherence/`.

## Handling Rules

- No pretend-builds; every step outputs concrete files.
- No cosine similarity, avoid unit-normalizing inputs unless instructed.
- If something is deferred, mark with `# TODO: @builder` and add a skipped test referencing it.
- All public functions include docstrings with type hints and shape notes.
- Performance counters and deterministic seeds are included.


---

## Documentation

- API Reference: `docs/API.md`
- Models Reference: `docs/Models.md`
- OpenAPI JSON: generate via the export script (see below) to `docs/openapi.json`.


## Features (What this software does)

- Define semantic axes and build axis packs (legacy and v1 artifact-backed flows).
- Compute embeddings, resonance, and detailed analysis over texts and/or vectors.
- Multi-scale token, span, frame representations with role projections.
- Frame indexing, search, tracing, and statistics over a SQLite-backed store.
- End-to-end API suitable for powering UIs with rich visualization (e.g., heatmaps).


## Architecture and Key Modules

- `src/coherence/api/main.py`: FastAPI app factory, router wiring, registry bootstrap.
- Routers under `src/coherence/api/routers/`:
  - `health.py`: readiness and state.
  - `embed.py`: text embedding.
  - `resonance.py`: resonance vs. axis packs (inline or stored), intermediate outputs.
  - `pipeline.py`: full analysis pipeline (tokens, spans, frames, vectors, roles).
  - `v1_axes.py`: build, activate, inspect, export axis packs (artifacts-backed).
  - `v1_frames.py`: index frames, search by axis ranges, trace entities, stats.
  - `axes.py`: legacy file-based axis packs; create from seed phrases.
  - `index.py`: legacy ANN indexing for docs.
  - `search.py`: legacy ANN recall + rerank for queries.
  - `whatif.py`: counterfactual stub.
  - `analyze.py`: legacy analyze over file-based packs.
- Shared models: `src/coherence/api/models.py` (see `docs/Models.md`).
- Artifacts: default at `artifacts/` (`COHERENCE_ARTIFACTS_DIR`), e.g., `frames.sqlite`, `axis_pack:<id>.npz` and metadata.


## API Overview (High level)

- `/health/ready` — readiness info (encoder, active pack, frames DB).
- `/embed` — encode texts to vectors.
- `/resonance` — compute resonance vs. an axis pack; supports inline packs.
- `/pipeline/analyze` — detailed analysis with spans, frames, role projections.
- `/v1/axes/*` — build, activate, get, export axis packs (production path).
- `/v1/frames/*` — index/search/trace/stats for frames (SQLite-backed).
- Legacy: `/axes`, `/index`, `/search`, `/analyze`, `/whatif`.


See `docs/API.md` for exhaustive endpoint specs and examples.

## Configuration

- `COHERENCE_ARTIFACTS_DIR` — where artifacts (axis packs, frames DB) are stored. Default: `artifacts/`.
- `COHERENCE_ENCODER` — optional encoder override for components that accept it.
- App config file: `configs/app.yaml` (log level, limits, etc.).
- Logging config: `configs/logging.yaml`.


## Artifacts and Data

- Axis packs (v1): `{ARTIFACTS}/axis_pack:<pack_id>.npz` + `.meta.json`.
- Frames DB: `{ARTIFACTS}/frames.sqlite`.
- Legacy axis packs (file-based): `data/axes/<axis_pack_id>.json`.


## Scripts

- Seed example axes:


```text

  python scripts/seed_axes.py --config configs/axis_packs/sample.json

```text

- Export OpenAPI schema (writes to `docs/openapi.json`):


```text

  python scripts/export_openapi.py --out docs/openapi.json

```text

- Generate client collections via Make (also regenerates OpenAPI):


```text

  # default BASE_URL is <<<http://localhost:8080>>>
  make postman      # writes docs/postman_collection.json
  make thunder      # writes docs/thunder-collection_Coherence_API.json
  make collections  # openapi + both collections

  # override base URL
  make collections BASE_URL=<<<http://127.0.0.1:8080>>>

```text

- PowerShell helper (Windows):


```text

  # Regenerate OpenAPI + Postman + Thunder into docs/
  .\tools\export-openapi.ps1 -BaseUrl "<<<http://localhost:8080">>> -OutDir "docs"

```text

## Testing

- Run tests:


```text

  pytest -q

```text

- Selected integration tests exercise API endpoints under `tests/`.


## Development Tips

- Use `uvicorn` with `--reload` for rapid iteration.
- Check `/health/ready` to verify active axis pack and frames DB presence.
- Prefer v1 endpoints (`/v1/axes`, `/v1/frames`) for production flows.
