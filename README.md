# Coherence

Coherence is a semantic analysis engine that allows users to define semantic axes and compute resonance and internal coherence over text. It outputs per-axis vectors at token, span, frame, and frame-span levels, with multi-scale diffusion and a lightweight logical frame layer.

This repository follows the authoritative build plan provided. We will implement the system milestone-by-milestone with concrete artifacts, tests, and a runnable API.

## Goals

- Axis-aware, magnitude-sensitive resonance (no cosine).
- Non-adjacent span coherence via SkipMesh and field diffusion.
- Semantic frames with logic operators and gating.
- Vectors everywhere: alpha, u, r, U (and C for spans) at multiple diffusion scales.

## Assumptions

- Default encoder: `sentence-transformers/all-mpnet-base-v2`.
- Device auto-selection (`cpu`/`cuda`) handled in encoder.
- Tokenization starts with simple whitespace (can swap later).
- Deterministic seeds set in config (`coherence.seed`).
- Choquet capacity disabled by default (linear aggregation) unless provided.

## Runbook (Quick Start)

1. Create a virtual environment and activate it

```bash
python -m venv .venv && source .venv/bin/activate
```

1. Install dependencies

```bash

pip install -r requirements.txt

```text

1. Run tests (initially mostly skipped)

```text

pytest -q

```text

1. Run the API server

```text

uvicorn coherence.api.main:app --reload --host 0.0.0.0 --port 8080

```text

1. Seed an example axis pack

```text

python scripts/seed_axes.py --config configs/axis_packs/sample.json

```text

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
