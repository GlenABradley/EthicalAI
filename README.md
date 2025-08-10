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
2. Install dependencies
```bash
pip install -r requirements.txt
```
3. Run tests (initially mostly skipped)
```bash
pytest -q
```
4. Run the API server
```bash
uvicorn coherence.api.main:app --reload --host 0.0.0.0 --port 8080
```
5. Seed an example axis pack
```bash
python scripts/seed_axes.py --config configs/axis_packs/sample.json
```

## Repository Layout
See the repository tree printed by the build steps. The Python source lives under `src/coherence/`.

## Handling Rules
- No pretend-builds; every step outputs concrete files.
- No cosine similarity, avoid unit-normalizing inputs unless instructed.
- If something is deferred, mark with `# TODO: @builder` and add a skipped test referencing it.
- All public functions include docstrings with type hints and shape notes.
- Performance counters and deterministic seeds are included.
