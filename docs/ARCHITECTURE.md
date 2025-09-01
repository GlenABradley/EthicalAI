# EthicalAI Architecture (Overview)

**Layers**
1. **coherence** (engine): embeddings, pipeline, FastAPI app.
2. **ethicalai** (integration): axes, evaluator, constitution (reranker), interaction (policy middleware).

**Artifacts**
- `artifacts/axis_pack:<id>.npz` — axis vectors
- `artifacts/axis_pack:<id>.meta.json` — metadata + thresholds
- `reports/calibration:<id>/` — CSVs + summary.json

**APIs**
- `/v1/axes/*` — build/activate/active
- `/v1/eval/text` — spans + DecisionProof
- `/v1/constitution/rank` — candidate reranker
- `/v1/interaction/respond` — middleware with proof/alternatives

**Flow**
1) Build Pack → 2) Calibrate with JSONL → 3) Activate → 4) Evaluate/Interact.
