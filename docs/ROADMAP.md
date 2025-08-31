# Roadmap

## Overview

EthicalAI implements intrinsic AI alignment through layered components that evaluate, rerank, and interact with AI systems to ensure ethical behavior.

## Phase 1: Evaluator Layer

**Goal**: Core ethical evaluation engine

- Build orthonormal ethical axes (virtue, deontology, consequentialism, autonomy, truthfulness, non-aggression, fairness)
- Project text onto axes for resonance scoring
- Apply thresholds for veto decisions
- Generate DecisionProof with spans and rationale

**Deliverables**:

- Axis pack persistence and calibration
- Text evaluation API with veto spans
- Orthonormal axis construction

## Phase 2: Constitution Layer

**Goal**: Safe generation through reranking

- Decoding-time candidate reranking
- Reject unethical generations
- Prioritize autonomy and truth in responses
- Maintain helpfulness while ensuring safety

**Deliverables**:

- Reranker API for N-best candidates
- Composite scoring (autonomy + truth + LM logprob)
- Breach detection and rejection

## Phase 3: Interaction Layer

**Goal**: User-facing ethical interface
- Transparent user-model interactions
- Proof generation for all responses
- Safe alternative suggestions
- Consent and transparency features

**Deliverables**:

- Interaction API with proofs
- UI for ethical evaluation and interaction
- Alternative path generation

## Future Phases

- Advanced calibration with larger datasets
- Multi-modal ethical evaluation
- Integration with major AI frameworks
- Community governance features

## Implementation Status

- ✅ Phase 1: Complete (Evaluator with axis packs, evaluation, proofs)
- ✅ Phase 2: Complete (Constitution reranker)
- ✅ Phase 3: Complete (Interaction API and UI)
- 🔄 Phase 4: In progress (Calibration, docs, CI/CD)
