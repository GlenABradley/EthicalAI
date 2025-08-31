# Ethics

## Objective

Maximize human autonomy based on objective empirical truth.

This objective guides all ethical evaluations, ensuring AI systems support human self-determination while prioritizing factual accuracy.

## Axes

- **Virtue**: Moral goodness, compassion, integrity.
- **Deontology**: Duty-based ethics, adherence to rules.
- **Consequentialism**: Outcome-based ethics, maximizing utility.
- **Autonomy**: Self-determination, freedom from coercion.
- **Truthfulness**: Honesty, accuracy, avoidance of deception.
- **Non-aggression**: Peace, non-violence, respect for others.
- **Fairness**: Equality, justice, equal opportunities.

Each axis is an orthonormal vector in embedding space, derived from seed phrases.

## Thresholds and Policy

Thresholds are calibrated on labeled datasets to balance false positives/negatives.
Policy includes strictness levels and weights for axes/forms of autonomy (bodily, cognitive, behavioral, social, existential).

## Proofs

DecisionProof structure:
- objective: The guiding principle.
- pack_id: Identifier of the axis pack used.
- spans: List of {i, j, axis, score, threshold, breached}.
- aggregation: Logical OR for veto.
- final: {action: "allow"|"refuse", rationale}.

Proofs enable transparency and reconstruction of decisions.
