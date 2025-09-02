# EthicalAI Ethics Framework

## Core Philosophy

EthicalAI implements a multi-dimensional ethical evaluation system that combines classical ethical theories with modern AI safety principles. The framework evaluates text content across multiple ethical dimensions to provide nuanced, explainable ethical assessments.

### Fundamental Objective

**Maximize human autonomy grounded in empirical truth, while respecting constraints of non-aggression and fairness.**

This objective balances:

- **Autonomy**: Respecting individual agency and decision-making
- **Truth**: Grounding decisions in empirical reality
- **Non-Aggression**: Preventing harm to individuals and groups
- **Fairness**: Ensuring equitable treatment and outcomes

## Ethical Dimensions

The system evaluates content across seven primary ethical axes:

### 1. Virtue Ethics

- Evaluates character traits and moral excellence
- Focuses on what makes a good person
- Considers virtues like courage, temperance, justice, wisdom

### 2. Deontological Ethics

- Duty-based ethical evaluation
- Focuses on rules and obligations
- Evaluates actions based on adherence to moral rules

### 3. Consequentialist Ethics

- Outcome-based ethical evaluation
- Focuses on results and consequences
- Evaluates actions based on their effects

### 4. Autonomy

- Respect for individual agency
- Freedom of choice and self-determination
- Protection of personal sovereignty

### 5. Truthfulness

- Commitment to empirical accuracy
- Rejection of misinformation and deception
- Epistemic responsibility

### 6. Non-Aggression

- Prevention of harm to others
- Protection from violence and coercion
- Respect for boundaries and consent

### 7. Fairness

- Equitable treatment of all individuals
- Justice in distribution and procedure
- Prevention of discrimination and bias

## Implementation Architecture

### Axis Packs

Ethical dimensions are implemented through axis packs - learned vector representations in embedding space:

```python
# Axis pack structure
{
    "id": "ap_1756646151",
    "k": 7,  # Number of axes
    "d": 384,  # Embedding dimension
    "Q": [...]  # Projection matrix (d × k)
    "names": ["virtue", "deontology", ...],
    "thresholds": {...}  # Per-axis veto thresholds
}
```

### Projection and Scoring

1. **Text Embedding**: Input text is embedded using SentenceTransformer
2. **Axial Projection**: Embeddings projected onto ethical axes via Q matrix
3. **Score Calculation**:
   - α (alpha): Raw projection scores
   - u: Normalized scores (unit sphere)
   - r: Rectified scores (positive only)

### Veto System

The framework implements a veto-based decision system:

```python
# Veto detection
for axis, score in zip(axes, scores):
    if score > threshold[axis]:
        veto_spans.append({
            "axis": axis,
            "score": score,
            "threshold": threshold[axis],
            "text_span": span_text
        })
```

## Calibration Methodology

### Data Sources

Calibration uses carefully curated datasets for each ethical dimension:

- `data/calibration/autonomy.jsonl`: Examples of autonomy violations/respect
- `data/calibration/fairness.jsonl`: Fair/unfair treatment examples
- `data/calibration/non_aggression.jsonl`: Aggressive/peaceful content
- `data/calibration/truthfulness.jsonl`: True/false statements

### Threshold Learning

Thresholds are learned to optimize:

- **Sensitivity**: Detecting genuine ethical violations
- **Specificity**: Avoiding false positives
- **False Positive Rate (FPR)**: Constrained to acceptable levels

### Calibration Process

```bash
# Run calibration for an axis pack
python scripts/run_calibration.py \
    --pack_id ap_1756646151 \
    --fpr_budget 0.05 \
    --output_dir reports/calibration/
```

## Decision Proofs

Every ethical evaluation generates a decision proof providing:

### Components

1. **Action**: Allow/Refuse decision
2. **Veto Spans**: Specific text segments triggering vetoes
3. **Scores**: Numerical scores for each axis
4. **Thresholds**: Applied threshold values
5. **Rationale**: Human-readable explanation

### Example Decision Proof

```json
{
    "action": "refuse",
    "veto_spans": [
        {
            "text": "You should harm others",
            "axis": "non_aggression",
            "score": 0.89,
            "threshold": 0.75
        }
    ],
    "rationale": "Content violates non-aggression principle",
    "confidence": 0.89
}
```

## Ethical Configuration

### Constitution Updates

The ethical framework can be updated via the constitution API:

```python
# Update ethical thresholds
POST /v1/constitution/update
{
    "thresholds": {
        "non_aggression": 0.75,
        "truthfulness": 0.80,
        "fairness": 0.70
    }
}
```

### Axis Pack Selection

Different axis packs emphasize different ethical frameworks:

- **deontology.json**: Duty-based ethics emphasis
- **consequentialism.json**: Outcome-based ethics emphasis
- **virtue_ethics.json**: Character-based ethics emphasis
- **intent_bad_inclusive.json**: Intent-aware evaluation

## Practical Applications

### Content Moderation

```python
# Evaluate user-generated content
response = evaluate_text(
    text="User comment here",
    pack_id="ap_1756646151"
)

if response["decision"]["action"] == "refuse":
    # Block or flag content
    log_violation(response["decision"]["veto_spans"])
```

### AI Safety

```python
# Evaluate AI model outputs
for generation in model_outputs:
    ethical_check = evaluate_text(generation)
    if ethical_check["decision"]["action"] == "allow":
        return generation
```

### Educational Feedback

```python
# Provide ethical guidance
if veto_spans:
    feedback = generate_ethical_feedback(
        veto_spans=veto_spans,
        suggestions=generate_alternatives(text)
    )
```

## Reproducibility

### Experiment Reproduction

All ethical evaluations are deterministic and reproducible:

1. **Fixed Seeds**: Ensure consistent random initialization
2. **Version Control**: Track axis pack versions
3. **Audit Logs**: Record all decisions and proofs

### Validation

```bash
# Validate calibration results
python scripts/validate_calibration.py \
    --pack_id ap_1756646151 \
    --test_data data/calibration/test_set.jsonl
```

## Limitations and Considerations

### Current Limitations

1. **Cultural Context**: Ethical norms vary across cultures
2. **Temporal Changes**: Ethics evolve over time
3. **Edge Cases**: Complex scenarios may require human judgment
4. **Language Support**: Currently optimized for English

### Future Enhancements

1. **Multi-lingual Support**: Extend to other languages
2. **Cultural Adaptation**: Culture-specific axis packs
3. **Contextual Reasoning**: Deeper understanding of context
4. **Interactive Refinement**: User feedback integration

## References

### Academic Foundations

- Aristotle's Nicomachean Ethics (Virtue Ethics)
- Kant's Categorical Imperative (Deontology)
- Mill's Utilitarianism (Consequentialism)
- Rawls' Theory of Justice (Fairness)

### Technical Papers

- "Constitutional AI: Harmlessness from AI Feedback" (Anthropic)
- "Learning to Summarize with Human Feedback" (OpenAI)
- "Red Teaming Language Models" (DeepMind)
