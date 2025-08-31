# EthicalAI Ethics (Objective & Proofs)

**Objective:** Maximize human autonomy grounded in empirical truth, constrained by non-aggression and fairness.

**Axes (7):** virtue, deontology, consequentialism, autonomy, truthfulness, non-aggression, fairness.

**Thresholds:** Learned from tiny calibration sets (`data/calibration/*.jsonl`). We pick τ per axis with a max FPR budget.

**Proofs:** Every refusal/allow decision comes with a **DecisionProof** listing spans, thresholds, and final action.

**Reproducibility:** Re-run calibration via:
```bash
python -m ethicalai.axes.calibrate --pack <id> \
  --dataset data/calibration/autonomy.jsonl \
  --dataset data/calibration/truth.jsonl \
  --dataset data/calibration/non_aggression.jsonl \
  --dataset data/calibration/fairness.jsonl
```
Outputs to `reports/calibration:<id>/summary.json` plus per-axis CSVs.
