from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Literal
import yaml, pathlib

STRICTNESS_PRESETS = {
    "permissive": 1.25,
    "balanced": 1.0,
    "strict": 0.85,
    "paranoid": 0.7,
}

@dataclass
class Policy:
    strictness: Literal["permissive","balanced","strict","paranoid"]
    thresholds_multiplier: float
    weights: Dict[str, float]
    forms: Dict[str, float]

    def effective_tau(self, tau: float) -> float:
        return self.thresholds_multiplier * tau

def load_policy(path: str | pathlib.Path = "src/ethicalai/interaction/policy.yaml") -> Policy:
    p = pathlib.Path(path)
    data = yaml.safe_load(p.read_text())
    strictness = data.get("strictness", "balanced")
    mult = data.get("thresholds", {}).get("multiplier")
    if mult is None:
        mult = STRICTNESS_PRESETS.get(strictness, 1.0)
    return Policy(
        strictness=strictness,
        thresholds_multiplier=float(mult),
        weights={**{"autonomy":1,"truth":1,"non_aggression":1,"fairness":1}, **data.get("weights", {})},
        forms={**{"bodily":1,"cognitive":1,"behavioral":1,"social":1,"existential":1}, **data.get("forms", {})},
    )
