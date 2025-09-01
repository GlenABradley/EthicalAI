from __future__ import annotations
from typing import List, Dict, Callable

Candidate = Dict[str, float | str]

def naive_generator(prompt: str, n: int = 3) -> List[Candidate]:
    # Replace with real LLM calls later (OpenAI/HF/local)
    return [{"text": f"{prompt} (option {i})", "logprob": -0.1*i} for i in range(n)]

GEN_REGISTRY: dict[str, Callable[[str, int], List[Candidate]]] = {
    "naive": naive_generator
}
