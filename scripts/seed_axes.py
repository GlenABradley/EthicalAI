#!/usr/bin/env python3
"""Seed axis packs from configs/axis_packs JSON.

Reads seeds from `configs/axis_packs/sample.json` if available. Expected format:
{
  "axes": [
    {"name": "Agency", "positives": ["allow"], "negatives": ["force"]},
    {"name": "Transparency", "positives": ["disclose"], "negatives": ["hide"]}
  ]
}
If missing or incompatible, falls back to a minimal demo seed set.  # TODO(@builder): replace sample config with real seeds.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from coherence.axis.builder import build_axis_pack_from_seeds
from coherence.encoders.registry import get_encoder


def load_seeds(path: Path) -> Dict[str, Dict[str, list[str]]]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text())
    except Exception:
        return {}
    axes = obj.get("axes")
    if not isinstance(axes, list):
        return {}
    seeds: Dict[str, Dict[str, list[str]]] = {}
    for a in axes:
        name = a.get("name")
        pos = a.get("positives", [])
        neg = a.get("negatives", [])
        if isinstance(name, str) and isinstance(pos, list) and isinstance(neg, list):
            seeds[name] = {"positive": list(map(str, pos)), "negative": list(map(str, neg))}
    return seeds


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    cfg_dir = repo_root / "configs" / "axis_packs"
    sample = cfg_dir / "sample.json"
    out_dir = repo_root / "data" / "axes"
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = load_seeds(sample)
    if not seeds:
        # Minimal fallback seeds
        seeds = {
            "Agency": {"positive": ["allow", "choose"], "negative": ["force", "compel"]},
            "Transparency": {"positive": ["disclose", "consent"], "negative": ["hide", "obscure"]},
        }
        print("No valid seeds found in sample.json; using fallback demo seeds. TODO(@builder): provide real seeds.")

    enc = get_encoder()
    pack = build_axis_pack_from_seeds(seeds, encode_fn=enc.encode, meta={"built_from": "scripts.seed_axes"})

    out_path = out_dir / "ap_sample.json"
    pack.save(out_path)
    print(f"Saved AxisPack: k={pack.k}, names={pack.names}, path={out_path}")


if __name__ == "__main__":
    main()
