from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from fastapi.testclient import TestClient

from coherence.api.main import create_app


def run_case(client: TestClient, X: np.ndarray, axis_pack: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    r = client.post(
        "/pipeline/analyze",
        json={
            "vectors": X.astype(np.float32).tolist(),
            "axis_pack": axis_pack,
            "params": params,
        },
    )
    r.raise_for_status()
    return r.json()


def main() -> None:
    out_dir = Path("artifacts/baseline")
    out_dir.mkdir(parents=True, exist_ok=True)

    app = create_app()
    client = TestClient(app)

    # Small fixed corpus: 3 texts as 1-token vectors each (deterministic)
    rng = np.random.default_rng(1234)
    X = rng.normal(size=(6, 4)).astype(np.float32)  # simulate token-level vectors

    # Simple orthonormal 2-axis pack in 4D
    Q = np.zeros((4, 2), dtype=np.float32)
    Q[0, 0] = 1.0
    Q[1, 1] = 1.0
    axis_pack = {
        "names": ["axis0", "axis1"],
        "Q": Q.tolist(),
        "lambda": [1.0, 1.0],
        "beta": [0.0, 0.0],
        "weights": [0.5, 0.5],
        "meta": {},
    }

    # Cases
    cases = {
        "default": {"max_span_len": 3, "max_skip": 2},
        "diffusion": {"max_span_len": 3, "max_skip": 2, "diffusion_tau": 0.5},
        "span_len_4": {"max_span_len": 4, "max_skip": 2},
    }

    results: Dict[str, Any] = {}
    for name, params in cases.items():
        results[name] = run_case(client, X, axis_pack, params)

    # Write individual artifacts
    # Use the default case for top-level files
    default = results["default"]
    (out_dir / "tokens.json").write_text(json.dumps(default["tokens"], ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "spans.json").write_text(json.dumps(default["spans"], ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "frames.json").write_text(json.dumps(default["frames"], ensure_ascii=False, indent=2), encoding="utf-8")

    # Save frame vectors as .npz
    fv = np.asarray(default["frame_vectors"], dtype=np.float32)
    np.savez(out_dir / "frame_vectors.npz", frame_vectors=fv)

    # Manifest: encoder/pack metadata and parameters
    # Pull encoder/active pack from health endpoint
    try:
        hr = client.get("/health/ready")
        health = hr.json() if hr.status_code == 200 else {}
    except Exception:
        health = {}

    manifest = {
        "encoder_model": health.get("encoder_model"),
        "encoder_dim": health.get("encoder_dim"),
        "active_pack": health.get("active_pack"),
        "axis_pack_inline": {
            "names": axis_pack["names"],
            "dim": 4,
            "k": 2,
        },
        "cases": list(cases.keys()),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # Save the multi-case bundle for reference
    (out_dir / "bundle.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Baseline artifacts written to {out_dir}")


if __name__ == "__main__":
    main()
