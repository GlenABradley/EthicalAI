from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from coherence.api.main import app

client = TestClient(app)


def make_axis_pack(d: int = 4, k: int = 2):
    # Simple orthonormal Q with first k identity columns
    Q = np.eye(d, k, dtype=np.float32)
    names = [f"a{i}" for i in range(k)]
    lam = np.ones(k, dtype=np.float32)
    beta = np.zeros(k, dtype=np.float32)
    weights = np.ones(k, dtype=np.float32) / k
    return {
        "names": names,
        "Q": Q.tolist(),
        "lambda": lam.tolist(),
        "beta": beta.tolist(),
        "weights": weights.tolist(),
        "mu": {},
        "meta": {},
    }


def test_resonance_with_vectors():
    axis = make_axis_pack(d=4, k=2)
    X = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    resp = client.post(
        "/resonance",
        json={
            "vectors": X,
            "axis_pack": axis,
            "return_intermediate": True,
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "scores" in data and isinstance(data["scores"], list)
    assert len(data["scores"]) == 2
    assert "coords" in data and "utilities" in data


def test_pipeline_analyze_with_vectors():
    axis = make_axis_pack(d=4, k=2)
    # 5 tokens, 4-dim vectors
    X = np.zeros((5, 4), dtype=np.float32)
    X[2, 0] = 2.0  # one salient token
    resp = client.post(
        "/pipeline/analyze",
        json={
            "vectors": X.tolist(),
            "axis_pack": axis,
            "params": {"max_span_len": 3, "max_skip": 2, "diffusion_tau": 0.1},
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert set(["tokens", "spans", "frames", "frame_vectors"]).issubset(set(data.keys()))
    assert isinstance(data["frames"], list)


def test_embed_endpoint_smoke():
    # Small smoke test with short texts; this may download a model on first run
    texts = ["Hello world", "Semantic coherence"]
    resp = client.post(
        "/embed",
        json={
            "texts": texts,
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "embeddings" in data and "shape" in data
    assert data["shape"][0] == 2
