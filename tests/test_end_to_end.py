"""
End-to-end tests for the EthicalAI API.

This module contains tests that verify the complete functionality of the API,
including text analysis, vector generation, and ethical evaluation.
"""
import json
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

def _ensure_axis_pack() -> str:
    """Ensure a minimal, valid axis pack (d=768, k=2) exists under data/axes/.

    Returns the axis_pack_id string usable with /analyze.
    """
    axis_pack_id = "apitest_768_2"
    root = Path(__file__).resolve().parents[1]
    axes_dir = root / "data" / "axes"
    axes_dir.mkdir(parents=True, exist_ok=True)
    path = axes_dir / f"{axis_pack_id}.json"

    d = 768
    k = 2

    def _write_pack(p: Path):
        Q = np.zeros((d, k), dtype=np.float32)
        Q[0, 0] = 1.0
        Q[1, 1] = 1.0
        names = [f"a{i}" for i in range(k)]
        lam = np.ones(k, dtype=np.float32)
        beta = np.zeros(k, dtype=np.float32)
        weights = np.ones(k, dtype=np.float32) / float(k)
        obj = {
            "names": names,
            "Q": Q.tolist(),
            "lambda": lam.astype(float).tolist(),
            "beta": beta.astype(float).tolist(),
            "weights": weights.astype(float).tolist(),
            "mu": {},
            "meta": {},
        }
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            Q = data.get("Q", [])
            names = data.get("names", [])
            d_ok = isinstance(Q, list) and len(Q) == d and all(isinstance(row, list) for row in Q)
            k_ok = d_ok and len(Q[0]) == k and len(names) == k
            if not (d_ok and k_ok):
                _write_pack(path)
        except Exception:
            _write_pack(path)
    else:
        _write_pack(path)

    return axis_pack_id

# Test data
SAMPLE_TEXTS = [
    "This is an example sentence about ethical AI development.",
    "Autonomous weapons raise significant ethical concerns in modern warfare.",
    "Bias in machine learning models can lead to unfair treatment of certain groups.",
    "Transparency in AI decision-making is crucial for accountability.",
    "Privacy-preserving techniques help protect user data in AI applications."
]

# Sample axis pack for testing
TEST_AXIS_PACK = {
    "id": "test_axis_pack",
    "name": "Test Axis Pack",
    "description": "Test axis pack for unit tests",
    "version": "1.0.0",
    "axes": [
        {
            "id": "ethics",
            "name": "Ethical Considerations",
            "description": "Measures the ethical considerations in the text",
            "positive_examples": ["ethical", "fair", "just", "responsible"],
            "negative_examples": ["unethical", "biased", "harmful", "unfair"],
            "weight": 1.0
        },
        {
            "id": "safety",
            "name": "Safety Concerns",
            "description": "Identifies potential safety concerns in the text",
            "positive_examples": ["safe", "secure", "reliable", "robust"],
            "negative_examples": ["dangerous", "harmful", "risky", "unreliable"],
            "weight": 0.8
        }
    ]
}


def test_health_check(api_client: TestClient):
    """Test the health check endpoint."""
    print("\n=== Starting health check test ===")
    print("Making request to /health/ready...")
    
    try:
        response = api_client.get("/health/ready")
        print(f"Response status: {response.status_code}")
        print(f"Response text: {response.text[:200]}...")
        
        assert response.status_code == 200, f"Expected status code 200, got {response.status_code}: {response.text}"
        data = response.json()
        print("Response JSON parsed successfully")
        
        # Check for expected keys in the health check response
        required_keys = ["encoder_model", "encoder_dim", "active_pack", "frames_db_present"]
        print("Checking for required keys in response...")
        for key in required_keys:
            assert key in data, f"Missing expected key in health check response: {key}"
            print(f"  ✓ Found key: {key}")
            
        print("=== Health check test completed successfully ===\n")
        
    except Exception as e:
        print(f"!!! Test failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def test_embed_endpoint(api_client: TestClient):
    """Test the text embedding endpoint."""
    test_text = SAMPLE_TEXTS[0]
    response = api_client.post(
        "/embed",
        json={"texts": [test_text]},
        timeout=30.0
    )
    
    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert "embeddings" in data, "Response missing 'embeddings' key"
    assert len(data["embeddings"]) == 1, f"Expected 1 embedding, got {len(data['embeddings'])}"
    assert len(data["embeddings"][0]) > 100, f"Expected high-dimensional vector, got length {len(data['embeddings'][0])}"


def test_analyze_endpoint(api_client: TestClient):
    """Test the text analysis endpoint."""
    test_text = SAMPLE_TEXTS[0]
    axis_pack_id = _ensure_axis_pack()
    response = api_client.post(
        "/analyze",
        json={
            "axis_pack_id": axis_pack_id,
            "texts": [test_text],
            "options": {}
        },
        timeout=60.0
    )
    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}: {response.text}"
    data = response.json()
    # Validate AnalyzeResponse structure
    for key in ["axes", "tokens", "spans", "frames", "frame_spans", "tau_used"]:
        assert key in data, f"Missing expected key in analyze response: {key}"
    # Tokens object should have the expected sub-keys
    for key in ["alpha", "u", "r", "U"]:
        assert key in data["tokens"], f"Missing tokens sub-key: {key}"


def test_batch_analysis(api_client: TestClient):
    """Test batch text analysis with multiple texts."""
    # The /analyze endpoint processes only the first text from the list.
    # Call it per-text to simulate a batch.
    axis_pack_id = _ensure_axis_pack()
    for i, text in enumerate(SAMPLE_TEXTS):
        response = api_client.post(
            "/analyze",
            json={
                "axis_pack_id": axis_pack_id,
                "texts": [text],
                "options": {}
            },
            timeout=120.0
        )
        assert response.status_code == 200, f"Text {i}: expected 200, got {response.status_code}: {response.text}"
        data = response.json()
        for key in ["axes", "tokens", "spans", "frames", "frame_spans", "tau_used"]:
            assert key in data, f"Text {i}: missing key {key}"


def test_whatif_analysis(api_client: TestClient):
    """Test what-if analysis with modified text."""
    axis_pack_id = _ensure_axis_pack()
    response = api_client.post(
        "/whatif",
        json={
            "axis_pack_id": axis_pack_id,
            "doc_id": "doc1",
            "edits": [
                {"type": "replace_text", "start": 0, "end": 7, "value": "unethical"}
            ]
        },
        timeout=90.0
    )
    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert "deltas" in data and isinstance(data["deltas"], list), "WhatIf response should contain 'deltas' list"


if __name__ == "__main__":
    pytest.main([__file__])
