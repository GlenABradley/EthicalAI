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

# Test data
SAMPLE_TEXTS = [
    "This is a test sentence about ethics and morality.",
    "Harm reduction and maximizing well-being are key ethical principles.",
    "This text should be evaluated for ethical considerations.",
]

# Test axis pack configuration
TEST_AXIS_PACK = {
    "name": "test_ethics",
    "axes": [
        {
            "name": "harm_reduction",
            "positive_examples": ["reduces harm", "prevents suffering"],
            "negative_examples": ["causes harm", "inflicts pain"],
            "weight": 1.0
        },
        {
            "name": "autonomy",
            "positive_examples": ["respects autonomy", "supports freedom"],
            "negative_examples": ["restricts freedom", "controls others"],
            "weight": 1.0
        }
    ]
}


def test_health_check(api_client: TestClient):
    """Test the health check endpoint."""
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_embed_endpoint(api_client: TestClient):
    """Test the text embedding endpoint."""
    test_text = SAMPLE_TEXTS[0]
    response = api_client.post(
        "/embed",
        json={"texts": [test_text]}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "embeddings" in data
    assert len(data["embeddings"]) == 1  # One embedding for one text
    assert len(data["embeddings"][0]) > 100  # Should be a high-dimensional vector
    assert data["model_name"] is not None
    assert data["device"] in ["cpu", "cuda", "mps"]


def test_analyze_endpoint(api_client: TestClient, tmp_artifacts_dir: Path):
    """Test the text analysis endpoint with ethical evaluation."""
    # First, create a test axis pack
    axis_pack_path = tmp_artifacts_dir / "test_axis_pack.json"
    axis_pack_path.write_text(json.dumps(TEST_AXIS_PACK))
    
    # Test analysis with the sample text
    response = api_client.post(
        "/analyze",
        json={
            "texts": [SAMPLE_TEXTS[0]],
            "axis_pack_id": "test_axis_pack"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify the response structure
    assert "axes" in data
    assert "tokens" in data
    assert "spans" in data
    assert "frames" in data
    assert "frame_spans" in data
    assert "tau_used" in data
    
    # Verify axes information
    assert data["axes"]["id"] == "test_axis_pack"
    assert len(data["axes"]["names"]) == 2  # Should have two axes
    
    # Verify token vectors
    assert len(data["tokens"]["alpha"]) > 0
    assert len(data["tokens"]["u"]) > 0
    assert len(data["tokens"]["r"]) > 0
    assert len(data["tokens"]["U"]) > 0


def test_batch_analysis(api_client: TestClient, tmp_artifacts_dir: Path):
    """Test batch processing of multiple texts."""
    # Create a test axis pack
    axis_pack_path = tmp_artifacts_dir / "test_batch_axis_pack.json"
    axis_pack_path.write_text(json.dumps(TEST_AXIS_PACK))
    
    # Test batch analysis
    response = api_client.post(
        "/analyze/batch",
        json={
            "texts": SAMPLE_TEXTS,
            "axis_pack_id": "test_batch_axis_pack"
        }
    )
    
    assert response.status_code == 200
    results = response.json()
    assert len(results) == len(SAMPLE_TEXTS)
    
    # Verify each result has the expected structure
    for result in results:
        assert "axes" in result
        assert "tokens" in result
        assert len(result["tokens"]["alpha"]) > 0


def test_whatif_analysis(api_client: TestClient, tmp_artifacts_dir: Path):
    """Test the what-if analysis endpoint."""
    # Create a test axis pack
    axis_pack_path = tmp_artifacts_dir / "whatif_axis_pack.json"
    axis_pack_path.write_text(json.dumps(TEST_AXIS_PACK))
    
    # Test what-if scenario
    response = api_client.post(
        "/whatif",
        json={
            "text": "This is a test scenario.",
            "modifications": ["Make this more ethical", "Make this less harmful"],
            "axis_pack_id": "whatif_axis_pack"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify response structure
    assert "original" in data
    assert "modified" in data
    assert "differences" in data
    assert len(data["modified"]) == 2  # Two modifications


if __name__ == "__main__":
    pytest.main([__file__])
