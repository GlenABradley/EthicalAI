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
    response = api_client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_embed_endpoint(api_client: TestClient):
    """Test the text embedding endpoint."""
    test_text = SAMPLE_TEXTS[0]
    response = api_client.post(
        "/api/v1/embed",
        json={"texts": [test_text]}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "embeddings" in data
    assert len(data["embeddings"]) == 1  # One embedding for one text
    assert len(data["embeddings"][0]) > 100  # Should be a high-dimensional vector
    assert data["model_name"] is not None
    assert data["device"] in ["cpu", "cuda", "mps"]


def test_analyze_endpoint(api_client: TestClient, tmp_path):
    """Test the text analysis endpoint."""
    # Create a temporary axis pack
    pack_dir = tmp_path / "test_pack"
    pack_dir.mkdir()
    (pack_dir / "axes").mkdir()
    
    axis_file = pack_dir / "axes" / "test_axis.json"
    axis_file.write_text(json.dumps(TEST_AXIS_PACK))
    
    # Test with a single text
    test_text = SAMPLE_TEXTS[0]
    response = api_client.post(
        "/api/v1/analyze",
        json={
            "texts": [test_text],
            "axis_pack_path": str(pack_dir)
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == 1
    assert "scores" in data["results"][0]
    assert "overall_score" in data["results"][0]
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


def test_batch_analysis(api_client: TestClient, tmp_path):
    """Test batch text analysis with multiple texts."""
    # Create a temporary axis pack
    pack_dir = tmp_path / "test_pack"
    pack_dir.mkdir()
    (pack_dir / "axes").mkdir()
    
    axis_file = pack_dir / "axes" / "test_axis.json"
    axis_file.write_text(json.dumps(TEST_AXIS_PACK))
    
    # Test with multiple texts
    response = api_client.post(
        "/api/v1/analyze/batch",
        json={
            "texts": SAMPLE_TEXTS,
            "axis_pack_path": str(pack_dir)
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == len(SAMPLE_TEXTS)
    for result in data["results"]:
        assert "scores" in result
        assert "overall_score" in result
        assert len(result["tokens"]["alpha"]) > 0


def test_whatif_analysis(api_client: TestClient, tmp_path):
    """Test what-if analysis with modified text."""
    # Create a temporary axis pack
    pack_dir = tmp_path / "test_pack"
    pack_dir.mkdir()
    (pack_dir / "axes").mkdir()
    
    axis_file = pack_dir / "axes" / "test_axis.json"
    axis_file.write_text(json.dumps(TEST_AXIS_PACK))
    
    # Test what-if scenario
    original_text = SAMPLE_TEXTS[0]
    modified_text = original_text.replace("ethical", "unethical")
    
    response = api_client.post(
        "/api/v1/analyze/whatif",
        json={
            "original_text": original_text,
            "modified_text": modified_text,
            "axis_pack_path": str(pack_dir)
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "original" in data
    assert "modified" in data
    assert "differences" in data
    assert isinstance(data["differences"], list) 


if __name__ == "__main__":
    pytest.main([__file__])
