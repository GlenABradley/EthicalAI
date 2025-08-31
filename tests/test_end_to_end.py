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
    response = api_client.post(
        "/analyze",
        json={
            "texts": [test_text],
            "pack_id": None,  # Use active pack
            "compute_embeddings": True,
            "compute_frames": True,
            "compute_metrics": True
        },
        timeout=60.0
    )
    
    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}: {response.text}"
    data = response.json()
    
    assert "results" in data, "Response missing 'results' key"
    assert len(data["results"]) == 1, f"Expected 1 result, got {len(data['results'])}"
    
    result = data["results"][0]
    required_keys = ["text", "embedding", "frames", "metrics"]
    for key in required_keys:
        assert key in result, f"Missing expected key in result: {key}"


def test_batch_analysis(api_client: TestClient):
    """Test batch text analysis with multiple texts."""
    # Test with multiple texts
    response = api_client.post(
        "/analyze",
        json={
            "texts": SAMPLE_TEXTS,
            "pack_id": None,  # Use active pack
            "compute_embeddings": True,
            "compute_frames": True,
            "compute_metrics": True
        },
        timeout=120.0  # Longer timeout for batch processing
    )
    
    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}: {response.text}"
    data = response.json()
    
    assert "results" in data, "Response missing 'results' key"
    assert len(data["results"]) == len(SAMPLE_TEXTS), \
        f"Expected {len(SAMPLE_TEXTS)} results, got {len(data['results'])}"
    
    # Verify each result has the expected structure
    required_keys = ["text", "embedding", "frames", "metrics"]
    for i, result in enumerate(data["results"]):
        for key in required_keys:
            assert key in result, f"Missing key '{key}' in result {i}"


def test_whatif_analysis(api_client: TestClient):
    """Test what-if analysis with modified text."""
    # Test what-if scenario
    original_text = SAMPLE_TEXTS[0]
    modified_text = original_text.replace("ethical", "unethical")
    
    response = api_client.post(
        "/whatif",
        json={
            "original_text": original_text,
            "modified_text": modified_text,
            "pack_id": None,  # Use active pack
            "compute_embeddings": True,
            "compute_frames": True,
            "compute_metrics": True
        },
        timeout=90.0  # Longer timeout for what-if analysis
    )
    
    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}: {response.text}"
    data = response.json()
    
    # Check both original and modified results
    required_keys = ["text", "embedding", "frames", "metrics"]
    for result_type in ["original", "modified"]:
        assert result_type in data, f"Missing result type: {result_type}"
        result = data[result_type]
        for key in required_keys:
            assert key in result, f"Missing key '{key}' in {result_type} result"
    
    # Verify the texts were modified as expected
    assert data["original"]["text"] == original_text, \
        f"Original text mismatch: {data['original']['text']} != {original_text}"
    assert data["modified"]["text"] == modified_text, \
        f"Modified text mismatch: {data['modified']['text']} != {modified_text}"


if __name__ == "__main__":
    pytest.main([__file__])
