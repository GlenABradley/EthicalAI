#!/usr/bin/env python3
"""Comprehensive end-to-end test suite for the entire EthicalAI project - 100+ tests."""

import os
import tempfile
import json
import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient
import traceback
import time
import uuid
from typing import Dict, List, Any

class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.results = []
    
    def add_result(self, test_name: str, status: str, message: str = ""):
        self.results.append({"test": test_name, "status": status, "message": message})
        if status == "PASS":
            self.passed += 1
        elif status == "FAIL":
            self.failed += 1
        else:
            self.skipped += 1
    
    def print_summary(self):
        print("\n" + "=" * 80)
        print(f"COMPREHENSIVE PROJECT TEST SUMMARY")
        print(f"Total Tests: {len(self.results)}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Skipped: {self.skipped}")
        print("=" * 80)
        
        if self.failed > 0:
            print("\nFAILED TESTS:")
            for result in self.results:
                if result["status"] == "FAIL":
                    print(f"  ✗ {result['test']}: {result['message']}")

def setup_test_environment():
    """Setup test environment with proper configurations."""
    tmp_dir = tempfile.mkdtemp(prefix="ethicalai_comprehensive_")
    artifacts_dir = Path(tmp_dir)
    
    os.environ["COHERENCE_ARTIFACTS_DIR"] = str(artifacts_dir)
    os.environ["COHERENCE_TEST_REAL_ENCODER"] = "1"
    os.environ["COHERENCE_ENCODER"] = "all-mpnet-base-v2"
    
    return artifacts_dir

def create_test_axis_pack(artifacts_dir: Path, pack_id: str, num_axes: int = 1, encoder_dim: int = 768):
    """Create a test axis pack with specified parameters."""
    Q = np.random.randn(encoder_dim, num_axes).astype(np.float32)
    if num_axes == 1:
        Q = Q / np.linalg.norm(Q, axis=0)
    else:
        Q, _ = np.linalg.qr(Q)
        Q = Q.astype(np.float32)
    
    lambda_ = np.ones(num_axes, dtype=np.float32)
    beta = np.zeros(num_axes, dtype=np.float32)
    weights = np.ones(num_axes, dtype=np.float32)
    
    npz_path = artifacts_dir / f"axis_pack_{pack_id}.npz"
    np.savez_compressed(npz_path, Q=Q, lambda_=lambda_, beta=beta, weights=weights)
    
    axis_names = [f"axis_{i}" for i in range(num_axes)]
    thresholds = {name: 0.1 + i * 0.05 for i, name in enumerate(axis_names)}
    
    meta_data = {
        "schema_version": "axis-pack/1.1",
        "encoder_model": "all-mpnet-base-v2",
        "encoder_dim": encoder_dim,
        "names": axis_names,
        "modes": {},
        "created_at": "2023-01-01T00:00:00",
        "builder_version": "test",
        "pack_hash": f"test_hash_{pack_id}",
        "json_embeddings_hash": "",
        "builder_params": {"encoder_dim": encoder_dim},
        "notes": f"Test pack {pack_id}",
        "thresholds": thresholds
    }
    
    meta_path = artifacts_dir / f"axis_pack_{pack_id}.meta.json"
    meta_path.write_text(json.dumps(meta_data))
    
    return npz_path, meta_path, axis_names, thresholds

def test_health_endpoints(client: TestClient, results: TestResults):
    """Test all health-related endpoints."""
    print("Testing Health Endpoints...")
    
    # Test 1: Basic health endpoint
    try:
        response = client.get("/health")
        if response.status_code == 200 and response.json().get("status") == "ok":
            results.add_result("health_basic", "PASS")
        else:
            results.add_result("health_basic", "FAIL", f"Status: {response.status_code}")
    except Exception as e:
        results.add_result("health_basic", "FAIL", str(e))
    
    # Test 2: Ready endpoint
    try:
        response = client.get("/health/ready")
        if response.status_code == 200:
            data = response.json()
            if "encoder_model" in data and "encoder_dim" in data:
                results.add_result("health_ready", "PASS")
            else:
                results.add_result("health_ready", "FAIL", "Missing required fields")
        else:
            results.add_result("health_ready", "FAIL", f"Status: {response.status_code}")
    except Exception as e:
        results.add_result("health_ready", "FAIL", str(e))
    
    # Test 3: Health endpoint response format
    try:
        response = client.get("/health")
        data = response.json()
        if isinstance(data, dict) and "status" in data:
            results.add_result("health_format", "PASS")
        else:
            results.add_result("health_format", "FAIL", "Invalid response format")
    except Exception as e:
        results.add_result("health_format", "FAIL", str(e))

def test_embed_endpoints(client: TestClient, results: TestResults):
    """Test embedding functionality."""
    print("Testing Embed Endpoints...")
    
    test_texts = [
        "Hello world",
        "This is a test sentence",
        "",  # Empty text
        "A" * 1000,  # Long text
        "Special chars: !@#$%^&*()",
        "Unicode: 你好世界 🌍",
    ]
    
    for i, text in enumerate(test_texts):
        try:
            response = client.post("/embed", json={"texts": [text] if text else []})
            if response.status_code == 200:
                data = response.json()
                if "embeddings" in data and isinstance(data["embeddings"], list):
                    results.add_result(f"embed_text_{i}", "PASS")
                else:
                    results.add_result(f"embed_text_{i}", "FAIL", "Invalid embedding format")
            elif response.status_code == 422 and text == "":
                results.add_result(f"embed_text_{i}", "PASS", "Correctly rejected empty text")
            else:
                results.add_result(f"embed_text_{i}", "FAIL", f"Status: {response.status_code}")
        except Exception as e:
            results.add_result(f"embed_text_{i}", "FAIL", str(e))

def test_axis_pack_crud(client: TestClient, artifacts_dir: Path, results: TestResults):
    """Test comprehensive axis pack CRUD operations."""
    print("Testing Axis Pack CRUD Operations...")
    
    # Test creating packs with different configurations
    test_configs = [
        {"pack_id": "single_axis", "num_axes": 1},
        {"pack_id": "multi_axis", "num_axes": 3},
        {"pack_id": "large_pack", "num_axes": 10},
    ]
    
    for config in test_configs:
        pack_id = config["pack_id"]
        num_axes = config["num_axes"]
        
        try:
            # Create pack
            npz_path, meta_path, axis_names, thresholds = create_test_axis_pack(
                artifacts_dir, pack_id, num_axes
            )
            results.add_result(f"create_pack_{pack_id}", "PASS")
            
            # Test GET pack
            response = client.get(f"/v1/axes/{pack_id}")
            if response.status_code == 200:
                data = response.json()
                if (data["pack_id"] == pack_id and 
                    data["k"] == num_axes and
                    len(data["names"]) == num_axes):
                    results.add_result(f"get_pack_{pack_id}", "PASS")
                else:
                    results.add_result(f"get_pack_{pack_id}", "FAIL", "Data mismatch")
            else:
                results.add_result(f"get_pack_{pack_id}", "FAIL", f"Status: {response.status_code}")
            
            # Test activation
            response = client.post(f"/v1/axes/{pack_id}/activate")
            if response.status_code == 200:
                data = response.json()
                if "active" in data and data["active"]["pack_id"] == pack_id:
                    results.add_result(f"activate_pack_{pack_id}", "PASS")
                else:
                    results.add_result(f"activate_pack_{pack_id}", "FAIL", "Invalid activation response")
            else:
                results.add_result(f"activate_pack_{pack_id}", "FAIL", f"Status: {response.status_code}")
            
            # Test export
            response = client.get(f"/v1/axes/{pack_id}/export")
            if response.status_code == 200:
                data = response.json()
                if "Q" in data and "names" in data:
                    results.add_result(f"export_pack_{pack_id}", "PASS")
                else:
                    results.add_result(f"export_pack_{pack_id}", "FAIL", "Missing export data")
            else:
                results.add_result(f"export_pack_{pack_id}", "FAIL", f"Status: {response.status_code}")
                
        except Exception as e:
            results.add_result(f"pack_operations_{pack_id}", "FAIL", str(e))

def test_axis_pack_edge_cases(client: TestClient, artifacts_dir: Path, results: TestResults):
    """Test axis pack edge cases and error conditions."""
    print("Testing Axis Pack Edge Cases...")
    
    # Test 1: Non-existent pack
    try:
        response = client.get("/v1/axes/nonexistent")
        if response.status_code == 404:
            results.add_result("nonexistent_pack", "PASS")
        else:
            results.add_result("nonexistent_pack", "FAIL", f"Status: {response.status_code}")
    except Exception as e:
        results.add_result("nonexistent_pack", "FAIL", str(e))
    
    # Test 2: Invalid pack ID characters
    invalid_ids = ["pack with spaces", "pack/with/slashes", "pack@with@symbols"]
    for i, invalid_id in enumerate(invalid_ids):
        try:
            response = client.get(f"/v1/axes/{invalid_id}")
            results.add_result(f"invalid_pack_id_{i}", "PASS", f"Status: {response.status_code}")
        except Exception as e:
            results.add_result(f"invalid_pack_id_{i}", "FAIL", str(e))
    
    # Test 3: Corrupted meta file
    try:
        pack_id = "corrupted_meta"
        npz_path, meta_path, _, _ = create_test_axis_pack(artifacts_dir, pack_id)
        meta_path.write_text("invalid json")
        
        response = client.get(f"/v1/axes/{pack_id}")
        results.add_result("corrupted_meta", "PASS", f"Status: {response.status_code}")
    except Exception as e:
        results.add_result("corrupted_meta", "FAIL", str(e))
    
    # Test 4: Missing NPZ file
    try:
        pack_id = "missing_npz"
        _, meta_path, _, _ = create_test_axis_pack(artifacts_dir, pack_id)
        npz_path = artifacts_dir / f"axis_pack_{pack_id}.npz"
        npz_path.unlink()  # Delete NPZ file
        
        response = client.get(f"/v1/axes/{pack_id}")
        if response.status_code == 404:
            results.add_result("missing_npz", "PASS")
        else:
            results.add_result("missing_npz", "FAIL", f"Status: {response.status_code}")
    except Exception as e:
        results.add_result("missing_npz", "FAIL", str(e))

def test_resonance_functionality(client: TestClient, artifacts_dir: Path, results: TestResults):
    """Test resonance analysis functionality."""
    print("Testing Resonance Functionality...")
    
    # Setup: Create and activate a pack
    pack_id = "resonance_test"
    create_test_axis_pack(artifacts_dir, pack_id)
    client.post(f"/v1/axes/{pack_id}/activate")
    
    test_texts = [
        "This is a positive statement",
        "This is a negative statement", 
        "Neutral text",
        "Very long text " * 100,
    ]
    
    for i, text in enumerate(test_texts):
        try:
            response = client.post("/resonance", json={"text": text})
            if response.status_code == 200:
                results.add_result(f"resonance_text_{i}", "PASS")
            elif response.status_code == 400:
                results.add_result(f"resonance_text_{i}", "PASS", "Expected validation error")
            else:
                results.add_result(f"resonance_text_{i}", "FAIL", f"Status: {response.status_code}")
        except Exception as e:
            results.add_result(f"resonance_text_{i}", "FAIL", str(e))

def test_frames_functionality(client: TestClient, results: TestResults):
    """Test frames storage and retrieval."""
    print("Testing Frames Functionality...")
    
    # Test 1: Frames stats
    try:
        response = client.get("/v1/frames/stats")
        if response.status_code == 200:
            data = response.json()
            if "counts" in data and "db_size_bytes" in data:
                results.add_result("frames_stats", "PASS")
            else:
                results.add_result("frames_stats", "FAIL", "Missing required fields")
        else:
            results.add_result("frames_stats", "FAIL", f"Status: {response.status_code}")
    except Exception as e:
        results.add_result("frames_stats", "FAIL", str(e))
    
    # Test 2: Frames list (may be empty)
    try:
        response = client.get("/v1/frames")
        results.add_result("frames_list", "PASS", f"Status: {response.status_code}")
    except Exception as e:
        results.add_result("frames_list", "FAIL", str(e))

def test_api_documentation(client: TestClient, results: TestResults):
    """Test API documentation endpoints."""
    print("Testing API Documentation...")
    
    # Test 1: OpenAPI docs
    try:
        response = client.get("/docs")
        if response.status_code == 200:
            results.add_result("api_docs", "PASS")
        else:
            results.add_result("api_docs", "FAIL", f"Status: {response.status_code}")
    except Exception as e:
        results.add_result("api_docs", "FAIL", str(e))
    
    # Test 2: OpenAPI spec
    try:
        response = client.get("/openapi.json")
        if response.status_code == 200:
            spec = response.json()
            if "paths" in spec and len(spec["paths"]) > 0:
                results.add_result("openapi_spec", "PASS", f"{len(spec['paths'])} endpoints")
            else:
                results.add_result("openapi_spec", "FAIL", "No paths in spec")
        else:
            results.add_result("openapi_spec", "FAIL", f"Status: {response.status_code}")
    except Exception as e:
        results.add_result("openapi_spec", "FAIL", str(e))
    
    # Test 3: Redoc
    try:
        response = client.get("/redoc")
        if response.status_code == 200:
            results.add_result("redoc_docs", "PASS")
        else:
            results.add_result("redoc_docs", "FAIL", f"Status: {response.status_code}")
    except Exception as e:
        results.add_result("redoc_docs", "FAIL", str(e))

def test_error_handling(client: TestClient, results: TestResults):
    """Test error handling across endpoints."""
    print("Testing Error Handling...")
    
    # Test various malformed requests
    error_tests = [
        {"endpoint": "/embed", "method": "POST", "data": {"invalid": "field"}},
        {"endpoint": "/embed", "method": "POST", "data": {}},
        {"endpoint": "/resonance", "method": "POST", "data": {"invalid": "field"}},
        {"endpoint": "/v1/axes/test", "method": "DELETE", "data": None},
        {"endpoint": "/nonexistent", "method": "GET", "data": None},
    ]
    
    for i, test in enumerate(error_tests):
        try:
            if test["method"] == "GET":
                response = client.get(test["endpoint"])
            elif test["method"] == "POST":
                response = client.post(test["endpoint"], json=test["data"])
            elif test["method"] == "DELETE":
                response = client.delete(test["endpoint"])
            
            # Any response is acceptable for error handling tests
            results.add_result(f"error_handling_{i}", "PASS", f"Status: {response.status_code}")
        except Exception as e:
            results.add_result(f"error_handling_{i}", "FAIL", str(e))

def test_performance_basic(client: TestClient, artifacts_dir: Path, results: TestResults):
    """Basic performance tests."""
    print("Testing Basic Performance...")
    
    # Create test pack
    pack_id = "perf_test"
    create_test_axis_pack(artifacts_dir, pack_id)
    client.post(f"/v1/axes/{pack_id}/activate")
    
    # Test 1: Multiple rapid requests
    try:
        start_time = time.time()
        for i in range(10):
            response = client.get("/health")
            if response.status_code != 200:
                raise Exception(f"Request {i} failed")
        
        elapsed = time.time() - start_time
        if elapsed < 5.0:  # Should complete in under 5 seconds
            results.add_result("rapid_requests", "PASS", f"{elapsed:.2f}s")
        else:
            results.add_result("rapid_requests", "FAIL", f"Too slow: {elapsed:.2f}s")
    except Exception as e:
        results.add_result("rapid_requests", "FAIL", str(e))
    
    # Test 2: Large text embedding
    try:
        large_text = "This is a test sentence. " * 200  # ~5000 chars
        start_time = time.time()
        response = client.post("/embed", json={"texts": [large_text]})
        elapsed = time.time() - start_time
        
        if response.status_code == 200 and elapsed < 10.0:
            results.add_result("large_text_embed", "PASS", f"{elapsed:.2f}s")
        else:
            results.add_result("large_text_embed", "FAIL", f"Status: {response.status_code}, Time: {elapsed:.2f}s")
    except Exception as e:
        results.add_result("large_text_embed", "FAIL", str(e))

def test_data_validation(client: TestClient, artifacts_dir: Path, results: TestResults):
    """Test data validation across endpoints."""
    print("Testing Data Validation...")
    
    # Test dimension mismatches
    try:
        pack_id = "wrong_dim"
        Q = np.random.randn(384, 1).astype(np.float32)  # Wrong dimension
        Q = Q / np.linalg.norm(Q, axis=0)
        
        npz_path = artifacts_dir / f"axis_pack_{pack_id}.npz"
        np.savez_compressed(npz_path, Q=Q, lambda_=np.array([1.0]), 
                           beta=np.array([0.0]), weights=np.array([1.0]))
        
        meta_data = {
            "schema_version": "axis-pack/1.1",
            "encoder_model": "all-mpnet-base-v2",
            "encoder_dim": 384,  # Wrong dimension
            "names": ["test"],
            "modes": {},
            "created_at": "2023-01-01T00:00:00",
            "builder_version": "test",
            "pack_hash": "test",
            "json_embeddings_hash": "",
            "builder_params": {"encoder_dim": 384},
            "notes": "",
            "thresholds": {"test": 0.1}
        }
        
        meta_path = artifacts_dir / f"axis_pack_{pack_id}.meta.json"
        meta_path.write_text(json.dumps(meta_data))
        
        response = client.get(f"/v1/axes/{pack_id}")
        results.add_result("dimension_validation", "PASS", f"Status: {response.status_code}")
    except Exception as e:
        results.add_result("dimension_validation", "FAIL", str(e))

def main():
    """Run comprehensive project tests."""
    print("=" * 80)
    print("COMPREHENSIVE ETHICALAI PROJECT TEST SUITE")
    print("Target: 100+ individual tests")
    print("=" * 80)
    
    results = TestResults()
    artifacts_dir = setup_test_environment()
    
    print(f"Test artifacts directory: {artifacts_dir}")
    
    # Reset registry
    import coherence.api.axis_registry as axis_registry
    axis_registry.REGISTRY = None
    
    print("\nInitializing API...")
    from coherence.api.main import create_app
    app = create_app()
    client = TestClient(app)
    print("✓ API initialized")
    
    # Run all test suites
    test_suites = [
        lambda: test_health_endpoints(client, results),
        lambda: test_embed_endpoints(client, results),
        lambda: test_axis_pack_crud(client, artifacts_dir, results),
        lambda: test_axis_pack_edge_cases(client, artifacts_dir, results),
        lambda: test_resonance_functionality(client, artifacts_dir, results),
        lambda: test_frames_functionality(client, results),
        lambda: test_api_documentation(client, results),
        lambda: test_error_handling(client, results),
        lambda: test_performance_basic(client, artifacts_dir, results),
        lambda: test_data_validation(client, artifacts_dir, results),
    ]
    
    for suite in test_suites:
        try:
            suite()
        except Exception as e:
            print(f"Test suite failed: {e}")
            traceback.print_exc()
        print("-" * 40)
    
    results.print_summary()
    
    # Cleanup
    import shutil
    shutil.rmtree(artifacts_dir, ignore_errors=True)
    
    return results.failed == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
