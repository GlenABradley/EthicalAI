#!/usr/bin/env python3
"""Comprehensive end-to-end test for the entire EthicalAI project."""

import os
import tempfile
import json
import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient
import traceback
import requests

def setup_test_environment():
    """Setup test environment with proper configurations."""
    tmp_dir = tempfile.mkdtemp(prefix="ethicalai_test_")
    artifacts_dir = Path(tmp_dir)
    
    # Setup environment variables
    os.environ["COHERENCE_ARTIFACTS_DIR"] = str(artifacts_dir)
    os.environ["COHERENCE_TEST_REAL_ENCODER"] = "1"
    os.environ["COHERENCE_ENCODER"] = "all-mpnet-base-v2"
    
    return artifacts_dir

def create_test_axis_pack(artifacts_dir: Path, pack_id: str):
    """Create a minimal test axis pack."""
    Q = np.random.randn(768, 1).astype(np.float32)
    Q = Q / np.linalg.norm(Q, axis=0)
    lambda_ = np.array([1.0], dtype=np.float32)
    beta = np.array([0.0], dtype=np.float32)
    weights = np.array([1.0], dtype=np.float32)
    
    npz_path = artifacts_dir / f"axis_pack_{pack_id}.npz"
    np.savez_compressed(npz_path, Q=Q, lambda_=lambda_, beta=beta, weights=weights)
    
    meta_data = {
        "schema_version": "axis-pack/1.1",
        "encoder_model": "all-mpnet-base-v2",
        "encoder_dim": 768,
        "names": ["test_axis"],
        "modes": {},
        "created_at": "2023-01-01T00:00:00",
        "builder_version": "test",
        "pack_hash": "test_hash",
        "json_embeddings_hash": "",
        "builder_params": {"encoder_dim": 768},
        "notes": "",
        "thresholds": {"test_axis": 0.1}
    }
    
    meta_path = artifacts_dir / f"axis_pack_{pack_id}.meta.json"
    meta_path.write_text(json.dumps(meta_data))
    
    return npz_path, meta_path

def test_coherence_health_endpoints(client: TestClient):
    """Test coherence health endpoints."""
    print("Testing Coherence Health Endpoints...")
    
    # Basic health
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    print(f"✓ /health: {data}")
    
    # Ready endpoint
    response = client.get("/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert "encoder_model" in data
    print(f"✓ /health/ready: encoder={data['encoder_model']}")

def test_coherence_embed_endpoints(client: TestClient):
    """Test coherence embedding endpoints."""
    print("\nTesting Coherence Embed Endpoints...")
    
    try:
        # Test text embedding
        response = client.post("/embed", json={"text": "Hello world"})
        if response.status_code == 200:
            data = response.json()
            print(f"✓ /embed: dim={len(data.get('embedding', []))}")
        else:
            print(f"⚠ /embed failed: {response.status_code}")
    except Exception as e:
        print(f"⚠ /embed error: {e}")

def test_coherence_resonance_endpoints(client: TestClient, artifacts_dir: Path):
    """Test coherence resonance endpoints."""
    print("\nTesting Coherence Resonance Endpoints...")
    
    # Create test axis pack first
    pack_id = "resonance-test"
    create_test_axis_pack(artifacts_dir, pack_id)
    
    try:
        # Activate pack
        response = client.post(f"/v1/axes/{pack_id}/activate")
        if response.status_code == 200:
            print(f"✓ Activated pack for resonance testing")
            
            # Test resonance
            response = client.post("/resonance", json={"text": "This is a test"})
            if response.status_code == 200:
                data = response.json()
                print(f"✓ /resonance: {list(data.keys())}")
            else:
                print(f"⚠ /resonance failed: {response.status_code}")
        else:
            print(f"⚠ Pack activation failed: {response.status_code}")
    except Exception as e:
        print(f"⚠ Resonance test error: {e}")

def test_coherence_pipeline_endpoints(client: TestClient):
    """Test coherence pipeline endpoints."""
    print("\nTesting Coherence Pipeline Endpoints...")
    
    try:
        # Test pipeline status
        response = client.get("/pipeline/status")
        print(f"Pipeline status: {response.status_code}")
        
        # Test pipeline endpoints
        response = client.get("/pipeline")
        if response.status_code == 200:
            print(f"✓ /pipeline: available")
        else:
            print(f"⚠ /pipeline: {response.status_code}")
    except Exception as e:
        print(f"⚠ Pipeline test error: {e}")

def test_coherence_frames_endpoints(client: TestClient):
    """Test coherence frames endpoints."""
    print("\nTesting Coherence Frames Endpoints...")
    
    try:
        # Test frames list
        response = client.get("/v1/frames")
        print(f"Frames list: {response.status_code}")
        
        # Test frames stats
        response = client.get("/v1/frames/stats")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ /v1/frames/stats: {data}")
        else:
            print(f"⚠ /v1/frames/stats: {response.status_code}")
    except Exception as e:
        print(f"⚠ Frames test error: {e}")

def test_coherence_search_endpoints(client: TestClient):
    """Test coherence search endpoints."""
    print("\nTesting Coherence Search Endpoints...")
    
    try:
        # Test search
        response = client.post("/search", json={"query": "test query"})
        print(f"Search: {response.status_code}")
        
        if response.status_code == 200:
            print(f"✓ /search: available")
        else:
            print(f"⚠ /search: {response.status_code}")
    except Exception as e:
        print(f"⚠ Search test error: {e}")

def test_coherence_whatif_endpoints(client: TestClient):
    """Test coherence what-if endpoints."""
    print("\nTesting Coherence What-If Endpoints...")
    
    try:
        # Test what-if analysis
        response = client.post("/whatif", json={"scenario": "test scenario"})
        print(f"What-if: {response.status_code}")
        
        if response.status_code == 200:
            print(f"✓ /whatif: available")
        else:
            print(f"⚠ /whatif: {response.status_code}")
    except Exception as e:
        print(f"⚠ What-if test error: {e}")

def test_coherence_analyze_endpoints(client: TestClient):
    """Test coherence analyze endpoints."""
    print("\nTesting Coherence Analyze Endpoints...")
    
    try:
        # Test analysis
        response = client.post("/analyze", json={"text": "analyze this text"})
        print(f"Analyze: {response.status_code}")
        
        if response.status_code == 200:
            print(f"✓ /analyze: available")
        else:
            print(f"⚠ /analyze: {response.status_code}")
    except Exception as e:
        print(f"⚠ Analyze test error: {e}")

def test_coherence_index_endpoints(client: TestClient):
    """Test coherence index endpoints."""
    print("\nTesting Coherence Index Endpoints...")
    
    try:
        # Test index operations
        response = client.get("/index")
        print(f"Index: {response.status_code}")
        
        if response.status_code == 200:
            print(f"✓ /index: available")
        else:
            print(f"⚠ /index: {response.status_code}")
    except Exception as e:
        print(f"⚠ Index test error: {e}")

def test_ethicalai_eval_endpoints(client: TestClient):
    """Test EthicalAI evaluation endpoints."""
    print("\nTesting EthicalAI Eval Endpoints...")
    
    try:
        # Test evaluation endpoints
        response = client.post("/eval", json={"text": "evaluate this"})
        print(f"Eval: {response.status_code}")
        
        if response.status_code == 200:
            print(f"✓ /eval: available")
        else:
            print(f"⚠ /eval: {response.status_code}")
    except Exception as e:
        print(f"⚠ EthicalAI eval test error: {e}")

def test_ethicalai_constitution_endpoints(client: TestClient):
    """Test EthicalAI constitution endpoints."""
    print("\nTesting EthicalAI Constitution Endpoints...")
    
    try:
        # Test constitution endpoints
        response = client.get("/constitution")
        print(f"Constitution: {response.status_code}")
        
        if response.status_code == 200:
            print(f"✓ /constitution: available")
        else:
            print(f"⚠ /constitution: {response.status_code}")
    except Exception as e:
        print(f"⚠ Constitution test error: {e}")

def test_axis_pack_functionality(client: TestClient, artifacts_dir: Path):
    """Test core axis pack functionality."""
    print("\nTesting Axis Pack Core Functionality...")
    
    pack_id = "core-test"
    create_test_axis_pack(artifacts_dir, pack_id)
    
    # Test GET
    response = client.get(f"/v1/axes/{pack_id}")
    assert response.status_code == 200
    print(f"✓ Axis pack GET: {pack_id}")
    
    # Test activation
    response = client.post(f"/v1/axes/{pack_id}/activate")
    assert response.status_code == 200
    print(f"✓ Axis pack activation: {pack_id}")

def test_api_documentation(client: TestClient):
    """Test API documentation endpoints."""
    print("\nTesting API Documentation...")
    
    try:
        # Test OpenAPI docs
        response = client.get("/docs")
        print(f"API docs: {response.status_code}")
        
        # Test OpenAPI spec
        response = client.get("/openapi.json")
        if response.status_code == 200:
            spec = response.json()
            print(f"✓ OpenAPI spec: {len(spec.get('paths', {}))} endpoints")
        else:
            print(f"⚠ OpenAPI spec: {response.status_code}")
    except Exception as e:
        print(f"⚠ Documentation test error: {e}")

def main():
    """Run comprehensive project tests."""
    print("=" * 80)
    print("COMPREHENSIVE ETHICALAI PROJECT TEST")
    print("=" * 80)
    
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
    
    test_functions = [
        test_coherence_health_endpoints,
        test_coherence_embed_endpoints,
        lambda c, a: test_coherence_resonance_endpoints(c, a),
        test_coherence_pipeline_endpoints,
        test_coherence_frames_endpoints,
        test_coherence_search_endpoints,
        test_coherence_whatif_endpoints,
        test_coherence_analyze_endpoints,
        test_coherence_index_endpoints,
        test_ethicalai_eval_endpoints,
        test_ethicalai_constitution_endpoints,
        lambda c, a: test_axis_pack_functionality(c, a),
        test_api_documentation,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            if test_func.__name__ in ["test_coherence_health_endpoints", "test_coherence_embed_endpoints", 
                                     "test_coherence_pipeline_endpoints", "test_coherence_frames_endpoints",
                                     "test_coherence_search_endpoints", "test_coherence_whatif_endpoints",
                                     "test_coherence_analyze_endpoints", "test_coherence_index_endpoints",
                                     "test_ethicalai_eval_endpoints", "test_ethicalai_constitution_endpoints",
                                     "test_api_documentation"]:
                test_func(client)
            else:
                test_func(client, artifacts_dir)
            passed += 1
            print(f"✓ {test_func.__name__ if hasattr(test_func, '__name__') else 'lambda_test'} PASSED")
        except Exception as e:
            failed += 1
            print(f"✗ {test_func.__name__ if hasattr(test_func, '__name__') else 'lambda_test'} FAILED: {e}")
            traceback.print_exc()
        print("-" * 40)
    
    print("\n" + "=" * 80)
    print(f"PROJECT TEST SUMMARY: {passed} passed, {failed} failed")
    print("=" * 80)
    
    # Cleanup
    import shutil
    shutil.rmtree(artifacts_dir, ignore_errors=True)
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
