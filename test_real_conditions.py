#!/usr/bin/env python3
"""Test script that mimics the exact test conditions."""

import os
import tempfile
import json
import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient

def main():
    print("Testing with exact test conditions...")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        artifacts_dir = Path(tmp_dir)
        
        # Reset registry like the test fixture does
        import coherence.api.axis_registry as axis_registry
        axis_registry.REGISTRY = None
        
        # Set environment variables exactly like the test fixture
        os.environ["COHERENCE_ARTIFACTS_DIR"] = str(artifacts_dir)
        os.environ["COHERENCE_TEST_REAL_ENCODER"] = "1"
        os.environ["COHERENCE_ENCODER"] = "all-mpnet-base-v2"  # Same as test
        
        print(f"Artifacts dir: {artifacts_dir}")
        print(f"Encoder: {os.environ['COHERENCE_ENCODER']}")
        
        # Create test files exactly like the test does
        pack_id = "thresh-pack"
        
        # Use 768D for all-mpnet-base-v2 (same as test)
        v = np.ones(768, dtype=np.float32) / np.sqrt(768)
        Q = v.reshape(-1, 1)
        lambda_ = np.array([1.0], dtype=np.float32)
        beta = np.array([0.0], dtype=np.float32)
        weights = np.array([1.0], dtype=np.float32)
        
        npz_path = artifacts_dir / f"axis_pack_{pack_id}.npz"
        np.savez_compressed(npz_path, Q=Q, lambda_=lambda_, beta=beta, weights=weights)
        print(f"Created NPZ file: {npz_path}")
        
        # Create meta file exactly like the test
        meta_data = {
            "schema_version": "axis-pack/1.1",
            "encoder_model": "all-mpnet-base-v2",
            "encoder_dim": 768,
            "names": ["autonomy"],
            "modes": {},
            "created_at": "2023-01-01T00:00:00",
            "builder_version": "test",
            "pack_hash": "test_hash",
            "json_embeddings_hash": "",
            "builder_params": {"encoder_dim": 768},
            "notes": "",
            "thresholds": {"autonomy": 0.123}
        }
        meta_path = artifacts_dir / f"axis_pack_{pack_id}.meta.json"
        meta_path.write_text(json.dumps(meta_data))
        print(f"Created meta file: {meta_path}")
        
        # Create app exactly like the test fixture
        from coherence.api.main import create_app
        app = create_app()
        client = TestClient(app)
        print("App created")
        
        # Test the exact same calls as the test
        print(f"Testing POST /v1/axes/{pack_id}/activate")
        r = client.post(f"/v1/axes/{pack_id}/activate")
        print(f"Activate response: {r.status_code}")
        if r.status_code != 200:
            print(f"Activate error: {r.text}")
            return False
        
        print(f"Testing GET /v1/axes/{pack_id}")
        r2 = client.get(f"/v1/axes/{pack_id}")
        print(f"Get response: {r2.status_code}")
        if r2.status_code != 200:
            print(f"Get error: {r2.text}")
            return False
        
        data = r2.json()
        if data["pack_id"] != pack_id:
            print(f"Pack ID mismatch: expected {pack_id}, got {data['pack_id']}")
            return False
        
        print("All tests passed!")
        return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
