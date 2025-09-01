#!/usr/bin/env python3
"""Debug script to test route resolution without hanging tests."""

import os
import tempfile
import json
import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient

# Set environment variables
os.environ["COHERENCE_TEST_MODE"] = "true"
os.environ["COHERENCE_ENCODER"] = "all-MiniLM-L6-v2"

def main():
    print("Creating temporary artifacts directory...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        artifacts_dir = Path(tmp_dir)
        os.environ["COHERENCE_ARTIFACTS_DIR"] = str(artifacts_dir)
        
        print(f"Artifacts dir: {artifacts_dir}")
        
        # Create test files
        pack_id = "test-pack"
        print(f"Creating test files for pack: {pack_id}")
        
        # Create NPZ file with underscore naming (like the test does)
        v = np.ones(384, dtype=np.float32) / np.sqrt(384)  # Use 384D for all-MiniLM-L6-v2
        Q = v.reshape(-1, 1)
        lambda_ = np.array([1.0], dtype=np.float32)
        beta = np.array([0.0], dtype=np.float32)
        weights = np.array([1.0], dtype=np.float32)
        
        npz_path = artifacts_dir / f"axis_pack_{pack_id}.npz"
        np.savez_compressed(npz_path, Q=Q, lambda_=lambda_, beta=beta, weights=weights)
        print(f"Created NPZ file: {npz_path}")
        
        # Create meta file
        meta_path = artifacts_dir / f"axis_pack_{pack_id}.meta.json"
        meta_data = {
            "schema_version": "axis-pack/1.1",
            "encoder_model": "all-MiniLM-L6-v2",
            "encoder_dim": 384,
            "names": ["test_axis"],
            "modes": {},
            "created_at": "2023-01-01T00:00:00",
            "builder_version": "test",
            "pack_hash": "test_hash",
            "json_embeddings_hash": "",
            "builder_params": {"encoder_dim": 384},
            "notes": "",
            "thresholds": {"test_axis": 0.123}
        }
        meta_path.write_text(json.dumps(meta_data))
        print(f"Created meta file: {meta_path}")
        
        print("Files created, now testing API...")
        
        # Import and create app
        try:
            from coherence.api.main import create_app
            app = create_app()
            client = TestClient(app)
            print("App created successfully")
            
            # Test the routes
            print(f"Testing POST /v1/axes/{pack_id}/activate")
            response = client.post(f"/v1/axes/{pack_id}/activate")
            print(f"Activate response: {response.status_code}")
            if response.status_code != 200:
                print(f"Activate error: {response.text}")
            
            print(f"Testing GET /v1/axes/{pack_id}")
            response = client.get(f"/v1/axes/{pack_id}")
            print(f"Get response: {response.status_code}")
            if response.status_code != 200:
                print(f"Get error: {response.text}")
            else:
                print(f"Get success: {response.json()}")
                
        except Exception as e:
            print(f"Error during API testing: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
