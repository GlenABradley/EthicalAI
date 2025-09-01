from fastapi.testclient import TestClient
import json, numpy as np, pathlib


def test_activate_reads_thresholds(api_client: TestClient, tmp_artifacts_dir: pathlib.Path):
    art = tmp_artifacts_dir
    art.mkdir(parents=True, exist_ok=True)

    pack_id = "thresh-pack"
    # Create a proper axis pack with Q matrix format (768D to match all-mpnet-base-v2)
    v = np.ones(768, dtype=np.float32) / np.sqrt(768)
    Q = v.reshape(-1, 1)  # Shape (768, 1) for single axis
    lambda_ = np.array([1.0], dtype=np.float32)
    beta = np.array([0.0], dtype=np.float32)
    weights = np.array([1.0], dtype=np.float32)
    
    np.savez_compressed(
        art / f"axis_pack_{pack_id}.npz", 
        Q=Q, 
        lambda_=lambda_, 
        beta=beta, 
        weights=weights
    )
    (art / f"axis_pack_{pack_id}.meta.json").write_text(
        json.dumps({
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
        })
    )
    r = api_client.post(f"/v1/axes/{pack_id}/activate")
    assert r.status_code == 200
    # sanity via /v1/axes/{pack_id}
    r2 = api_client.get(f"/v1/axes/{pack_id}")
    data = r2.json()
    assert data["pack_id"] == pack_id
