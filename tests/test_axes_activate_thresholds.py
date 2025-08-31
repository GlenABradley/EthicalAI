from fastapi.testclient import TestClient
import json, numpy as np, pathlib


def test_activate_reads_thresholds(api_client: TestClient, tmp_artifacts_dir: pathlib.Path):
    art = tmp_artifacts_dir
    art.mkdir(parents=True, exist_ok=True)

    pack_id = "thresh-pack"
    v = np.ones(8, dtype=np.float32) / np.sqrt(8)
    np.savez_compressed(art / f"axis_pack:{pack_id}.npz", autonomy=v)
    (art / f"axis_pack:{pack_id}.meta.json").write_text(
        json.dumps({"meta": {}, "thresholds": {"autonomy": 0.123}})
    )
    r = api_client.post(f"/v1/axes/{pack_id}/activate")
    assert r.status_code == 200
    # sanity via /v1/axes/{pack_id}
    r2 = api_client.get(f"/v1/axes/{pack_id}")
    data = r2.json()
    assert data["pack_id"] == pack_id
