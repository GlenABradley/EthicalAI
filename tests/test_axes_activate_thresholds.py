from fastapi.testclient import TestClient
import json, numpy as np, pathlib
import ethicalai.api.axes as e_axes


def test_activate_reads_thresholds(api_client: TestClient, tmp_artifacts_dir: pathlib.Path, monkeypatch):
    # Ensure EthicalAI axes module uses our temp artifacts dir
    art = tmp_artifacts_dir
    art.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(e_axes, "ART_DIR", art, raising=False)

    pack_id = "thresh-pack"
    v = np.ones(8, dtype=np.float32) / np.sqrt(8)
    np.savez_compressed(art / f"axis_pack:{pack_id}.npz", autonomy=v)
    (art / f"axis_pack:{pack_id}.meta.json").write_text(
        json.dumps({"meta": {}, "thresholds": {"autonomy": 0.123}})
    )
    r = api_client.post("/v1/axes/activate", params={"pack_id": pack_id})
    assert r.status_code == 200
    # sanity via /v1/axes/active
    r2 = api_client.get("/v1/axes/active")
    data = r2.json()
    assert data["pack_id"] == pack_id
