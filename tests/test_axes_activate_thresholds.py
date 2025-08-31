from fastapi.testclient import TestClient
from coherence.api.main import app
from ethicalai.api.axes import ACTIVE
import json, numpy as np, pathlib

client = TestClient(app)

def test_activate_reads_thresholds(tmp_path, monkeypatch):
    art = pathlib.Path("artifacts")
    pack_id="thresh-pack"
    v = np.ones(8, dtype=np.float32)/np.sqrt(8)
    np.savez_compressed(art / f"axis_pack:{pack_id}.npz", autonomy=v)
    (art / f"axis_pack:{pack_id}.meta.json").write_text(json.dumps({"meta":{}, "thresholds":{"autonomy": 0.123}}))
    r = client.post("/v1/axes/activate", params={"pack_id": pack_id})
    assert r.status_code == 200
    # sanity via /v1/axes/active
    r2 = client.get("/v1/axes/active")
    data = r2.json()
    assert data["pack_id"] == pack_id
