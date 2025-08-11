from __future__ import annotations

from typing import List

from fastapi.testclient import TestClient


def test_frames_stats_ok(api_client: TestClient, sample_axis_jsons: List[str]):
    c = api_client
    # Build a minimal pack to populate active pack info
    r0 = c.post("/v1/axes/build", json={"json_paths": [str(p) for p in sample_axis_jsons]})
    assert r0.status_code in (200, 201), r0.text

    r = c.get("/v1/frames/stats")
    assert r.status_code == 200, r.text
    data = r.json()
    assert set(["db_path", "db_size_bytes", "counts", "last_ingest_ts", "active_pack"]).issubset(data.keys())
    assert isinstance(data["db_size_bytes"], int) and data["db_size_bytes"] >= 0
    counts = data["counts"]
    assert {"frames", "frame_axis", "frame_vectors"}.issubset(counts.keys())
