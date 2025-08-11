from __future__ import annotations

from typing import Any, Dict, List

from fastapi.testclient import TestClient


def test_frames_index_empty_ok(api_client: TestClient, sample_axis_jsons: List[str]):
    c = api_client
    # Build a minimal pack so frames endpoints have active pack
    r0 = c.post("/v1/axes/build", json={"json_paths": [str(p) for p in sample_axis_jsons]})
    assert r0.status_code in (200, 201), r0.text
    build = r0.json()
    assert build["k"] >= 1

    payload: Dict[str, Any] = {
        "doc_id": "doc0",
        "frames": [{"id": "f0"}],
    }
    r = c.post("/v1/frames/index", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data.get("ingested") == 1
    assert data.get("k") == build["k"]


def test_frames_search_and_trace_empties(api_client: TestClient, sample_axis_jsons: List[str]):
    c = api_client
    r0 = c.post("/v1/axes/build", json={"json_paths": [str(p) for p in sample_axis_jsons]})
    assert r0.status_code in (200, 201), r0.text

    r = c.get("/v1/frames/search", params={"axis": 0, "min": -1.0, "max": 1.0, "limit": 5})
    assert r.status_code == 200, r.text
    assert r.json() == {"items": []}

    r2 = c.get("/v1/frames/trace/foo", params={"limit": 5})
    assert r2.status_code == 200, r2.text
    assert r2.json() == {"items": []}
