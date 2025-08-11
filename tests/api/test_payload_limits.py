from __future__ import annotations

from typing import List

from fastapi.testclient import TestClient


def test_analyze_payload_limit_413(api_client: TestClient, sample_axis_jsons: List[str]):
    c = api_client
    # Build a minimal pack so analyze has an active pack by default
    r0 = c.post("/v1/axes/build", json={"json_paths": [str(p) for p in sample_axis_jsons]})
    assert r0.status_code in (200, 201), r0.text

    too_long = "a" * (100000 + 1)
    r = c.post("/pipeline/analyze", json={"texts": [too_long]})
    assert r.status_code == 413, r.text
    assert r.json().get("detail") == "max_doc_chars exceeded"
