from fastapi.testclient import TestClient


def test_health_ready(api_client: TestClient):
    resp = api_client.get("/health/ready")
    assert resp.status_code == 200
    assert resp.json().get("status") == "ok"
