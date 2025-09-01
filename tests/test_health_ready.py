from fastapi.testclient import TestClient


def test_health_ready_fast(api_client: TestClient):
    r = api_client.get("/health/ready")
    assert r.status_code == 200
