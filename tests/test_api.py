from fastapi.testclient import TestClient
from coherence.api.main import app


def test_health_ready():
    client = TestClient(app)
    resp = client.get("/health/ready")
    assert resp.status_code == 200
    assert resp.json().get("status") == "ok"
