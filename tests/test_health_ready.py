from fastapi.testclient import TestClient
from coherence.api.main import app

def test_health_ready_fast():
    client = TestClient(app)
    r = client.get("/health/ready")
    assert r.status_code == 200
