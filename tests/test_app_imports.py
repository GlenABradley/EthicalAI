from fastapi.testclient import TestClient


def test_app_importable(api_client: TestClient):
    app = api_client.app
    assert "coherence" in getattr(app, "title", "").lower()
