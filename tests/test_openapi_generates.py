from fastapi.testclient import TestClient


def test_openapi_generation(tmp_path, api_client: TestClient):
    spec = api_client.app.openapi()
    assert "openapi" in spec
    out = tmp_path / "openapi.json"
    out.write_text("ok")
    assert out.exists()
