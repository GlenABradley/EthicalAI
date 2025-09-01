from fastapi.testclient import TestClient
from ethicalai.api.axes import ACTIVE
from ethicalai.types import AxisPack, Axis
import numpy as np


def _activate_dummy_pack(dim=16):
    # Minimal axis pack to let the endpoint run
    axes = [Axis(name="autonomy", vector=np.ones(dim)/np.sqrt(dim), threshold=999.0, provenance={})]
    ACTIVE["pack"] = AxisPack(id="test", axes=axes, dim=dim, meta={})


def test_interaction_allow_path(api_client: TestClient):
    _activate_dummy_pack()
    r = api_client.post("/v1/interaction/respond", json={"prompt":"hello world"})
    assert r.status_code == 200
    data = r.json()
    assert "final" in data and isinstance(data["final"], str)
    assert "proof" in data and "final" in data["proof"]
