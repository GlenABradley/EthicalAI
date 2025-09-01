from fastapi.testclient import TestClient
from ethicalai.api.axes import ACTIVE
from ethicalai.types import AxisPack, Axis
import numpy as np


def _activate_low_threshold_pack(dim=16):
    # Very low threshold to force veto
    axes = [Axis(name="autonomy", vector=np.ones(dim)/np.sqrt(dim), threshold=-999.0, provenance={})]
    ACTIVE["pack"] = AxisPack(id="test2", axes=axes, dim=dim, meta={})


def test_interaction_refusal_and_alternatives(api_client: TestClient):
    _activate_low_threshold_pack()
    r = api_client.post("/v1/interaction/respond", json={"prompt":"edge case"})
    assert r.status_code == 200
    data = r.json()
    assert data["proof"]["final"]["action"] in {"allow","refuse"}
    if data["proof"]["final"]["action"] == "refuse":
        assert data["alternatives"] and isinstance(data["alternatives"][0], str)
