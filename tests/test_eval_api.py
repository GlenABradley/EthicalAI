from fastapi.testclient import TestClient
from ethicalai.api.axes import ACTIVE
from ethicalai.types import Axis, AxisPack
import numpy as np

def _activate_dummy_pack():
    D = 384
    axes = [
        Axis("autonomy", np.eye(D, dtype=np.float32)[0], 999.0, {}),  # high τ to avoid breaches
        Axis("truth",    np.eye(D, dtype=np.float32)[1], 999.0, {}),
    ]
    ACTIVE["pack"] = AxisPack(id="dev", axes=axes, dim=D, meta={})

def test_eval_text_flow(api_client: TestClient):
    _activate_dummy_pack()
    r = api_client.post("/v1/eval/text", json={"text":"hello world", "window":8, "stride":4})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["proof"]["final"]["action"] in ("allow","refuse")
