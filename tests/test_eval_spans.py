import numpy as np
from ethicalai.types import Axis, AxisPack
from ethicalai.eval.spans import project_scores

def test_project_scores_shapes():
    D = 16
    X = np.ones((40, D), dtype=np.float32)
    axes = [Axis("a", np.eye(D, dtype=np.float32)[0], 0.0, {}),
            Axis("b", np.eye(D, dtype=np.float32)[1], 0.0, {})]
    pack = AxisPack(id="t", axes=axes, dim=D, meta={})
    spans = project_scores(X, pack, window=10, stride=10)
    # 4 windows × 2 axes
    assert len(spans) == 8
    assert all(set(s.keys()) == {"i","j","axis","score","threshold","breached"} for s in spans)
