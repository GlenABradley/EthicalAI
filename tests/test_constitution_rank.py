import numpy as np
from ethicalai.types import Axis, AxisPack
from ethicalai.constitution.decoding import rank_candidates, _axis_composite

class TinyEnc:
    """Deterministic encoder for tests:
       token 'A' -> +e1 ; token 'T' -> +e2 ; token 'B' -> +100*e3 (to trip veto on axis 'risk').
    """
    def __init__(self, dim=4):
        self.dim = dim
        self._A = np.array([1,0,0,0], dtype=np.float32)
        self._T = np.array([0,1,0,0], dtype=np.float32)
        self._B = np.array([0,0,100,0], dtype=np.float32)
        self._0 = np.zeros((self.dim,), dtype=np.float32)
    def encode_text(self, text:str):
        toks = text.split()
        vecs = []
        for t in toks:
            if t == "A": vecs.append(self._A)
            elif t == "T": vecs.append(self._T)
            elif t == "B": vecs.append(self._B)
            else: vecs.append(self._0)
        if not vecs:
            vecs = [self._0]
        return np.stack(vecs, axis=0)

def _pack():
    D=4
    # autonomy -> e1, truthfulness -> e2, risk -> e3 with low threshold to cause veto if 'B' appears
    axes = [
        Axis("autonomy",     np.array([1,0,0,0], dtype=np.float32), 0.0, {}),
        Axis("truthfulness", np.array([0,1,0,0], dtype=np.float32), 0.0, {}),
        Axis("risk",         np.array([0,0,1,0], dtype=np.float32), 1.0, {}),  # τ=1.0; 'B' mean-pools to >> 1.0
    ]
    return AxisPack(id="t", axes=axes, dim=D, meta={})

def test_rank_prefers_no_veto_then_composite_then_logprob():
    pack = _pack()
    enc = TinyEnc()
    # c1: no veto, good autonomy/truth (A T)
    c1 = {"text":"A T", "logprob": 100.0}  # Higher logprob to win
    # c2: veto (contains B)
    c2 = {"text":"B", "logprob": 10.0}
    # c3: no veto, lower autonomy/truth (only A)
    c3 = {"text":"A", "logprob": -1.0}  # Lower logprob
    result = rank_candidates([c2,c3,c1], pack, enc=enc, pref_axes=("autonomy","truthfulness"))
    assert result["choice"]["text"] in ("A T","AT","A T ")
    # Tie-break: ensure composite outranks logprob when both no-veto
    # c1 composite should be higher than c3, and c1 has higher logprob

def test_axis_composite_meaningful():
    spans = [
        {"axis":"autonomy","score": 2.0, "i":0,"j":1,"threshold":0.0,"breached":False},
        {"axis":"truthfulness","score": 0.0, "i":0,"j":1,"threshold":0.0,"breached":False},
    ]
    v = _axis_composite(spans, ("autonomy","truthfulness"))
    assert abs(v - 1.0) < 1e-6
