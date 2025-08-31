from ethicalai.constitution.regularizer import regularizer_numpy

def test_regularizer_numpy_basic():
    # Two axes; one score above tau
    span_scores = {
        "autonomy": [(0.2, 0.0), (0.1, 0.0)],  # small penalties
        "risk":     [(1.5, 1.0)],              # larger penalty
    }
    weights = {"autonomy": 0.5, "risk": 2.0}
    val = regularizer_numpy(span_scores, weights, lam=0.1)
    assert val > 0.0
    # Simple sanity range check
    assert val < 1.0
