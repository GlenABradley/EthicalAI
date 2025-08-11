import numpy as np
from coherence.axis.builder import build_axis_pack_from_vectors
from coherence.axis.pack import AxisPack

def test_384d_axis_pack():
    """Test creating and using a 384D axis pack that matches the encoder's output dimension."""
    d = 384  # Dimension of encoder output
    k = 2    # Number of axes
    rng = np.random.default_rng(42)
    
    # Create synthetic seeds with 384D vectors
    # axis_1 roughly along first dimension
    pos_1 = [rng.normal(scale=0.1, size=d).astype(np.float32) for _ in range(5)]
    pos_1 = [v + np.array([1.0] + [0.0]*(d-1)) for v in pos_1]  # Bias along first dimension
    neg_1 = [rng.normal(scale=0.1, size=d).astype(np.float32) for _ in range(5)]
    neg_1 = [v - np.array([1.0] + [0.0]*(d-1)) for v in neg_1]  # Bias opposite first dimension
    
    # axis_2 roughly along second dimension
    pos_2 = [rng.normal(scale=0.1, size=d).astype(np.float32) for _ in range(5)]
    pos_2 = [v + np.array([0.0, 1.0] + [0.0]*(d-2)) for v in pos_2]  # Bias along second dimension
    neg_2 = [rng.normal(scale=0.1, size=d).astype(np.float32) for _ in range(5)]
    neg_2 = [v - np.array([0.0, 1.0] + [0.0]*(d-2)) for v in neg_2]  # Bias opposite second dimension
    
    seeds_vecs = {
        "axis_1": (pos_1, neg_1),
        "axis_2": (pos_2, neg_2),
    }

    # Build the axis pack
    pack: AxisPack = build_axis_pack_from_vectors(seeds_vecs)
    
    # Validate shapes and orthonormality
    assert pack.Q.shape == (d, k), f"Expected Q shape (384, 2), got {pack.Q.shape}"
    assert len(pack.names) == k, f"Expected {k} axis names, got {len(pack.names)}"
    
    # Check orthonormality
    qtq = pack.Q.T @ pack.Q
    assert np.allclose(qtq, np.eye(k), atol=1e-5), "Q columns are not orthonormal"
    
    # Test projecting a random vector
    test_vec = rng.normal(size=d).astype(np.float32)
    projection = pack.Q.T @ test_vec
    assert projection.shape == (k,), f"Expected projection shape ({k},), got {projection.shape}"
    
    # Test projecting multiple vectors
    test_vecs = rng.normal(size=(3, d)).astype(np.float32)
    projections = test_vecs @ pack.Q
    assert projections.shape == (3, k), f"Expected projections shape (3, {k}), got {projections.shape}"
    
    print("✓ 384D axis pack test passed")
