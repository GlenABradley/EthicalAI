import pytest
import numpy as np

from coherence.axis.builder import build_axis_pack_from_vectors
from coherence.axis.pack import AxisPack

def test_axis_pack_builds_and_is_orthonormal(tmp_path):
    rng = np.random.default_rng(123)
    d = 8
    # Create synthetic seeds for two axes with clear separation
    # axis A roughly along e0
    pos_A = [np.eye(d, dtype=np.float32)[0] + 0.01 * rng.standard_normal(d).astype(np.float32) for _ in range(5)]
    neg_A = [np.eye(d, dtype=np.float32)[1] + 0.01 * rng.standard_normal(d).astype(np.float32) for _ in range(5)]
    # axis B roughly along e2
    pos_B = [np.eye(d, dtype=np.float32)[2] + 0.01 * rng.standard_normal(d).astype(np.float32) for _ in range(5)]
    neg_B = [np.eye(d, dtype=np.float32)[3] + 0.01 * rng.standard_normal(d).astype(np.float32) for _ in range(5)]

    seeds_vecs = {
        "axis_A": (pos_A, neg_A),
        "axis_B": (pos_B, neg_B),
    }

    pack: AxisPack = build_axis_pack_from_vectors(seeds_vecs)
    # Shapes
    assert pack.Q.shape == (d, 2)
    assert len(pack.names) == 2
    # Orthonormality
    qtq = pack.Q.T @ pack.Q
    assert np.allclose(qtq, np.eye(2), atol=1e-5)

    # IO round-trip
    outp = tmp_path / "pack.json"
    pack.save(outp)
    loaded = AxisPack.load(outp)
    assert loaded.names == pack.names
    assert loaded.Q.shape == pack.Q.shape
    assert np.allclose(loaded.Q.T @ loaded.Q, np.eye(2), atol=1e-5)
