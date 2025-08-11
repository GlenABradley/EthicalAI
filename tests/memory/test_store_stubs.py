from __future__ import annotations

import tempfile
from pathlib import Path

from coherence.memory.store import create_store


def test_store_put_and_k_infer_and_stubs():
    db = Path(tempfile.mkdtemp()) / "frames.sqlite"
    store = create_store(db)

    frames = [
        {"id": "f0", "predicate": [0, 1], "roles": {}, "coords": [0, 0, 0, 0, 0], "role_coords": {"predicate": [0, 0, 0, 0, 0]}},
        {"id": "f1", "predicate": [2, 3], "roles": {}, "coords": [0.1, 0.2, 0.3, 0.4, 0.5], "role_coords": {"predicate": [1, 1, 1, 1, 1]}},
    ]

    ing = store.put(
        doc_id="docA",
        frames=frames,
        frame_vectors=[[0.0] * (3 * 2), [1.0] * (3 * 2)],
        pack_id="packX",
        pack_hash="hashX",
        k=5,
        d=2,
    )
    assert ing == 2

    # Search/trace smoke
    assert isinstance(store.search(axis_idx=0, min_val=-1.0, max_val=1.0, limit=10), list)
    assert isinstance(store.trace(entity_str="x", limit=10), list)
