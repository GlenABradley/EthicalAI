from __future__ import annotations

from typing import Dict, List, Any
from pathlib import Path
import numpy as np

from coherence.axis.pack import AxisPack
from coherence.encoders.registry import get_encoder
from coherence.metrics.resonance import project, utilities, aggregate
from coherence.coherence.skipmesh import span_coherence
from coherence.index.ann import init_index, add as ann_add
from coherence.index.store import write_spans, write_frames
from coherence.frames.srl_lite import build_frames
from coherence.cfg.loader import load_app_config


def _tokenize(text: str, mode: str) -> List[str]:
    if mode == "simple_split":
        return text.split()
    # TODO(@builder): spaCy later
    return text.split()


def run_index(axis_pack_id: str, docs: List[Dict[str, str]], options: Dict[str, Any]) -> Dict[str, Any]:
    cfg = load_app_config()
    index_cfg = cfg.get("index", {})
    search_cfg = cfg.get("search", {})
    tokenizer = index_cfg.get("tokenizer", "simple_split")
    span_window = int(index_cfg.get("span_window", 64))
    span_stride = int(index_cfg.get("span_stride", 32))
    squash = bool(search_cfg.get("squash", True))

    # Load pack
    pack_path = Path("data/axes") / f"{axis_pack_id}.json"
    if not pack_path.exists():
        raise ValueError("Axis pack not found. Build via /axes/create or seed script.")
    pack = AxisPack.load(pack_path)

    enc = get_encoder()

    # Init ANN on k-dim u vectors
    init_index(axis_pack_id, k=pack.k, backend=cfg.get("ann", {}).get("backend", "numpy"))

    indexed_ids: List[str] = []

    for doc in docs:
        doc_id = doc["doc_id"]
        text = doc["text"]
        tokens = _tokenize(text, tokenizer)
        if not tokens:
            continue
        X = enc.encode(tokens).astype(np.float32)  # (n,d)
        n, d = X.shape

        # Sliding windows
        span_records: List[dict] = []
        ann_items: List[np.ndarray] = []
        ann_ids: List[str] = []
        ann_payloads: List[dict] = []

        for start in range(0, n, span_stride):
            end = min(start + span_window, n)
            if end - start <= 0:
                break
            x = X[start:end].mean(axis=0)  # (d,)
            alpha = project(x, pack)  # (k,)
            u = utilities(alpha, pack)
            if squash:
                u = 1.0 / (1.0 + np.exp(-u))
            U = float(aggregate(u, pack))
            C = float(span_coherence(X, pack, start, end, max_skip=2))
            rec = {
                "doc_id": doc_id,
                "start": start,
                "end": end,
                "text": " ".join(tokens[start:end]),
                "alpha": alpha.astype(np.float32).tolist(),
                "u": u.astype(np.float32).tolist(),
                "r": u.astype(np.float32).tolist(),  # TODO(@builder): gating
                "U": U,
                "C": C,
                "t": 1.0,
                "tau": 0.0,
            }
            span_records.append(rec)
            ann_items.append(u.astype(np.float32))
            ann_ids.append(f"{doc_id}:{start}-{end}")
            ann_payloads.append(rec)

        # Persist and add to ANN
        if span_records:
            write_spans(axis_pack_id, doc_id, span_records)
            ann_add(axis_pack_id, np.vstack(ann_items), ann_ids, ann_payloads)

        # Frames indexing and persistence
        frames = build_frames(X, pack, saliency_thresh=0.0, arg_band=0.5, max_arg_len=2)
        frame_records: List[dict] = []
        for fr in frames:
            # Mean embedding over predicate + all role tokens
            idxs: List[int] = list(range(fr.predicate[0], fr.predicate[1]))
            for _, (s, e) in fr.roles.items():
                idxs.extend(range(s, e))
            idxs = [ix for ix in idxs if 0 <= ix < X.shape[0]]
            if not idxs:
                continue
            x = X[idxs].mean(axis=0)
            a = project(x, pack)
            u = utilities(a, pack)
            if squash:
                u = 1.0 / (1.0 + np.exp(-u))
            U = float(aggregate(u, pack))
            rec_f = {
                "doc_id": doc_id,
                "frame_id": str(fr.id),
                "pred_start": int(fr.predicate[0]),
                "pred_end": int(fr.predicate[1]),
                "roles": {k: [int(s), int(e)] for k, (s, e) in fr.roles.items()},
                "alpha": a.astype(np.float32).tolist(),
                "u": u.astype(np.float32).tolist(),
                "r": u.astype(np.float32).tolist(),
                "U": U,
                "t": 1.0,
                "tau": 0.0,
            }
            frame_records.append(rec_f)
        if frame_records:
            write_frames(axis_pack_id, doc_id, frame_records)

        indexed_ids.append(doc_id)

    return {"indexed": indexed_ids, "anns_built": True, "tau_used": [0.0]}
