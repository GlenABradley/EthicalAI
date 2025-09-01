from __future__ import annotations
import argparse, json, pathlib
import numpy as np
from typing import Dict, List, Tuple, Iterable
from ..types import AxisPack, Axis
from ..encoders import get_encoder, align_dim

def pick_thresholds(pack: AxisPack, scores: Dict[str, List[Tuple[float,int]]], fpr_max: float=0.05) -> AxisPack:
    """scores[axis] = [(score, label{0/1}), ...] ; set ax.threshold via simple ROC sweep."""
    for ax in pack.axes:
        pts = sorted(scores.get(ax.name, []))
        if not pts:
            ax.threshold = 0.0
            continue
        # candidate taus are observed scores
        best_tau, best_f1 = 0.0, -1.0
        for tau in [p[0] for p in pts]:
            tp = sum(1 for s,l in pts if s>tau and l==1)
            fp = sum(1 for s,l in pts if s>tau and l==0)
            fn = sum(1 for s,l in pts if s<=tau and l==1)
            tn = sum(1 for s,l in pts if s<=tau and l==0)
            fpr = fp / max(fp+tn,1)
            if fpr <= fpr_max:
                prec = tp / max(tp+fp,1)
                rec  = tp / max(tp+fn,1)
                f1   = 2*prec*rec / max(prec+rec,1e-12)
                if f1 > best_f1:
                    best_f1, best_tau = f1, tau
        ax.threshold = float(best_tau)
    return pack

# -------------------- CLI & utilities --------------------

def _load_pack_from_artifacts(pack_id: str, artifacts_dir: str = "artifacts") -> AxisPack:
    art = pathlib.Path(artifacts_dir)
    npz_path = art / f"axis_pack:{pack_id}.npz"
    meta_path = art / f"axis_pack:{pack_id}.meta.json"
    if not npz_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Pack {pack_id} not found in {art}")
    arrs = np.load(npz_path)
    meta = json.loads(meta_path.read_text())
    axes: List[Axis] = []
    thresholds = meta.get("thresholds", {})
    for name in arrs.files:
        vec = arrs[name]
        tau = float(thresholds.get(name, 0.0))
        axes.append(Axis(name=name, vector=vec, threshold=tau, provenance=meta.get("meta", {})))
    dim = axes[0].vector.shape[0] if axes else 0
    return AxisPack(id=pack_id, axes=axes, dim=dim, meta=meta.get("meta", {}))

def _save_thresholds(pack: AxisPack, artifacts_dir: str = "artifacts") -> None:
    meta_path = pathlib.Path(artifacts_dir) / f"axis_pack:{pack.id}.meta.json"
    meta = {"meta": pack.meta, "thresholds": {ax.name: ax.threshold for ax in pack.axes}}
    meta_path.write_text(json.dumps(meta, indent=2))

def _iter_jsonl(path: pathlib.Path) -> Iterable[Dict]:
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        yield json.loads(line)

def _score_text_per_axis(text: str, pack: AxisPack, encoder) -> Dict[str, float]:
    """
    Embed text -> token embeddings [T,D?]; mean-pool -> [D?]; align to pack.dim; project onto each axis.
    """
    X = encoder.encode_text(text)  # [T,D] or [D]
    v = X if X.ndim == 1 else X.mean(axis=0)
    v = align_dim(v, pack.dim)
    out = {}
    for ax in pack.axes:
        out[ax.name] = float(v @ align_dim(ax.vector, pack.dim))
    return out

def _collect_scores(datasets: List[str], pack: AxisPack, encoder) -> Dict[str, List[Tuple[float,int]]]:
    scores: Dict[str, List[Tuple[float,int]]] = {ax.name: [] for ax in pack.axes}
    for d in datasets:
        p = pathlib.Path(d)
        for ex in _iter_jsonl(p):
            text = ex["text"]
            label = int(ex["label"])
            s = _score_text_per_axis(text, pack, encoder)
            for ax in pack.axes:
                scores[ax.name].append((s[ax.name], label))
    return scores

def _metrics(points: List[Tuple[float,int]]):
    # Simple AUROC/AUPRC estimate via threshold sweep
    if not points:
        return {"auroc": 0.0, "auprc": 0.0}
    pts = sorted(points, key=lambda x: x[0])
    P = sum(l for _,l in pts)
    N = len(pts) - P
    if P == 0 or N == 0:
        return {"auroc": 1.0, "auprc": 1.0}
    # ROC/AUC (rank-sum)
    # Convert to ranks:
    # AUC = (sum of ranks of positives - P(P+1)/2) / (P*N)
    # Use 1-based ranks
    rank_sum = 0
    for idx, (_, label) in enumerate(pts, start=1):
        if label == 1:
            rank_sum += idx
    auroc = (rank_sum - P*(P+1)/2) / (P*N)
    # PR curve via sweep
    auprc = 0.0
    tp=fp=0
    last_recall = 0.0
    for score,label in reversed(pts):  # high->low threshold
        if label==1: tp +=1
        else: fp+=1
        prec = tp / max(tp+fp,1)
        rec  = tp / max(P,1)
        auprc += prec * max(rec - last_recall, 0)
        last_recall = rec
    return {"auroc": float(auroc), "auprc": float(auprc)}

def _sanitize_for_fs(name: str) -> str:
    """Make a name safe for cross-platform filesystem usage.

    Keep alnum and ._-, replace everything else with '_'.
    """
    allowed = set("._-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    return "".join((ch if ch in allowed else "_") for ch in name)

def main(argv: List[str] | None = None):
    ap = argparse.ArgumentParser(description="Calibrate axis thresholds from labeled datasets.")
    ap.add_argument("--pack", required=True, help="Axis pack id (saved in artifacts/)")
    ap.add_argument("--dataset", action="append", required=True, help="Path to a calibration JSONL file (can repeat)")
    ap.add_argument("--reports", default="reports", help="Output directory for calibration reports")
    ap.add_argument("--fpr-max", type=float, default=0.05, help="Max false positive rate when selecting thresholds")
    args = ap.parse_args(argv)

    pack = _load_pack_from_artifacts(args.pack)
    enc = get_encoder()
    scores = _collect_scores(args.dataset, pack, enc)
    # Metrics before
    metrics_before = {ax.name: _metrics(scores[ax.name]) for ax in pack.axes}
    # Fit thresholds
    pack = pick_thresholds(pack, scores, fpr_max=args.fpr_max)
    _save_thresholds(pack)
    # Metrics after (same scores, updated thresholds used only at inference; keep for report symmetry)
    metrics_after = {ax.name: _metrics(scores[ax.name]) for ax in pack.axes}

    safe_pack_id = _sanitize_for_fs(pack.id)
    outdir = pathlib.Path(args.reports) / f"calibration_{safe_pack_id}"
    outdir.mkdir(parents=True, exist_ok=True)
    # Save per-axis CSV for audit
    for ax in pack.axes:
        arr = np.array(scores[ax.name], dtype=float)
        np.savetxt(outdir / f"{ax.name}.csv", arr, delimiter=",", header="score,label", comments="")
    # Save summary JSON
    summary = {
        "pack_id": pack.id,
        "fpr_max": args.fpr_max,
        "thresholds": {ax.name: ax.threshold for ax in pack.axes},
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "datasets": args.dataset,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"ok": True, **summary}, indent=2))

if __name__ == "__main__":
    main()
