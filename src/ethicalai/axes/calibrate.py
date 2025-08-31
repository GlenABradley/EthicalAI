from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple
from ..types import AxisPack

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
