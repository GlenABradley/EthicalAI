from __future__ import annotations

"""Thin, function-style wrappers for agent use.

These wrap internal services to be callable directly by an agent runtime.
Rules (embedded):
- Always return vectors (alpha,u,r,U) and C for spans where applicable.
- Do not answer from memory: call services in this process.
- If index missing: raise ValueError with guidance.
- For NL queries, use u_from_nl.
- If filters empty results, relax thresholds by 20% once.
"""
from typing import Dict, List, Any
import numpy as np

from coherence.api.models import CreateAxisPack
from coherence.api.routers import axes as axes_router
from coherence.axis.pack import AxisPack
from coherence.index.pipeline import run_index
from coherence.index.ann import has_index
from coherence.agent.query_map import u_from_nl
from pathlib import Path


def create_axis_pack(payload: CreateAxisPack) -> Dict[str, Any]:
    resp = axes_router.create_pack(payload)
    return resp.model_dump()


def index_texts(payload: Dict[str, Any]) -> Dict[str, Any]:
    axis_pack_id = payload["axis_pack_id"]
    docs = payload["texts"]
    options = payload.get("options", {})
    return run_index(axis_pack_id=axis_pack_id, docs=docs, options=options)


def search(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Agent search calling in-process API router for full scoring.

    Returns the same shape as /search response.
    """
    from coherence.api.routers.search import search as api_search
    from coherence.api.models import SearchRequest, SearchFilters, SearchHyper

    axis_pack_id: str = payload["axis_pack_id"]
    if not has_index(axis_pack_id):
        raise ValueError("Index not built for axis_pack_id; call /index first.")

    req = SearchRequest(
        axis_pack_id=axis_pack_id,
        query=payload["query"],
        filters=payload.get("filters", SearchFilters()),
        hyper=payload.get("hyper", SearchHyper()),
        top_k=int(payload.get("top_k", 10)),
    )
    resp = api_search(req)
    return resp.model_dump()
