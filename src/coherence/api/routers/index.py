from __future__ import annotations

from fastapi import APIRouter, HTTPException
from coherence.api.models import IndexRequest, IndexResponse

router = APIRouter()


@router.post("", response_model=IndexResponse)
def index_docs(req: IndexRequest) -> IndexResponse:
    """Index documents for a given axis pack id.

    Returns which doc_ids were indexed and whether ANN was built.
    """
    # Lazy import to avoid importing pyarrow at module import time
    from coherence.index.pipeline import run_index
    try:
        res = run_index(req.axis_pack_id, [d.model_dump() for d in req.texts], req.options)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return IndexResponse(**res)
