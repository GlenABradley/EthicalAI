from __future__ import annotations

from fastapi import APIRouter
from typing import Dict

router = APIRouter()


@router.get("/ready")
def ready() -> Dict[str, str]:
    """Readiness probe endpoint.

    Returns
    - JSON {"status": "ok"}
    """
    return {"status": "ok"}
