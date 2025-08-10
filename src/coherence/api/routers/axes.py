from __future__ import annotations

# TODO: @builder — Implement axis pack endpoints in Milestone 7.
from fastapi import APIRouter

router = APIRouter()


@router.get("")
def list_packs() -> dict:
    """List available axis packs.
    Placeholder until Milestone 7.
    """
    return {"items": []}
