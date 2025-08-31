"""API routes for the Coherence service."""
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any

# Import the v1 axes router directly from the module
from coherence.api.routers.v1_axes import router as v1_axes_router

router = APIRouter()

# Include the v1 axes router with versioned prefix
router.include_router(v1_axes_router, prefix="/v1/axes", tags=["v1_axes"])

@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}

# Removed the duplicate /axes endpoint
