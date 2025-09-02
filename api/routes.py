"""API routes for the Coherence service.

This module aggregates all API routes and sub-routers for the Coherence API.
It acts as the main routing hub, organizing endpoints by version and functionality.

The module structure supports:
- API versioning (v1, v2, etc.)
- Modular endpoint organization
- Clean separation of concerns
- Easy route discovery and management

Current API structure:
- /v1/axes: Axis pack operations and ethical evaluation
- /v1/frames: Frame storage and semantic memory (if enabled)
"""
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any

# Import the v1 axes router directly from the coherence module
# This router handles all axis-related operations including:
# - Creating and managing axis packs
# - Text analysis and ethical evaluation
# - Vector operations and projections
from coherence.api.routers.v1_axes import router as v1_axes_router

# Create the main API router that will aggregate all sub-routers
router = APIRouter()

# Include the v1 axes router with versioned prefix
# This maintains clean API versioning and allows future expansion
router.include_router(
    v1_axes_router, 
    prefix="/v1/axes",  # Version 1 axes endpoints
    tags=["v1_axes"]     # OpenAPI documentation tag
)

# Note: The duplicate /health and /axes endpoints have been removed
# to avoid conflicts with the versioned API structure
