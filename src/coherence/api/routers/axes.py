"""
Axes Router for EthicalAI API

This module provides API endpoints for managing and creating axis packs in the EthicalAI system.
Axis packs are collections of semantic axes used for analyzing and evaluating text based on
various ethical dimensions.

Endpoints:
- GET /list: List all available axis packs
- GET /{axis_pack_id}: Get details of a specific axis pack
- POST /create: Create a new axis pack from seed phrases
"""
from __future__ import annotations

from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from pathlib import Path
import time
import numpy as np
import logging

from coherence.api.models import CreateAxisPack
from coherence.axis.builder import build_axis_pack_from_seeds
from coherence.axis.pack import AxisPack
from coherence.encoders.registry import get_encoder

# Set up logging
logger = logging.getLogger(__name__)

# Directory to store axis pack JSON files
DATA_AXES_DIR = Path("data/axes")
DATA_AXES_DIR.mkdir(parents=True, exist_ok=True)

# Create FastAPI router with tags for API documentation
router = APIRouter(
    prefix="/axes",
    tags=["axes"],
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Axis pack not found"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid request parameters"}
    }
)


class CreateAxisPackResponse(BaseModel):
    """Response model returned after successfully creating an axis pack.
    
    Attributes:
        axis_pack_id: Unique identifier for the created axis pack
        k: Number of axes in the created pack
        names: List of axis names in the pack
    """
    axis_pack_id: str = Field(..., description="Unique identifier for the created axis pack")
    k: int = Field(..., description="Number of axes in the created pack")
    names: List[str] = Field(..., description="List of axis names in the pack")


@router.get(
    "/list",
    response_model=Dict[str, List[Dict[str, object]]],
    summary="List all available axis packs",
    response_description="List of axis packs with their metadata"
)
async def list_packs() -> Dict[str, List[Dict[str, object]]]:
    """Retrieve a list of all available axis packs.
    
    Scans the data/axes directory for JSON files containing axis pack definitions
    and returns their metadata. Each pack includes its ID, axis names, and 
    dimensionality.
    
    Returns:
        Dictionary with a single key 'items' containing a list of axis pack metadata:
        - id: Unique identifier of the axis pack (filename without .json extension)
        - names: List of axis names in the pack
        - k: Number of axes in the pack
        
    Raises:
        HTTPException: If there's an error reading the axis packs directory
    """
    items: List[Dict[str, object]] = []
    try:
        for p in sorted(DATA_AXES_DIR.glob("*.json")):
            try:
                pack = AxisPack.load(p)
                items.append({
                    "id": p.stem, 
                    "names": pack.names, 
                    "k": pack.k,
                    "description": getattr(pack.meta, 'description', '')
                })
            except Exception as e:
                logger.warning(f"Skipping invalid axis pack {p}: {str(e)}")
                continue
        return {"items": items}
    except Exception as e:
        logger.error(f"Error listing axis packs: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list axis packs"
        )


@router.get(
    "/{axis_pack_id}",
    response_model=Dict[str, object],
    summary="Get axis pack by ID",
    responses={
        200: {"description": "Axis pack details"},
        404: {"description": "Axis pack not found"}
    }
)
async def get_pack(
    axis_pack_id: str = Path(..., description="ID of the axis pack to retrieve")
) -> Dict[str, object]:
    """Retrieve detailed information about a specific axis pack.
    
    Args:
        axis_pack_id: The unique identifier of the axis pack to retrieve
        
    Returns:
        Dictionary containing the axis pack details:
        - id: The axis pack ID (same as the input parameter)
        - names: List of axis names in the pack
        - k: Number of axes in the pack
        - d: Dimensionality of the embedding space
        - meta: Additional metadata associated with the axis pack
        
    Raises:
        HTTPException: 404 if the specified axis pack is not found
    """
    f = DATA_AXES_DIR / f"{axis_pack_id}.json"
    if not f.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Axis pack '{axis_pack_id}' not found"
        )
    
    try:
        pack = AxisPack.load(f)
        return {
            "id": axis_pack_id,
            "names": pack.names,
            "k": pack.k,
            "d": pack.d,
            "meta": pack.meta,
            "created_at": getattr(pack.meta, 'created_at', None),
            "version": getattr(pack.meta, 'version', '1.0.0')
        }
    except Exception as e:
        logger.error(f"Error loading axis pack {axis_pack_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load axis pack"
        )


@router.post(
    "/create",
    response_model=CreateAxisPackResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new axis pack",
    responses={
        201: {"description": "Axis pack created successfully"},
        400: {"description": "Invalid input parameters"},
        500: {"description": "Failed to create axis pack"}
    }
)
async def create_pack(
    payload: CreateAxisPack = Body(
        ...,
        example={
            "axes": [
                {
                    "name": "ethical_concern",
                    "positives": ["helpful", "beneficial", "moral"],
                    "negatives": ["harmful", "damaging", "unethical"]
                }
            ],
            "lambda_": [1.0],
            "beta": [0.0],
            "weights": [1.0]
        }
    )
) -> CreateAxisPackResponse:
    """Create a new axis pack from seed phrases and save it to disk.
    
    This endpoint creates a new semantic axis pack using the provided seed words.
    It uses the default encoder (configured in settings) and a diff-of-means
    builder to create the semantic axes.
    
    Args:
        payload: CreateAxisPack model containing:
            - axes: List of axis definitions with names and seed words
            - lambda_: Optional list of lambda values for each axis (default: 1.0)
            - beta: Optional list of beta values for each axis (default: 0.0)
            - weights: Optional list of weights for each axis (default: 1.0)
            
    Returns:
        CreateAxisPackResponse containing the ID, number of axes, and axis names
        
    Raises:
        HTTPException: 400 if input validation fails
        HTTPException: 500 if there's an error creating the axis pack
    """
    if not payload.axes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No axes provided in the request"
        )

    try:
        # Prepare seeds mapping expected by builder
        seeds = {a.name: {"positive": a.positives, "negative": a.negatives} for a in payload.axes}

        # Get the default encoder from configuration
        enc = get_encoder()
        
        # Build the axis pack from seed words
        pack = build_axis_pack_from_seeds(
            seeds,
            encode_fn=enc.encode,
            lambda_init=1.0 if payload.lambda_ is None else None,
            beta_init=0.0 if payload.beta is None else None,
            weights_init=None if payload.weights is None else payload.weights,
            meta={
                "built_from": "api.create",
                "created_at": time.time(),
                "version": "1.0.0",
                "description": "Generated via API"
            },
        )

        # Apply overrides if provided
        if payload.lambda_ is not None:
            if len(payload.lambda_) != pack.k:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"lambda_ length ({len(payload.lambda_)}) must equal number of axes ({pack.k})"
                )
            pack.lambda_ = np.asarray(payload.lambda_, dtype=np.float32)
            
        if payload.beta is not None:
            if len(payload.beta) != pack.k:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"beta length ({len(payload.beta)}) must equal number of axes ({pack.k})"
                )
            pack.beta = np.asarray(payload.beta, dtype=np.float32)
            
        if payload.weights is not None:
            if len(payload.weights) != pack.k:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"weights length ({len(payload.weights)}) must equal number of axes ({pack.k})"
                )
            pack.weights = np.asarray(payload.weights, dtype=np.float32)

        # Generate a unique ID and save the pack
        ts = int(time.time())
        base = f"ap_{ts}"
        out_path = DATA_AXES_DIR / f"{base}.json"
        
        # Ensure the output directory exists
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the pack to disk
        pack.save(out_path)
        logger.info(f"Created new axis pack: {base} with {pack.k} axes")
        
        return CreateAxisPackResponse(
            axis_pack_id=base,
            k=pack.k,
            names=pack.names
        )
        
    except Exception as e:
        logger.error(f"Error creating axis pack: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create axis pack: {str(e)}"
        )
