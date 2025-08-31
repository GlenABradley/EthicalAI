"""API routers for the Coherence service."""

# Import all routers to make them available when importing from this package
from .v1_axes import router as axes_router
from .v1_axes import router as v1_axes_router
from .pipeline import router as pipeline_router
from .resonance import router as resonance_router

__all__ = [
    'axes_router',
    'v1_axes_router',
    'pipeline_router',
    'resonance_router',
]
