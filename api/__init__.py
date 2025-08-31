"""Coherence API module.

This module contains the FastAPI application and API endpoints for the Coherence service.
"""

from fastapi import FastAPI

__version__ = "0.1.0"

app = FastAPI(
    title="Coherence API",
    description="API for semantic analysis and axis operations",
    version=__version__
)

# Import routes to register them with the app
from . import routes  # noqa
