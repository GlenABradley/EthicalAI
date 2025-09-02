"""Main FastAPI application module.

This module initializes the FastAPI application for the Coherence API,
which provides semantic analysis and ethical evaluation capabilities.
It sets up the main application instance, configures API documentation,
and includes the versioned API routes.

The API supports:
- Text embedding and vector generation
- Ethical axis evaluation
- Frame storage and retrieval
- Semantic memory operations
"""
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from . import __version__
from .routes import router

# Initialize the FastAPI application with metadata
app = FastAPI(
    title="Coherence API",
    description="API for semantic analysis and axis operations",
    version=__version__,
    docs_url="/docs",  # Interactive API documentation
    redoc_url="/redoc",  # Alternative API documentation
    openapi_url="/openapi.json"  # OpenAPI schema endpoint
)

# Include the main router with API version prefix
# This allows for future API versioning without breaking existing clients
app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    """Root endpoint that provides API information.
    
    Returns:
        dict: API metadata including version, documentation URLs,
              and available endpoint paths.
    """
    return {
        "name": "Coherence API",
        "version": __version__,
        "docs": "/docs",
        "redoc": "/redoc",
        "endpoints": [
            "/health",
            "/api/v1/axes",
            "/api/v1/frames",
            "/docs",
            "/redoc"
        ]
    }

# Health check endpoint for monitoring and service discovery
@app.get("/health")
async def health_check():
    """Health check endpoint for service monitoring.
    
    Used by load balancers, orchestrators, and monitoring systems
    to verify that the service is running and responsive.
    
    Returns:
        JSONResponse: Status 200 with service health information.
    """
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"status": "ok", "version": __version__}
    )
