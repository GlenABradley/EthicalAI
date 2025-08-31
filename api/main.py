"""Main FastAPI application module."""
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from . import __version__
from .routes import router

app = FastAPI(
    title="Coherence API",
    description="API for semantic analysis and axis operations",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Include the router with API version prefix
app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    """Root endpoint that provides API information."""
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

# Add health check at root level
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"status": "ok", "version": __version__}
    )
