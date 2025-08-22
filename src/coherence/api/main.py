from __future__ import annotations

import logging
from fastapi import FastAPI

from coherence.api.routers import health, axes
from coherence.api.routers import index as index_router
from coherence.api.routers import search as search_router
from coherence.api.routers import whatif as whatif_router
from coherence.api.routers import analyze as analyze_router
from coherence.cfg.loader import load_app_config
from coherence.cfg.logging import configure_logging


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Loads logging and application config, then mounts routers.
    """
    configure_logging()
    cfg = load_app_config()
    app = FastAPI(title="Coherence API", version="0.0.1")

    app.include_router(health.router, prefix="/health", tags=["health"])
    app.include_router(axes.router, prefix="/axes", tags=["axes"])
    app.include_router(index_router.router, prefix="/index", tags=["index"])
    app.include_router(search_router.router, prefix="/search", tags=["search"])
    app.include_router(whatif_router.router, prefix="/whatif", tags=["whatif"])
    app.include_router(analyze_router.router, prefix="/analyze", tags=["analyze"])

    # TODO: @builder — expand analyze options (multi-τ, gating)

    log = logging.getLogger("coherence.api")
    log.info("App created with config: %s", {"server": cfg.get("server", {}), "api": cfg.get("api", {})})
    return app


app = create_app()

