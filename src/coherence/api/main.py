from __future__ import annotations

import logging
from fastapi import FastAPI

from coherence.api.routers import health
from coherence.api.routers import embed as embed_router
from coherence.api.routers import resonance as resonance_router
from coherence.api.routers import pipeline as pipeline_router
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
    app.include_router(embed_router.router, tags=["embed"])
    app.include_router(resonance_router.router, tags=["resonance"])
    app.include_router(pipeline_router.router, prefix="/pipeline", tags=["pipeline"])

    # TODO: @builder — in later milestones add axes/analyze/counterfactual routers

    log = logging.getLogger("coherence.api")
    log.info("App created with config: %s", {"server": cfg.get("server", {}), "api": cfg.get("api", {})})
    return app


app = create_app()
