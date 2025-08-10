from __future__ import annotations

import logging
from fastapi import FastAPI

from coherence.api.routers import health
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

    # TODO: @builder — in later milestones add axes/analyze/counterfactual routers

    log = logging.getLogger("coherence.api")
    log.info("App created with config: %s", {"server": cfg.get("server", {}), "api": cfg.get("api", {})})
    return app


app = create_app()
