from __future__ import annotations

import logging, time
from fastapi import FastAPI
from fastapi import Depends
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import uuid
import contextvars

from coherence.api.routers import health, axes
from coherence.api.routers import index as index_router
from coherence.api.routers import search as search_router
from coherence.api.routers import whatif as whatif_router
from coherence.api.routers import analyze as analyze_router
from coherence.api.routers import embed as embed_router
from coherence.api.routers import resonance as resonance_router
from coherence.api.routers import pipeline as pipeline_router
from coherence.api.routers import v1_axes as v1_axes_router
from coherence.api.axis_registry import init_registry
from coherence.encoders.text_sbert import get_default_encoder
from coherence.cfg.loader import load_app_config
from coherence.cfg.logging import configure_logging


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Loads logging and application config, then mounts routers.
    """
    configure_logging()
    cfg = load_app_config()
    log = logging.getLogger("coherence.api")
    t0 = time.perf_counter()
    log.info("create_app: start")
    app = FastAPI(title="Coherence API", version="0.0.1")

    # Simple Request-ID propagation
    request_id_var = contextvars.ContextVar("request_id", default="-")

    class RequestIDMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
            request_id_var.set(rid)
            response = await call_next(request)
            response.headers["X-Request-ID"] = rid
            return response

    app.add_middleware(RequestIDMiddleware)

    # Optional CORS for local dev
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, prefix="/health", tags=["health"])
    # Routers from both branches
    app.include_router(embed_router.router, tags=["embed"])
    app.include_router(resonance_router.router, tags=["resonance"])
    app.include_router(pipeline_router.router, prefix="/pipeline", tags=["pipeline"])
    app.include_router(v1_axes_router.router, prefix="/v1/axes", tags=["axes"]) 
    from coherence.api.routers import v1_frames as v1_frames_router  # lazy import to avoid early DB setup
    app.include_router(v1_frames_router.router, prefix="/v1/frames", tags=["frames"]) 

    app.include_router(axes.router, prefix="/axes", tags=["axes"])
    app.include_router(index_router.router, prefix="/index", tags=["index"])
    app.include_router(search_router.router, prefix="/search", tags=["search"])
    app.include_router(whatif_router.router, prefix="/whatif", tags=["whatif"])
    app.include_router(analyze_router.router, prefix="/analyze", tags=["analyze"])

    # EthicalAI integration (non-fatal if not present)
    try:
        from ethicalai.api.eval import router as ethical_eval_router
        app.include_router(ethical_eval_router)
        from ethicalai.api.axes import router as ethical_axes_router
        app.include_router(ethical_axes_router)
        from ethicalai.api.interaction import router as ethical_interaction_router
        app.include_router(ethical_interaction_router)
    except Exception as e:
        print("EthicalAI router not loaded:", e)

    # TODO: @builder — expand analyze options (multi-τ, gating)

    # Initialize AxisRegistry once at startup using encoder dimension
    try:
        enc = get_default_encoder()
        encoder_dim = enc._model.get_sentence_embedding_dimension()
        init_registry(encoder_dim=encoder_dim)
    except Exception as e:
        logging.getLogger("coherence.api").warning(f"Registry init skipped (encoder load failed): {e}")

    log.info("App created with config: %s", {"server": cfg.get("server", {}), "api": cfg.get("api", {})})
    log.info("create_app: done in %.3fs", time.perf_counter() - t0)
    return app


app = create_app()
