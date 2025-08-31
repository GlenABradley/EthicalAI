import json, os, pathlib
from fastapi.openapi.utils import get_openapi

# If your app lives elsewhere, adjust import below:
from src.coherence.api.main import app  # <— update if needed

out = pathlib.Path("docs")
out.mkdir(parents=True, exist_ok=True)
schema = get_openapi(
    title=app.title if hasattr(app, "title") else "Coherence API",
    version=getattr(app, "version", "0.0.0"),
    routes=app.routes,
)
(out / "openapi.json").write_text(json.dumps(schema, indent=2))
print(f"Wrote {out / 'openapi.json'}")
