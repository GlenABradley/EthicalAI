import sys
import os
import json
import shutil
import tempfile
import importlib
from pathlib import Path
from typing import Iterator, List

import pytest
from fastapi.testclient import TestClient

# Ensure src/ is importable when running tests without installation
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _make_sample_axis(path: Path, name: str):
    data = {
        "name": name,
        "inclusive_mode": False,
        "plain_language_ontology": "test axis",
        "max_examples": [
            "maximize good",
            "increase welfare",
        ],
        "min_examples": [
            "cause harm",
            "reduce autonomy",
        ],
        "weight": 1.0,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


@pytest.fixture()
def tmp_artifacts_dir() -> Iterator[Path]:
    tmp = Path(tempfile.mkdtemp(prefix="coh_artifacts_"))
    try:
        os.environ["COHERENCE_ARTIFACTS_DIR"] = str(tmp)
        yield tmp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture()
def sample_axis_jsons(tmp_artifacts_dir: Path) -> List[Path]:
    a1 = tmp_artifacts_dir / "a1.json"
    a2 = tmp_artifacts_dir / "a2.json"
    _make_sample_axis(a1, "a1")
    _make_sample_axis(a2, "a2")
    return [a1, a2]


@pytest.fixture()
def api_client(tmp_artifacts_dir: Path) -> TestClient:
    # Reset registry and load app fresh
    import coherence.api.axis_registry as axis_registry
    axis_registry.REGISTRY = None
    # Reload frames router to pick up new env and recreate STORE at correct DB path
    try:
        vr = importlib.import_module("coherence.api.routers.v1_frames")
        importlib.reload(vr)
    except Exception:
        pass
    mod = importlib.import_module("coherence.api.main")
    importlib.reload(mod)
    app = getattr(mod, "app", None)
    if app is None:
        # Use factory if module-level app is not exported
        app = mod.create_app()
    client = TestClient(app)
    # Trigger lazy init of registry for health
    client.get("/health/ready")
    return client
