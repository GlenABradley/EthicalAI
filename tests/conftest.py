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


@pytest.fixture(scope="session", autouse=True)
def _preload_real_encoder() -> None:
    """Pre-load the real encoder once per test session.

    Forces real model usage in tests (no mocks/stubs) and primes the cache
    so repeated app startups reuse the already-initialized encoder.
    """
    # Ensure real encoder is used even under pytest
    os.environ["COHERENCE_TEST_REAL_ENCODER"] = "1"
    try:
        from coherence.encoders.text_sbert import get_default_encoder
        get_default_encoder()
    except Exception as e:
        print(f"Warning: failed to preload real encoder: {e}")


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


@pytest.fixture(scope="module")
def tmp_artifacts_dir() -> Iterator[Path]:
    tmp = Path(tempfile.mkdtemp(prefix="coh_artifacts_"))
    try:
        os.environ["COHERENCE_ARTIFACTS_DIR"] = str(tmp)
        yield tmp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture(scope="module")
def sample_axis_jsons(tmp_artifacts_dir: Path) -> List[Path]:
    a1 = tmp_artifacts_dir / "a1.json"
    a2 = tmp_artifacts_dir / "a2.json"
    _make_sample_axis(a1, "a1")
    _make_sample_axis(a2, "a2")
    return [a1, a2]


@pytest.fixture(scope="function")
def api_client(tmp_artifacts_dir: Path) -> TestClient:
    # Reset registry and load app fresh
    import coherence.api.axis_registry as axis_registry
    axis_registry.REGISTRY = None

    # Set test environment variables for real encoder usage
    os.environ["COHERENCE_ARTIFACTS_DIR"] = str(tmp_artifacts_dir)
    os.environ["COHERENCE_TEST_REAL_ENCODER"] = "1"
    os.environ["COHERENCE_ENCODER"] = "all-mpnet-base-v2"

    from coherence.api.main import create_app
    app = create_app()
    client = TestClient(app)
    return client


@pytest.fixture(scope="function")
def api_client_real_encoder(tmp_artifacts_dir: Path) -> TestClient:
    """API client fixture that uses the real encoder (no mocking)."""
    import coherence.api.axis_registry as axis_registry
    axis_registry.REGISTRY = None

    os.environ["COHERENCE_ARTIFACTS_DIR"] = str(tmp_artifacts_dir)
    os.environ["COHERENCE_TEST_REAL_ENCODER"] = "1"
    os.environ["COHERENCE_ENCODER"] = "all-mpnet-base-v2"

    from coherence.api.main import create_app
    app = create_app()
    client = TestClient(app)
    return client
