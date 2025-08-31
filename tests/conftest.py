import sys
import os
import json
import shutil
import tempfile
import importlib
import unittest.mock
from pathlib import Path
from typing import Iterator, List
from unittest.mock import MagicMock

import pytest
try:
    import numpy as np
except ImportError:
    print("Warning: numpy not installed. Some tests may fail.")
    np = None
from fastapi.testclient import TestClient

# Ensure src/ is importable when running tests without installation
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session", autouse=True)
def _preload_real_encoder_if_requested():
    """Optionally pre-load the real encoder once per test session.

    Enable by setting COHERENCE_TEST_REAL_ENCODER to true/1/yes.
    This avoids re-initializing the SentenceTransformer for each test case.
    """
    use_real = os.getenv("COHERENCE_TEST_REAL_ENCODER", "").lower() in ("1", "true", "yes")
    if use_real:
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
    
    # Set test environment variables
    os.environ["COHERENCE_ARTIFACTS_DIR"] = str(tmp_artifacts_dir)
    os.environ["COHERENCE_TEST_MODE"] = "true"
    os.environ["COHERENCE_ENCODER"] = "all-mpnet-base-v2"
    
    # Allow opting into real encoder via env var without changing tests
    use_real = os.getenv("COHERENCE_TEST_REAL_ENCODER", "").lower() in ("1", "true", "yes")
    if use_real:
        from coherence.api.main import create_app
        app = create_app()
        client = TestClient(app)
        return client
    
    # Mock the encoder to prevent model downloading during tests
    from coherence.encoders.text_sbert import SBERTEncoder
    
    # Create a mock encoder that doesn't actually load the model
    mock_encoder = MagicMock()
    # Expose minimal _model API expected by health.create_app/init
    mock_encoder._model = MagicMock()
    mock_encoder._model.get_sentence_embedding_dimension.return_value = 768
    # Proper encode() implementation returning arrays/lists
    def mock_encode(texts):
        if np is not None:
            return np.random.rand(len(texts), 768).astype(np.float32)
        else:
            return [[0.1] * 768 for _ in texts]
    mock_encoder.encode.side_effect = mock_encode
    mock_encoder.model_name = "all-mpnet-base-v2"
    mock_encoder.device = "cpu"
    
    with unittest.mock.patch('coherence.encoders.text_sbert.get_default_encoder', return_value=mock_encoder):
        # Import and create app with mocked encoder
        from coherence.api.main import create_app
        app = create_app()
        
        # Create test client
        client = TestClient(app)
        
        return client


@pytest.fixture(scope="function")
def api_client_real_encoder(tmp_artifacts_dir: Path) -> TestClient:
    """API client fixture that uses the real encoder (no mocking)."""
    # Reset registry and load app fresh
    import coherence.api.axis_registry as axis_registry
    axis_registry.REGISTRY = None
    
    # Set test environment variables
    os.environ["COHERENCE_ARTIFACTS_DIR"] = str(tmp_artifacts_dir)
    os.environ["COHERENCE_TEST_MODE"] = "true"
    os.environ["COHERENCE_ENCODER"] = "all-mpnet-base-v2"
    
    # Import and create app with real encoder (no mocking)
    from coherence.api.main import create_app
    app = create_app()
    
    # Create test client
    client = TestClient(app)
    
    return client
