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
    
    # Mock the encoder to prevent model downloading during tests
    from coherence.encoders.text_sbert import SBERTEncoder
    
    # Create a mock encoder that doesn't actually load the model
    mock_encoder = MagicMock()
    def mock_encode(texts):
        # Return proper numpy array with shape (num_texts, embedding_dim)
        # Mock the encoder to return proper numpy arrays
        if np is not None:
            mock_encoder.encode.return_value = np.random.rand(len(texts), 384).astype(np.float32)
        else:
            # Fallback if numpy is not available
            mock_encoder.encode.return_value = [[0.1] * 384 for _ in texts]
        mock_encoder._model.get_sentence_embedding_dimension.return_value = 384  
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
