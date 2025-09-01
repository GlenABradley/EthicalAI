# Create virtual environment if it doesn't exist
if (-not (Test-Path .venv)) {
    python -m venv .venv
}

# Activate virtual environment
.venv\Scripts\Activate.ps1

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install fastapi uvicorn httpx pytest pytest-asyncio pytest-cov

# Run the tests
$env:PYTHONPATH="$PWD"
python -m pytest tests/test_end_to_end.py -v -s --log-cli-level=DEBUG

# Deactivate virtual environment
Deactivate
