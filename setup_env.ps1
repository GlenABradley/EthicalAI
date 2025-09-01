# Set environment variables
$env:PYTHONPATH = "$PWD"
$pythonPath = "C:\Users\glen4\AppData\Local\Programs\Python\Python312\python.exe"

# Install required packages
Write-Host "Installing required packages..."
& $pythonPath -m pip install --upgrade pip
& $pythonPath -m pip install pytest fastapi uvicorn httpx pytest-asyncio pytest-cov

# Run the tests
Write-Host "Running tests..."
& $pythonPath -m pytest tests/test_minimal.py -v

# Check if the test file exists and run it directly if it does
if (Test-Path "tests/test_end_to_end.py") {
    Write-Host "Running end-to-end tests..."
    & $pythonPath -m pytest tests/test_end_to_end.py -v
}

Write-Host "Done!"
