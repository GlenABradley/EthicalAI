# Test runner script for EthicalAI

# Set error action preference
$ErrorActionPreference = "Stop"

# Function to print with timestamp
function Write-TimestampedMessage {
    param (
        [string]$Message,
        [string]$Color = "White"
    )
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] $Message" -ForegroundColor $Color
}

try {
    # Check if Python is available
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Python is not installed or not in PATH"
    }
    Write-TimestampedMessage "Using Python: $pythonVersion" -Color "Green"

    # Check if pip is available
    $pipVersion = pip --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "pip is not installed or not in PATH"
    }
    Write-TimestampedMessage "Using pip: $($pipVersion -split ' ')[1]" -Color "Green"

    # Install test dependencies
    Write-TimestampedMessage "Installing test dependencies..." -Color "Cyan"
    pip install -r requirements.txt
    pip install pytest pytest-asyncio aiohttp pytest-cov

    # Install the package in development mode
    Write-TimestampedMessage "Installing package in development mode..." -Color "Cyan"
    pip install -e .

    # Run the tests with coverage
    Write-TimestampedMessage "Running tests with coverage..." -Color "Cyan"
    $testResults = python -m pytest tests/ -v --cov=src --cov-report=term-missing

    # Display test results
    Write-Host $testResults -ForegroundColor "Green"

    # Generate HTML coverage report
    Write-TimestampedMessage "Generating HTML coverage report..." -Color "Cyan"
    python -m pytest --cov=src --cov-report=html:htmlcov

    Write-TimestampedMessage "Test execution completed successfully!" -Color "Green"
    Write-TimestampedMessage "Coverage report available at: $PWD\htmlcov\index.html" -Color "Green"
}
catch {
    Write-TimestampedMessage "Error: $_" -Color "Red"
    exit 1
}
