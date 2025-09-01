# Test runner script for EthicalAI UI
# This script ensures a clean test environment by managing the dev server lifecycle

# Configuration
$PORT = 3000
$TEST_COMMAND = "npx playwright test tests/navigation.spec.ts --timeout=30000 --workers=1"
$DEV_SERVER_CMD = "npm run dev -- --port $PORT"

# Kill any existing node processes
Write-Host "[1/4] Cleaning up existing processes..."
taskkill /F /IM node.exe 2>&1 | Out-Null
Start-Sleep -Seconds 2  # Give it a moment to clean up

# Start the development server
Write-Host "[2/4] Starting development server on port $PORT..."
$serverProcess = Start-Process -NoNewWindow -PassThru -FilePath "npm" -ArgumentList "run dev -- --port $PORT"

# Wait for server to be ready
Write-Host "[3/4] Waiting for server to be ready..."
$maxAttempts = 30
$attempt = 0
$serverReady = $false

while ($attempt -lt $maxAttempts) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:$PORT" -Method Head -UseBasicParsing -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            $serverReady = $true
            break
        }
    } catch {
        # Server not ready yet
    }
    $attempt++
    Start-Sleep -Seconds 1
}

if (-not $serverReady) {
    Write-Error "Failed to start development server on port $PORT"
    exit 1
}

Write-Host "[4/4] Running tests..."
# Run the tests
$testProcess = Start-Process -NoNewWindow -PassThru -FilePath "npx" -ArgumentList "playwright test tests/navigation.spec.ts --timeout=30000 --workers=1" -Wait -PassThru

# Clean up
Write-Host "Cleaning up..."
if ($serverProcess) {
    Stop-Process -Id $serverProcess.Id -Force -ErrorAction SilentlyContinue
}

# Exit with the test process exit code
exit $testProcess.ExitCode
