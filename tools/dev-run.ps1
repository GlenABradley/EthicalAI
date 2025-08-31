param([string]$host = '127.0.0.1', [int]$port = 8080)

if (Test-Path .env) {
  Write-Host 'Loading .env'
  foreach ($line in Get-Content .env) {
    if (-not [string]::IsNullOrWhiteSpace($line) -and -not $line.Trim().StartsWith('#')) {
      $parts = $line.Split('=',2)
      if ($parts.Count -eq 2) { [System.Environment]::SetEnvironmentVariable($parts[0], $parts[1]) }
    }
  }
}

if (Test-Path .\scripts\seed_axes.py) {
  if (-not (Test-Path .\output\axis_packs)) { New-Item -ItemType Directory .\output\axis_packs | Out-Null }
  python .\scripts\seed_axes.py --out .\output\axis_packs\advanced_axis_pack.json
}

Write-Host "Starting API at http://${host}:${port}/health"
python -m uvicorn coherence.api.main:app --host $host --port $port --reload
