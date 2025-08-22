param(
  [string]$BaseUrl = "http://localhost:8080",
  [string]$OutDir = "docs"
)

$ErrorActionPreference = "Stop"

# Resolve repo root (this script is in tools/)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir

# Paths
$OpenApiJson = Join-Path $RepoRoot (Join-Path $OutDir "openapi.json")
$PostmanJson = Join-Path $RepoRoot (Join-Path $OutDir "postman_collection.json")
$ThunderJson = Join-Path $RepoRoot (Join-Path $OutDir "thunder-collection_Coherence_API.json")

# Ensure output directory exists
New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot $OutDir) | Out-Null

# Generate OpenAPI
python scripts/export_openapi.py --out $OpenApiJson

# Generate Postman collection
python scripts/openapi_to_postman.py --in $OpenApiJson --out $PostmanJson --base-url $BaseUrl

# Generate Thunder Client collection
python scripts/openapi_to_thunder.py --in $OpenApiJson --out $ThunderJson --base-url $BaseUrl

Write-Host "Export complete:" -ForegroundColor Green
Write-Host "  OpenAPI:   $OpenApiJson"
Write-Host "  Postman:   $PostmanJson"
Write-Host "  Thunder:   $ThunderJson"
