# Runs the same gates as .github/workflows/checks.yml and scripts/check.sh.
$ErrorActionPreference = "Stop"

function Run-Check([string]$Name, [scriptblock]$Body) {
    Write-Host "== $Name"
    & $Body
    if ($LASTEXITCODE -ne 0) {
        Write-Error "== $Name FAILED (exit $LASTEXITCODE)"
        exit 1
    }
}

Run-Check "compileall"  { python -m compileall urban_canopy -q }
Run-Check "ruff"        { python -m ruff check urban_canopy }
Run-Check "black"       { python -m black --check urban_canopy }
Run-Check "pytest"      { python -m pytest }
Run-Check "diagnostics" { python -m urban_canopy.diagnostics }

Write-Host "All checks passed"
