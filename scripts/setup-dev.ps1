# Create a venv and install the package with the requested optional layers.
#   powershell -ExecutionPolicy Bypass -File .\scripts\setup-dev.ps1 [-WithApi] [-WithMl]
param(
    [switch]$WithApi,
    [switch]$WithMl
)
$ErrorActionPreference = "Stop"

$extras = "dev"
if ($WithApi) { $extras = "$extras,api" }
if ($WithMl)  { $extras = "$extras,ml" }

python -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[$extras]"
Write-Host "Done. Activate with: .\.venv\Scripts\Activate.ps1"
