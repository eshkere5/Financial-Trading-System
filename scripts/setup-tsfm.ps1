param([switch]$SkipInstall, [switch]$Offline)
$ErrorActionPreference = "Stop"
Push-Location (Join-Path $PSScriptRoot "..")
try {
    if (-not $SkipInstall) {
        if ($Offline) { throw "Use -SkipInstall for offline preparation; install dependencies separately." }
        & python -m pip install -e ".[dev,tsfm]"
        if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed" }
    }
    $prepareArgs = @("-m", "finvex.tsfm", "prepare")
    if ($Offline) { $prepareArgs += "--offline" }
    & python @prepareArgs
    if ($LASTEXITCODE -ne 0) { throw "Weight preparation failed" }
    & python -m finvex.tsfm doctor --load
    if ($LASTEXITCODE -ne 0) { throw "TSFM did not load" }
} finally {
    Pop-Location
}
