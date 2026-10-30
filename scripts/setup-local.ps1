[CmdletBinding()]
param()
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path -LiteralPath (Join-Path $root "pyproject.toml"))) {
    throw "Run this script from the Finvex project. pyproject.toml is missing."
}
$configDir = Join-Path $root "configs"
New-Item -ItemType Directory -Path $configDir -Force | Out-Null

function Copy-IfMissing([string]$Source, [string]$Target) {
    if (Test-Path -LiteralPath $Target) {
        Write-Host "Kept existing: $Target"
        return
    }
    if (-not (Test-Path -LiteralPath $Source)) {
        throw "Template missing: $Source"
    }
    Copy-Item -LiteralPath $Source -Destination $Target -ErrorAction Stop
    Write-Host "Created: $Target"
}

# Add ignore rules BEFORE creating private files. This does not untrack existing files.
$ignorePath = Join-Path $root ".gitignore"
$text = if (Test-Path -LiteralPath $ignorePath) { [System.IO.File]::ReadAllText($ignorePath) } else { "" }
$text = $text -replace '(?m)^Kronos/\r?$', '/Kronos/'
$text = $text -replace '(?m)^data/\r?$', '/data/'
$text = $text -replace '(?m)^reports/\r?$', '/reports/'
$text = $text -replace '(?m)^logs/\r?$', '/logs/'
$rules = @(
    '/configs/llm_shared.yaml',
    '/configs/bybit_testnet_key.yaml',
    '/configs/sandbox_key.yaml',
    '/configs/bybit_mainnet_key.yaml',
    '/configs/prod_key.yaml',
    '/configs/deepseek_news.yaml',
    '/configs/*.user.yaml',
    '/configs/providers.local.yaml',
    '/.env',
    '/.venv/',
    '/data/',
    '/reports/',
    '/logs/'
)
$existingLines = @($text -split '\r?\n')
foreach ($rule in $rules) {
    if ($existingLines -notcontains $rule) {
        $text = $text.TrimEnd() + "`r`n" + $rule + "`r`n"
        $existingLines += $rule
    }
}
$utf8 = [System.Text.UTF8Encoding]::new($false)
[System.IO.File]::WriteAllText($ignorePath, $text, $utf8)

Copy-IfMissing (Join-Path $configDir "examples/llm.example.yaml") (Join-Path $configDir "llm_shared.yaml")
Copy-IfMissing (Join-Path $configDir "examples/bybit-testnet.example.yaml") (Join-Path $configDir "bybit_testnet_key.yaml")
Copy-IfMissing (Join-Path $configDir "examples/t-invest-sandbox.example.yaml") (Join-Path $configDir "sandbox_key.yaml")

foreach ($name in @("engine", "llm_strategist")) {
    $source = Join-Path $configDir ($name + ".yaml")
    if (Test-Path -LiteralPath $source) {
        Copy-IfMissing $source (Join-Path $configDir ($name + ".user.yaml"))
    }
}
Write-Host "Local configuration prepared. Existing keys were not overwritten."
Write-Host "Templates contain no credentials. Fill only the configurations needed for your scenario."
Write-Host "No broker calls were made; execution was not enabled."
Write-Host "Important: .user.yaml files are copies, not automatically active configurations."
