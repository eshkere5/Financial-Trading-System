# Start LiteLLM next to Finvex. The provider key is sent only to the provider you configure.
param([int]$Port = 4000, [switch]$Install)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$cfg = Join-Path $root "deploy\litellm\litellm_config.yaml"
$example = Join-Path $root "deploy\litellm\litellm_config.example.yaml"
$envFile = Join-Path $root ".env.litellm"

if (-not (Get-Command litellm -ErrorAction SilentlyContinue)) {
    if (-not $Install) { throw "LiteLLM not found. Run this script with -Install, or: python -m pip install litellm[proxy]" }
    python -m pip install "litellm[proxy]"
    if ($LASTEXITCODE -ne 0) { throw "Could not install litellm" }
}
if (-not (Test-Path $cfg)) {
    Copy-Item $example $cfg
    Write-Host "Created $cfg"
    Write-Host "Open it, set api_base and the model id from your provider, then run this script again."
    exit 0
}
if (-not (Test-Path $envFile)) {
    Set-Content -Path $envFile -Value @("PROVIDER_API_KEY=", "LITELLM_MASTER_KEY=") -Encoding ASCII
}

$vars = [ordered]@{}
foreach ($line in (Get-Content $envFile)) {
    if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)=(.*)$') { $vars[$Matches[1]] = $Matches[2].Trim() }
}
if (-not $vars.Contains("PROVIDER_API_KEY")) { $vars["PROVIDER_API_KEY"] = "" }
if ([string]::IsNullOrEmpty($vars["LITELLM_MASTER_KEY"])) {
    $bytes = New-Object byte[] 32
    $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    $rng.GetBytes($bytes)
    $vars["LITELLM_MASTER_KEY"] = "sk-" + (($bytes | ForEach-Object { $_.ToString("x2") }) -join "")
    $lines = foreach ($k in $vars.Keys) { "$k=$($vars[$k])" }
    Set-Content -Path $envFile -Value $lines -Encoding ASCII
    Write-Host "Created LITELLM_MASTER_KEY in $envFile"
}
if ([string]::IsNullOrEmpty($vars["PROVIDER_API_KEY"])) {
    Write-Warning "PROVIDER_API_KEY is empty. For a cloud model put the provider key into $envFile"
}
foreach ($k in $vars.Keys) { Set-Item -Path ("Env:" + $k) -Value $vars[$k] }

Write-Host ""
Write-Host "Finvex: Settings -> provider LiteLLM, model finvex-main, key = LITELLM_MASTER_KEY from $envFile"
Write-Host "Stop with Ctrl+C"
litellm --config $cfg --port $Port
