param(
    [switch]$SkipBackend,
    [switch]$BundleDeep
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

function Invoke-Checked {
    param([string]$Program, [string[]]$CommandArgs)
    & $Program @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "$Program failed with exit code $LASTEXITCODE"
    }
}

if (-not $SkipBackend) {
    Invoke-Checked -Program python -CommandArgs @('-m', 'pip', 'install', '-r', 'requirements.txt')
    Invoke-Checked -Program python -CommandArgs @('-m', 'pip', 'install', '-r', 'requirements-tauri.txt')
    if ($BundleDeep) {
        $env:PANO_BUNDLE_DEEP = "1"
    } else {
        Remove-Item Env:PANO_BUNDLE_DEEP -ErrorAction SilentlyContinue
    }
    Invoke-Checked -Program python -CommandArgs @('-m', 'PyInstaller', 'tools/pano_backend.spec', '--noconfirm')

    $Triple = (rustc --print host-tuple).Trim()
    if ($LASTEXITCODE -ne 0) { throw 'rustc failed to report its host tuple.' }
    if (-not $Triple) {
        throw "Unable to determine Rust host tuple."
    }

    $IsWin = [System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform(
        [System.Runtime.InteropServices.OSPlatform]::Windows
    )
    $Ext = if ($IsWin) { ".exe" } else { "" }
    $Source = Join-Path $Root "dist/pano-backend$Ext"
    if (-not (Test-Path $Source)) {
        throw "Backend sidecar was not produced at $Source"
    }

    $BinDir = Join-Path $Root "src-tauri/binaries"
    New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
    Copy-Item $Source (Join-Path $BinDir "pano-backend-$Triple$Ext") -Force
}

Invoke-Checked -Program npm -CommandArgs @('ci')
Invoke-Checked -Program npm -CommandArgs @('run', 'tauri:bundle')
