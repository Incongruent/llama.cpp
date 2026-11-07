# A16: post-build SASS gate for the dual-GPU fork (P40 cc6.1 + RTX 3080 cc8.6).
# The llama.cpp convention is that CMAKE_CUDA_ARCHITECTURES controls what lands
# in the binary; this script is the readback gate - it never trusts the build
# flags alone. It runs cuobjdump on the linked ggml-cuda.dll, extracts the set
# of compiled architectures and fails the build if any expected arch is missing
# (an unintended -DCMAKE_CUDA_ARCHITECTURES drift is exactly the class of bug
# that silently produced P40 JIT/PTX-only runs).
#
# Usage (from a repo checkout):
#   powershell -ExecutionPolicy Bypass -File scripts\check-arch.ps1 `
#       -Binary build-dev\bin\ggml-cuda.dll       # (paths are relative to repo root)
#       -Expected 61,86                            # default: 61;86
#       -CUDAPath "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
#
# Exit codes: 0 = PASS (all expected archs in binary), 1 = FAIL, 2 = tooling error.

param(
    [Parameter(Mandatory = $false)][string]$Binary,
    [string[]]$Expected = @('61', '86'),
    [string]$CUDAPath = ""
)

$ErrorActionPreference = "Stop"

# Normalize: "61,86" (single string), 61;86 or a real array all become ['61','86'].
# PS binds -Expected 61,86 as one literal string element of a string[] param.
$expList = @()
foreach ($item in $Expected) { $expList += @(($item -split '[,;:\s]+' | Where-Object { $_ })) }
$Expected = @($expList | Where-Object { $_ })

if (-not $Binary) {
    Write-Error "check-arch.ps1: -Binary is required (e.g. build\bin\ggml-cuda.dll)"
    exit 2
}
if (-not (Test-Path -LiteralPath $Binary)) {
    Write-Error "check-arch.ps1: binary not found: $Binary"
    exit 2
}

# --- locate cuobjdump ----------------------------------------------------
$cuobjdump = $null
if ($CUDAPath) {
    $candidate = Join-Path $CUDAPath "bin\cuobjdump.exe"
    if (Test-Path -LiteralPath $candidate) { $cuobjdump = $candidate }
}
if (-not $cuobjdump) {
    # fall back: CUDA_PATH env, then PATH search
    if ($env:CUDA_PATH) {
        $candidate = Join-Path $env:CUDA_PATH "bin\cuobjdump.exe"
        if (Test-Path -LiteralPath $candidate) { $cuobjdump = $candidate }
    }
}
if (-not $cuobjdump) {
    $pathHit = Get-Command cuobjdump.exe -ErrorAction SilentlyContinue
    if ($pathHit) {
        $cuobjdump = $pathHit.Source
    }
}
if (-not $cuobjdump -or -not (Test-Path -LiteralPath $cuobjdump)) {
    Write-Error "check-arch.ps1: cuobjdump.exe not found (set -CUDAPath or CUDA_PATH)"
    exit 2
}

# --- extract compiled arch set -------------------------------------------
$elfListing = & $cuobjdump -lelf $Binary 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error ("check-arch.ps1: cuobjdump failed (exit {0}):`n{1}" -f $LASTEXITCODE, ($elfListing | Select-Object -First 20 | Out-String))
    exit 2
}
$binaryName = Split-Path -Leaf $Binary
$archSet = @{}
foreach ($line in $elfListing) {
    # typical line: "arch = sm_61" (matches sm_<num> tokens anywhere on the line)
    if ($line -match "sm_(\d+)") {
        $archSet[$Matches[1]] = $true
    }
}
$printed = ($archSet.Keys | Sort-Object) -join ", "
Write-Host ("check-arch: {0} compiled archs: [ {1} ]" -f $binaryName, $printed)

# --- check expected archs -------------------------------------------------
$missing = @()
foreach ($e in $Expected) {
    if (-not $archSet.ContainsKey($e)) {
        $missing += $e
    }
}
if ($missing.Count -gt 0) {
    Write-Error ("check-arch: MISSING expected archs in {0}: {1} (found: {2})." -f $Binary, ($missing -join ","), $printed)
    exit 1
}
Write-Host ("check-arch: PASS - {0} contains SASS for all expected archs ({1})" -f $Binary, ($Expected -join ","))
exit 0
