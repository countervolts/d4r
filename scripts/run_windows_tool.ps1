param(
    [Parameter(Mandatory = $true, Position = 0)][string]$Tool,   # e.g. replay_dlss_kernel, d4r-k, kvk
    [string]$Zluda = $(if ($env:ZLUDA_ROOT) { $env:ZLUDA_ROOT } else { $env:D4R_ZLUDA_DIR }),
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$ToolArgs
)
# Windows counterpart of the scripts/run_*.sh wrappers: runs a tool from build_windows_tools.ps1 with
# ZLUDA's nvcuda.dll first on PATH (the CUDA-driver tools load it by name; HIP comes from the AMD driver).
#   scripts/run_windows_tool.ps1 replay_dlss_kernel -Zluda C:/zluda/target/release CAPTURE.fatbin KERNEL
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$candidates = @("build/tools/$Tool.exe", "build/engine-win/Release/$Tool.exe", "build/$Tool.exe") |
    ForEach-Object { Join-Path $root $_ }
$exe = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $exe) { throw "$Tool.exe not built; run scripts/build_windows_tools.ps1 (looked in: $($candidates -join ', '))" }
if ($Zluda) {
    if (-not (Test-Path -LiteralPath (Join-Path $Zluda 'nvcuda.dll'))) { throw "no nvcuda.dll in $Zluda" }
    $env:PATH = "$Zluda;$env:PATH"
}
& $exe @ToolArgs
exit $LASTEXITCODE
