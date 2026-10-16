[CmdletBinding()]
param([string]$AdlxRoot, [string]$BuildDirectory, [string]$Compiler)
$ErrorActionPreference='Stop'
$repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$commit='32b5a740d42295c5dfe9026b9f52683da0f3af91'
if (!$AdlxRoot) { $AdlxRoot=Join-Path $repo 'external/ADLX' }
if (!$BuildDirectory) { $BuildDirectory=Join-Path $repo 'build/gpu-telemetry' }
if (!$Compiler) { $Compiler=Join-Path $repo '.tools/llvm-mingw-20260922-ucrt-x86_64/bin/x86_64-w64-mingw32-clang.exe' }
if (!(Test-Path -LiteralPath $AdlxRoot)) {
    & git clone --no-checkout https://github.com/GPUOpen-LibrariesAndSDKs/ADLX.git $AdlxRoot
    if ($LASTEXITCODE) { throw 'ADLX SDK download failed.' }
    & git -C $AdlxRoot checkout --detach $commit
    if ($LASTEXITCODE) { throw 'ADLX SDK checkout failed.' }
}
if ((& git -C $AdlxRoot rev-parse HEAD).Trim() -ne $commit -or (& git -C $AdlxRoot status --porcelain)) {
    throw "Expected clean ADLX SDK $commit; use a separate AdlxRoot."
}
if (!(Test-Path -LiteralPath $Compiler)) { throw 'Set Compiler to an x64 LLVM-MinGW clang.exe.' }
New-Item -ItemType Directory -Force $BuildDirectory | Out-Null
$exe=Join-Path $BuildDirectory 'd4r_gpu_telemetry.exe'
& $Compiler -std=c11 -O2 -Wall -Wextra -Wno-unknown-pragmas -DWIN32_LEAN_AND_MEAN -D_M_AMD64 `
    -isystem "$AdlxRoot/SDK/Include" (Join-Path $repo 'tools/windows/gpu_telemetry.c') -o $exe
if ($LASTEXITCODE) { throw "GPU telemetry compile failed: $LASTEXITCODE" }
@{sdkCommit=$commit; sdkLicense=(Join-Path $AdlxRoot 'ADLX SDK License Agreement.pdf');
    executable=$exe; compiler=$Compiler; sha256=(Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash} |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $BuildDirectory 'build-info.json') -Encoding UTF8
Write-Host "Built local read-only telemetry: $exe"
# The SDK, driver DLL and SDK-dependent binary are not added to public packages.
