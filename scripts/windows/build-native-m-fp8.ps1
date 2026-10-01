[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$DlssDll, [string]$HipRoot, [string]$OutputDirectory, [switch]$HardwarePacking)
$ErrorActionPreference='Stop'
$repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (!$HipRoot) { $HipRoot=Join-Path $repo '.tools/therock-10.2.0a20260929/_rocm_sdk_core' }
if (!$OutputDirectory) { $OutputDirectory=Join-Path $repo $(if ($HardwarePacking) { 'build/native-m-fp8-hardware-pack' } else { 'build/native-m-fp8-experimental' }) }
$OutputDirectory=[IO.Path]::GetFullPath($OutputDirectory)
if ($OutputDirectory -eq [IO.Path]::GetFullPath((Join-Path $repo 'build/native-m-gfx1201'))) {
    throw 'Keep the validated FP16 baseline separate from the experimental FP8 build.'
}
$DlssDll=(Get-Item -LiteralPath $DlssDll -ErrorAction Stop).FullName
$compiler=Join-Path $HipRoot 'lib/llvm/bin/clang++.exe'
if (!(Test-Path -LiteralPath $compiler)) { $compiler=Join-Path $HipRoot 'bin/clang++.exe' }
if (!(Test-Path -LiteralPath $compiler)) { throw "HIP compiler missing under $HipRoot" }
New-Item -ItemType Directory -Force $OutputDirectory | Out-Null
$disassembler=Join-Path (Split-Path $compiler) 'llvm-objdump.exe'
if (!(Test-Path -LiteralPath $disassembler)) { throw 'HIP llvm-objdump is required to confirm native FP8 instructions.' }
foreach ($source in Get-ChildItem -LiteralPath (Join-Path $repo 'kernels/m') -Filter '*.hip' | Sort-Object Name) {
    $flags=@()
    if ((Get-Content -LiteralPath $source.FullName -Raw) -match '// d4r-build-flags: ([^\r\n]+)') { $flags=$Matches[1].Trim().Split(' ') }
    if ($HardwarePacking) { $flags+='-DD4R_FP8_HW_PACK' }
    $object=Join-Path $OutputDirectory ($source.BaseName+'.hsaco')
    $ErrorActionPreference='Continue'
    & $compiler -x hip -std=c++17 --offload-arch=gfx1201 --offload-device-only --no-gpu-bundle-output `
        -mno-wavefrontsize64 -nogpuinc -nogpulib -O3 -DD4R_DEVICE_ONLY_MINIMAL `
        -DSWIN_EXACT -DSWIN_EXACT_PV -DD4R_FP8_WMMA @flags $source.FullName `
        -o $object > (Join-Path $OutputDirectory ($source.BaseName+'.build.stdout.log')) `
        2> (Join-Path $OutputDirectory ($source.BaseName+'.build.stderr.log'))
    $ErrorActionPreference='Stop'
    if ($LASTEXITCODE) { throw "Experimental FP8 compile failed: $($source.Name)" }
    $isa=Join-Path $OutputDirectory ($source.BaseName+'.isa.txt')
    & $disassembler --disassemble --mcpu=gfx1201 $object > $isa
    if ($LASTEXITCODE -or !(Select-String -LiteralPath $isa -Pattern 'v_wmma_f32_16x16x16_fp8_fp8')) {
        throw "Native FP8 WMMA is absent from $($source.Name); no FP16 fallback is accepted."
    }
    if ($HardwarePacking -and !(Select-String -LiteralPath $isa -Pattern 'v_cvt_pk_fp8_f32')) { throw 'Native FP8 operand packing is absent from ISA.' }
    Write-Host "Compiled gfx1201 FP8 WMMA: $($source.BaseName)"
}
if ($HardwarePacking) {
    $probeDirectory=Join-Path $OutputDirectory 'packing-probe'
    New-Item -ItemType Directory -Force $probeDirectory | Out-Null
    & $compiler -x hip -std=c++17 --offload-arch=gfx1201 --offload-device-only --no-gpu-bundle-output `
        -mno-wavefrontsize64 -nogpuinc -nogpulib -O3 -DD4R_DEVICE_ONLY_MINIMAL `
        -DD4R_FP8_WMMA -DD4R_FP8_HW_PACK -DSWIN_EXACT -DSWIN_EXACT_PV `
        (Join-Path $repo 'tools/windows/fp8_pack_probe.hip') -o (Join-Path $probeDirectory 'fp8_pack_probe.hsaco')
    if ($LASTEXITCODE) { throw 'Native hardware packing diagnostic compile failed.' }
}
& python (Join-Path $repo 'kernels/tools/kernel_manifest.py') $OutputDirectory $DlssDll
if ($LASTEXITCODE) { throw 'Experimental FP8 manifest generation failed.' }
@{architecture='gfx1201'; family='M'; experimental=$true; nativeFp8=$true; accuracy=$true;
    numerics='SWIN_EXACT; SWIN_EXACT_PV; FP8 WMMA for e4m3 GEMMs; exact FP16 attention';
    validation='not validated; do not install before layer/full-network checks';
    compiler=$compiler; compilerVersion=(& $compiler --version | Select-Object -First 1);
    sourceCommit=(& git -C $repo rev-parse HEAD).Trim(); nativeFp8IsaVerified=$true; hardwarePacking=[bool]$HardwarePacking;
    sourceWorkingTreeDirty=[bool](& git -C $repo status --porcelain);
    dlssSha256=(Get-FileHash -LiteralPath $DlssDll -Algorithm SHA256).Hash;
    modules=@(Get-ChildItem -LiteralPath $OutputDirectory -Filter '*.hsaco' | Get-FileHash -Algorithm SHA256)} |
    ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $OutputDirectory 'build-info.json') -Encoding UTF8
Write-Host "Experimental native FP8 M objects: $OutputDirectory (baseline unchanged)"
