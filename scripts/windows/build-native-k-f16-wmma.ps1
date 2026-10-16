[CmdletBinding()]
param([ValidateSet('enc0','enc1','enc2','enc3','enc4','dec5','dec4','dec3','dec2','dec1','dec0')]
    [string]$Layer='enc1', [string]$HipRoot, [string]$OutputDirectory)
$ErrorActionPreference='Stop'
$repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (!$HipRoot) { $HipRoot=Join-Path $repo '.tools/therock-10.2.0a20260929/_rocm_sdk_core' }
if (!$OutputDirectory) { $OutputDirectory=Join-Path $repo 'build/native-k-f16-wmma-experimental' }
$OutputDirectory=[IO.Path]::GetFullPath($OutputDirectory)
if ($OutputDirectory -eq [IO.Path]::GetFullPath((Join-Path $repo 'build/native-k-gfx1201'))) {
    throw 'Keep the validated baseline separate from this experimental build.'
}
$compiler=Join-Path $HipRoot 'lib/llvm/bin/clang++.exe'
if (!(Test-Path -LiteralPath $compiler)) { $compiler=Join-Path $HipRoot 'bin/clang++.exe' }
if (!(Test-Path -LiteralPath $compiler)) { throw "HIP compiler missing under $HipRoot" }
$name="dltss_pwin_${Layer}_layer"
$source=Join-Path $repo "kernels/k/$name.hip"
$flags=@()
if ((Get-Content -LiteralPath $source -Raw) -match '// d4r-build-flags: ([^\r\n]+)') { $flags=$Matches[1].Trim().Split(' ') }
New-Item -ItemType Directory -Force $OutputDirectory | Out-Null
$object=Join-Path $OutputDirectory "$name.hsaco"
$ErrorActionPreference='Continue'
& $compiler -x hip -std=c++17 --offload-arch=gfx1201 --offload-device-only --no-gpu-bundle-output `
    -mno-wavefrontsize64 -nogpuinc -nogpulib -O3 -DD4R_DEVICE_ONLY_MINIMAL `
    -DD4R_K_FP16_BASELINE -DD4R_K_F16_WMMA @flags $source -o $object `
    > (Join-Path $OutputDirectory "$name.build.stdout.log") 2> (Join-Path $OutputDirectory "$name.build.stderr.log")
$ErrorActionPreference='Stop'
if ($LASTEXITCODE) { throw "Experimental K FP16 WMMA compile failed: $Layer" }
$isa=Join-Path $OutputDirectory "$name.isa.txt"
& (Join-Path (Split-Path $compiler) 'llvm-objdump.exe') --disassemble --mcpu=gfx1201 $object > $isa
if ($LASTEXITCODE -or !(Select-String -LiteralPath $isa -Pattern 'v_wmma_f16_16x16x16_f16')) {
    throw 'Packed FP16 WMMA is absent from ISA; no alternative instruction is accepted.'
}
& (Join-Path (Split-Path $compiler) 'llvm-readobj.exe') --notes $object `
    > (Join-Path $OutputDirectory "$name.metadata.txt")
if ($LASTEXITCODE) { throw 'Native object metadata extraction failed.' }
@{layer=$Layer; architecture='gfx1201'; experimental=$true; numericalValidation='pending';
    compiler=$compiler; sha256=(Get-FileHash -LiteralPath $object -Algorithm SHA256).Hash} |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $OutputDirectory "$name.build-info.json") -Encoding UTF8
Write-Host "Compiled experimental gfx1201 packed FP16 WMMA: $Layer. Validate before selecting another layer."
# This script does not generate an override manifest or install into the game.
