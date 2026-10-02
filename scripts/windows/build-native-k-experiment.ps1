[CmdletBinding()]
param([ValidateSet('enc0','enc1','enc2','enc3','enc4','dec5','dec4','dec3','dec2','dec1','dec0')]
    [string]$Layer='enc1', [ValidateSet('source','cu','wgp')][string]$Mode='source',
    [ValidateSet('baseline','load','store','both')][string]$NonTemporal='baseline',
    [ValidateSet(0,1,2,4)][int]$PosTiles=0, [switch]$ShareValues, [switch]$PrivateValues,
    [switch]$PackedAccumulator,
    [string]$HipRoot, [string]$OutputDirectory)
$ErrorActionPreference='Stop'
$repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (!$HipRoot) { $HipRoot=Join-Path $repo '.tools/therock-10.2.0a20260929/_rocm_sdk_core' }
if ($ShareValues -and $PrivateValues) { throw 'Choose shared or private V storage.' }
if (!$OutputDirectory) { $OutputDirectory=Join-Path $repo "build/native-k-experiment-$Layer-$Mode-$NonTemporal-pos$PosTiles-share$([int][bool]$ShareValues)-private$([int][bool]$PrivateValues)-packed$([int][bool]$PackedAccumulator)" }
$OutputDirectory=[IO.Path]::GetFullPath($OutputDirectory)
foreach ($baseline in @('build/native-k-gfx1201','build/windows-rdna4-therock','dist/windows-rdna4-command-list')) {
    if ($OutputDirectory.StartsWith([IO.Path]::GetFullPath((Join-Path $repo $baseline)), [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Keep experimental objects outside the validated baseline/package.'
    }
}
$compiler=Join-Path $HipRoot 'lib/llvm/bin/clang++.exe'
if (!(Test-Path -LiteralPath $compiler)) { $compiler=Join-Path $HipRoot 'bin/clang++.exe' }
$name="dltss_pwin_${Layer}_layer"
$source=Join-Path $repo "kernels/k/$name.hip"
$flags=@()
if ((Get-Content -LiteralPath $source -Raw) -match '// d4r-build-flags: ([^\r\n]+)') { $flags=$Matches[1].Trim().Split(' ') }
if ($Mode -ne 'source') {
    $flags=@($flags | Where-Object { $_ -ne '-mcumode' -and $_ -ne '-mno-cumode' })
    $flags+=if ($Mode -eq 'cu') { '-mcumode' } else { '-mno-cumode' }
}
if ($NonTemporal -in @('load','both')) { $flags+='-DPWIN_NT_LOAD' }
if ($NonTemporal -in @('store','both')) { $flags+='-DPWIN_NT_STORE' }
if ($PosTiles) {
    if ($Layer -notin @('enc0','enc1','dec0')) { throw 'Token tiling is audited for position-only layers only.' }
    $flags+=if ($Layer -eq 'enc1') { "-DD4R_K_ENC1_POS_MT=$PosTiles" } else { "-DD4R_K_POS_MT=$PosTiles" }
}
if ($ShareValues) {
    if ($Layer -notin @('enc0','dec0') -and !($Layer -eq 'enc1' -and $PosTiles)) {
        throw 'ShareValues requires a position-only template.'
    }
    $flags+='-DPOS_VSHARE'
}
if ($PackedAccumulator) { $flags+='-DD4R_K_PACKED_ACC' }
if ($PrivateValues) { $flags+='-DD4R_K_PRIVATE_V' }
New-Item -ItemType Directory -Force $OutputDirectory | Out-Null
$object=Join-Path $OutputDirectory "$name.hsaco"
$arguments=@('-x','hip','-std=c++17','--offload-arch=gfx1201','--offload-device-only',
    '--no-gpu-bundle-output','-mno-wavefrontsize64','-nogpuinc','-nogpulib','-O3',
    '-DD4R_DEVICE_ONLY_MINIMAL','-DD4R_K_FP16_BASELINE')+$flags+@($source,'-o',$object)
$ErrorActionPreference='Continue'
& $compiler @arguments > (Join-Path $OutputDirectory "$name.build.stdout.log") 2> (Join-Path $OutputDirectory "$name.build.stderr.log")
$ErrorActionPreference='Stop'
if ($LASTEXITCODE) { throw "Experimental K compile failed: $Layer" }
& (Join-Path (Split-Path $compiler) 'llvm-objdump.exe') --disassemble --mcpu=gfx1201 $object `
    > (Join-Path $OutputDirectory "$name.isa.txt")
if ($LASTEXITCODE) { throw 'Native object ISA extraction failed.' }
& (Join-Path (Split-Path $compiler) 'llvm-readobj.exe') --notes $object `
    > (Join-Path $OutputDirectory "$name.metadata.txt")
if ($LASTEXITCODE) { throw 'Native object metadata extraction failed.' }
@{layer=$Layer; architecture='gfx1201'; experimental=$true; numericalValidation='pending';
    mode=$Mode; nonTemporal=$NonTemporal; tokenTilesPerWave=$PosTiles; shareValues=[bool]$ShareValues;
    privateValues=[bool]$PrivateValues;
    packedAccumulator=[bool]$PackedAccumulator;
    compiler=$compiler; arguments=$arguments;
    sourceCommit=(git -C $repo rev-parse HEAD); workingTreeDirty=[bool](& git -C $repo status --porcelain);
    compilerVersion=(& $compiler --version | Out-String).Trim();
    sha256=(Get-FileHash -LiteralPath $object -Algorithm SHA256).Hash} |
    ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $OutputDirectory "$name.build-info.json") -Encoding UTF8
Write-Host "Compiled gfx1201 K experiment: $Layer mode=$Mode nonTemporal=$NonTemporal posTiles=$PosTiles shareValues=$([bool]$ShareValues). Validate before installing."
# No override manifest or game installation is produced by this script.
