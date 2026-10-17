[CmdletBinding()]
param(
    [ValidateSet('lds','ds','native')][string]$LaneExchange='lds',
    [ValidateSet('gfx1100','gfx1101','gfx1102','gfx1103','gfx1200','gfx1201')][string]$GpuArch='gfx1101',
    [switch]$ReferencePv,
    [string]$HipRoot, [string]$OutputDirectory
)
$ErrorActionPreference='Stop'
$repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if ($GpuArch -match '^gfx12') {
    if (!$PSBoundParameters.ContainsKey('LaneExchange')) { $LaneExchange='native' }
    if ($LaneExchange -ne 'native') { throw 'RDNA4 experiments use native lane exchange.' }
}
if (!$HipRoot) { $HipRoot=Join-Path $repo '.tools/therock-10.2.0a20260929/_rocm_sdk_core' }
if (!$OutputDirectory) {
    $suffix=if ($ReferencePv) { '-reference-pv' } else { '-bounded-pv' }
    $OutputDirectory=Join-Path $repo "build/native-m-experiment-$GpuArch-$LaneExchange$suffix"
}
$OutputDirectory=[IO.Path]::GetFullPath($OutputDirectory)
# Experiments never overwrite a package or an earlier candidate.
New-Item -ItemType Directory $OutputDirectory -ErrorAction Stop | Out-Null
$compiler=Join-Path $HipRoot 'lib/llvm/bin/clang++.exe'
if (!(Test-Path -LiteralPath $compiler)) { $compiler=Join-Path $HipRoot 'bin/clang++.exe' }
$flags=@('-x','hip','-std=c++17',"--offload-arch=$GpuArch",'--offload-device-only',
    '--no-gpu-bundle-output','-mno-wavefrontsize64','-nogpuinc','-nogpulib','-O3',
    '-DD4R_DEVICE_ONLY_MINIMAL','-DSWIN_EXACT','-DSWIN_EXACT_PV')
if ($GpuArch -match '^gfx11' -and $LaneExchange -ne 'native') { $flags+='-DD4R_WINDOWS_GFX11_LANE_EXCHANGE' }
if ($LaneExchange -eq 'ds') { $flags+='-DD4R_WINDOWS_GFX11_DS_EXCHANGE' }
if ($ReferencePv) { $flags+='-DSWIN_EXACT_PV_REFERENCE' }
$modules=@()
foreach ($source in Get-ChildItem (Join-Path $repo 'kernels/m') -Filter '*.hip') {
    $object=Join-Path $OutputDirectory ($source.BaseName+'.hsaco')
    & $compiler @flags $source.FullName -o $object `
        > (Join-Path $OutputDirectory ($source.BaseName+'.stdout.log')) `
        2> (Join-Path $OutputDirectory ($source.BaseName+'.stderr.log'))
    if ($LASTEXITCODE) { throw "M compile failed: $($source.Name)" }
    $modules+=@{name=$source.BaseName; sha256=(Get-FileHash -LiteralPath $object -Algorithm SHA256).Hash}
}
@{architecture=$GpuArch; laneExchange=$LaneExchange; experimental=$true;
    numerics='SWIN_EXACT; SWIN_EXACT_PV'; validation='pending';
    compiler=$compiler; compilerVersion=(& $compiler --version | Out-String).Trim();
    flags=$flags; modules=$modules} | ConvertTo-Json -Depth 5 |
    Set-Content (Join-Path $OutputDirectory 'build-info.json') -Encoding UTF8
Write-Host "Built strict M $LaneExchange candidate: $OutputDirectory. Validate before installing."
