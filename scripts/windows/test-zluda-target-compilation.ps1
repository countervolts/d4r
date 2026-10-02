[CmdletBinding()]
param([string]$ZludaRoot, [string[]]$GpuArchitectures, [string]$OutputDirectory)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'gpu-target.ps1')
$repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (!$ZludaRoot) { $ZludaRoot=Join-Path $repo 'dist/zluda-windows-final' }
if (!$OutputDirectory) { $OutputDirectory=Join-Path $repo 'test-results/zluda-target-compilation' }
if (!$GpuArchitectures) { $GpuArchitectures=@(Get-D4RGpuTargets | ForEach-Object { $_.architecture }) }
foreach ($arch in $GpuArchitectures) { [void](Get-D4RGpuTargetInfo $arch) }
$emitter=Join-Path $ZludaRoot 'd4r_emit.exe'
if (!(Test-Path -LiteralPath $emitter)) { throw "Rebuild ZLUDA and its offline emitter: $emitter" }
New-Item -ItemType Directory -Force $OutputDirectory | Out-Null
$settings=@{PATH="$ZludaRoot;$env:PATH"; D4R_ZLUDA_WMMA='1'; D4R_ZLUDA_WMMA_FP8='1';
    D4R_ZLUDA_WMMA_F16_REFERENCE='1'; D4R_ZLUDA_WMMA_FP8_NATIVE='0'; D4R_ZLUDA_WMMA_LAYOUT=$null;
    D4R_ZLUDA_WAVE64=$null; D4R_ZLUDA_FAST_MATH=$null; D4R_ZLUDA_IGNORE_DENORMAL=$null;
    D4R_ZLUDA_WMMA_F32ACC=$null; D4R_ZLUDA_EXTRA_BC=$null; D4R_ZLUDA_WGP=$null}
$old=@{}; $rows=[Collections.Generic.List[object]]::new()
try {
    foreach ($key in $settings.Keys) {
        $old[$key]=[Environment]::GetEnvironmentVariable($key,'Process')
        if ($null -eq $settings[$key]) { Remove-Item -LiteralPath "Env:$key" -ErrorAction SilentlyContinue }
        else { [Environment]::SetEnvironmentVariable($key,$settings[$key],'Process') }
    }
    foreach ($fixture in @(@{name='pattern'; source='cuda_driver_probe.cpp'; reference='1'},
            @{name='f16-mma-reference'; source='cuda_mma_probe.cpp'; reference='1'},
            @{name='f16-mma-wmma'; source='cuda_mma_probe.cpp'; reference='0'})) {
        $env:D4R_ZLUDA_WMMA_F16_REFERENCE=$fixture.reference
        # Reuse the public synthetic PTX exercised by the hardware diagnostics.
        # Reading source does not execute a probe or load HIP/NVIDIA libraries.
        $match=[regex]::Match((Get-Content (Join-Path $repo ('tools/windows/'+$fixture.source)) -Raw),'(?s)R"PTX\((.*?)\)PTX"')
        if (!$match.Success) { throw 'Public PTX fixture not found.' }
        $ptx=Join-Path $OutputDirectory ($fixture.name+'.ptx')
        $match.Groups[1].Value | Set-Content -LiteralPath $ptx -Encoding ASCII
        foreach ($arch in $GpuArchitectures) {
            $output=Join-Path $OutputDirectory ($arch+'/'+$fixture.name)
            New-Item -ItemType Directory -Force $output | Out-Null
            $log=Join-Path $output 'compile.log'
            $oldPreference=$ErrorActionPreference
            try { $ErrorActionPreference='Continue'; & $emitter $ptx $output $arch *> $log; $code=$LASTEXITCODE }
            finally { $ErrorActionPreference=$oldPreference }
            if ($code) { throw "Offline ZLUDA compilation failed: $arch $($fixture.name), exit=$code; see $log" }
            Assert-D4RCodeObjectTarget (Join-Path $output 'module.hsaco') $arch
            $text=Get-Content -LiteralPath $log -Raw
            if ($text -match 'not a recognized processor|ignoring processor|unsupported processor') { throw "Target rejected by compiler: $log" }
            $asm=Get-Content (Join-Path $output 'asm.s') -Raw
            if ($asm -notmatch '\.amdhsa_wavefront_size32\s+1') { throw "Missing wave32 code generation: $output" }
            $hasWmma=($asm -match 'v_wmma_')
            if (($fixture.name -eq 'f16-mma-wmma' -or ($fixture.name -eq 'f16-mma-reference' -and $arch.StartsWith('gfx11'))) -and !$hasWmma) {
                throw "WMMA lowering missing: $output"
            }
            # gfx12 strict F16 accumulation intentionally uses the existing
            # double-precision reference helper. Do not label it native WMMA.
            if ($fixture.name -eq 'f16-mma-reference' -and $arch.StartsWith('gfx12') -and
                ($hasWmma -or $asm -notmatch 'v_fma_f64')) { throw "Strict gfx12 reference lowering changed: $output" }
            $rows.Add(@{architecture=$arch; fixture=$fixture.name; compiled=$true; elfTargetCorrect=$true;
                nativeWmma=$hasWmma; hardwareExecuted=$false})
            Write-Host "PASS ZLUDA_OFFLINE target=$arch fixture=$($fixture.name) wave=32 hardware_executed=0"
        }
    }
    @{passed=$true; fixtures=@($rows); emitterSha256=(Get-FileHash -LiteralPath $emitter).Hash; gpuWorkExecuted=$false} |
        ConvertTo-Json -Depth 6 | Set-Content (Join-Path $OutputDirectory 'summary.json') -Encoding UTF8
} finally {
    foreach ($key in $old.Keys) {
        if ($null -eq $old[$key]) { Remove-Item -LiteralPath "Env:$key" -ErrorAction SilentlyContinue }
        else { [Environment]::SetEnvironmentVariable($key,$old[$key],'Process') }
    }
}
