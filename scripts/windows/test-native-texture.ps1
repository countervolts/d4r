[CmdletBinding()]
param([string]$TextureRoot, [string]$ControlTextureRoot, [string]$CommonTextureRoot, [string]$ZludaRoot, [string]$PackageRoot,
    [string]$OutputDirectory, [int]$Iterations=8, [string]$OutputResolution='512x288', [string]$InputResolution,
    [switch]$CandidateFirst,
    [ValidateSet('output','input')][string]$KernelStage='output')
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'gpu-target.ps1')
$repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (!$PackageRoot) { $PackageRoot=Join-Path $repo 'dist/windows-rdna4-command-list' }
$GpuArch=Get-D4RGpuTarget $PackageRoot $null
if (!$TextureRoot) { $TextureRoot=Join-Path $repo "build/private-textures-$GpuArch" }
if (!$ZludaRoot) { $ZludaRoot=Join-Path $repo 'dist/zluda-windows-final' }
if (!$OutputDirectory) { $OutputDirectory=Join-Path $repo 'test-results/k-native-texture-reference' }
if ($OutputResolution -notmatch '^([0-9]+)x([0-9]+)$') { throw 'OutputResolution must be WIDTHxHEIGHT.' }
$outputWidth=[int]$Matches[1]; $outputHeight=[int]$Matches[2]
$combined=Join-Path $OutputDirectory 'private-native'
New-Item -ItemType Directory -Force $combined | Out-Null
foreach ($directory in @((Join-Path $repo "build/native-k-$GpuArch"),$CommonTextureRoot,$TextureRoot) | Where-Object { $_ }) {
    Get-ChildItem -LiteralPath $directory -Filter '*.hsaco' | Copy-Item -Destination $combined -Force
}
& python (Join-Path $repo 'kernels/tools/kernel_manifest.py') $combined (Join-Path $repo 'nvngx_dlss.dll')
if ($LASTEXITCODE) { throw 'Combined private manifest failed.' }
$controlNative=Join-Path $repo "build/native-k-$GpuArch"
if ($ControlTextureRoot -or $CommonTextureRoot) {
    $controlNative=Join-Path $OutputDirectory 'control-private-native'
    New-Item -ItemType Directory -Force $controlNative | Out-Null
    foreach ($directory in @((Join-Path $repo "build/native-k-$GpuArch"),$CommonTextureRoot,$ControlTextureRoot) | Where-Object { $_ }) {
        Get-ChildItem -LiteralPath $directory -Filter '*.hsaco' | Copy-Item -Destination $controlNative -Force
    }
    & python (Join-Path $repo 'kernels/tools/kernel_manifest.py') $controlNative (Join-Path $repo 'nvngx_dlss.dll')
    if ($LASTEXITCODE) { throw 'Control private manifest failed.' }
}
$settings=@{D4R_VALIDATE_OUTPUT='1'; D4R_PROFILE_STAGES='1'; D4R_ZLUDA_PROFILE='1'; D4R_QUIET_API='1';
    D4R_ZLUDA_WMMA='1'; D4R_ZLUDA_WMMA_FP8='1'; D4R_ZLUDA_WMMA_FP8_NATIVE='0';
    D4R_ZLUDA_WMMA_F16_REFERENCE='1'; D4R_ZLUDA_PROFILE_DEFERRED=$null;
    ZLUDA_CACHE_DIR=(Join-Path $repo 'build/zluda-cache-windows'); PYTHONPATH=(Join-Path $repo '.tools/python/vendor');
    D4R_ZLUDA_NATIVE_DIR=$null}
$old=@{}
$stagedObjects=@{}
foreach ($stage in @(@{name='control'; root=$controlNative},@{name='candidate'; root=$combined})) {
    $stagedObjects[$stage.name]=@{}
    foreach ($object in Get-ChildItem -LiteralPath $stage.root -Filter '*.hsaco') {
        Assert-D4RCodeObjectTarget $object.FullName $GpuArch
        $stagedObjects[$stage.name][$object.Name]=(Get-FileHash -LiteralPath $object.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}
try {
    foreach ($key in $settings.Keys) { $old[$key]=[Environment]::GetEnvironmentVariable($key,'Process'); [Environment]::SetEnvironmentVariable($key,$settings[$key],'Process') }
    foreach ($flags in @(0,11)) {
        $variant=if ($flags -eq 11) { 'depthinv_mvlo_hdr' } else { 'depthreg_mvhi_ldr' }
        $suffix=if ($KernelStage -eq 'output') { '_max_v2_rel' } else { '_v2_rel' }
        $kernel="hiluma_engine_${KernelStage}_${variant}${suffix}"
        $order=if ($CandidateFirst) { @('candidate','control') } else { @('control','candidate') }
        foreach ($mode in $order) {
            $env:D4R_ZLUDA_NATIVE_DIR=if ($mode -eq 'control') { $controlNative } else { $combined }
            $expectedBackend=if (Test-Path -LiteralPath (Join-Path $env:D4R_ZLUDA_NATIVE_DIR ($kernel + '.hsaco'))) { 'native' } else { 'translated' }
            if ($mode -eq 'candidate' -and $expectedBackend -ne 'native') { throw "Missing candidate object: $kernel" }
            $arguments=@{RuntimeProfile='therock'; ZludaRoot=$ZludaRoot;
                PackageRoot=$PackageRoot; NgxCore=(Join-Path $repo '_nvngx.dll');
                DlssDll=(Join-Path $repo 'nvngx_dlss.dll'); NgxMode='d3d12'; NgxAbi='project-legacy'; Preset=11;
                NgxCreateFlags=$flags; NgxOutputResolution=$OutputResolution; NgxOnly=$true; RequireNativeNetwork=$true; CommandListBackend=$true;
                NgxInputResolution=$InputResolution;
                PixelProfile='depth-stencil'; BarrierMode='inherited-legacy'; CaptureExceptions=$true; EarlyIndirectProbe=$true;
                Iterations=$Iterations; TimeoutSeconds=300; OptiScalerDll=(Join-Path $repo 'dist/optiscaler-windows-d4r/OptiScaler.dll');
                OutputDirectory=(Join-Path $OutputDirectory "$mode-$flags")}
            & (Join-Path $PSScriptRoot 'test-windows-rdna4.ps1') @arguments
            if ($LASTEXITCODE) { throw "Texture $mode flags=$flags workload failed." }
            & python (Join-Path $repo 'tools/windows/profile_report.py') $arguments.OutputDirectory --output (Join-Path $arguments.OutputDirectory 'profile.json')
            if ($LASTEXITCODE) { throw 'Texture profile report failed.' }
            $profile=Get-Content -LiteralPath (Join-Path $arguments.OutputDirectory 'profile.json') -Raw | ConvertFrom-Json
            $samples=@($profile.kernels | Where-Object { $_.kernel -eq $kernel -and $_.backend -eq $expectedBackend -and $_.phase -eq 'main' -and $_.serializing })
            if ($samples.Count -ne 1 -or $samples[0].gpuMs.samples -ne $Iterations) {
                throw "Missing valid serializing $KernelStage-kernel timings ($expectedBackend) for $mode flags=$flags; retain this run as correctness evidence only."
            }
        }
        $control=Join-Path $OutputDirectory "control-$flags"; $candidate=Join-Path $OutputDirectory "candidate-$flags"
        $stderr=Get-Content -LiteralPath (Join-Path $candidate 'd3d12-evaluate-preset-11.stderr.log') -Raw
        $hits=[regex]::Matches($stderr, ('\[d4r-launch\] kernel="' + $kernel + '" backend=native')).Count
        if ($hits -ne $Iterations) { throw "Texture override was not executed on every frame: $kernel hits=$hits" }
        & python (Join-Path $repo 'tools/windows/frame_compare.py') --reference $control --actual $candidate --width $outputWidth --height $outputHeight --exact
        if ($LASTEXITCODE) { throw "Texture flags=$flags full-frame RGB mismatch; preserve baseline." }
    }
    @{passed=$true; architecture=$GpuArch; framesPerVariant=$Iterations; outputResolution=$OutputResolution; createFlags=@(0,11); strictRgb=$true;
        inputResolution=$InputResolution;
        serializingProfile=$true; candidateFirst=[bool]$CandidateFirst; kernelStage=$KernelStage;
        controlTextureRoot=$ControlTextureRoot; commonTextureRoot=$CommonTextureRoot;
        stagedObjects=$stagedObjects;
        manifestSha256=(Get-FileHash -LiteralPath (Join-Path $TextureRoot 'd4r-kernels.txt') -Algorithm SHA256).Hash.ToLowerInvariant();
        privateNvidiaDerivedCode=$true; source=(Get-Content -LiteralPath (Join-Path $TextureRoot 'build-info.json') -Raw | ConvertFrom-Json)} |
        ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $TextureRoot 'validation.json') -Encoding UTF8
    Write-Host "PASS: both private K $KernelStage variants executed and matched every RGB component. $OutputDirectory"
} finally {
    foreach ($key in $old.Keys) { [Environment]::SetEnvironmentVariable($key,$old[$key],'Process') }
}
