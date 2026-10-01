[CmdletBinding()]
param([string]$TextureRoot, [string]$OutputDirectory, [int]$Iterations=8, [string]$OutputResolution='512x288')
$ErrorActionPreference='Stop'
$repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (!$TextureRoot) { $TextureRoot=Join-Path $repo 'build/private-textures-gfx1201' }
if (!$OutputDirectory) { $OutputDirectory=Join-Path $repo 'test-results/k-native-texture-reference' }
if ($OutputResolution -notmatch '^([0-9]+)x([0-9]+)$') { throw 'OutputResolution must be WIDTHxHEIGHT.' }
$outputWidth=[int]$Matches[1]; $outputHeight=[int]$Matches[2]
$combined=Join-Path $OutputDirectory 'private-native'
New-Item -ItemType Directory -Force $combined | Out-Null
foreach ($directory in @((Join-Path $repo 'build/native-k-gfx1201'),$TextureRoot)) {
    Get-ChildItem -LiteralPath $directory -Filter '*.hsaco' | Copy-Item -Destination $combined -Force
}
& python (Join-Path $repo 'kernels/tools/kernel_manifest.py') $combined (Join-Path $repo 'nvngx_dlss.dll')
if ($LASTEXITCODE) { throw 'Combined private manifest failed.' }
$settings=@{D4R_VALIDATE_OUTPUT='1'; D4R_PROFILE_STAGES='1'; D4R_ZLUDA_PROFILE='1'; D4R_QUIET_API='1';
    ZLUDA_CACHE_DIR=(Join-Path $repo 'build/zluda-cache-windows'); PYTHONPATH=(Join-Path $repo '.tools/python/vendor');
    D4R_ZLUDA_NATIVE_DIR=$null}
$old=@{}
try {
    foreach ($key in $settings.Keys) { $old[$key]=[Environment]::GetEnvironmentVariable($key,'Process'); [Environment]::SetEnvironmentVariable($key,$settings[$key],'Process') }
    foreach ($flags in @(0,11)) {
        foreach ($mode in @('control','candidate')) {
            $env:D4R_ZLUDA_NATIVE_DIR=if ($mode -eq 'control') { Join-Path $repo 'build/native-k-gfx1201' } else { $combined }
            $arguments=@{RuntimeProfile='therock'; ZludaRoot=(Join-Path $repo 'dist/zluda-windows-profile');
                PackageRoot=(Join-Path $repo 'dist/windows-rdna4-command-list'); NgxCore=(Join-Path $repo '_nvngx.dll');
                DlssDll=(Join-Path $repo 'nvngx_dlss.dll'); NgxMode='d3d12'; NgxAbi='project-legacy'; Preset=11;
                NgxCreateFlags=$flags; NgxOutputResolution=$OutputResolution; NgxOnly=$true; RequireNativeNetwork=$true; CommandListBackend=$true;
                PixelProfile='depth-stencil'; BarrierMode='inherited-legacy'; CaptureExceptions=$true; EarlyIndirectProbe=$true;
                Iterations=$Iterations; TimeoutSeconds=300; OptiScalerDll=(Join-Path $repo 'dist/optiscaler-windows-d4r/OptiScaler.dll');
                OutputDirectory=(Join-Path $OutputDirectory "$mode-$flags")}
            & (Join-Path $PSScriptRoot 'test-windows-rdna4.ps1') @arguments
            if ($LASTEXITCODE) { throw "Texture $mode flags=$flags workload failed." }
            & python (Join-Path $repo 'tools/windows/profile_report.py') $arguments.OutputDirectory --output (Join-Path $arguments.OutputDirectory 'profile.json')
            if ($LASTEXITCODE) { throw 'Texture profile report failed.' }
        }
        $control=Join-Path $OutputDirectory "control-$flags"; $candidate=Join-Path $OutputDirectory "candidate-$flags"
        $kernel=if ($flags -eq 11) { 'hiluma_engine_output_depthinv_mvlo_hdr_max_v2_rel' } else { 'hiluma_engine_output_depthreg_mvhi_ldr_max_v2_rel' }
        $stderr=Get-Content -LiteralPath (Join-Path $candidate 'd3d12-evaluate-preset-11.stderr.log') -Raw
        $hits=[regex]::Matches($stderr, ('\[d4r-launch\] kernel="' + $kernel + '" backend=native')).Count
        if ($hits -ne $Iterations) { throw "Texture override was not executed on every frame: $kernel hits=$hits" }
        & python (Join-Path $repo 'tools/windows/frame_compare.py') --reference $control --actual $candidate --width $outputWidth --height $outputHeight --exact
        if ($LASTEXITCODE) { throw "Texture flags=$flags full-frame RGB mismatch; preserve baseline." }
    }
    @{passed=$true; architecture='gfx1201'; framesPerVariant=$Iterations; outputResolution=$OutputResolution; createFlags=@(0,11); strictRgb=$true;
        manifestSha256=(Get-FileHash -LiteralPath (Join-Path $TextureRoot 'd4r-kernels.txt') -Algorithm SHA256).Hash.ToLowerInvariant();
        privateNvidiaDerivedCode=$true; source=(Get-Content -LiteralPath (Join-Path $TextureRoot 'build-info.json') -Raw | ConvertFrom-Json)} |
        ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $TextureRoot 'validation.json') -Encoding UTF8
    Write-Host "PASS: both private K output variants executed and matched every RGB component. $OutputDirectory"
} finally {
    foreach ($key in $old.Keys) { [Environment]::SetEnvironmentVariable($key,$old[$key],'Process') }
}
