[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$CaptureDirectory, [string]$HipRoot,
    [string]$PackageRoot, [string]$OutputDirectory, [int]$Blocks = 12,
    [Parameter(Mandatory=$true)][string]$DlssDll, [int]$PtxBlocks = 2)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'gpu-target.ps1')
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (!$HipRoot) { $HipRoot = Join-Path $repo '.tools/therock-10.2.0a20260929/_rocm_sdk_core' }
if (!$PackageRoot) { $PackageRoot = Join-Path $repo 'dist/windows-rdna4-therock' }
$GpuArch=Get-D4RGpuTarget $PackageRoot $null
if (!$OutputDirectory) { $OutputDirectory = Join-Path $repo ('test-results/m-real-reference-' + (Get-Date -Format 'yyyyMMdd-HHmmss')) }
New-Item -ItemType Directory -Force $OutputDirectory | Out-Null
$previousPython=$env:PYTHONPATH; $previousDiag=$env:D4R_DIAG_DIR
$env:PYTHONPATH = "$(Join-Path $repo '.tools/python/vendor');$previousPython"
$summary = [ordered]@{utc=[DateTime]::UtcNow.ToString('o'); captureDirectory=$CaptureDirectory; layers=@(); passed=$false}
try {
    foreach ($layer in @('enc1','enc2','enc3_tube','dec2','dec1')) {
        $name = "rrlite_${layer}_4x4"
        $captures = @(Get-ChildItem -LiteralPath $CaptureDirectory -Directory | Where-Object Name -Match ("^replay-[0-9]+-" + $name + '$'))
        if (!$captures.Count) { throw "Missing actual M capture: $name" }
        $capture = $captures | Sort-Object Name | Select-Object -First 1
        $folder = Join-Path $OutputDirectory $layer
        $output = Join-Path $folder 'output'
        New-Item -ItemType Directory -Force $folder | Out-Null
        $env:D4R_DIAG_DIR = $folder
        & (Join-Path $PackageRoot 'bin/d4r_native_replay_probe.exe') --hip-root $HipRoot `
            --module (Join-Path $PackageRoot "experimental/m/${name}_${GpuArch}.hsaco") `
            --fixture-dir $capture.FullName --output-dir $output --iterations 1 `
            > (Join-Path $folder 'gpu.stdout.log') 2> (Join-Path $folder 'gpu.stderr.log')
        if ($LASTEXITCODE) { throw "M $layer actual GPU replay failed: $folder" }
        & python (Join-Path $PackageRoot 'm_replay_validate.py') validate --layer $layer `
            --fixture-dir $capture.FullName --output-dir $output --blocks $Blocks --real-capture `
            > (Join-Path $folder 'reference.stdout.log') 2> (Join-Path $folder 'reference.stderr.log')
        if ($LASTEXITCODE) { throw "M $layer mismatch; stopping before the next layer: $folder" }
        Get-Content -LiteralPath (Join-Path $folder 'reference.stdout.log') | Write-Host
        & python -u (Join-Path $PackageRoot 'm_ptx_validate.py') --layer $layer `
            --capture-dir $capture.FullName --native-dir $output --dlss-dll $DlssDll `
            --output-dir (Join-Path $folder 'private-ptx') --blocks $PtxBlocks `
            > (Join-Path $folder 'ptx-reference.stdout.log') 2> (Join-Path $folder 'ptx-reference.stderr.log')
        if ($LASTEXITCODE) { throw "M $layer independent PTX mismatch; stopping: $folder" }
        Get-Content -LiteralPath (Join-Path $folder 'ptx-reference.stdout.log') | Write-Host
        $summary.layers += @{layer=$layer; passed=$true}
    }
    $summary.passed=$true
} catch { $summary.error=$_.Exception.Message; Write-Warning $summary.error }
finally {
    $env:PYTHONPATH=$previousPython; $env:D4R_DIAG_DIR=$previousDiag
    $summary | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $OutputDirectory 'summary.json') -Encoding UTF8
    Write-Host "Private real-weight replay results: $OutputDirectory (do not redistribute captures/allocations)"
}
if (!$summary.passed) { exit 1 }
