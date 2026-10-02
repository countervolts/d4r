[CmdletBinding()]
param([string]$HipRoot, [string]$PackageRoot, [string]$OutputDirectory,
    [string[]]$Layers = @('enc1','enc2','enc3_tube','dec2','dec1'),
    [ValidateRange(1,10000)][int]$Iterations = 32)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'gpu-target.ps1')
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (!$HipRoot) { $HipRoot = Join-Path $repo '.tools/therock-10.2.0a20260929/_rocm_sdk_core' }
if (!$PackageRoot) { $PackageRoot = Join-Path $repo 'dist/windows-rdna4-therock' }
$GpuArch=Get-D4RGpuTarget $PackageRoot $null
if (!$OutputDirectory) { $OutputDirectory = Join-Path $repo ('test-results/m-replay-' + (Get-Date -Format 'yyyyMMdd-HHmmss')) }
New-Item -ItemType Directory -Force $OutputDirectory | Out-Null
$previousPython = $env:PYTHONPATH
$previousDiag = $env:D4R_DIAG_DIR
$env:PYTHONPATH = "$(Join-Path $repo '.tools/python/vendor');$previousPython"
$summary = [ordered]@{utc=[DateTime]::UtcNow.ToString('o'); hipRoot=$HipRoot; architecture=$GpuArch; layers=@(); passed=$false}
try {
    foreach ($layer in $Layers) {
        if ($layer -notmatch '^(enc[12]|enc3_tube|dec[12])$') { throw "Invalid M layer $layer" }
        $folder = Join-Path $OutputDirectory $layer
        $fixture = Join-Path $folder 'fixture'
        $output = Join-Path $folder 'output'
        New-Item -ItemType Directory -Force $folder | Out-Null
        $env:D4R_DIAG_DIR = $folder
        & python (Join-Path $PackageRoot 'm_replay_validate.py') generate --layer $layer --fixture-dir $fixture `
            > (Join-Path $folder 'fixture.stdout.log') 2> (Join-Path $folder 'fixture.stderr.log')
        if ($LASTEXITCODE) { throw "M $layer fixture generation failed: $folder" }
        & (Join-Path $PackageRoot 'bin/d4r_native_replay_probe.exe') --hip-root $HipRoot `
            --module (Join-Path $PackageRoot "experimental/m/rrlite_${layer}_4x4_${GpuArch}.hsaco") `
            --fixture-dir $fixture --output-dir $output --iterations $Iterations `
            > (Join-Path $folder 'gpu.stdout.log') 2> (Join-Path $folder 'gpu.stderr.log')
        if ($LASTEXITCODE) { throw "M $layer GPU replay failed: $folder" }
        & python (Join-Path $PackageRoot 'm_replay_validate.py') validate --layer $layer --fixture-dir $fixture --output-dir $output `
            > (Join-Path $folder 'reference.stdout.log') 2> (Join-Path $folder 'reference.stderr.log')
        if ($LASTEXITCODE) { throw "M $layer reference mismatch; stopping before the next layer: $folder" }
        Get-Content -LiteralPath (Join-Path $folder 'reference.stdout.log') | Write-Host
        $summary.layers += @{layer=$layer; passed=$true; iterations=$Iterations}
    }
    $summary.passed = $true
} catch { $summary.error=$_.Exception.Message; Write-Warning $summary.error }
finally {
    $env:PYTHONPATH=$previousPython; $env:D4R_DIAG_DIR=$previousDiag
    $summary | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $OutputDirectory 'summary.json') -Encoding UTF8
    Compress-Archive -LiteralPath $OutputDirectory -DestinationPath "$OutputDirectory.zip" -Force
    Write-Host "M replay bundle: $OutputDirectory.zip"
}
if (!$summary.passed) { exit 1 }
