[CmdletBinding()]
param([string]$HipRoot, [string]$PackageRoot, [string]$ZludaRoot, [string]$OutputDirectory)
$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (!$HipRoot) { $HipRoot = Join-Path $repo '.tools/therock-10.2.0a20260929/_rocm_sdk_core' }
if (!$PackageRoot) { $PackageRoot = Join-Path $repo 'dist/windows-rdna4-therock' }
if (!$ZludaRoot) { $ZludaRoot = Join-Path $repo 'dist/zluda-windows-final' }
if (!$OutputDirectory) { $OutputDirectory = Join-Path $repo ('test-results/native-identity-' + (Get-Date -Format 'yyyyMMdd-HHmmss')) }
New-Item -ItemType Directory -Force $OutputDirectory | Out-Null
$previous = @{D4R_ZLUDA_NATIVE_DIR=$env:D4R_ZLUDA_NATIVE_DIR; D4R_ZLUDA_VERBOSE=$env:D4R_ZLUDA_VERBOSE; D4R_DIAG_DIR=$env:D4R_DIAG_DIR}
function Invoke-NativeProbe([string]$Name) {
    $info = New-Object Diagnostics.ProcessStartInfo
    $info.FileName = Join-Path $PackageRoot 'bin/d4r_cuda_driver_probe.exe'
    $info.Arguments = '--hip-root "' + $HipRoot + '" --cuda-dll "' + (Join-Path $ZludaRoot 'nvcuda.dll') + '" --iterations 32'
    $info.UseShellExecute=$false; $info.CreateNoWindow=$true; $info.RedirectStandardOutput=$true; $info.RedirectStandardError=$true
    $process = New-Object Diagnostics.Process
    $process.StartInfo=$info
    if (!$process.Start()) { throw 'Cannot start native override probe' }
    $stdout=$process.StandardOutput.ReadToEndAsync(); $stderr=$process.StandardError.ReadToEndAsync()
    if (!$process.WaitForExit(120000)) { $process.Kill(); $process.WaitForExit(); throw 'Native override test timed out' }
    [IO.File]::WriteAllText((Join-Path $OutputDirectory "$Name.stdout.log"),$stdout.GetAwaiter().GetResult())
    [IO.File]::WriteAllText((Join-Path $OutputDirectory "$Name.stderr.log"),$stderr.GetAwaiter().GetResult())
    $code=$process.ExitCode; $process.Dispose()
    return $code
}
try {
    $env:D4R_ZLUDA_VERBOSE = '1'; $env:D4R_DIAG_DIR = $OutputDirectory
    $bad = Join-Path $OutputDirectory 'wrong-identity'
    $good = Join-Path $OutputDirectory 'matching-identity'
    New-Item -ItemType Directory -Force $bad,$good | Out-Null
    foreach ($directory in @($bad,$good)) {
        Copy-Item -LiteralPath (Join-Path $PackageRoot 'bin/probe_gfx1201.hsaco') -Destination (Join-Path $directory 'd4r_ptx_pattern.hsaco') -Force
    }
    'd4r_ptx_pattern 0000000000000000' | Set-Content -LiteralPath (Join-Path $bad 'd4r-kernels.txt') -Encoding ASCII
    $env:D4R_ZLUDA_NATIVE_DIR = $bad
    if (Invoke-NativeProbe 'wrong') { throw 'Wrong identity must use the correct translated fallback' }
    if (!(Select-String -LiteralPath (Join-Path $OutputDirectory 'wrong.stderr.log') -Pattern 'reason=ptx_identity.*fallback=translated' -Quiet)) {
        throw 'Wrong identity was not explicitly rejected'
    }
    $identity = Select-String -LiteralPath (Join-Path $OutputDirectory 'wrong.stdout.log') -Pattern 'fnv1a64=([0-9a-f]{16})'
    if (!$identity) { throw 'PTX probe did not report its identity' }
    "d4r_ptx_pattern $($identity.Matches[0].Groups[1].Value)" | Set-Content -LiteralPath (Join-Path $good 'd4r-kernels.txt') -Encoding ASCII
    $env:D4R_ZLUDA_NATIVE_DIR = $good
    if (Invoke-NativeProbe 'matching') { throw 'Matching native override produced incorrect output' }
    $hits = @(Select-String -LiteralPath (Join-Path $OutputDirectory 'matching.stderr.log') -Pattern '\[d4r-launch\].*backend=native')
    if ($hits.Count -ne 32) { throw "Expected 32 native launches, got $($hits.Count)" }
    Write-Host 'PASS NATIVE_IDENTITY wrong_hash=fallback matching_hash=native launches=32'
} finally {
    foreach ($name in $previous.Keys) { [Environment]::SetEnvironmentVariable($name,$previous[$name],'Process') }
}
