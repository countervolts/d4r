[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Probe,
    [Parameter(Mandatory=$true)][string]$DenoiserDll,
    [Parameter(Mandatory=$true)][string]$OutputDirectory,
    [switch]$QueryOnly,
    [ValidateRange(1,600)][int]$TimeoutSeconds=120
)
$ErrorActionPreference='Stop'
$repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$Probe=(Get-Item -LiteralPath $Probe).FullName
$DenoiserDll=(Get-Item -LiteralPath $DenoiserDll).FullName
$expected='48f1e5888ba6a0a3d59a98b9751e37c392b0f7b8c223d0082d5c1f40642879d3'
if((Get-FileHash -LiteralPath $DenoiserDll -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected){
    throw 'Expected unmodified AMD SDK 2.3 denoiser 1.2.0.2740.'
}
$OutputDirectory=[IO.Path]::GetFullPath($OutputDirectory)
$buildRoot=[IO.Path]::GetFullPath((Join-Path $repo 'build'))
if(!$OutputDirectory.StartsWith($buildRoot+'\',[StringComparison]::OrdinalIgnoreCase)){
    throw 'Probe output must stay under the ignored build directory.'
}
if(Test-Path -LiteralPath $OutputDirectory){throw 'Use a new probe output directory.'}
New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
# Start-Process joins arguments into one command line. Reject embedded quotes;
# quote absolute file paths so spaces in the workspace are supported.
foreach($path in @($DenoiserDll,$OutputDirectory)){
    if($path.Contains('"')){throw 'Embedded quotes are not supported in probe paths.'}
}
$arguments='"'+$DenoiserDll+'" "'+$OutputDirectory+'" --size 64x64 --frames 48'
if($QueryOnly){$arguments+=' --query-only'}
$process=Start-Process -FilePath $Probe -ArgumentList $arguments -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $OutputDirectory 'stdout.log') `
    -RedirectStandardError (Join-Path $OutputDirectory 'stderr.log')
try{
    $deadline=[DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    while(!$process.WaitForExit(1000)){
        if([DateTime]::UtcNow -ge $deadline){
            Stop-Process -Id $process.Id -ErrorAction SilentlyContinue
            throw "FSR RR probe timed out after $TimeoutSeconds seconds; exact probe process stopped."
        }
    }
    $process.WaitForExit()
    Get-Content -LiteralPath (Join-Path $OutputDirectory 'stdout.log')
    if($process.ExitCode -ne 0){throw "FSR RR probe failed ($($process.ExitCode)); see $OutputDirectory"}
}finally{$process.Dispose()}
