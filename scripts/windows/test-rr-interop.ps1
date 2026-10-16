[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Probe,
    [Parameter(Mandatory=$true)][string]$OutputDirectory
)
$ErrorActionPreference='Stop'
$repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$Probe=(Get-Item -LiteralPath $Probe).FullName
$OutputDirectory=[IO.Path]::GetFullPath($OutputDirectory)
if(!$OutputDirectory.StartsWith([IO.Path]::GetFullPath((Join-Path $repo 'build'))+'\',[StringComparison]::OrdinalIgnoreCase)){
    throw 'RR interop probe reports must stay under ignored build/.'
}
if(Test-Path -LiteralPath $OutputDirectory){throw 'Use a new output directory.'}
New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
$cases=@(
    @{name='smoke';bytes=1048576;iterations=8},
    @{name='render-colour-1712x960-rgba16f-padded';bytes=13271040;iterations=3;width=1712;height=960},
    @{name='output-colour-2560x1440-rgba16f';bytes=29491200;iterations=3;width=2560;height=1440}
)
$results=@()
foreach($case in $cases){
    $stdout=Join-Path $OutputDirectory "$($case.name).stdout.log"
    $stderr=Join-Path $OutputDirectory "$($case.name).stderr.log"
    $arguments=@('--bytes',"$($case.bytes)",'--iterations',"$($case.iterations)")
    if($case.width){$arguments+=@('--image-width',"$($case.width)",'--image-height',"$($case.height)")}
    $process=Start-Process -FilePath $Probe -ArgumentList $arguments `
        -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    try{
        if(!$process.WaitForExit(30000)){
            Stop-Process -Id $process.Id -ErrorAction SilentlyContinue
            throw "Exact RR interop probe stopped after 30 seconds: $($case.name)"
        }
        $process.WaitForExit()
        $text=Get-Content -LiteralPath $stdout -Raw
        $records=[regex]::Matches($text,'(?m)^RR_INTEROP_ITERATION .* exact=1 ')
        if($process.ExitCode -ne 0 -or $records.Count -ne $case.iterations -or
            $text -notmatch 'PASS RR_INTEROP_TRANSFER .*gpu_fence_handoff=1 .*inference_executed=0'){
            throw "RR interop case failed ($($process.ExitCode)): $($case.name); see $OutputDirectory"
        }
        if($case.width -and $text -notmatch 'canonical_image=1'){throw 'Canonical image path was not exercised.'}
        $results+=@{name=$case.name;bytes=$case.bytes;iterations=$case.iterations;exactBothDirections=$true;canonicalImage=[bool]$case.width}
        Write-Host "PASS RR_INTEROP $($case.name): $($case.iterations) iterations, $($case.bytes) bytes"
    }finally{$process.Dispose()}
}
@{passed=$true;probeSha256=(Get-FileHash -LiteralPath $Probe).Hash;cases=$results;
    gpuFenceHandoff=$true;inferenceExecuted=$false;gameAccessed=$false} |
    ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $OutputDirectory 'summary.json') -Encoding UTF8
