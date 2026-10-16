[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Probe,
    [Parameter(Mandatory=$true)][string]$ModelDirectory,
    [Parameter(Mandatory=$true)][string]$InputDirectory,
    [Parameter(Mandatory=$true)][string]$ReferenceDirectory,
    [Parameter(Mandatory=$true)][string]$OutputDirectory
)
$ErrorActionPreference='Stop'
$repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
foreach($name in @('Probe','ModelDirectory','InputDirectory','ReferenceDirectory')){
    Set-Variable -Name $name -Value ((Get-Item -LiteralPath (Get-Variable -Name $name -ValueOnly)).FullName)
}
$OutputDirectory=[IO.Path]::GetFullPath($OutputDirectory)
if(!$OutputDirectory.StartsWith([IO.Path]::GetFullPath((Join-Path $repo 'build'))+'\',[StringComparison]::OrdinalIgnoreCase)){
    throw 'RR bridge outputs must stay under ignored build/.'
}
if(Test-Path -LiteralPath $OutputDirectory){throw 'Use a fresh RR bridge output directory.'}
New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
$frames=@(Get-Content -LiteralPath (Join-Path $InputDirectory 'frames.txt') | Where-Object {$_ -match '^frame '}).Count
if(!$frames -or $frames -gt 32){throw 'Need 1..32 RR fixture frames.'}
$arguments=@('--rr-model',$ModelDirectory,'--rr-inputs',$InputDirectory,'--rr-reference',$ReferenceDirectory,
    '--rr-output',(Join-Path $OutputDirectory 'frames'))
foreach($value in $arguments){if($value.Contains('"')){throw 'Embedded quotes are not supported in probe paths.'}}
$quoted=($arguments | ForEach-Object {'"'+$_+'"'}) -join ' '
$process=Start-Process -FilePath $Probe -ArgumentList $quoted -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $OutputDirectory 'stdout.log') `
    -RedirectStandardError (Join-Path $OutputDirectory 'stderr.log')
try{
    $deadline=[DateTime]::UtcNow.AddSeconds(180)
    while(!$process.WaitForExit(1000)){
        if([DateTime]::UtcNow -ge $deadline){Stop-Process -Id $process.Id;throw 'Exact RR bridge process stopped at 180-second limit.'}
    }
    $process.WaitForExit()
    $text=Get-Content -LiteralPath (Join-Path $OutputDirectory 'stdout.log') -Raw
    Get-Content -LiteralPath (Join-Path $OutputDirectory 'stdout.log')
    $records=[regex]::Matches($text,'(?m)^RR_BRIDGED_FRAME .*finite=1 nonuniform=1 standalone_exact=1 ')
    if($process.ExitCode -ne 0 -or $records.Count -ne $frames -or $text -notmatch 'PASS RR_BRIDGED_INFERENCE'){
        Get-Content -LiteralPath (Join-Path $OutputDirectory 'stderr.log')
        throw "RR bridged inference failed ($($process.ExitCode)); see $OutputDirectory"
    }
    $outputs=@(Get-ChildItem -LiteralPath (Join-Path $OutputDirectory 'frames') -Filter '*-out.bin')
    if($outputs.Count -ne $frames){throw 'Incomplete RR output file set.'}
    foreach($output in $outputs){
        if((Get-FileHash -LiteralPath $output.FullName).Hash -ne (Get-FileHash -LiteralPath (Join-Path $ReferenceDirectory $output.Name)).Hash){
            throw 'Independent bridged/reference output hash comparison failed.'
        }
    }
    @{passed=$true;frames=$frames;guides=7;standaloneExact=$true;gpuFenceHandoff=$true;
      gameAccessed=$false;qualityReference=$false;d3d12Textures=$true;probeSha256=(Get-FileHash -LiteralPath $Probe).Hash;
      model=$ModelDirectory;inputs=$InputDirectory;reference=$ReferenceDirectory} |
      ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $OutputDirectory 'summary.json') -Encoding UTF8
}finally{$process.Dispose()}
