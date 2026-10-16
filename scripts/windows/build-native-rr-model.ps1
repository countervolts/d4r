[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$DlssdDll,
    [ValidateSet('D','E')][string]$Preset='D',
    [string]$OutputDirectory,
    [string]$GitBash='C:\Program Files\Git\bin\bash.exe'
)
$ErrorActionPreference='Stop'
$repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$DlssdDll=(Get-Item -LiteralPath $DlssdDll -ErrorAction Stop).FullName
$GitBash=(Get-Item -LiteralPath $GitBash -ErrorAction Stop).FullName
$expectedHash='4bc7ea5fcb2f32cf86bc2cb072e8d2860914a0be511cbdcb2fc206c1d9d83b80'
$hash=(Get-FileHash -LiteralPath $DlssdDll -Algorithm SHA256).Hash.ToLowerInvariant()
if($hash -ne $expectedHash){throw "Unsupported nvngx_dlssd.dll SHA256 $hash; expected NVIDIA DLSS 310.9.1 ($expectedHash)."}
if(!$OutputDirectory){$OutputDirectory=Join-Path $repo ("build/native-rr-$($Preset.ToLower())-windows-gfx1101")}
$OutputDirectory=[IO.Path]::GetFullPath($OutputDirectory)
$buildRoot=[IO.Path]::GetFullPath((Join-Path $repo 'build'))
if(!$OutputDirectory.StartsWith($buildRoot+'\',[StringComparison]::OrdinalIgnoreCase)){
    throw "RR model output must stay under the ignored build directory: $OutputDirectory"
}
if(Test-Path -LiteralPath $OutputDirectory){throw "Use a new RR model output directory: $OutputDirectory"}
$glslang=Join-Path $repo '.tools/glslang-16.6.0/bin'
$pythonHome=Split-Path (Get-Command python.exe -ErrorAction Stop).Source
$pythonShim=Join-Path $repo '.tools/rr-build-bin'
New-Item -ItemType Directory -Force $pythonShim | Out-Null
$python3=Join-Path $pythonShim 'python3.exe'
Copy-Item -LiteralPath (Get-Command python.exe).Source -Destination $python3 -Force
$pythonVendor=Join-Path $repo '.tools/python/vendor'
$cygpath=Join-Path (Split-Path (Split-Path $GitBash -Parent) -Parent) 'usr/bin/cygpath.exe'
if(!(Test-Path -LiteralPath $cygpath)){throw "Git for Windows cygpath missing: $cygpath"}
foreach($tool in @($python3,(Join-Path $glslang 'glslangValidator.exe'))){if(!(Test-Path -LiteralPath $tool)){throw "RR build tool missing: $tool"}}
$unixRepo=(& $cygpath -u $repo).Trim()
$unixDll=(& $cygpath -u $DlssdDll).Trim()
$unixOut=(& $cygpath -u $OutputDirectory).Trim()
if(!$unixRepo -or !$unixDll -or !$unixOut){throw 'Git Bash path conversion failed.'}
$oldPath=$env:PATH
$oldPythonPath=$env:PYTHONPATH
$env:PATH="$pythonShim;$pythonHome;$glslang;$(Split-Path $GitBash -Parent);$oldPath"
$env:PYTHONPATH=$pythonVendor
$env:D4R_RR_GFX11='1'
$env:D4R_RR_RELAXED='0'
$log=Join-Path $buildRoot ("native-rr-$($Preset.ToLower())-windows-build.log")
try{
    & $python3 -c 'import numpy; print("numpy", numpy.__version__)'
    if($LASTEXITCODE){throw 'Pinned NumPy could not load in the Windows RR build environment.'}
    $command="cd '$unixRepo' && D4R_RR_GFX11=1 D4R_RR_RELAXED=0 bash engine/rr/build_native_model.sh --dll '$unixDll' '$Preset' '$unixOut'"
    $previousPreference=$ErrorActionPreference
    try{
        $ErrorActionPreference='Continue'
        & $GitBash -lc $command *> $log
        $bashExit=$LASTEXITCODE
    }finally{$ErrorActionPreference=$previousPreference}
    if($bashExit){throw "RR model build failed with exit $bashExit; see $log"}
    foreach($required in @('rr.txt','model.json','enc0_m.spv','dec0_m.spv','hkpn.spv','enc0.args','hkpn.args')){
        if(!(Test-Path -LiteralPath (Join-Path $OutputDirectory $required))){throw "RR build incomplete; missing $required"}
    }
    Write-Host "Built local preset $Preset native RR model: $OutputDirectory"
    Write-Host "DLL SHA256: $hash. Build log: $log"
}finally{$env:PATH=$oldPath;$env:PYTHONPATH=$oldPythonPath}
