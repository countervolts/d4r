[CmdletBinding()]
param(
    [string]$SourceRoot,
    [string]$BuildToolsRoot,
    [string]$InstallRoot,
    [int]$Jobs = 6
)
$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (!$SourceRoot) { $SourceRoot = Join-Path $repo 'external/OptiScaler' }
if (!$InstallRoot) { $InstallRoot = Join-Path $repo 'dist/optiscaler-windows-d4r' }
$SourceRoot = [IO.Path]::GetFullPath($SourceRoot)
$InstallRoot = [IO.Path]::GetFullPath($InstallRoot)
$base = '45a2001303ddff632e279f77aef85ceede5832cb'
$patch = Join-Path $repo 'patches/optiscaler/0001-windows-external-d4r-backend.patch'
$previousPath = $env:PATH
try {
    # Git for Windows' submodule helper needs its POSIX utility directory.
    $gitExe = (Get-Command git -ErrorAction Stop).Source
    $gitInstall = [IO.Path]::GetFullPath((Join-Path (Split-Path $gitExe) '..'))
    $gitUtilities = Join-Path $gitInstall 'usr/bin'
    if (Test-Path -LiteralPath $gitUtilities) { $env:PATH = "$gitUtilities;$env:PATH" }
    if (!(Test-Path -LiteralPath (Join-Path $SourceRoot '.git'))) {
        & git clone --filter=blob:none --no-checkout https://github.com/optiscaler/OptiScaler.git $SourceRoot
        if ($LASTEXITCODE) { throw 'OptiScaler clone failed.' }
        & git -C $SourceRoot checkout --detach $base
        if ($LASTEXITCODE) { throw 'Pinned OptiScaler checkout failed.' }
    }
    & git -C $SourceRoot merge-base --is-ancestor $base HEAD
    if ($LASTEXITCODE) { throw "OptiScaler must be based on audited commit $base" }
    $ErrorActionPreference = 'Continue'
    & git -C $SourceRoot apply --reverse --check $patch 2>$null
    $alreadyApplied = $LASTEXITCODE -eq 0
    $ErrorActionPreference = 'Stop'
    if (!$alreadyApplied) {
        & git -C $SourceRoot apply --check $patch
        if ($LASTEXITCODE) { throw 'External backend patch conflicts with the existing source; no files changed.' }
        & git -C $SourceRoot apply $patch
        if ($LASTEXITCODE) { throw 'External backend patch failed.' }
    }
    & git -C $SourceRoot submodule update --init --depth 1
    if ($LASTEXITCODE) { throw 'Pinned OptiScaler submodules failed.' }
    if (!$BuildToolsRoot -and (Test-Path -LiteralPath (Join-Path $repo '.tools/vs2022/MSBuild/Current/Bin/MSBuild.exe'))) {
        $BuildToolsRoot = Join-Path $repo '.tools/vs2022'
    }
    if (!$BuildToolsRoot) {
        $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
        if (Test-Path -LiteralPath $vswhere) {
            $BuildToolsRoot = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        }
    }
    if (!$BuildToolsRoot) { throw 'MSVC v143 Build Tools and Windows SDK 10.0.26100 are required; see patches/optiscaler/README.md.' }
    $msbuild = Join-Path $BuildToolsRoot 'MSBuild/Current/Bin/MSBuild.exe'
    if (!(Test-Path -LiteralPath $msbuild)) { throw "MSBuild missing: $msbuild" }
    New-Item -ItemType Directory -Force $InstallRoot | Out-Null
    $log = Join-Path $InstallRoot 'build.log'
    $ErrorActionPreference = 'Continue'
    & $msbuild (Join-Path $SourceRoot 'OptiScaler.sln') "/m:$Jobs" /p:Configuration=Release /p:Platform=x64 `
        /p:PostBuildEventUseInBuild=false /verbosity:quiet "/flp:logfile=$log;verbosity=normal"
    $ErrorActionPreference = 'Stop'
    if ($LASTEXITCODE) { throw "OptiScaler build failed ($LASTEXITCODE); see $log" }
    foreach ($file in @('OptiScaler.dll', 'OptiScaler.pdb')) {
        Copy-Item -LiteralPath (Join-Path $SourceRoot "x64/Release/$file") -Destination $InstallRoot -Force
    }
    Copy-Item -LiteralPath (Join-Path $SourceRoot 'LICENSE') -Destination (Join-Path $InstallRoot 'GPL-3.0.txt') -Force
    Copy-Item -LiteralPath $patch -Destination $InstallRoot -Force
    $metadata = @{base=$base; sourceCommit=(& git -C $SourceRoot rev-parse HEAD).Trim();
        sourceDirty=[bool](& git -C $SourceRoot status --porcelain); backendAbi=1;
        buildTools=$BuildToolsRoot; msbuildVersion=(& $msbuild -version -nologo | Select-Object -Last 1);
        patchSha256=(Get-FileHash -LiteralPath $patch -Algorithm SHA256).Hash;
        dllSha256=(Get-FileHash -LiteralPath (Join-Path $InstallRoot 'OptiScaler.dll') -Algorithm SHA256).Hash;
        submodules=@(& git -C $SourceRoot submodule status)}
    $metadata | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $InstallRoot 'build-info.json') -Encoding UTF8
    Write-Host "Built OptiScaler with external Windows d4r backend: $InstallRoot"
} finally { $env:PATH = $previousPath }
