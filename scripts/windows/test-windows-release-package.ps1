# Drag-in release packaging with synthetic per-target inputs. No GPU, no
# NVIDIA DLLs, no game. Code objects carry real AMDGPU ELF target headers.
[CmdletBinding()]
param([string]$OutputDirectory)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'gpu-target.ps1')
$repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (!$OutputDirectory) { $OutputDirectory=Join-Path $repo ('test-results/windows-release-package/'+(Get-Date -Format 'yyyyMMdd-HHmmss-fff')) }
$OutputDirectory=[IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Force $OutputDirectory | Out-Null
$targets=@(Get-D4RGpuTargets)

function Write-CodeObject([string]$Path,[int]$Machine) {
    $header=New-Object byte[] 64
    $header[0]=127; $header[1]=69; $header[2]=76; $header[3]=70; $header[4]=2; $header[5]=1; $header[18]=224; $header[48]=$Machine
    New-Item -ItemType Directory -Force (Split-Path $Path) | Out-Null
    [IO.File]::WriteAllBytes($Path,$header)
}
function New-Inputs([string]$Prefix,[scriptblock]$Mutate) {
    foreach ($target in $targets) {
        $arch=$target.architecture; $root="$Prefix-$arch"
        $text=@{'OptiScaler.dll'='optiscaler'; 'd4r/_nvngx.dll'="shim $arch"; 'd4r/nvapi/nvapi64.dll'="nvapi $arch";
            'd4r/hip/bin/amdhip64_7.dll'='hip'; 'd4r/zluda/nvcuda.dll'='zluda'; 'licenses/d4r.txt'='license';
            'd4r/d4r_gpu_inventory.exe'='inventory'; 'd4r/d4r_debug_launcher.exe'='launcher'; 'windows-game.ps1'='installer';
            'd4r/native/d4r-kernels.txt'='manifest'}
        foreach ($entry in $text.GetEnumerator()) {
            $path=Join-Path $root $entry.Key; New-Item -ItemType Directory -Force (Split-Path $path) | Out-Null
            Set-Content -LiteralPath $path $entry.Value -Encoding ASCII
        }
        $objects=@("d4r/pixel_convert_$arch.hsaco")+@(1..16 | ForEach-Object { "d4r/native/kernel$_.hsaco" })
        foreach ($object in $objects) { Write-CodeObject (Join-Path $root $object) $target.elfMachine }
        if ($Mutate) { & $Mutate $root $arch }
        $files=@(Get-ChildItem -LiteralPath $root -Recurse -File | Where-Object { $_.Name -ne 'package.json' } | ForEach-Object {
            @{path=$_.FullName.Substring($root.Length+1).Replace('\','/'); sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash}
        })
        @{architecture=$arch; d4rWorkingTreeDirty=$false; d4rCommit='fixture'; hardwareValidation='fixture';
          dlssSha256='3975567b8943c53acce397f2b72380092f84f162d00b0d2c7d08a1025c563983';
          zludaBuild=@{sourceCommit='fixture'}; optiScaler=@{sourceCommit='fixture'}; hipRuntime='fixture'; files=$files} |
            ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $root 'package.json') -Encoding UTF8
    }
}
function Invoke-Package([string]$Prefix,[string]$Root) {
    & (Join-Path $PSScriptRoot 'package-windows-release.ps1') -GamePackagePrefix $Prefix -LoaderDll $loader -PackageRoot $Root | Out-Null
}
$loader=Join-Path $OutputDirectory 'build/d4r_nvngx_loader.dll'
New-Item -ItemType Directory -Force (Split-Path $loader) | Out-Null
Set-Content -LiteralPath $loader 'loader fixture' -Encoding ASCII
$results=@()

# 1. Layout of a good release.
$prefix=Join-Path $OutputDirectory 'inputs/good/game'
New-Inputs $prefix
$release=Join-Path $OutputDirectory 'release'
Invoke-Package $prefix $release
foreach ($path in @('dxgi.dll','OptiScaler.ini','d4r/_nvngx.dll','d4r/d4r.ini','d4r/dll-pins.txt','d4r/package.json',
                    'd4r/hip/bin/amdhip64_7.dll','d4r/zluda/nvcuda.dll','d4r/licenses/d4r.txt','d4r/ngx/PUT _nvngx.dll HERE.txt')) {
    if (!(Test-Path -LiteralPath (Join-Path $release $path))) { throw "Release lacks $path" }
}
if ((Get-Content -LiteralPath (Join-Path $release 'd4r/_nvngx.dll') -Raw).Trim() -ne 'loader fixture') { throw 'd4r/_nvngx.dll is not the loader.' }
foreach ($target in $targets) {
    $arch=$target.architecture
    if ((Get-Content -LiteralPath (Join-Path $release "d4r/_nvngx_$arch.dll") -Raw).Trim() -ne "shim $arch") { throw "Wrong shim for $arch" }
    if ((Get-Content -LiteralPath (Join-Path $release "d4r/nvapi/$arch/nvapi64.dll") -Raw).Trim() -ne "nvapi $arch") { throw "Wrong NVAPI for $arch" }
    Assert-D4RCodeObjectTarget (Join-Path $release "d4r/pixel_convert_$arch.hsaco") $arch
    $native=@(Get-ChildItem -LiteralPath (Join-Path $release "d4r/native/$arch") -Filter '*.hsaco')
    if ($native.Count -ne 16 -or !(Test-Path -LiteralPath (Join-Path $release "d4r/native/$arch/d4r-kernels.txt"))) { throw "Native folder for $arch is incomplete" }
    foreach ($object in $native) { Assert-D4RCodeObjectTarget $object.FullName $arch }
}
foreach ($excluded in @('d4r/d4r_gpu_inventory.exe','d4r/d4r_debug_launcher.exe','windows-game.ps1','d4r/native/kernel1.hsaco','d4r/nvapi/nvapi64.dll')) {
    if (Test-Path -LiteralPath (Join-Path $release $excluded)) { throw "Installer-only or untargeted file shipped: $excluded" }
}
$manifest=Get-Content -LiteralPath (Join-Path $release 'd4r/package.json') -Raw | ConvertFrom-Json
foreach ($file in $manifest.files) {
    if ((Get-FileHash -LiteralPath (Join-Path $release $file.path) -Algorithm SHA256).Hash -ne $file.sha256) { throw "Manifest hash wrong: $($file.path)" }
}
$shipped=@(Get-ChildItem -LiteralPath $release -Recurse -File | ForEach-Object { $_.FullName.Substring($release.Length+1).Replace('\','/') } | Where-Object { $_ -ne 'd4r/package.json' })
if (Compare-Object $shipped @($manifest.files.path)) { throw 'Manifest and shipped files differ.' }
$results+=@{case='drag-in-layout'; passed=$true; targets=$targets.Count; files=$shipped.Count}

# 2. A code object for another GPU in a target's input is rejected.
$prefix=Join-Path $OutputDirectory 'inputs/mixed/game'
New-Inputs $prefix { param($root,$arch) if ($arch -eq 'gfx1101') { Write-CodeObject (Join-Path $root 'd4r/native/kernel3.hsaco') 65 } }
$rejected=$false
try { Invoke-Package $prefix (Join-Path $OutputDirectory 'release-mixed') } catch { $rejected=$_.Exception.Message -match 'does not target gfx1101' }
if (!$rejected) { throw 'A gfx1100 code object in the gfx1101 input was not rejected.' }
$results+=@{case='mixed-target-rejected'; passed=$true}

# 3. Shared runtime files must be identical across targets.
$prefix=Join-Path $OutputDirectory 'inputs/runtime/game'
New-Inputs $prefix { param($root,$arch) if ($arch -eq 'gfx1200') { Set-Content -LiteralPath (Join-Path $root 'd4r/zluda/nvcuda.dll') 'other zluda' -Encoding ASCII } }
$rejected=$false
try { Invoke-Package $prefix (Join-Path $OutputDirectory 'release-runtime') } catch { $rejected=$_.Exception.Message -match 'Shared runtime differs' }
if (!$rejected) { throw 'A differing shared runtime was not rejected.' }
$results+=@{case='shared-runtime-mismatch-rejected'; passed=$true}

$results | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $OutputDirectory 'summary.json') -Encoding UTF8
Write-Host "PASS WINDOWS_RELEASE_PACKAGE cases=$($results.Count) Report: $OutputDirectory"
