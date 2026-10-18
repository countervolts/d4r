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
# Stand-in compatibility NVAPI: names its build target the way the real DLL does (nvapi_trace.cpp).
function Write-Nvapi([string]$Path,[string]$Target) {
    New-Item -ItemType Directory -Force (Split-Path $Path) | Out-Null
    [IO.File]::WriteAllBytes($Path,[Text.Encoding]::ASCII.GetBytes("MZ nvapi fixture D4R_NVAPI_TARGET=$Target")+[byte]0)
}
function New-Inputs([string]$Prefix,[scriptblock]$Mutate) {
    foreach ($target in $targets) {
        $arch=$target.architecture; $root="$Prefix-$arch"
        $text=@{'OptiScaler.dll'='optiscaler'; 'd4r/_nvngx.dll'="shim $arch";             'd4r/nvapi/backend/nvapi64.dll'='raw backend nvapi'; 'd4r/hip/bin/amdhip64_7.dll'='hip'; 'd4r/zluda/nvcuda.dll'='zluda'; 'licenses/d4r.txt'='license';
            'd4r/d4r_gpu_inventory.exe'='inventory'; 'd4r/d4r_debug_launcher.exe'='launcher'; 'windows-game.ps1'='installer';
            'd4r/native/d4r-kernels.txt'='manifest'}
        foreach ($entry in $text.GetEnumerator()) {
            $path=Join-Path $root $entry.Key; New-Item -ItemType Directory -Force (Split-Path $path) | Out-Null
            Set-Content -LiteralPath $path $entry.Value -Encoding ASCII
        }
        # As the game package stages it: the target's own build, in both places.
        Write-Nvapi (Join-Path $root 'd4r/nvapi/nvapi64.dll') $arch
        Write-Nvapi (Join-Path $root 'd4r/zluda/nvapi64.dll') $arch
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
                    'd4r/hip/bin/amdhip64_7.dll','d4r/zluda/nvcuda.dll','d4r/licenses/d4r.txt','d4r/ngx/PUT _nvngx.dll HERE.txt',
                    'd4r/gpu-targets.json','d4r/d4r_gpu_inventory.exe','d4r/d4r-check.ps1','WINDOWS_README.txt',
                    'install.bat','install-dependencies.py','d4r/model-tools/python/python.exe','d4r/model-tools/rr-tools.json')) {
    if (!(Test-Path -LiteralPath (Join-Path $release $path))) { throw "Release lacks $path" }
}
if ((Get-Content -LiteralPath (Join-Path $release 'd4r/_nvngx.dll') -Raw).Trim() -ne 'loader fixture') { throw 'd4r/_nvngx.dll is not the loader.' }
foreach ($target in $targets) {
    $arch=$target.architecture
    if ((Get-Content -LiteralPath (Join-Path $release "d4r/_nvngx_$arch.dll") -Raw).Trim() -ne "shim $arch") { throw "Wrong shim for $arch" }
    Assert-D4RNvapiTarget (Join-Path $release "d4r/nvapi/$arch/nvapi64.dll") $arch
    foreach ($other in $targets) {
        if ($other.architecture -ne $arch) {
            $mismatch=$false
            try { Assert-D4RNvapiTarget (Join-Path $release "d4r/nvapi/$arch/nvapi64.dll") $other.architecture } catch { $mismatch=$_.Exception.Message -match "built for $arch" }
            if (!$mismatch) { throw "NVAPI for $arch passed as $($other.architecture)" }
        }
    }
    Assert-D4RCodeObjectTarget (Join-Path $release "d4r/pixel_convert_$arch.hsaco") $arch
    $native=@(Get-ChildItem -LiteralPath (Join-Path $release "d4r/native/$arch") -Filter '*.hsaco')
    if ($native.Count -ne 16 -or !(Test-Path -LiteralPath (Join-Path $release "d4r/native/$arch/d4r-kernels.txt"))) { throw "Native folder for $arch is incomplete" }
    foreach ($object in $native) { Assert-D4RCodeObjectTarget $object.FullName $arch }
}
if ((Get-Content -LiteralPath (Join-Path $release 'd4r/nvapi/backend/nvapi64.dll') -Raw).Trim() -ne 'raw backend nvapi') { throw 'The raw ZLUDA NVAPI backend was not staged separately.' }
if (Test-Path -LiteralPath (Join-Path $release 'd4r/zluda/nvapi64.dll')) { throw 'A shared NVAPI beside nvcuda.dll would be loaded for every GPU; it must not ship.' }
if (@(Get-ChildItem -LiteralPath $release -Recurse -Filter 'd4r_gpu_inventory.exe').Count -ne 1) { throw 'The GPU query tool must ship exactly once.' }
foreach ($excluded in @('d4r/d4r_debug_launcher.exe','windows-game.ps1','d4r/native/kernel1.hsaco','d4r/nvapi/nvapi64.dll','d4r/zluda/nvapi64.dll')) {
    if (Test-Path -LiteralPath (Join-Path $release $excluded)) { throw "Installer-only or untargeted file shipped: $excluded" }
}
$manifest=Get-Content -LiteralPath (Join-Path $release 'd4r/package.json') -Raw | ConvertFrom-Json
foreach ($file in $manifest.files) {
    if ((Get-FileHash -LiteralPath (Join-Path $release $file.path) -Algorithm SHA256).Hash -ne $file.sha256) { throw "Manifest hash wrong: $($file.path)" }
}
$shipped=@(Get-ChildItem -LiteralPath $release -Recurse -File | ForEach-Object { $_.FullName.Substring($release.Length+1).Replace('\','/') } | Where-Object { $_ -ne 'd4r/package.json' })
if (Compare-Object $shipped @($manifest.files.path)) { throw 'Manifest and shipped files differ.' }
$results+=@{case='drag-in-layout'; passed=$true; targets=$targets.Count; files=$shipped.Count}

# 1b. A subset release carries only the requested targets.
$subset=Join-Path $OutputDirectory 'release-subset'
& (Join-Path $PSScriptRoot 'package-windows-release.ps1') -GamePackagePrefix $prefix -LoaderDll $loader -PackageRoot $subset -Architectures gfx1101,gfx1201 | Out-Null
$kept=@(Get-ChildItem -LiteralPath (Join-Path $subset 'd4r') -Filter '_nvngx_gfx*.dll' | ForEach-Object { $_.Name })
if (($kept | Sort-Object) -join ',' -ne '_nvngx_gfx1101.dll,_nvngx_gfx1201.dll') { throw "Subset shipped the wrong shims: $($kept -join ',')" }
$rejected=$false
try { & (Join-Path $PSScriptRoot 'package-windows-release.ps1') -GamePackagePrefix $prefix -LoaderDll $loader -PackageRoot (Join-Path $OutputDirectory 'release-unknown') -Architectures gfx9999 | Out-Null } catch { $rejected=$_.Exception.Message -match 'Unknown GPU target' }
if (!$rejected) { throw 'An unknown architecture was not rejected.' }
$results+=@{case='architecture-subset'; passed=$true; shipped=$kept}

# 1c. The optional FSR payload (vkd3d-proton core + amdxc64 shim) ships under d4r/fsr with its licences and source
#     offer; AMD's denoiser never ships; a payload that claims to bundle it, or a changed file, is rejected.
function New-FsrPayload([string]$Root,[bool]$ClaimsAmd) {
    $files=[ordered]@{'fsr/d4r_vkd3d_core.dll'='core fixture'; 'fsr/amdxc64.dll'='shim fixture'; 'SOURCE.txt'='source offer';
                      'licenses/vkd3d-proton-LGPL-2.1.txt'='lgpl'; 'licenses/dxil-spirv-MIT.txt'='mit'}
    $hashes=[ordered]@{}
    foreach ($entry in $files.GetEnumerator()) {
        $path=Join-Path $Root $entry.Key; New-Item -ItemType Directory -Force (Split-Path $path) | Out-Null
        Set-Content -LiteralPath $path $entry.Value -Encoding ASCII
        $hashes[$entry.Key]=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    @{schema=1; kind='d4r-fsr-payload'; amdDenoiserBundled=$ClaimsAmd; files=$hashes} | ConvertTo-Json -Depth 4 |
        Set-Content -LiteralPath (Join-Path $Root 'fsr-payload.json') -Encoding UTF8
}
$goodPrefix=Join-Path $OutputDirectory 'inputs/good/game'
$payload=Join-Path $OutputDirectory 'fsr-payload'; New-FsrPayload $payload $false
$fsrRelease=Join-Path $OutputDirectory 'release-fsr'
& (Join-Path $PSScriptRoot 'package-windows-release.ps1') -GamePackagePrefix $goodPrefix -LoaderDll $loader -PackageRoot $fsrRelease -FsrPayload $payload | Out-Null
foreach ($path in @('d4r/fsr/d4r_vkd3d_core.dll','d4r/fsr/amdxc64.dll','d4r/fsr/SOURCE.txt','d4r/licenses/vkd3d-proton-LGPL-2.1.txt',
                    'd4r/licenses/dxil-spirv-MIT.txt','d4r/fsr/patches/0001-dxil-spirv-include-exception-for-libcxx.patch')) {
    if (!(Test-Path -LiteralPath (Join-Path $fsrRelease $path))) { throw "FSR release lacks $path" }
}
if (Get-ChildItem -LiteralPath $fsrRelease -Recurse -Filter 'amd_fidelityfx_denoiser_dx12.dll') { throw "AMD's denoiser must never ship." }
$fsrManifest=Get-Content -LiteralPath (Join-Path $fsrRelease 'd4r/package.json') -Raw | ConvertFrom-Json
foreach ($file in $fsrManifest.files) {
    if ((Get-FileHash -LiteralPath (Join-Path $fsrRelease $file.path) -Algorithm SHA256).Hash -ne $file.sha256) { throw "FSR manifest hash wrong: $($file.path)" }
}
$fsrShipped=@(Get-ChildItem -LiteralPath $fsrRelease -Recurse -File | ForEach-Object { $_.FullName.Substring($fsrRelease.Length+1).Replace('\','/') } | Where-Object { $_ -ne 'd4r/package.json' })
if (Compare-Object $fsrShipped @($fsrManifest.files.path)) { throw 'FSR release manifest and shipped files differ.' }
if (Test-Path -LiteralPath (Join-Path $release 'd4r/fsr')) { throw 'The default release must not carry the FSR payload.' }
$claims=Join-Path $OutputDirectory 'fsr-payload-claims-amd'; New-FsrPayload $claims $true
$rejected=$false
try { & (Join-Path $PSScriptRoot 'package-windows-release.ps1') -GamePackagePrefix $goodPrefix -LoaderDll $loader -PackageRoot (Join-Path $OutputDirectory 'release-fsr-claims') -FsrPayload $claims | Out-Null } catch { $rejected=$_.Exception.Message -match 'Unsupported FSR payload' }
if (!$rejected) { throw 'A payload claiming to bundle AMD software was not rejected.' }
$tampered=Join-Path $OutputDirectory 'fsr-payload-tampered'; New-FsrPayload $tampered $false
Set-Content -LiteralPath (Join-Path $tampered 'fsr/amdxc64.dll') 'changed after hashing' -Encoding ASCII
$rejected=$false
try { & (Join-Path $PSScriptRoot 'package-windows-release.ps1') -GamePackagePrefix $goodPrefix -LoaderDll $loader -PackageRoot (Join-Path $OutputDirectory 'release-fsr-tampered') -FsrPayload $tampered | Out-Null } catch { $rejected=$_.Exception.Message -match 'FSR payload file changed' }
if (!$rejected) { throw 'A changed FSR payload file was not rejected.' }
$results+=@{case='fsr-payload'; passed=$true; files=$fsrShipped.Count}

# 2. A code object for another GPU in a target's input is rejected.
$prefix=Join-Path $OutputDirectory 'inputs/mixed/game'
New-Inputs $prefix { param($root,$arch) if ($arch -eq 'gfx1101') { Write-CodeObject (Join-Path $root 'd4r/native/kernel3.hsaco') 65 } }
$rejected=$false
try { Invoke-Package $prefix (Join-Path $OutputDirectory 'release-mixed') } catch { $rejected=$_.Exception.Message -match 'does not target gfx1101' }
if (!$rejected) { throw 'A gfx1100 code object in the gfx1101 input was not rejected.' }
$results+=@{case='mixed-target-rejected'; passed=$true}

# 2b. NVAPI selection inputs: an NVAPI built for another GPU, one without a target marker (built before the
#     marker existed) and a missing one are rejected; the d4r 0.1.5 input shape (the gfx1101 copy shared beside
#     nvcuda.dll in every input) packages without that copy, and every target still gets its own build.
$cases=[ordered]@{
    'wrong-target'=@{mutate={ param($root,$arch) if ($arch -eq 'gfx1151') { Write-Nvapi (Join-Path $root 'd4r/nvapi/nvapi64.dll') 'gfx1101' } }; expect='built for gfx1101, not gfx1151'}
    'no-marker'=@{mutate={ param($root,$arch) if ($arch -eq 'gfx1102') { Set-Content -LiteralPath (Join-Path $root 'd4r/nvapi/nvapi64.dll') 'legacy nvapi' -Encoding ASCII } }; expect='does not name exactly one GPU target'}
    'missing'=@{mutate={ param($root,$arch) if ($arch -eq 'gfx1200') { Remove-Item -LiteralPath (Join-Path $root 'd4r/nvapi/nvapi64.dll') } }; expect='No d4r/nvapi/nvapi64.dll in the gfx1200 input'}
}
foreach ($case in $cases.GetEnumerator()) {
    $prefix=Join-Path $OutputDirectory "inputs/nvapi-$($case.Key)/game"
    New-Inputs $prefix $case.Value.mutate
    $rejected=$false
    try { Invoke-Package $prefix (Join-Path $OutputDirectory "release-nvapi-$($case.Key)") } catch { $rejected=$_.Exception.Message -like "*$($case.Value.expect)*" }
    if (!$rejected) { throw "NVAPI case '$($case.Key)' was not rejected with '$($case.Value.expect)'." }
}
$prefix=Join-Path $OutputDirectory 'inputs/shared-nvapi/game'
New-Inputs $prefix { param($root,$arch) Write-Nvapi (Join-Path $root 'd4r/zluda/nvapi64.dll') 'gfx1101' }
$sharedRelease=Join-Path $OutputDirectory 'release-shared-nvapi'
Invoke-Package $prefix $sharedRelease
if (Test-Path -LiteralPath (Join-Path $sharedRelease 'd4r/zluda/nvapi64.dll')) { throw 'The shared gfx1101 NVAPI shipped.' }
foreach ($target in $targets) { Assert-D4RNvapiTarget (Join-Path $sharedRelease "d4r/nvapi/$($target.architecture)/nvapi64.dll") $target.architecture }
$results+=@{case='nvapi-per-target'; passed=$true; targets=$targets.Count}

# 3. Shared runtime files must be identical across targets.
$prefix=Join-Path $OutputDirectory 'inputs/runtime/game'
New-Inputs $prefix { param($root,$arch) if ($arch -eq 'gfx1200') { Set-Content -LiteralPath (Join-Path $root 'd4r/zluda/nvcuda.dll') 'other zluda' -Encoding ASCII } }
$rejected=$false
try { Invoke-Package $prefix (Join-Path $OutputDirectory 'release-runtime') } catch { $rejected=$_.Exception.Message -match 'Shared runtime differs' }
if (!$rejected) { throw 'A differing shared runtime was not rejected.' }
$results+=@{case='shared-runtime-mismatch-rejected'; passed=$true}

# Local ZIPs may include uncommitted inputs, but remain under ignored build/ and
# preserve all target/hash/runtime checks. This fixture is never a usable game ZIP.
$prefix=Join-Path $OutputDirectory 'inputs/development/game'
New-Inputs $prefix
foreach($target in $targets){
    $metadataPath="$prefix-$($target.architecture)/package.json"
    $metadata=Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
    $metadata.d4rWorkingTreeDirty=$true
    $metadata | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $metadataPath -Encoding UTF8
}
$developmentRoot=Join-Path $repo ('build/release-development-fixture-'+[Guid]::NewGuid().ToString('N'))
$zip=Join-Path $developmentRoot 'fixture.zip'
& (Join-Path $PSScriptRoot 'package-windows-release.ps1') -GamePackagePrefix $prefix -LoaderDll $loader `
    -PackageRoot (Join-Path $developmentRoot 'package') -ArchivePath $zip -LocalDevelopment | Out-Null
$metadata=Get-Content -LiteralPath (Join-Path $developmentRoot 'package/d4r/package.json') -Raw | ConvertFrom-Json
if(!$metadata.localDevelopment){throw 'Local ZIP lacks development marker.'}
$archive=[IO.Compression.ZipFile]::OpenRead($zip)
try{
    foreach($required in @('install.bat','install-dlss.ps1','install-dependencies.py','d4r/d4r.ini','d4r/model-tools/python/python.exe')){
        if(!$archive.GetEntry($required)){throw "Local ZIP lacks $required"}
    }
}finally{$archive.Dispose()}
$results+=@{case='local-development-zip'; passed=$true}
$rejected=$false
try{
    & (Join-Path $PSScriptRoot 'package-windows-release.ps1') -GamePackagePrefix $prefix -LoaderDll $loader `
        -PackageRoot (Join-Path $repo 'dist/rejected-local-fixture') -LocalDevelopment | Out-Null
}catch{$rejected=$_.Exception.Message -match 'must stay under ignored build/'}
if(!$rejected){throw 'Local package escaped build/.'}
$results+=@{case='local-development-path-rejected'; passed=$true}

$results | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $OutputDirectory 'summary.json') -Encoding UTF8
Write-Host "PASS WINDOWS_RELEASE_PACKAGE cases=$($results.Count) Report: $OutputDirectory"
