# d4r-check.ps1 against fixture installs. No GPU, no NVIDIA binaries: the packaging test's
# synthetic release is the install, a fake inventory file stands in for the GPU query, and
# fixture text files stand in for the NVIDIA DLLs (pinned in a fixture dll-pins.txt).
[CmdletBinding()]
param([string]$OutputDirectory)
$ErrorActionPreference='Stop'
$repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (!$OutputDirectory) { $OutputDirectory=Join-Path $repo ('test-results/d4r-check/'+(Get-Date -Format 'yyyyMMdd-HHmmss-fff')) }
$OutputDirectory=[IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Force $OutputDirectory | Out-Null
& (Join-Path $PSScriptRoot 'test-windows-release-package.ps1') -OutputDirectory (Join-Path $OutputDirectory 'package') | Out-Null
$release=Join-Path $OutputDirectory 'package/release'
$check=Join-Path $repo 'scripts/windows/d4r-check.ps1'
$powershell="$env:SystemRoot/System32/WindowsPowerShell/v1.0/powershell.exe"
$results=@()
function Sha([string]$path) { (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash }
function Set-ManifestHash([string]$game,[string]$relative) {
    $path=Join-Path $game 'd4r/package.json'
    $manifest=Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
    foreach ($entry in $manifest.files) { if ($entry.path -eq $relative) { $entry.sha256=Sha (Join-Path $game ($relative -replace '/','\')) } }
    $manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $path -Encoding UTF8
}
function New-Install([string]$name) {
    $game=Join-Path $OutputDirectory "cases/$name"
    New-Item -ItemType Directory -Force (Split-Path $game) | Out-Null
    Copy-Item -LiteralPath $release -Destination $game -Recurse
    Set-Content -LiteralPath (Join-Path $game 'Game.exe') 'fixture' -Encoding ASCII
    Set-Content -LiteralPath (Join-Path $game 'd4r/nvngx_dlss.dll') 'fixture dlss' -Encoding ASCII
    Set-Content -LiteralPath (Join-Path $game 'd4r/ngx/_nvngx.dll') 'fixture core' -Encoding ASCII
    $unrelated=('B'*64)
    @('# fixture pins',
      "dlss $(Sha (Join-Path $game 'd4r/nvngx_dlss.dll')) 310.9.1 validated",
      "dlss $unrelated 310.7.0 unvalidated",
      "ngx $(Sha (Join-Path $game 'd4r/ngx/_nvngx.dll')) 32.0.16.1714 validated") |
        Set-Content -LiteralPath (Join-Path $game 'd4r/dll-pins.txt') -Encoding ASCII
    Set-ManifestHash $game 'd4r/dll-pins.txt'
    return $game
}
function Write-Inventory([string]$name,[string]$arch,[string]$gpu='AMD Radeon RX 7700 XT') {
    $path=Join-Path $OutputDirectory "inventory-$name.json"
    @{devices=@(@{name=$gpu; architecture=$arch; wavefrontSize=32})} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $path -Encoding UTF8
    return $path
}
function Run-Check([string]$game,[string]$inventory,[string[]]$More=@()) {
    $ErrorActionPreference='Continue' # native stderr must not become a terminating error
    $text=& $powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $game 'd4r/d4r-check.ps1') -GameFolder $game -InventoryFile $inventory @More 2>&1 | Out-String
    return @{exit=$LASTEXITCODE; text=$text}
}
function Expect([string]$case,$run,[int]$exit,[string[]]$patterns) {
    if ($run.exit -ne $exit) { throw "$case : exit $($run.exit), expected $exit`n$($run.text)" }
    foreach ($pattern in $patterns) { if ($run.text -notmatch $pattern) { throw "$case : output lacks /$pattern/`n$($run.text)" } }
    $script:results+=@{case=$case; passed=$true; gpuKernelsExecuted=$false}
}

# 1. Healthy install.
$game=New-Install 'healthy'; $inv=Write-Inventory 'a' 'gfx1101'
Expect 'healthy-install' (Run-Check $game $inv) 0 @('ok\s+d4r\\nvngx_dlss\.dll \(nvngx_dlss\.dll 310\.9\.1, tested on Windows\)','ok\s+d4r\\ngx\\_nvngx\.dll',
    'ok\s+GPU AMD Radeon RX 7700 XT \(gfx1101\): build present, 16 native kernels','ok\s+all 17 GPU code objects are built for gfx1101',
    'ok\s+no vkd3d/Wine d3d12\.dll','ok\s+d4r\.ini is valid \(model K\)','compile-tested','Everything d4r needs is in place')

# 2. NVIDIA DLL problems, each with the fix.
$game=New-Install 'dll-missing'; Remove-Item -LiteralPath (Join-Path $game 'd4r/nvngx_dlss.dll')
Expect 'dll-missing' (Run-Check $game $inv) 1 @("MISSING\s+d4r\\nvngx_dlss\.dll: copy NVIDIA's nvngx_dlss\.dll here \(version 310\.9\.1 or 310\.7\.0","1 thing\(s\) to fix")
$game=New-Install 'dll-wrong'; Set-Content -LiteralPath (Join-Path $game 'd4r/nvngx_dlss.dll') 'some other version' -Encoding ASCII
Expect 'dll-wrong-version' (Run-Check $game $inv) 1 @('MISSING\s+d4r\\nvngx_dlss\.dll is version \S+, which d4r does not accept\. d4r on Windows needs exactly 310\.9\.1 or 310\.7\.0')
$game=New-Install 'dll-less-tested'; Set-Content -LiteralPath (Join-Path $game 'd4r/nvngx_dlss.dll') 'fixture dlss 310.7' -Encoding ASCII
@('dlss '+(Sha (Join-Path $game 'd4r/nvngx_dlss.dll'))+' 310.7.0 unvalidated','ngx '+(Sha (Join-Path $game 'd4r/ngx/_nvngx.dll'))+' 32.0.16.1714 validated') | Set-Content -LiteralPath (Join-Path $game 'd4r/dll-pins.txt') -Encoding ASCII
Set-ManifestHash $game 'd4r/dll-pins.txt'
Expect 'dll-less-tested-accepted' (Run-Check $game $inv) 0 @('ok\s+d4r\\nvngx_dlss\.dll \(nvngx_dlss\.dll 310\.7\.0\)','note\s+nvngx_dlss\.dll 310\.7\.0 is accepted but has had less Windows testing')

# 3. Leftovers, edited files, a replaced dxgi.dll, and a broken OptiScaler.ini.
$game=New-Install 'vkd3d'; Set-Content -LiteralPath (Join-Path $game 'd3d12.dll') 'vkd3d-proton fixture' -Encoding ASCII
Expect 'vkd3d-proxy' (Run-Check $game $inv) 1 @('MISSING\s+d3d12\.dll in the game folder is vkd3d-proton/Wine')
$game=New-Install 'dxgi-replaced'; Set-Content -LiteralPath (Join-Path $game 'dxgi.dll') 'another dxgi' -Encoding ASCII
Expect 'foreign-dxgi' (Run-Check $game $inv) 0 @('note\s+dxgi\.dll is not the one that came with d4r')
$game=New-Install 'opti-path'; (Get-Content -LiteralPath (Join-Path $game 'OptiScaler.ini')) -replace '^NvngxPath=.*$','NvngxPath=nowhere' | Set-Content -LiteralPath (Join-Path $game 'OptiScaler.ini') -Encoding ASCII
Expect 'optiscaler-ini-path' (Run-Check $game $inv) 1 @('MISSING\s+OptiScaler\.ini NvngxPath = nowhere does not hold _nvngx\.dll')

# 4. d4r.ini mistakes are notes with what will happen instead.
$game=New-Install 'ini'; Set-Content -LiteralPath (Join-Path $game 'd4r/d4r.ini') "[DLSS]`nModel = E`nShowWatermark = sometimes`n[Latency]`nFrameAge = 1`n[Debug]`nLog = loud`n" -Encoding ASCII
Expect 'ini-mistakes' (Run-Check $game $inv) 0 @("note\s+d4r\.ini: \[DLSS\] Model 'E' is not available on Windows",'must be true or false, not ''sometimes''','\[Latency\] FrameAge is not a Windows setting','Log must be normal or verbose')

# 5. GPU cases.
$game=New-Install 'gpu-no-build'; Remove-Item -LiteralPath (Join-Path $game 'd4r/_nvngx_gfx1103.dll')
Expect 'gpu-without-build' (Run-Check $game (Write-Inventory 'b' 'gfx1103' 'AMD Radeon 780M')) 1 @('MISSING\s+GPU AMD Radeon 780M \(gfx1103\): this release has no complete build for it \(missing: d4r\\_nvngx_gfx1103\.dll\)')
$game=New-Install 'gpu-rdna2'
Expect 'gpu-unsupported' (Run-Check $game (Write-Inventory 'c' 'gfx1030' 'AMD Radeon RX 6800')) 1 @('MISSING\s+no supported RDNA3/RDNA4 GPU found by the AMD runtime\. It sees: AMD Radeon RX 6800 \(gfx1030\)')
$game=New-Install 'gpu-wrong-object'
$other=@(Get-ChildItem -LiteralPath (Join-Path $game 'd4r/native/gfx1100') -Filter '*.hsaco')[0].FullName
Copy-Item -LiteralPath $other -Destination (Join-Path $game 'd4r/native/gfx1101/kernel3.hsaco') -Force
Expect 'code-object-for-another-gpu' (Run-Check $game $inv) 1 @('MISSING\s+kernel3\.hsaco is not built for gfx1101','differs from the release')

# 6. What the last game start said, and the bug-report ZIP.
$game=New-Install 'log-failure'
Set-Content -LiteralPath (Join-Path $game 'd4r/d4r_nvngx.log') "d4r: DLSS disabled: Wrong nvngx_dlss.dll: you have version 1.2.3.4" -Encoding ASCII
Expect 'log-reports-failure' (Run-Check $game $inv) 1 @('MISSING\s+last game start: DLSS disabled: Wrong nvngx_dlss\.dll')
$game=New-Install 'log-ok'
Set-Content -LiteralPath (Join-Path $game 'd4r/d4r_nvngx.log') "d4r: player mode, model=K validate=0 async=0 log=normal, architecture=gfx1101" -Encoding ASCII
Expect 'log-reports-ok' (Run-Check $game $inv) 0 @('ok\s+last game start: player mode, model=K')
$game=New-Install 'report'; $reportDir=Join-Path $OutputDirectory 'reports'; New-Item -ItemType Directory -Force $reportDir | Out-Null
Set-Content -LiteralPath (Join-Path $game 'd4r/d4r_nvngx.log') "d4r: DLSS disabled: test failure" -Encoding ASCII
$run=Run-Check $game $inv @('-Report','-ReportDirectory',$reportDir)
$zip=@(Get-ChildItem -LiteralPath $reportDir -Filter 'd4r-report-*.zip')
if ($zip.Count -ne 1) { throw "Expected one report ZIP.`n$($run.text)" }
Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive=[IO.Compression.ZipFile]::OpenRead($zip[0].FullName)
try { $names=@($archive.Entries | ForEach-Object { $_.Name }) } finally { $archive.Dispose() }
foreach ($required in @('check.txt','d4r_nvngx.log','d4r.ini','OptiScaler.ini','files.sha256.txt','gpu-inventory.json')) { if ($names -notcontains $required) { throw "Report lacks $required (has: $($names -join ', '))" } }
if ($names -contains 'nvngx_dlss.dll' -or $names -contains '_nvngx.dll') { throw 'The report must never contain NVIDIA DLLs.' }
if ($run.text -notmatch 'Bug report written') { throw 'Report location was not printed.' }
$results+=@{case='bug-report-zip'; passed=$true; entries=$names}

# 7. No d4r folder at all.
$empty=Join-Path $OutputDirectory 'cases/empty'; New-Item -ItemType Directory -Force $empty | Out-Null
$ErrorActionPreference='Continue'
$text=& $powershell -NoProfile -ExecutionPolicy Bypass -File $check -GameFolder $empty 2>&1 | Out-String
$exitEmpty=$LASTEXITCODE; $ErrorActionPreference='Stop'
if ($exitEmpty -ne 1 -or $text -notmatch 'no d4r folder here') { throw "A folder without d4r must be reported.`n$text" }
$results+=@{case='no-d4r-folder'; passed=$true}

$results | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $OutputDirectory 'summary.json') -Encoding UTF8
Write-Host "PASS D4R_CHECK cases=$($results.Count) Report: $OutputDirectory"
