[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$tools = Join-Path $repo '.tools'
$downloads = Join-Path $tools 'downloads'
New-Item -ItemType Directory -Force -Path $downloads | Out-Null

function Ensure-PinnedGitCheckout([string]$Path, [string]$Url, [string]$Commit) {
    if (!(Test-Path -LiteralPath $Path)) {
        git clone --no-checkout $Url $Path
        if ($LASTEXITCODE) { throw "Git clone failed: $Url" }
    }
    git -C $Path fetch --depth 1 origin $Commit
    if ($LASTEXITCODE) { throw "Pinned dependency fetch failed: $Url@$Commit" }
    git -C $Path checkout --detach $Commit
    if ($LASTEXITCODE) { throw "Pinned dependency checkout failed: $Url@$Commit" }
    $actual = (git -C $Path rev-parse HEAD).Trim()
    if ($actual -ne $Commit) { throw "Dependency revision mismatch: $Path ($actual)" }
}

$headers = Join-Path $tools 'Vulkan-Headers-v1.4.341'
Ensure-PinnedGitCheckout $headers 'https://github.com/KhronosGroup/Vulkan-Headers.git' 'b5c8f996196ba4aa6d8f97e52b5d3b6e70f7e4e2'
$volk = Join-Path $tools 'volk'
Ensure-PinnedGitCheckout $volk 'https://github.com/zeux/volk.git' '0dc3ce00bf98b9f0b6fe708ca0f7eb74e2830173'

$glslangDirectory = Join-Path $tools 'glslang-16.6.0'
$validator = Join-Path $glslangDirectory 'bin/glslangValidator.exe'
$archive = Join-Path $downloads 'glslang-16.6.0-windows-x86_64-release.zip'
$expectedHash = '82bf434e69b9bb4829de7e2b4bc2c5e7a7861e53d66cf75e5cc70f5f694a8d9b'
if (!(Test-Path -LiteralPath $validator)) {
    if (!(Test-Path -LiteralPath $archive)) {
        Invoke-WebRequest -UseBasicParsing `
            -Uri 'https://github.com/KhronosGroup/glslang/releases/download/16.6.0/glslang-16.6.0-windows-x86_64-release.zip' `
            -OutFile $archive
    }
    $actualHash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -ne $expectedHash) { throw "glslang archive SHA256 mismatch: $actualHash" }
    Expand-Archive -LiteralPath $archive -DestinationPath $glslangDirectory -Force
    $releaseBinary = Join-Path $glslangDirectory 'bin/glslang.exe'
    if (!(Test-Path -LiteralPath $releaseBinary)) { throw "glslang archive layout changed: $releaseBinary" }
    Copy-Item -LiteralPath $releaseBinary -Destination $validator
}
if (!(Test-Path -LiteralPath $headers -PathType Container) -or
    !(Test-Path -LiteralPath (Join-Path $headers 'include/vulkan/vulkan.h')) -or
    !(Test-Path -LiteralPath (Join-Path $volk 'volk.c')) -or
    !(Test-Path -LiteralPath $validator)) {
    throw 'Native engine dependencies are incomplete.'
}
Write-Host "Native engine dependencies ready: Vulkan-Headers $((git -C $headers rev-parse --short HEAD).Trim()), Volk $((git -C $volk rev-parse --short HEAD).Trim()), glslang 16.6.0."
