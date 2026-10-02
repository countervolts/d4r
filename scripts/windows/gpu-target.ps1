# Shared by the build, diagnostics, staging and game package scripts.
function Get-D4RGpuTargets {
    $registry=Join-Path $PSScriptRoot 'gpu-targets.json'
    if (!(Test-Path -LiteralPath $registry)) { $registry=Join-Path $PSScriptRoot '../../tools/windows/gpu-targets.json' }
    $data=Get-Content -LiteralPath $registry -Raw | ConvertFrom-Json
    if ($data.schema -ne 1) { throw 'Unsupported GPU target registry schema.' }
    return $data.targets
}
function Get-D4RGpuTargetInfo([string]$Target) {
    $info=@(Get-D4RGpuTargets | Where-Object { $_.architecture -eq $Target })
    if ($info.Count -ne 1) { throw "Unsupported GPU target: $Target" }
    return $info[0]
}
function Get-D4RGpuTarget([string]$Root, [string]$Requested) {
    $configured=$null
    $targetFile=Join-Path $Root 'gpu-target.json'
    if (Test-Path -LiteralPath $targetFile) {
        $configured=(Get-Content -LiteralPath $targetFile -Raw | ConvertFrom-Json).architecture
        [void](Get-D4RGpuTargetInfo $configured)
    }
    $target=if ($Requested) { $Requested } elseif ($configured) { $configured } else { 'gfx1201' }
    [void](Get-D4RGpuTargetInfo $target)
    if (!$configured -and $target -ne 'gfx1201') { throw "A $target build requires gpu-target.json; rebuild with -GpuArch $target." }
    if ($configured -and $target -ne $configured) { throw "Requested $target differs from package target $configured" }
    return $target
}
function Assert-D4RCodeObjectTarget([string]$Path, [string]$Target) {
    $info=Get-D4RGpuTargetInfo $Target
    $reader=[IO.BinaryReader]::new([IO.File]::OpenRead($Path))
    try { $header=$reader.ReadBytes(64) } finally { $reader.Dispose() }
    if ($header.Length -ne 64 -or $header[0] -ne 127 -or $header[1] -ne 69 -or $header[2] -ne 76 -or $header[3] -ne 70 -or
        $header[4] -ne 2 -or $header[5] -ne 1 -or $header[18] -ne 224 -or $header[19] -ne 0) {
        throw "Invalid AMDGPU ELF64 code object: $Path"
    }
    # LLVM AMDGPU e_flags machine IDs; inspect, never patch or spoof them.
    $expected=$info.elfMachine
    if ($header[48] -ne $expected) { throw "Code object does not target ${Target}: $Path (ELF machine=$($header[48]))" }
}
