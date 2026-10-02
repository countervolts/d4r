# Shared by the build, diagnostics, staging and game package scripts.
function Get-D4RGpuTarget([string]$Root, [string]$Requested) {
    $configured=$null
    $targetFile=Join-Path $Root 'gpu-target.json'
    if (Test-Path -LiteralPath $targetFile) {
        $configured=(Get-Content -LiteralPath $targetFile -Raw | ConvertFrom-Json).architecture
        if ($configured -notin @('gfx1200','gfx1201')) { throw "Unsupported package GPU target: $configured" }
    }
    $target=if ($Requested) { $Requested } elseif ($configured) { $configured } else { 'gfx1201' }
    if ($target -notin @('gfx1200','gfx1201')) { throw "Unsupported GPU target: $target" }
    if (!$configured -and $target -ne 'gfx1201') { throw 'A gfx1200 build requires gpu-target.json; rebuild with -GpuArch gfx1200.' }
    if ($configured -and $target -ne $configured) { throw "Requested $target differs from package target $configured" }
    return $target
}
function Assert-D4RCodeObjectTarget([string]$Path, [string]$Target) {
    if ($Target -notin @('gfx1200','gfx1201')) { throw "Unsupported GPU target: $Target" }
    $reader=[IO.BinaryReader]::new([IO.File]::OpenRead($Path))
    try { $header=$reader.ReadBytes(64) } finally { $reader.Dispose() }
    if ($header.Length -ne 64 -or $header[0] -ne 127 -or $header[1] -ne 69 -or $header[2] -ne 76 -or $header[3] -ne 70 -or
        $header[4] -ne 2 -or $header[5] -ne 1 -or $header[18] -ne 224 -or $header[19] -ne 0) {
        throw "Invalid AMDGPU ELF64 code object: $Path"
    }
    # LLVM AMDGPU e_flags machine IDs; inspect, never patch or spoof them.
    $expected=if ($Target -eq 'gfx1200') { 0x48 } else { 0x4e }
    if ($header[48] -ne $expected) { throw "Code object does not target ${Target}: $Path (ELF machine=$($header[48]))" }
}
