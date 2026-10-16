[CmdletBinding()]
param([string]$SourceRoot, [string]$HipRoot = 'C:\Program Files\AMD\ROCm\7.2', [string]$WorkRoot)
$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (!$SourceRoot) { $SourceRoot = Join-Path $repo 'external/ZLUDA' }
if (!$WorkRoot) { $WorkRoot = Join-Path $repo 'build/zluda-helpers-windows' }
$toolchain = Join-Path $repo '.tools/llvm-mingw-20260922-ucrt-x86_64'
$compiler = Join-Path $HipRoot 'bin/clang++.exe'
$dis = Join-Path $HipRoot 'bin/llvm-dis.exe'
$assembler = Join-Path $HipRoot 'bin/llvm-as.exe'
$ocml = Join-Path $HipRoot 'amdgcn/bitcode/ocml.bc'
foreach ($file in @($compiler,$dis,$assembler,$ocml)) {
    if (!(Test-Path $file)) { throw "HIP helper dependency missing: $file" }
}
New-Item -ItemType Directory -Force $WorkRoot | Out-Null
$configInclude = Join-Path $WorkRoot 'libcxx-config'
New-Item -ItemType Directory -Force $configInclude | Out-Null
Copy-Item -LiteralPath (Join-Path $toolchain 'include/c++/v1/__config_site') -Destination $configInclude -Force
Copy-Item -LiteralPath (Join-Path $SourceRoot 'ext/llvm-project/libcxx/vendor/llvm/default_assertion_handler.in') `
    -Destination (Join-Path $configInclude '__assertion_handler') -Force
# HIP 7.2's long shuffle overloads assume every non-MSVC compiler has LP64.
# Windows GNU also has 32-bit long. Correct that public header in a private
# mirror; the installed SDK remains untouched and the GPU target is unchanged.
$hipInclude = Join-Path $WorkRoot 'hip-include'
New-Item -ItemType Directory -Force $hipInclude | Out-Null
Copy-Item -LiteralPath (Join-Path $HipRoot 'include/hip') -Destination $hipInclude -Recurse -Force
$warpHeader = Join-Path $hipInclude 'hip/amd_detail/amd_warp_functions.h'
$warpSource = [IO.File]::ReadAllText($warpHeader)
$warpSource = $warpSource.Replace('#ifndef _MSC_VER', '#if __SIZEOF_LONG__ == 8')
[IO.File]::WriteAllText($warpHeader,$warpSource,[Text.UTF8Encoding]::new($false))
$variants = @(
    @{Suffix=''; Flags=@()},
    @{Suffix='_constrained'; Flags=@('-ffp-model=strict','-ffp-exception-behavior=ignore')},
    @{Suffix='_w64'; Flags=@('-mwavefrontsize64','-DD4R_WAVE64')},
    @{Suffix='_constrained_w64'; Flags=@('-ffp-model=strict','-ffp-exception-behavior=ignore','-mwavefrontsize64','-DD4R_WAVE64')}
)
foreach ($variant in $variants) {
    $name = 'zluda_ptx_impl' + $variant.Suffix
    $raw = Join-Path $WorkRoot "$name.raw.bc"
    $ir = Join-Path $WorkRoot "$name.ll"
    $output = Join-Path $SourceRoot "ptx/lib/$name.bc"
    $arguments = @('-x','hip','-std=c++20','--target=x86_64-w64-windows-gnu','-stdlib=libc++',
        '--offload-arch=gfx1030','--offload-device-only','-emit-llvm','-c','-nogpulib','-O3',
        '-mno-wavefrontsize64','-DHIP_ENABLE_WARP_SYNC_BUILTINS',
        '-Xclang','-fdenormal-fp-math=dynamic',
        '-isystem',(Join-Path $SourceRoot 'ext/llvm-project/libcxx/include'),
        '-isystem',$configInclude,
        '-isystem',(Join-Path $toolchain 'include'),
        '-I',$hipInclude,
        '-Xclang','-mlink-bitcode-file','-Xclang',$ocml)
    $arguments += $variant.Flags
    $arguments += @('-o',$raw,(Join-Path $SourceRoot 'ptx/lib/zluda_ptx_impl.cpp'))
    $ErrorActionPreference = 'Continue'
    & $compiler @arguments 2>&1 | Tee-Object (Join-Path $WorkRoot "$name.compile.log")
    $ErrorActionPreference = 'Stop'
    if ($LASTEXITCODE) { throw "PTX helper compile failed: $name ($LASTEXITCODE)" }
    & $dis $raw -o $ir
    if ($LASTEXITCODE) { throw "Disassemble failed: $raw" }
    $clean = [Collections.Generic.List[string]]::new()
    foreach ($line in [IO.File]::ReadLines($ir)) {
        if ($line -match '@llvm.used|wchar_size|llvm.module.flags|__hip_cuid') { continue }
        $line = $line -replace 'optnone','' -replace 'define hidden','define linkonce_odr'
        $line = $line -replace '"target-cpu"="gfx1030"','' -replace '"target-features"="[^"]+"',''
        $clean.Add($line)
    }
    [IO.File]::WriteAllLines($ir,$clean,[Text.UTF8Encoding]::new($false))
    & $assembler $ir -o $output
    if ($LASTEXITCODE) { throw "Assemble failed: $ir" }
    Write-Host "Built $output ($((Get-Item $output).Length) bytes)"
}
