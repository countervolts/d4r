# Read-only inventory before touching game files. It allocates no GPU memory
# and executes no kernels; successful discovery is not hardware validation.
function Invoke-D4RGpuPreflight([string]$PackageRoot, [string]$GameDirectory, [string]$Target, [string]$OutputDirectory) {
    New-Item -ItemType Directory -Force $OutputDirectory | Out-Null
    $result=[ordered]@{passed=$false; target=$Target; gpuWorkExecuted=$false; failure=$null}
    try {
        $proxy=Join-Path $GameDirectory 'd3d12.dll'
        if (Test-Path -LiteralPath $proxy) {
            $item=Get-Item -LiteralPath $proxy
            $result.localD3D12=@{path=$proxy; sha256=(Get-FileHash -LiteralPath $proxy).Hash; product=$item.VersionInfo.ProductName}
            # Identify known Vulkan/Wine proxies by their own strings. Do not
            # load, remove or replace a third-party game DLL.
            if ($item.Length -le 64MB -and [Text.Encoding]::ASCII.GetString([IO.File]::ReadAllBytes($proxy)) -match '(?i)vkd3d|wined3d') {
                throw 'Local d3d12.dll identifies as vkd3d/Wine. Restore the game native D3D12 installation before using this Windows backend.'
            }
        }
        $info=[Diagnostics.ProcessStartInfo]::new()
        $info.FileName=Join-Path $PackageRoot 'd4r/d4r_gpu_inventory.exe'
        $hip=Join-Path $PackageRoot 'd4r/hip'
        foreach ($value in @($hip,$OutputDirectory)) { if ($value.Contains('"')) { throw 'Unsupported quote in inventory path.' } }
        $info.Arguments='--hip-root "' + $hip + '" --output-dir "' + $OutputDirectory + '"'
        $info.UseShellExecute=$false; $info.CreateNoWindow=$true
        $info.EnvironmentVariables['D4R_DIAG_DIR']=$OutputDirectory
        $info.RedirectStandardOutput=$true; $info.RedirectStandardError=$true
        $process=[Diagnostics.Process]::new(); $process.StartInfo=$info
        try {
            if (!$process.Start()) { throw 'Cannot start GPU inventory.' }
            $stdout=$process.StandardOutput.ReadToEndAsync(); $stderr=$process.StandardError.ReadToEndAsync()
            $timedOut=!$process.WaitForExit(30000)
            if ($timedOut) { $process.Kill(); $process.WaitForExit() }
            $stdout.GetAwaiter().GetResult() | Set-Content (Join-Path $OutputDirectory 'gpu-inventory.stdout.log') -Encoding UTF8
            $stderr.GetAwaiter().GetResult() | Set-Content (Join-Path $OutputDirectory 'gpu-inventory.stderr.log') -Encoding UTF8
            $result.exitCode=$process.ExitCode
            $result.exceptionOrExitHex='0x{0:X8}' -f [BitConverter]::ToUInt32([BitConverter]::GetBytes([int]$process.ExitCode),0)
            if ($timedOut) { $result.timedOut=$true; throw 'HIP inventory timed out after 30 seconds.' }
            if ($process.ExitCode) { throw "HIP inventory failed (exit $($process.ExitCode))." }
        } finally { $process.Dispose() }
        $inventory=Get-Content (Join-Path $OutputDirectory 'gpu-inventory.json') -Raw | ConvertFrom-Json
        $result.inventory=$inventory
        if ($inventory.compiledTarget -ne $Target) { throw "Inventory executable targets $($inventory.compiledTarget); package requires $Target." }
        $matching=@($inventory.devices | Where-Object { $_.architecture -eq $Target -and $_.wavefrontSize -eq 32 })
        if (!$matching.Count) {
            $actual=@($inventory.devices | ForEach-Object { "$($_.architecture) ($($_.name))" }) -join ', '
            throw "GPU target mismatch: detected $actual; package requires $Target. Use the package for the actual HIP architecture; do not rename code objects."
        }
        $result.passed=$true
        Write-Host "D4R_PREFLIGHT target=$Target devices=$($matching.Count) gpu_work_executed=0"
    } catch {
        $result.failure=$_.Exception.Message
        $result | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $OutputDirectory 'preflight.json') -Encoding UTF8
        try { Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,DriverDate | ConvertTo-Json |
            Set-Content (Join-Path $OutputDirectory 'driver.json') -Encoding UTF8 } catch {}
        $bundle=$OutputDirectory+'.zip'
        Compress-Archive -Path (Join-Path $OutputDirectory '*') -DestinationPath $bundle -Force
        throw "$($result.failure) Diagnostic bundle: $bundle"
    }
    $result | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $OutputDirectory 'preflight.json') -Encoding UTF8
}
