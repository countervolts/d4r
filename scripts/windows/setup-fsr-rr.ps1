[CmdletBinding()]
param()
$ErrorActionPreference='Stop'
$repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$directory=Join-Path $repo '.tools/fsr-rr-1.2'
$dll=Join-Path $directory 'amd_fidelityfx_denoiser_dx12.dll'
$expected='48f1e5888ba6a0a3d59a98b9751e37c392b0f7b8c223d0082d5c1f40642879d3'
$url='https://raw.githubusercontent.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/v2.3.0/Kits/FidelityFX/signedbin/amd_fidelityfx_denoiser_dx12.dll'
New-Item -ItemType Directory -Path $directory -Force | Out-Null
if(Test-Path -LiteralPath $dll){
    if((Get-FileHash -LiteralPath $dll -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected){
        throw "Existing denoiser hash mismatch; refusing to replace $dll"
    }
}else{
    $temporary=Join-Path $directory ([Guid]::NewGuid().ToString('N')+'.download')
    try{
        Invoke-WebRequest -Uri $url -OutFile $temporary -UseBasicParsing
        if((Get-FileHash -LiteralPath $temporary -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected){
            throw 'AMD denoiser download hash mismatch.'
        }
        Move-Item -LiteralPath $temporary -Destination $dll
    }finally{
        if(Test-Path -LiteralPath $temporary){Remove-Item -LiteralPath $temporary}
    }
}
Write-Host "Pinned AMD denoiser ready: $dll"
Write-Host 'Provider availability and execution still require test-fsr-rr.ps1.'
