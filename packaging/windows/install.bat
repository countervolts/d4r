@echo off
setlocal
if not exist "%~dp0d4r\model-tools\python\python.exe" (
  echo Missing portable tools. Extract the complete current d4r ZIP first.
  exit /b 1
)
if not "%~1"=="" goto arguments
echo d4r will download NVIDIA DLSS files and an NVIDIA driver archive.
echo Driver archive is about 945 MiB (991 MB). It will only extract _nvngx.dll; it will not install or change drivers.
choice /C YN /N /M "Continue? [Y/N] "
if errorlevel 2 exit /b 0
echo Optional Ray Reconstruction backends (disabled by default; synthetic validation only, game compatibility unverified):
echo   [D]/[E] native DLSS RR model built locally from NVIDIA's RR DLL (gfx1101 only).
echo   [F] experimental FSR Ray Regeneration: downloads AMD's pinned 14 MB denoiser DLL; runs it through the
echo       bundled vkd3d-proton (d4r\fsr). AMD does not support FSR RR on RDNA3; d4r emulates its FP8 math.
choice /C NDEF /N /M "Enable RR? [N]one / [D] / [E] / [F]SR: "
if errorlevel 4 goto fsr
if errorlevel 3 goto modelE
if errorlevel 2 goto modelD
"%~dp0d4r\model-tools\python\python.exe" "%~dp0install-dependencies.py" --package-root "%~dp0." --yes
goto result
:modelD
"%~dp0d4r\model-tools\python\python.exe" "%~dp0install-dependencies.py" --package-root "%~dp0." --yes --rr-preset D --enable-rr
goto result
:modelE
"%~dp0d4r\model-tools\python\python.exe" "%~dp0install-dependencies.py" --package-root "%~dp0." --yes --rr-preset E --enable-rr
goto result
:fsr
"%~dp0d4r\model-tools\python\python.exe" "%~dp0install-dependencies.py" --package-root "%~dp0." --yes --enable-fsr
goto result
:arguments
"%~dp0d4r\model-tools\python\python.exe" "%~dp0install-dependencies.py" --package-root "%~dp0." %*
:result
set "D4R_INSTALL_EXIT=%ERRORLEVEL%"
if not "%D4R_INSTALL_EXIT%"=="0" echo d4r install failed. Read the message above.
exit /b %D4R_INSTALL_EXIT%
