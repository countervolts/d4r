@echo off
setlocal
echo d4r will download NVIDIA DLSS files and an NVIDIA driver archive.
echo Driver archive is about 945 MiB (991 MB). It will only extract _nvngx.dll; it will not install or change drivers.
choice /C YN /N /M "Continue? [Y/N] "
if errorlevel 2 exit /b 0
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install-dlss.ps1" -PackageRoot "%~dp0." -Yes
set "D4R_INSTALL_EXIT=%ERRORLEVEL%"
if not "%D4R_INSTALL_EXIT%"=="0" echo d4r install failed. Read the message above.
exit /b %D4R_INSTALL_EXIT%
