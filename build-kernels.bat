@echo off
setlocal
rem Build all registered RDNA 3, 3.5 and 4 targets. No GPU or NVIDIA DLL needed.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\windows\build-windows-gpu-matrix.ps1" -KernelsOnly -BuildRoot "%~dp0build\windows-kernels" -OutputRoot "%~dp0dist\windows-kernels" %*
set "D4R_BUILD_EXIT=%ERRORLEVEL%"
if not "%D4R_BUILD_EXIT%"=="0" echo Kernel build failed. See build\windows-kernels for logs. Run scripts\windows\setup-windows-tools.ps1 -RuntimeProfile therock if tools are missing.
exit /b %D4R_BUILD_EXIT%
