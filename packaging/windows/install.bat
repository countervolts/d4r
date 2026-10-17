@echo off
setlocal
if not exist "%~dp0d4r\model-tools\python\python.exe" (
  echo Missing portable tools. Extract the complete current d4r ZIP first.
  exit /b 1
)
if not "%~1"=="" goto arguments
echo d4r will download the required NVIDIA DLSS files and NVIDIA driver archive.
echo Releases with the FSR backend also download AMD's Ray Regeneration denoiser (14 MB).
echo Driver archive is about 945 MiB (991 MB). It will only extract _nvngx.dll; it will not install or change drivers.
choice /C YN /N /M "Continue? [Y/N] "
if errorlevel 2 exit /b 0
"%~dp0d4r\model-tools\python\python.exe" "%~dp0install-dependencies.py" --package-root "%~dp0." --yes
set "D4R_INSTALL_EXIT=%ERRORLEVEL%"
set "D4R_PAUSE=1"
goto report
:arguments
"%~dp0d4r\model-tools\python\python.exe" "%~dp0install-dependencies.py" --package-root "%~dp0." %*
set "D4R_INSTALL_EXIT=%ERRORLEVEL%"
:report
if not "%D4R_INSTALL_EXIT%"=="0" echo d4r install failed. Read the message above.
if defined D4R_PAUSE pause
exit /b %D4R_INSTALL_EXIT%
