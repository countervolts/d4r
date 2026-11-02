@echo off
rem Runs install.ps1 (NVIDIA's DLLs and the engine's models for native Windows) without changing PowerShell's
rem execution policy. Options are passed on, e.g. install.cmd -Yes
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
if errorlevel 1 pause
