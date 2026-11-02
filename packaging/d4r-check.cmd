@echo off
rem Checks this d4r install (d4r-check.ps1) without changing PowerShell's execution policy
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0d4r-check.ps1" %*
pause
