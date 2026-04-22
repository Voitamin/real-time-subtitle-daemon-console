@echo off
setlocal
powershell -ExecutionPolicy Bypass -File "%~dp0start_dual_obs.ps1" %*
endlocal
