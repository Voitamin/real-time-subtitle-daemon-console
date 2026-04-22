@echo off
setlocal
taskkill /IM mediamtx.exe /F >nul 2>nul
if %ERRORLEVEL% EQU 0 (
  echo [rtmp-relay] mediamtx.exe stopped.
) else (
  echo [rtmp-relay] no running mediamtx.exe found.
)
pause

