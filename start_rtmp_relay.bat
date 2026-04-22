@echo off
setlocal
cd /d "%~dp0"

if not exist ".\mediamtx.exe" (
  echo [rtmp-relay] ERROR: mediamtx.exe not found in %CD%
  pause
  exit /b 1
)

if not exist ".\mediamtx_rtmp.yml" (
  echo [rtmp-relay] ERROR: mediamtx_rtmp.yml not found in %CD%
  pause
  exit /b 1
)

if not exist ".\OUT" mkdir ".\OUT"

echo [rtmp-relay] stopping existing mediamtx.exe (if any)...
taskkill /IM mediamtx.exe /F >nul 2>nul
timeout /t 1 >nul

echo.
echo [rtmp-relay] MediaMTX RTMP relay starting...
echo [rtmp-relay] OBS1 (publish):
echo   Server: rtmp://127.0.0.1:1935
echo   Stream Key: relaymain
echo.
echo [rtmp-relay] OBS2 (read):
echo   Source URL: rtmp://127.0.0.1:1935/relaymain
echo.
echo [rtmp-relay] Press Ctrl+C to stop.
echo.

.\mediamtx.exe .\mediamtx_rtmp.yml

echo.
echo [rtmp-relay] MediaMTX exited with code %ERRORLEVEL%.
pause
