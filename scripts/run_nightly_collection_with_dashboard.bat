@echo off
setlocal

cd /d "%~dp0\.."

set "PORT=8765"
set "DASHBOARD_URL=http://localhost:%PORT%/docs/nightly_network_feature_dashboard.html"

echo Checking dashboard control server...
powershell -NoProfile -Command "try { Invoke-RestMethod -Method Get -Uri 'http://127.0.0.1:%PORT%/api/collector/status' | Out-Null; exit 0 } catch { exit 1 }"

if errorlevel 1 (
  echo Dashboard control server is not running, starting it...
  start "NightlyDashboardServer" cmd /c "python scripts\nightly_dashboard_server.py"
  timeout /t 2 /nobreak >nul
) else (
  echo Dashboard control server already running.
)

echo Opening dashboard page...
start "" "%DASHBOARD_URL%"

echo Triggering collection start via local control API...
powershell -NoProfile -Command "try { Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:%PORT%/api/collector/start' | Out-Null; exit 0 } catch { Write-Error $_; exit 1 }"

if errorlevel 1 (
  echo FAILED to trigger collector start.
  exit /b 1
)

echo Nightly collection trigger sent.
exit /b 0
