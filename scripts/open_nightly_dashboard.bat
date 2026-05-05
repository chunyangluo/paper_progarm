@echo off
setlocal

cd /d "%~dp0\.."

set "PORT=8765"
set "URL=http://localhost:%PORT%/docs/nightly_network_feature_dashboard.html"

echo Starting dashboard control server on port %PORT% ...
echo You can start/stop collection from dashboard buttons.
start "" "%URL%"
python scripts/nightly_dashboard_server.py

exit /b 0
