@echo off
setlocal

REM Merge collected chunk outputs into one full dataset

cd /d "%~dp0\.."

if not exist "logs" mkdir "logs"

set "INPUT=data/versions/dataset_20260421_100k.csv"
set "RUN_NAME=dataset_20260421_100k_nightly"

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set TS=%%i
set "LOG_FILE=logs\nightly_network_merge_%TS%.log"

echo [%date% %time%] Starting merge...
echo Log: %LOG_FILE%

python scripts/nightly_network_feature_pipeline.py ^
  --input "%INPUT%" ^
  --run-name "%RUN_NAME%" ^
  --merge-only > "%LOG_FILE%" 2>&1

if errorlevel 1 (
  echo [%date% %time%] MERGE FAILED. Check log: %LOG_FILE%
  exit /b 1
)

echo [%date% %time%] MERGE DONE. Check log: %LOG_FILE%
exit /b 0
