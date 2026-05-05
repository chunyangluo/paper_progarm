@echo off
setlocal enabledelayedexpansion

REM Nightly chunked network feature collection
REM Usage:
REM   Double-click this file, or run via Windows Task Scheduler.

cd /d "%~dp0\.."

if not exist "logs" mkdir "logs"

set "INPUT=data/versions/dataset_20260421_100k.csv"
set "RUN_NAME=dataset_20260421_100k_nightly"
set "CHUNK_SIZE=20000"
set "CHUNKS_PER_RUN=2"
set "BATCH_SIZE=100"

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set TS=%%i
set "LOG_FILE=logs\nightly_network_collection_%TS%.log"

echo [%date% %time%] Starting nightly network collection...
echo Input: %INPUT%
echo RunName: %RUN_NAME%
echo ChunkSize: %CHUNK_SIZE%, ChunksPerRun: %CHUNKS_PER_RUN%, BatchSize: %BATCH_SIZE%
echo Log: %LOG_FILE%

python scripts/nightly_network_feature_pipeline.py ^
  --input "%INPUT%" ^
  --run-name "%RUN_NAME%" ^
  --chunk-size %CHUNK_SIZE% ^
  --chunks-per-run %CHUNKS_PER_RUN% ^
  --batch-size %BATCH_SIZE% ^
  --no-auto-merge > "%LOG_FILE%" 2>&1

if errorlevel 1 (
  echo [%date% %time%] FAILED. Check log: %LOG_FILE%
  exit /b 1
)

echo [%date% %time%] DONE. Check log: %LOG_FILE%
exit /b 0
