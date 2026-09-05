@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if exist "%PYTHON_EXE%" (
  "%PYTHON_EXE%" -c "import sys" >nul 2>&1
  if errorlevel 1 (
    set "PYTHON_EXE=python"
  )
) else (
  set "PYTHON_EXE=python"
)

set "PYTHONPATH="
start "" "http://127.0.0.1:8002/"
echo CogniTrack is running at http://127.0.0.1:8002/
echo Press Ctrl+C to stop it.
"%PYTHON_EXE%" -m uvicorn main:app --app-dir backend --host 0.0.0.0 --port 8002
pause
