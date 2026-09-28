@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Japo - Setup

where python >nul 2>nul
if errorlevel 1 (
  echo Python not found. Install Python 3.11 and tick "Add Python to PATH", then run this again.
  pause
  exit /b
)
python -c "import sys; exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)"
if errorlevel 1 (
  echo Warning: Japo is tested with Python 3.10 - 3.12. Your version:
  python --version
  echo Press any key to continue anyway, or close this window.
  pause >nul
)

echo === Creating Python environment ===
if not exist "venv\Scripts\python.exe" python -m venv venv
if not exist "venv\Scripts\python.exe" (
  echo Could not create the environment.
  pause
  exit /b
)
"venv\Scripts\python.exe" -m pip install --upgrade pip
"venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo Setup failed. See the messages above.
  pause
  exit /b
)
echo.
echo === Setup complete. Start Japo with run.bat ===
pause
