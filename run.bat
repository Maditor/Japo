@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
  echo Environment not found. Run setup.bat first.
  pause
  exit /b
)
"venv\Scripts\python.exe" japo.py
if errorlevel 1 pause
