@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Japo - Build installer

if not exist "dist\Japo\Japo.exe" (
  echo No build found. Run build_exe.bat first.
  pause
  exit /b
)

echo === 1/2 Looking for Inno Setup ===
call :find_iscc
if not defined ISCC (
  echo Inno Setup not found, installing...
  winget install -e --id JRSoftware.InnoSetup --accept-package-agreements --accept-source-agreements
  call :find_iscc
)
if not defined ISCC (
  echo Inno Setup not found. Download it from https://jrsoftware.org/isdl.php and run this again.
  pause
  exit /b
)

echo === 2/2 Compressing into one file (this takes a few minutes) ===
"%ISCC%" japo.iss
if errorlevel 1 (
  echo Installer build failed. See the messages above.
  pause
  exit /b
)

echo.
echo ============================================
echo  Done! Installer: %CD%\Output\Japo-Setup.exe
echo ============================================
explorer Output
pause
exit /b

:find_iscc
set ISCC=
for %%P in ("%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" "%ProgramFiles%\Inno Setup 6\ISCC.exe") do (
  if exist %%P set ISCC=%%~P
)
exit /b
