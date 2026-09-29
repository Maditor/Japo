@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Japo - Build

if not exist "venv\Scripts\python.exe" (
  echo Environment not found. Run setup.bat first.
  pause
  exit /b
)
set PY="%~dp0venv\Scripts\python.exe"

rem Close a running Japo build, otherwise its files are locked and the old build cannot be replaced
tasklist /fi "imagename eq Japo.exe" | find /i "Japo.exe" >nul
if not errorlevel 1 (
  echo Japo is running - closing it before building...
  taskkill /im Japo.exe /f >nul 2>nul
  timeout /t 2 /nobreak >nul
)

echo === 1/4 Installing PyInstaller ===
%PY% -m pip install -q pyinstaller

echo === 2/4 Building (this takes a few minutes) ===
%PY% -m PyInstaller --noconfirm --clean --windowed --onedir ^
  --name "Japo" ^
  --icon "icon.ico" ^
  --add-data "icon.ico;." ^
  --add-data "icon_light.ico;." ^
  --collect-all faster_whisper ^
  --collect-all ctranslate2 ^
  --collect-binaries pyaudiowpatch ^
  --collect-all cutlet ^
  --collect-all fugashi ^
  --collect-all unidic_lite ^
  --collect-binaries nvidia.cublas ^
  --collect-binaries nvidia.cudnn ^
  --collect-binaries nvidia.cuda_nvrtc ^
  japo.py
if errorlevel 1 (
  echo.
  echo Build failed. See the messages above.
  pause
  exit /b
)

echo === 3/4 Copying config and model ===
set OUT=dist\Japo
if exist cloudflare.txt copy /y cloudflare.txt "%OUT%\" >nul
if exist gemini_key.txt copy /y gemini_key.txt "%OUT%\" >nul
if exist ai_custom.json copy /y ai_custom.json "%OUT%\" >nul
if exist translators.json copy /y translators.json "%OUT%\" >nul
if exist model (
  echo Copying model folder...
  xcopy /e /i /y /q model "%OUT%\model" >nul
)

echo === 4/4 Creating desktop shortcut ===
powershell -NoProfile -Command ^
  "$s=(New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop')+'\Japo.lnk');" ^
  "$s.TargetPath='%CD%\%OUT%\Japo.exe';" ^
  "$s.WorkingDirectory='%CD%\%OUT%';" ^
  "$s.IconLocation='%CD%\%OUT%\Japo.exe,0';" ^
  "$s.Save()"

echo.
echo ============================================
echo  Done! Open Japo from the desktop shortcut.
echo  App folder: %CD%\%OUT%
echo  If something goes wrong, check japo_log.txt in the app folder.
echo ============================================
pause
