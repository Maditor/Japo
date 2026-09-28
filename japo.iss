; Japo installer (Inno Setup 6) - produces a single Japo-Setup.exe
#define AppName "Japo"
#define AppVersion "1.0"

[Setup]
AppId={{6E4B2C1A-7A3F-4C55-9B8E-4A0F1D2C3B5E}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=Maditor
AppPublisherURL=https://github.com/Maditor
AppSupportURL=https://github.com/Maditor/Japo/issues
AppUpdatesURL=https://github.com/Maditor/Japo/releases
; Per-user install: no admin rights needed, app can write its settings and log
DefaultDirName={localappdata}\Programs\{#AppName}
PrivilegesRequired=lowest
DisableProgramGroupPage=yes
OutputDir=Output
OutputBaseFilename=Japo-Setup
SetupIconFile=icon.ico
UninstallDisplayIcon={app}\Japo.exe
UninstallDisplayName={#AppName}
Compression=lzma2/ultra64
SolidCompression=yes
LZMAUseSeparateProcess=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"

[Files]
; App files (model, keys and personal settings excluded; the model downloads on first run)
Source: "dist\Japo\*"; DestDir: "{app}"; \
  Excludes: "model,cloudflare.txt,gemini_key.txt,japo_settings.json,japo_log.txt"; \
  Flags: ignoreversion recursesubdirs createallsubdirs
; API keys: included only if present, never overwrite existing ones
Source: "cloudflare.txt"; DestDir: "{app}"; Flags: skipifsourcedoesntexist onlyifdoesntexist
Source: "gemini_key.txt"; DestDir: "{app}"; Flags: skipifsourcedoesntexist onlyifdoesntexist

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\Japo.exe"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\Japo.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\Japo.exe"; Description: "Launch Japo"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: files; Name: "{app}\japo_settings.json"
Type: files; Name: "{app}\japo_log.txt"
