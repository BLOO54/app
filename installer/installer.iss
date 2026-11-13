; Inno Setup script -> AIImageStudio-Setup-<version>.exe
; Installs for the current user only, so it NEVER needs administrator rights.
; Contents: a private copy of Python (build\python) + the app (build\app).
; The big AI engine (PyTorch) is downloaded by the app on first start and kept across updates.

#ifndef MyAppVersion
  #define MyAppVersion "1.0.0"
#endif
#define MyAppName "AI Image Studio"

[Setup]
AppId={{4C7E2B91-8D3A-4F60-A1E5-93B2D7C64F18}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher=AI Image Studio
DefaultDirName={localappdata}\Programs\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
DisableDirPage=yes
PrivilegesRequired=lowest
OutputDir=..\Output
OutputBaseFilename=AIImageStudio-Setup-{#MyAppVersion}
SetupIconFile=..\assets\icon.ico
UninstallDisplayIcon={app}\icon.ico
Compression=lzma2/fast
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
CloseApplications=yes

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Shortcuts:"

[Files]
Source: "..\build\python\*"; DestDir: "{app}\python"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\build\app\*"; DestDir: "{app}\app"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\assets\icon.ico"; DestDir: "{app}"; Flags: ignoreversion

[InstallDelete]
; remove old app code before copying the new version (the AI engine in python\Lib is kept)
Type: filesandordirs; Name: "{app}\app"

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\python\pythonw.exe"; Parameters: "-m aistudio"; WorkingDir: "{app}\app"; IconFilename: "{app}\icon.ico"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\python\pythonw.exe"; Parameters: "-m aistudio"; WorkingDir: "{app}\app"; IconFilename: "{app}\icon.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\python\pythonw.exe"; Parameters: "-m aistudio"; WorkingDir: "{app}\app"; Description: "Start {#MyAppName} now"; Flags: nowait postinstall skipifsilent
; after an in-app update (silent install started with /RESTARTAPP=1) start the new version again
Filename: "{app}\python\pythonw.exe"; Parameters: "-m aistudio"; WorkingDir: "{app}\app"; Flags: nowait; Check: ShouldRelaunch

[Code]
function ShouldRelaunch: Boolean;
begin
  Result := ExpandConstant('{param:RESTARTAPP|0}') = '1';
end;

[UninstallRun]
Filename: "powershell.exe"; Parameters: "-NoProfile -Command ""Get-Process pythonw,python -ErrorAction SilentlyContinue | Where-Object {{ $_.Path -like '{app}\*' } | Stop-Process -Force"""; Flags: runhidden; RunOnceId: "KillApp"

[UninstallDelete]
; the AI engine (several GB) and downloaded models (~7 GB each). Your pictures are kept.
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\app"
Type: filesandordirs; Name: "{localappdata}\AIImageStudio"
