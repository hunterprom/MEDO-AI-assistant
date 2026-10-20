; Inno Setup script for MEDO — the one-click Windows installer.
; Build (after the three PyInstaller bundles are in dist\):  iscc packaging\installer.iss
; Produces MEDO-Setup.exe. The big AI model is NOT in here — the first-run wizard
; downloads it (keeps the installer small).

#define AppName    "MEDO"
#define AppVersion "2.0.0"
#define AppExe     "MEDO.exe"
#define Publisher  "MEDO"

[Setup]
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#Publisher}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
UninstallDisplayName={#AppName}
UninstallDisplayIcon={app}\medo.ico
OutputBaseFilename=MEDO-Setup
Compression=lzma2
SolidCompression=yes
PrivilegesRequired=admin
WizardStyle=modern
; The MEDO logo: icon of MEDO-Setup.exe itself and of the wizard window.
SetupIconFile=medo.ico
ArchitecturesInstallIn64BitMode=x64compatible
; CODE SIGNING (see docs/Packaging.md): without a signed exe, Windows SmartScreen
; will warn on first run. Sign dist\MEDO\MEDO.exe and MEDO-Setup.exe with
; signtool + an EV/OV cert, then set:  SignTool=mysigntool

[Languages]
Name: "en"; MessagesFile: "compiler:Default.isl"

[Files]
; Launcher/tray (the user-facing app) at the root...
Source: "..\dist\MEDO\*";         DestDir: "{app}";          Flags: recursesubdirs ignoreversion
; ...engine and vision as SEPARATE bundles in subfolders (vision = numpy<2 env).
Source: "..\dist\medo-engine\*";  DestDir: "{app}\engine";   Flags: recursesubdirs ignoreversion
Source: "..\dist\medo-vision\*";  DestDir: "{app}\vision";   Flags: recursesubdirs ignoreversion
; App icon — used by the shortcuts and Programs & Features (see [Icons] below).
Source: "medo.ico";               DestDir: "{app}";          Flags: ignoreversion

[Icons]
Name: "{group}\{#AppName}";          Filename: "{app}\{#AppExe}"; IconFilename: "{app}\medo.ico"
Name: "{commondesktop}\{#AppName}";  Filename: "{app}\{#AppExe}"; IconFilename: "{app}\medo.ico"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"

[Run]
; Launch the first-run wizard right after install.
Filename: "{app}\{#AppExe}"; Description: "Start MEDO"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}"

[Code]
// Uninstall: ASK before removing downloaded models + user data (don't assume).
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
  begin
    if MsgBox('Also remove your MEDO settings and downloaded AI models?' #13#10
      + 'Choose No to keep them for a future reinstall.',
      mbConfirmation, MB_YESNO) = IDYES then
    begin
      DelTree(ExpandConstant('{userappdata}\MEDO'), True, True, True);
      // Ollama models live in the Ollama store (%USERPROFILE%\.ollama or the
      // OLLAMA_MODELS path). We do NOT delete them automatically here: a shared
      // Ollama may hold models other apps use. docs/Packaging.md tells the user
      // how to remove them if they want the space back.
    end;
  end;
end;
