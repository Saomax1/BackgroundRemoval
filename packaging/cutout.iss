; Inno Setup script for the Windows installer. Build from the repo root:
;     iscc /DAppVersion=0.1.57 /DFirstRunGB=2.9 packaging\cutout.iss
; Input:  dist\setup\  (from packaging/make_release.py --setup-dir: the app, its libraries,
;         Qt and release.json; the GPU runtime and the AI models are left out)
; Output: dist\installer\Cutout-Setup-<version>.exe, a few hundred MB, small enough for a
;         GitHub release. On its first start the app downloads the parts left out
;         (cutout.update), and later updates download only what changed.

#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif
#ifndef FirstRunGB
  #define FirstRunGB "3"
#endif
#define AppName "Cutout"
#ifndef SourceDir
  #define SourceDir "..\dist\setup"
#endif

[Setup]
AppId={{6F1C2A7E-3B9D-4E52-9C1A-8B7D0E4F2A61}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=Cutout
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
; Per-user install (no admin prompt). It also keeps the install folder writable, which
; the in-app updater needs.
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir=..\dist\installer
OutputBaseFilename=Cutout-Setup-{#AppVersion}
Compression=lzma2/max
SolidCompression=yes
LZMAUseSeparateProcess=yes
DiskSpanning=no
UninstallDisplayIcon={app}\Cutout.exe
WizardStyle=modern
ShowLanguageDialog=no
CloseApplications=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[InstallDelete]
; Re-check every installed file on the next start. Models and GPU files that are still
; current are kept (no 3 GB download again); anything stale is replaced or removed then.
Type: files; Name: "{app}\install-state.json"

[UninstallDelete]
; Files the app downloaded itself, which Setup did not install.
Type: filesandordirs; Name: "{app}\models"
Type: filesandordirs; Name: "{app}\_internal"
Type: files; Name: "{app}\release.json"
Type: files; Name: "{app}\install-state.json"
; Optimised model graphs the app saved for faster starts (about the size of the models).
Type: filesandordirs; Name: "{localappdata}\Cutout\graph-cache"

[Messages]
FinishedLabel=Cutout is installed.%n%nThe first time it starts it downloads its AI models and GPU support (about {#FirstRunGB} GB, once). After that it works fully offline.

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\Cutout.exe"
; Same app with detailed logging, for reporting a problem (Help > Copy Debug Report).
Name: "{autoprograms}\{#AppName} (debug mode)"; Filename: "{app}\Cutout.exe"; Parameters: "--debug"; Comment: "Start Cutout with detailed logging"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\Cutout.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\Cutout.exe"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent

[Code]
function InitializeSetup(): Boolean;
begin
  Result := True;
  // nvcuda.dll ships with the NVIDIA display driver. Everything else CUDA needs is bundled.
  if (not WizardSilent) and (not FileExists(ExpandConstant('{sys}\nvcuda.dll'))) then
    Result := MsgBox('No NVIDIA graphics driver was found on this PC.' + #13#10#13#10 +
      'Cutout will still work, but on the CPU it is much slower. ' +
      'Install or update the NVIDIA driver to use the GPU.' + #13#10#13#10 +
      'Continue installing?', mbConfirmation, MB_YESNO) = IDYES;
end;
