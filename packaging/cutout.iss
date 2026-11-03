; Inno Setup script for the one-file Windows installer (needs Inno Setup 6.5.2 or newer,
; which lifted the old 2 GB single-exe limit to ~4 GB). Build from the repo root:
;     iscc /DAppVersion=0.1.0 packaging\cutout.iss
; Input:  dist\Cutout\  (PyInstaller one-folder build, models already staged in models\)
; Output: dist\installer\Cutout-Setup-<version>.exe

#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif
#define AppName "Cutout"
#define SourceDir "..\dist\Cutout"

[Setup]
AppId={{6F1C2A7E-3B9D-4E52-9C1A-8B7D0E4F2A61}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=Cutout
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
; Per-user install by default (no admin prompt); the dialog still offers "all users".
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir=..\dist\installer
OutputBaseFilename=Cutout-Setup-{#AppVersion}
; Most of the payload is model weights and CUDA DLLs. Non-solid LZMA2 keeps compile
; memory sane and lets Setup start extracting without decompressing everything first.
Compression=lzma2/max
SolidCompression=no
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
; Models are replaced wholesale on upgrade so a dropped or renamed model never lingers.
Type: filesandordirs; Name: "{app}\models"
Type: filesandordirs; Name: "{app}\_internal"

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\Cutout.exe"
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
