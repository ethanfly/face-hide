#ifndef MyAppVersion
  #define MyAppVersion "0.0.0"
#endif
#ifndef DistDir
  #define DistDir "..\dist\FaceHide"
#endif
#ifndef IconFile
  #define IconFile "FaceHide.ico"
#endif
#ifndef OutDir
  #define OutDir "..\dist"
#endif

[Setup]
AppId={{E6C8F3A1-4B27-4D9E-9A51-0C2E8B7D4A91}
AppName=当面隐藏
AppVersion={#MyAppVersion}
AppVerName=当面隐藏 {#MyAppVersion}
AppPublisher=FaceHide
DefaultDirName={autopf}\FaceHide
DefaultGroupName=当面隐藏
DisableProgramGroupPage=yes
OutputDir={#OutDir}
OutputBaseFilename=FaceHide-{#MyAppVersion}-win64-setup
SetupIconFile={#IconFile}
UninstallDisplayIcon={app}\FaceHide.exe
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
CloseApplications=yes
UsedUserAreasWarning=no
MinVersion=10.0

[Languages]
Name: "chinesesimp"; MessagesFile: "languages\ChineseSimplified.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
Source: "{#DistDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\当面隐藏"; Filename: "{app}\FaceHide.exe"; WorkingDir: "{app}"
Name: "{autodesktop}\当面隐藏"; Filename: "{app}\FaceHide.exe"; WorkingDir: "{app}"

[Run]
Filename: "{app}\FaceHide.exe"; Description: "启动当面隐藏"; Flags: nowait postinstall skipifsilent
