; DeepSeek OCR GUI 安装程序脚本
; 需要 Inno Setup 6 或更高版本
; 下载地址: https://jrsoftware.org/isdl.php

#define MyAppName "DeepSeek OCR"
#define MyAppVersion "1.0"
#define MyAppPublisher "DeepSeek AI"
#define MyAppExeName "DeepSeekOCR.exe"

[Setup]
; 应用程序基本信息
AppId={{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
OutputDir=installer
OutputBaseFilename=DeepSeekOCR_Setup_v{#MyAppVersion}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern

; 系统要求
MinVersion=10.0.19041
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64

; 权限
PrivilegesRequired=lowest

; 界面设置
DisableProgramGroupPage=yes
LicenseFile=LICENSE
InfoBeforeFile=打包说明.md

[Languages]
Name: "chinesesimplified"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; 主程序文件
Source: "dist\DeepSeekOCR\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "GUI使用说明.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "打包说明.md"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\使用说明"; Filename: "{app}\GUI使用说明.md"
Name: "{group}\卸载 {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[Messages]
WelcomeLabel2=这将在您的电脑上安装 [name/ver]。%n%n强烈建议在继续之前关闭所有其他应用程序。%n%n注意：首次运行时需要下载 AI 模型（约 5-10 GB），请确保有良好的网络连接。

[Code]
function InitializeSetup(): Boolean;
var
  ResultCode: Integer;
begin
  Result := True;
  
  // 检查是否有足够的磁盘空间 (至少 15 GB)
  if GetSpaceOnDisk(ExpandConstant('{app}'), False, Free) < 15 * 1024 * 1024 * 1024 then
  begin
    MsgBox('磁盘空间不足！安装至少需要 15 GB 可用空间。', mbError, MB_OK);
    Result := False;
    Exit;
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    // 安装完成后的提示
    MsgBox('安装完成！' + #13#10#13#10 + 
           '重要提示：' + #13#10 +
           '1. 首次运行时会自动下载 AI 模型（5-10 GB）' + #13#10 +
           '2. 下载过程需要稳定的网络连接' + #13#10 +
           '3. 建议在有 NVIDIA GPU 的电脑上运行以获得最佳性能' + #13#10 +
           '4. 详细使用说明请查看安装目录中的"使用说明.md"文件', 
           mbInformation, MB_OK);
  end;
end;
