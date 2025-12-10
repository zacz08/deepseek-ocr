; DeepSeek OCR Windows 安装程序脚本
; 使用 NSIS (Nullsoft Scriptable Install System) 生成
; 
; 使用方法：
; 在 Windows 上安装 NSIS，然后运行：
; "C:\Program Files (x86)\NSIS\makensis.exe" create_installer.nsi

;================================
; 包含现代化 UI
;================================

!include "MUI2.nsh"
!include "x64.nsh"
!include "FileFunc.nsh"

;================================
; 基本信息
;================================

Name "DeepSeek OCR v1.0"
OutFile "DeepSeek-OCR-Windows-v1.0-Setup.exe"
InstallDir "$PROGRAMFILES\DeepSeek OCR"
InstallDirRegKey HKCU "Software\DeepSeek OCR" ""

; 管理员权限检查
RequestExecutionLevel admin

;================================
; 版本信息
;================================

VIProductVersion "1.0.0.0"
VIAddVersionKey /LANG=2052 "ProductName" "DeepSeek OCR"
VIAddVersionKey /LANG=2052 "CompanyName" "DeepSeek"
VIAddVersionKey /LANG=2052 "FileDescription" "AI 文档识别工具"
VIAddVersionKey /LANG=2052 "FileVersion" "1.0.0.0"
VIAddVersionKey /LANG=2052 "LegalCopyright" "(c) 2024 DeepSeek"

;================================
; 用户界面
;================================

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_LANGUAGE "SimplifiedChinese"

;================================
; 语言设置
;================================

LangString DESC_Section1 ${LANG_SIMPCHINESE} "安装 DeepSeek OCR 主程序"
LangString DESC_Section2 ${LANG_SIMPCHINESE} "创建开始菜单快捷方式"
LangString DESC_Section3 ${LANG_SIMPCHINESE} "创建桌面快捷方式"
LangString APP_RUNNING ${LANG_SIMPCHINESE} "DeepSeek OCR 正在运行，请先关闭"
LangString UNINSTALL_SUCCESS ${LANG_SIMPCHINESE} "成功卸载 DeepSeek OCR"

;================================
; 安装部分
;================================

Section "主程序" Section1
  
  ; 检查程序是否正在运行
  FindProcDLL::FindProcess "DeepSeek-OCR.exe"
  IntCmp $R0 0 done
    MessageBox MB_YESNO|MB_ICONEXCLAMATION $(APP_RUNNING) IDYES done IDNO 0
    Abort
  done:
  
  SetOutPath "$INSTDIR"
  
  ; 复制所有文件
  File /r "dist\DeepSeek-OCR-Windows\*.*"
  
  ; 创建卸载程序
  WriteUninstaller "$INSTDIR\Uninstall.exe"
  
  ; 注册表项（用于卸载）
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\DeepSeek OCR" \
    "DisplayName" "DeepSeek OCR v1.0"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\DeepSeek OCR" \
    "UninstallString" "$INSTDIR\Uninstall.exe"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\DeepSeek OCR" \
    "DisplayIcon" "$INSTDIR\DeepSeek-OCR.exe"
  
  WriteRegStr HKCU "Software\DeepSeek OCR" "" "$INSTDIR"
  
  DetailPrint "主程序安装完成"
  
SectionEnd

Section "开始菜单快捷方式" Section2
  
  CreateDirectory "$SMPROGRAMS\DeepSeek OCR"
  CreateShortcut "$SMPROGRAMS\DeepSeek OCR\DeepSeek OCR.lnk" \
    "$INSTDIR\DeepSeek-OCR.exe" \
    "" "$INSTDIR\DeepSeek-OCR.exe" 0
  CreateShortcut "$SMPROGRAMS\DeepSeek OCR\卸载.lnk" \
    "$INSTDIR\Uninstall.exe" \
    "" "$INSTDIR\Uninstall.exe" 0
  
  DetailPrint "开始菜单快捷方式创建完成"
  
SectionEnd

Section "桌面快捷方式" Section3
  
  CreateShortcut "$DESKTOP\DeepSeek OCR.lnk" \
    "$INSTDIR\DeepSeek-OCR.exe" \
    "" "$INSTDIR\DeepSeek-OCR.exe" 0
  
  DetailPrint "桌面快捷方式创建完成"
  
SectionEnd

;================================
; 卸载部分
;================================

Section "Uninstall"
  
  ; 删除快捷方式
  Delete "$SMPROGRAMS\DeepSeek OCR\DeepSeek OCR.lnk"
  Delete "$SMPROGRAMS\DeepSeek OCR\卸载.lnk"
  Delete "$SMPROGRAMS\DeepSeek OCR\*.lnk"
  RMDir "$SMPROGRAMS\DeepSeek OCR"
  
  Delete "$DESKTOP\DeepSeek OCR.lnk"
  
  ; 删除应用目录
  RMDir /r "$INSTDIR"
  
  ; 删除注册表
  DeleteRegKey HKCU "Software\DeepSeek OCR"
  DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\DeepSeek OCR"
  
  MessageBox MB_OK $(UNINSTALL_SUCCESS)
  
SectionEnd

;================================
; 组件描述
;================================

!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
  !insertmacro MUI_DESCRIPTION_TEXT ${Section1} $(DESC_Section1)
  !insertmacro MUI_DESCRIPTION_TEXT ${Section2} $(DESC_Section2)
  !insertmacro MUI_DESCRIPTION_TEXT ${Section3} $(DESC_Section3)
!insertmacro MUI_FUNCTION_DESCRIPTION_END

;================================
; 安装后操作
;================================

Function .onInstSuccess
  ExecShell "open" "$INSTDIR"
FunctionEnd

;================================
; 卸载前检查
;================================

Function un.onInit
  MessageBox MB_ICONQUESTION|MB_YESNO|MB_DEFBUTTON2 \
    "确定要卸载 DeepSeek OCR 吗？" IDYES +2
  Abort
FunctionEnd

;================================
; 完整说明
;================================

/*
========================================
NSIS 脚本使用说明
========================================

【安装 NSIS】

Windows 上：
1. 下载：https://sourceforge.net/projects/nsis/files/
2. 下载最新稳定版（建议 v3.x）
3. 双击安装，选择默认路径 C:\Program Files (x86)\NSIS\

或使用 Chocolatey：
  choco install nsis

【生成安装程序】

在命令行中运行：
  "C:\Program Files (x86)\NSIS\makensis.exe" create_installer.nsi

或在 NSIS 的图形界面中打开 create_installer.nsi 并点击 Compile

【输出文件】

生成的安装程序：
  DeepSeek-OCR-Windows-v1.0-Setup.exe

文件大小：通常与 dist 文件夹相同（12-15GB）

【测试安装程序】

1. 双击 .exe 文件
2. 按照安装向导进行
3. 选择安装位置（默认 C:\Program Files\DeepSeek OCR）
4. 选择快捷方式选项
5. 完成安装

【注意事项】

✓ 确保 dist\DeepSeek-OCR-Windows\ 文件夹存在且完整
✓ 生成安装程序时会自动压缩文件，可能需要 10-20 分钟
✓ 最终 .exe 大小约为 dist 文件夹大小（因为包含所有文件）
✓ 用户安装后会占用约 12-15GB 的磁盘空间

【自定义安装程序】

修改以下内容可以定制安装程序：

- Name "DeepSeek OCR v1.0"           → 修改版本号
- OutFile "...Setup.exe"              → 修改输出文件名
- InstallDir "$PROGRAMFILES\..."      → 修改默认安装位置
- !insertmacro MUI_LANGUAGE ...       → 添加其他语言

========================================
*/
