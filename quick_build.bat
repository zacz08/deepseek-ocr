@echo off
chcp 65001 >nul
echo ============================================================
echo DeepSeek OCR GUI 一键打包工具
echo ============================================================
echo.

REM 激活 conda 环境
echo [1/3] 激活 conda 环境...
call C:\Zachary\03_Software\miniconda3\shell\condabin\conda-hook.ps1
call conda activate base

REM 检查并安装 PyInstaller
echo.
echo [2/3] 检查 PyInstaller...
python -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo PyInstaller 未安装，正在安装...
    pip install pyinstaller
) else (
    echo PyInstaller 已安装
)

REM 执行打包
echo.
echo [3/3] 开始打包...
echo.
python build_exe.py

echo.
echo ============================================================
echo 打包完成！
echo ============================================================
echo.
echo 可执行文件位置: dist\DeepSeekOCR\DeepSeekOCR.exe
echo.
echo 按任意键退出...
pause >nul
