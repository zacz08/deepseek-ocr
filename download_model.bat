@echo off
chcp 65001 >nul
echo ============================================================
echo DeepSeek OCR 模型下载工具
echo ============================================================
echo.
echo 此工具将下载模型到本地，用于离线部署
echo 需要下载约 5-10 GB 的数据，请确保：
echo   1. 网络连接稳定
echo   2. 有足够的磁盘空间（至少 15GB）
echo.
pause
echo.

cd /d "%~dp0"

REM 激活 conda 环境（如果使用）
if exist "C:\Zachary\03_Software\miniconda3\shell\condabin\conda-hook.ps1" (
    echo 激活 conda 环境...
    call conda activate deepseek-ocr
)

echo.
echo 开始下载模型...
echo.

python download_model.py

echo.
echo ============================================================
pause
