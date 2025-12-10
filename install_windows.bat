@echo off
REM ============================================================
REM DeepSeek OCR Windows 安装脚本
REM ============================================================
REM 使用方法：双击运行此文件
REM ============================================================

echo ============================================================
echo DeepSeek OCR Windows 环境安装
echo ============================================================
echo.

REM 检查 conda 是否安装
where conda >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [错误] 未检测到 Conda
    echo.
    echo 请先安装 Anaconda 或 Miniconda:
    echo https://www.anaconda.com/download
    echo 或 https://docs.conda.io/en/latest/miniconda.html
    echo.
    pause
    exit /b 1
)

echo [1/5] 检测到 Conda 环境
echo.

REM 创建 conda 环境
echo [2/5] 创建 Python 环境...
call conda create -n deepseek python=3.12 -y
if %ERRORLEVEL% NEQ 0 (
    echo [错误] 环境创建失败
    pause
    exit /b 1
)

echo.
echo [3/5] 激活环境...
call conda activate deepseek

echo.
echo [4/5] 安装依赖包（这可能需要 10-30 分钟）...
echo 正在安装 PyTorch...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo 正在安装 vLLM...
pip install vllm

echo.
echo 正在安装其他依赖...
pip install transformers pillow pymupdf tokenizers huggingface_hub

echo.
echo [5/5] 测试安装...
python -c "import torch; import vllm; import transformers; print('✓ 所有依赖安装成功')"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo ✓ 安装完成！
    echo ============================================================
    echo.
    echo 使用方法：
    echo   1. 双击 run_gui.bat 启动程序
    echo   2. 或在命令行中运行：
    echo      conda activate deepseek
    echo      python DeepSeek-OCR-master\DeepSeek-OCR-vllm\gui_ocr_vllm.py
    echo.
) else (
    echo.
    echo [错误] 依赖测试失败
    echo 请检查网络连接或手动安装依赖
)

echo.
pause
