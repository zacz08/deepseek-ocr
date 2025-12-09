@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================================
echo DeepSeek OCR GUI (vLLM版) 启动器
echo ============================================================
echo.

REM 设置工作目录
cd DeepSeek-OCR-master\DeepSeek-OCR-vllm

echo 正在启动 GUI...
python gui_ocr_vllm.py

pause
