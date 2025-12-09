#!/bin/bash
# Linux 环境 GUI 测试脚本

echo "=========================================="
echo "DeepSeek OCR vLLM GUI - Linux 测试"
echo "=========================================="
echo ""

# 检查是否在正确的目录
if [ ! -f "gui_ocr_vllm.py" ]; then
    echo "❌ 错误: 未找到 gui_ocr_vllm.py"
    echo "请在 DeepSeek-OCR-master/DeepSeek-OCR-vllm 目录下运行"
    exit 1
fi

# 检查 Python 环境
echo "检查 Python 环境..."
python3 --version

# 检查显示环境
if [ -z "$DISPLAY" ]; then
    echo "⚠️  警告: DISPLAY 环境变量未设置"
    echo "尝试设置为 :0"
    export DISPLAY=:0
fi

echo "DISPLAY = $DISPLAY"
echo ""

# 检查依赖
echo "检查依赖..."
python3 -c "import tkinter; print('✓ tkinter')" 2>/dev/null || echo "❌ tkinter 未安装"
python3 -c "import torch; print('✓ torch')" 2>/dev/null || echo "❌ torch 未安装"
python3 -c "import PIL; print('✓ PIL')" 2>/dev/null || echo "❌ PIL 未安装"
python3 -c "import fitz; print('✓ PyMuPDF')" 2>/dev/null || echo "❌ PyMuPDF 未安装"

# 检查 vLLM (可选)
python3 -c "import vllm; print('✓ vLLM 可用')" 2>/dev/null || echo "⚠️  vLLM 不可用，将使用 HF Transformers"

# 检查 Transformers (必需)
python3 -c "import transformers; print('✓ transformers')" 2>/dev/null || {
    echo "❌ transformers 未安装"
    echo "正在安装..."
    pip install transformers
}

echo ""
echo "=========================================="
echo "启动 GUI..."
echo "=========================================="
echo ""

# 运行 GUI
python3 gui_ocr_vllm.py

echo ""
echo "GUI 已关闭"
