#!/bin/bash
set -e

source $(conda info --base)/etc/profile.d/conda.sh
conda activate deepseek

cd /home/zc/deepseek-ocr

# 清理旧文件
rm -rf build dist/DeepSeek-OCR-Windows 2>/dev/null || true

# 运行 PyInstaller
echo "开始打包..."
pyinstaller DeepSeek-OCR.spec --distpath ./dist --workpath ./build --noconfirm

echo ""
echo "✅ 打包完成！"
echo "输出目录: /home/zc/deepseek-ocr/dist/DeepSeek-OCR-Windows"
