#!/bin/bash

# ============================================================
# DeepSeek OCR Windows EXE 打包脚本
# ============================================================
# 功能: 使用 PyInstaller 和 deepseek conda 环境打包程序
# 使用: bash package_exe.sh
# ============================================================

set -e  # 任何错误时退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=======================================================================${NC}"
echo -e "${BLUE}DeepSeek OCR Windows EXE 打包工具${NC}"
echo -e "${BLUE}=======================================================================${NC}"

# 1. 检查 conda 环境
echo ""
echo -e "${YELLOW}Step 1: 验证 conda 环境${NC}"
echo "-------"

# 获取当前 shell 的初始化脚本
source $(conda info --base)/etc/profile.d/conda.sh

# 验证 deepseek 环境存在
if ! conda env list | grep -q "^deepseek "; then
    echo -e "${RED}❌ 错误: 找不到 'deepseek' conda 环境${NC}"
    echo "请先创建环境: conda create -n deepseek python=3.10 -y"
    exit 1
fi

echo -e "${GREEN}✅ 找到 deepseek conda 环境${NC}"

# 2. 激活环境
echo ""
echo -e "${YELLOW}Step 2: 激活环境${NC}"
echo "-------"

conda activate deepseek

PYTHON_PATH=$(which python)
PYTHON_VERSION=$(python --version)

echo -e "${GREEN}✅ 已激活环境${NC}"
echo "   Python 位置: ${PYTHON_PATH}"
echo "   Python 版本: ${PYTHON_VERSION}"

# 3. 验证关键依赖
echo ""
echo -e "${YELLOW}Step 3: 验证关键依赖${NC}"
echo "-------"

REQUIRED_PACKAGES=("torch" "vllm" "transformers" "pyinstaller" "tkinter" "PIL" "fitz")

MISSING_PACKAGES=()

for package in "${REQUIRED_PACKAGES[@]}"; do
    if python -c "import ${package}" 2>/dev/null; then
        echo -e "${GREEN}✅ ${package} 已安装${NC}"
    else
        echo -e "${YELLOW}⚠️  ${package} 缺失${NC}"
        MISSING_PACKAGES+=("${package}")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}缺失的包: ${MISSING_PACKAGES[@]}${NC}"
    echo "请运行以下命令安装:"
    echo "  conda activate deepseek"
    echo "  pip install ${MISSING_PACKAGES[@]}"
    exit 1
fi

# 4. 验证必要文件
echo ""
echo -e "${YELLOW}Step 4: 验证必要文件${NC}"
echo "-------"

cd /home/zc/deepseek-ocr

# 检查 spec 文件
if [ ! -f "DeepSeek-OCR.spec" ]; then
    echo -e "${RED}❌ 错误: 找不到 DeepSeek-OCR.spec 文件${NC}"
    exit 1
fi
echo -e "${GREEN}✅ DeepSeek-OCR.spec 文件存在${NC}"

# 检查主程序文件
if [ ! -f "DeepSeek-OCR-master/DeepSeek-OCR-vllm/gui_ocr_vllm.py" ]; then
    echo -e "${RED}❌ 错误: 找不到主程序文件${NC}"
    exit 1
fi
echo -e "${GREEN}✅ 主程序文件 (gui_ocr_vllm.py) 存在${NC}"

# 检查模型目录
if [ -d "models/DeepSeek-OCR" ]; then
    MODEL_SIZE=$(du -sh models/DeepSeek-OCR | awk '{print $1}')
    echo -e "${GREEN}✅ 本地模型目录存在 (大小: ${MODEL_SIZE})${NC}"
else
    echo -e "${YELLOW}⚠️  未找到本地模型目录 (models/DeepSeek-OCR)${NC}"
    echo "    注意: 打包时会从 HuggingFace 下载模型（需要网络）"
fi

# 5. 清理旧的打包文件
echo ""
echo -e "${YELLOW}Step 5: 清理旧的打包文件${NC}"
echo "-------"

if [ -d "build" ]; then
    echo "删除 build 目录..."
    rm -rf build
fi

if [ -d "dist/DeepSeek-OCR-Windows" ]; then
    echo "删除 dist/DeepSeek-OCR-Windows 目录..."
    rm -rf dist/DeepSeek-OCR-Windows
fi

echo -e "${GREEN}✅ 清理完成${NC}"

# 6. 运行 PyInstaller
echo ""
echo -e "${YELLOW}Step 6: 运行 PyInstaller${NC}"
echo "-------"

echo "这可能需要 30-60 分钟，请耐心等待..."
echo ""

START_TIME=$(date +%s)

# 运行 PyInstaller
pyinstaller DeepSeek-OCR.spec \
    --distpath ./dist \
    --buildpath ./build \
    --noconfirm

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))

echo ""
echo -e "${GREEN}✅ PyInstaller 执行完成 (耗时: ${ELAPSED_MIN} 分钟)${NC}"

# 7. 验证打包结果
echo ""
echo -e "${YELLOW}Step 7: 验证打包结果${NC}"
echo "-------"

if [ ! -d "dist/DeepSeek-OCR-Windows" ]; then
    echo -e "${RED}❌ 错误: 打包失败，找不到输出目录${NC}"
    exit 1
fi

echo -e "${GREEN}✅ 输出目录存在${NC}"

# 检查关键文件
EXE_FILE="dist/DeepSeek-OCR-Windows/DeepSeek-OCR.exe"
if [ -f "${EXE_FILE}" ]; then
    EXE_SIZE=$(ls -lh "${EXE_FILE}" | awk '{print $5}')
    echo -e "${GREEN}✅ 主程序文件 (DeepSeek-OCR.exe) 存在 (大小: ${EXE_SIZE})${NC}"
else
    echo -e "${RED}⚠️  警告: 找不到 DeepSeek-OCR.exe${NC}"
fi

# 计算总大小
TOTAL_SIZE=$(du -sh dist/DeepSeek-OCR-Windows | awk '{print $1}')

echo ""
echo -e "${BLUE}=======================================================================${NC}"
echo -e "${GREEN}✅ 打包完成！${NC}"
echo -e "${BLUE}=======================================================================${NC}"
echo ""
echo "📦 输出目录: dist/DeepSeek-OCR-Windows"
echo "📊 总大小: ${TOTAL_SIZE}"
echo ""
echo "📝 后续步骤:"
echo "  1. 将 dist/DeepSeek-OCR-Windows 目录复制到 Windows 机器"
echo "  2. 在 Windows 上安装 NSIS"
echo "  3. 运行以下命令生成安装程序:"
echo "     makensis.exe create_installer.nsi"
echo ""
echo "或者直接使用 dist/DeepSeek-OCR-Windows 中的 exe 文件"
echo ""
