# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec 文件：用于打包 DeepSeek OCR GUI 为 Windows EXE
使用方法：pyinstaller DeepSeek-OCR.spec
"""

import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# ==============================================================================
# 配置项
# ==============================================================================

# 使用 SPECPATH (PyInstaller 提供的变量，指向 spec 文件所在目录)
PROJECT_DIR = SPECPATH
GUI_FILE = os.path.join(PROJECT_DIR, 'DeepSeek-OCR-master/DeepSeek-OCR-vllm/gui_ocr_vllm.py')

# ==============================================================================
# 收集数据文件和模块
# ==============================================================================

# 收集 transformers 相关数据
datas = []
datas += collect_data_files('transformers')
datas += collect_data_files('tokenizers')
datas += collect_data_files('huggingface_hub')

# 收集 vLLM 相关数据
datas += collect_data_files('vllm')

# ⚠️ 不打包模型文件（13GB 太大，用户单独下载）
# 用户需要将 models/DeepSeek-OCR 文件夹放在 EXE 同目录下
print("⚠ 模型文件未打包，用户需要单独提供 models/DeepSeek-OCR 文件夹")

# 添加 GUI 资源（如有）
if os.path.exists(os.path.join(PROJECT_DIR, 'assets')):
    datas.append(('assets', 'assets'))
    print("✓ 已包含 GUI 资源")

# ==============================================================================
# 二进制文件（CUDA/cuDNN）
# ==============================================================================

binaries = []

# 注意：需要根据你的系统调整以下路径
# 例如，如果你的 CUDA 在 /usr/local/cuda-12.1，应该这样：
# binaries = [
#     ('/usr/local/cuda-12.1/lib64', 'cuda/lib64'),
#     ('/usr/local/cuda-12.1/bin', 'cuda/bin'),
# ]

# 如果有 cuDNN，添加：
# binaries += [
#     ('/path/to/cudnn/lib', 'cudnn/lib'),
# ]

# ==============================================================================
# 隐藏导入
# ==============================================================================

hiddenimports = [
    # Transformers
    'transformers',
    'transformers.models',
    'transformers.models.auto',
    
    # vLLM
    'vllm',
    'vllm.model_executor',
    'vllm.model_executor.models',
    'vllm.model_executor.layers',
    'vllm.model_executor.layers.attention',
    'vllm.worker',
    'vllm.worker.worker',
    'vllm.worker.worker_base',
    
    # GPU/CUDA
    'flash_attn',
    'flash_attn.flash_attn_interface',
    
    # PIL
    'PIL',
    'PIL.Image',
    
    # PyMuPDF
    'fitz',
    
    # 其他
    'torch',
    'numpy',
    'regex',
]

# 添加 vLLM 的所有子模块
hiddenimports += collect_submodules('vllm')

# ==============================================================================
# 分析
# ==============================================================================

a = Analysis(
    [GUI_FILE],
    pathex=[PROJECT_DIR],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludedimports=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    noarchive=False,
    # 关键：不使用符号链接，复制实际文件
    module_collection_mode={
        '*': 'pyz+py',  # 所有模块都收集
    },
)

# 移除符号链接，使用实际文件
import shutil

datas_fixed = []
for d in a.datas:
    if os.path.islink(d[1]):
        real_path = os.path.realpath(d[1])
        if os.path.exists(real_path):
            datas_fixed.append((d[0], real_path, d[2]))
        else:
            datas_fixed.append(d)
    else:
        datas_fixed.append(d)
a.datas = datas_fixed

binaries_fixed = []
for b in a.binaries:
    if os.path.islink(b[1]):
        real_path = os.path.realpath(b[1])
        if os.path.exists(real_path):
            binaries_fixed.append((b[0], real_path, b[2]))
        else:
            binaries_fixed.append(b)
    else:
        binaries_fixed.append(b)
a.binaries = binaries_fixed

# ==============================================================================
# PYZ
# ==============================================================================

pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=None
)

# ==============================================================================
# EXE (使用 onedir 模式避免 4GB 限制)
# ==============================================================================

exe = EXE(
    pyz,
    a.scripts,
    [],  # 不包含 binaries (onedir 模式)
    exclude_binaries=True,  # 重要：使用 onedir 模式
    name='DeepSeek-OCR',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # True = 显示控制台窗口（用于日志输出）
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

# ==============================================================================
# 集合（最终打包）
# ==============================================================================

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='DeepSeek-OCR-Windows',
)

# ==============================================================================
# 打包说明
# ==============================================================================

"""
使用方法：

1. 确保已安装 PyInstaller：
   pip install pyinstaller==6.0.0

2. 在项目根目录运行：
   pyinstaller DeepSeek-OCR.spec

3. 打包完成后，输出文件夹：
   dist/DeepSeek-OCR-Windows/

4. 验证输出：
   ls -lh dist/DeepSeek-OCR-Windows/
   # 应该包含 DeepSeek-OCR.exe 和所有依赖

注意事项：

✓ 如果模型很大，打包可能需要 30 分钟到 1 小时
✓ 最终包大小约 12-15GB（包含完整模型）
✓ 确保磁盘空间充足（至少 50GB）
✓ 如果有 CUDA，需要在 binaries 中指定路径

如果遇到问题：

1. 检查是否所有依赖都已安装：
   python -c "import vllm, transformers, torch; print('OK')"

2. 查看 PyInstaller 的详细输出：
   pyinstaller DeepSeek-OCR.spec --debug=all

3. 如果某个库无法找到，可以在 hiddenimports 中添加
"""
