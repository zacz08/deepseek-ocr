"""
DeepSeek OCR GUI 打包脚本
使用 PyInstaller 将 GUI 打包为独立的 Windows 可执行文件
"""

import os
import subprocess
import sys

def check_pyinstaller():
    """检查并安装 PyInstaller"""
    try:
        import PyInstaller
        print("✓ PyInstaller 已安装")
        return True
    except ImportError:
        print("× PyInstaller 未安装")
        print("正在安装 PyInstaller...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
            print("✓ PyInstaller 安装成功")
            return True
        except Exception as e:
            print(f"✗ PyInstaller 安装失败: {e}")
            return False

def create_spec_file():
    """创建自定义的 spec 文件"""
    spec_content = """# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['DeepSeek-OCR-master\\\\DeepSeek-OCR-hf\\\\gui_ocr.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'transformers',
        'torch',
        'PIL',
        'einops',
        'easydict',
        'addict',
        'tokenizers',
        'safetensors',
        'huggingface_hub',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='DeepSeekOCR',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # 保留控制台以显示调试信息
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='DeepSeekOCR',
)
"""
    
    with open('DeepSeekOCR.spec', 'w', encoding='utf-8') as f:
        f.write(spec_content)
    print("✓ 已创建 spec 文件")

def build_exe():
    """执行打包"""
    print("\n开始打包...")
    print("=" * 60)
    print("注意: 打包过程可能需要 10-30 分钟，请耐心等待")
    print("=" * 60)
    
    try:
        # 使用 spec 文件打包
        cmd = [
            'pyinstaller',
            '--clean',
            'DeepSeekOCR.spec'
        ]
        
        subprocess.check_call(cmd)
        print("\n" + "=" * 60)
        print("✓ 打包完成!")
        print("=" * 60)
        print(f"\n可执行文件位置: dist\\DeepSeekOCR\\DeepSeekOCR.exe")
        print("\n使用说明:")
        print("1. 整个 dist\\DeepSeekOCR 文件夹就是完整的应用程序")
        print("2. 可以将该文件夹复制到任何 Windows 电脑上使用")
        print("3. 首次运行时会自动下载模型（需要网络连接）")
        print("4. 建议创建快捷方式到桌面方便使用")
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ 打包失败: {e}")
        return False
    except Exception as e:
        print(f"\n✗ 发生错误: {e}")
        return False
    
    return True

def main():
    print("=" * 60)
    print("DeepSeek OCR GUI 打包工具")
    print("=" * 60)
    print()
    
    # 检查并安装 PyInstaller
    if not check_pyinstaller():
        print("\n请手动安装 PyInstaller: pip install pyinstaller")
        return
    
    # 创建 spec 文件
    create_spec_file()
    
    # 执行打包
    build_exe()

if __name__ == "__main__":
    main()
