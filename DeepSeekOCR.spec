# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['DeepSeek-OCR-master\\DeepSeek-OCR-hf\\gui_ocr.py'],
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
