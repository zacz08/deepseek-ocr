"""
打包脚本 - 包含离线模型
用于创建完整的离线部署包
"""

import os
import shutil
from pathlib import Path

def create_offline_package():
    print("="*60)
    print("创建离线部署包")
    print("="*60)
    print()
    
    # 检查模型文件是否存在
    model_path = Path("./models/DeepSeek-OCR")
    if not model_path.exists():
        print("❌ 错误: 模型文件不存在！")
        print()
        print("请先运行 download_model.py 下载模型")
        print("或运行: download_model.bat")
        return False
    
    # 检查 EXE 是否存在
    exe_path = Path("./dist/DeepSeekOCR")
    if not exe_path.exists():
        print("❌ 错误: 打包的程序不存在！")
        print()
        print("请先运行 build_exe.py 打包程序")
        return False
    
    print("✓ 找到模型文件")
    print("✓ 找到打包程序")
    print()
    
    # 创建离线包目录
    output_dir = Path("./dist/DeepSeekOCR_Offline")
    if output_dir.exists():
        print(f"删除旧的离线包: {output_dir}")
        shutil.rmtree(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("正在复制文件...")
    print()
    
    # 复制程序文件
    print("1/3 复制程序文件...")
    shutil.copytree(exe_path, output_dir / "DeepSeekOCR", dirs_exist_ok=True)
    
    # 复制模型文件
    print("2/3 复制模型文件 (这可能需要几分钟)...")
    models_dest = output_dir / "models"
    models_dest.mkdir(exist_ok=True)
    shutil.copytree(model_path, models_dest / "DeepSeek-OCR", dirs_exist_ok=True)
    
    # 复制文档
    print("3/3 复制文档和说明...")
    docs = [
        "用户快速入门.txt",
        "GUI使用说明.md",
        "离线部署指南.md"
    ]
    
    for doc in docs:
        if Path(doc).exists():
            shutil.copy(doc, output_dir)
    
    # 创建启动脚本
    startup_script = output_dir / "启动DeepSeekOCR.bat"
    with open(startup_script, 'w', encoding='utf-8') as f:
        f.write("""@echo off
cd /d "%~dp0"
start "" "DeepSeekOCR\\DeepSeekOCR.exe"
""")
    
    # 创建使用说明
    readme = output_dir / "使用说明.txt"
    with open(readme, 'w', encoding='utf-8') as f:
        f.write("""=================================================
DeepSeek OCR 离线版使用说明
=================================================

本软件包为离线版本，包含完整的 AI 模型，
无需联网即可使用。

快速开始：
1. 双击 "启动DeepSeekOCR.bat" 启动程序
2. 点击 "加载模型" 按钮
3. 在模型路径中输入: ./models/DeepSeek-OCR
4. 等待模型加载完成
5. 选择图片开始识别

注意事项：
- 模型文件夹（models）必须与程序在同一目录
- 首次加载模型需要 1-2 分钟
- 建议系统内存 8GB 以上
- 推荐使用 NVIDIA GPU 加速

文件说明：
- DeepSeekOCR/     程序文件夹
- models/           模型文件夹（重要，不可删除）
- 启动DeepSeekOCR.bat  快速启动脚本
- 使用说明.txt     本文件
- 用户快速入门.txt 详细使用教程

技术支持：
如有问题，请查看 GUI使用说明.md 或联系技术支持。

=================================================
""")
    
    # 计算总大小
    total_size = 0
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            total_size += Path(root, file).stat().st_size
    
    total_size_gb = total_size / (1024 ** 3)
    
    print()
    print("="*60)
    print("✓ 离线包创建成功！")
    print("="*60)
    print()
    print(f"输出位置: {output_dir.absolute()}")
    print(f"总大小: {total_size_gb:.2f} GB")
    print()
    print("文件结构:")
    print("  DeepSeekOCR_Offline/")
    print("  ├── DeepSeekOCR/          # 程序文件")
    print("  ├── models/                # 模型文件")
    print("  ├── 启动DeepSeekOCR.bat   # 启动脚本")
    print("  ├── 使用说明.txt           # 使用说明")
    print("  └── 其他文档...")
    print()
    print("下一步:")
    print("1. 将整个 DeepSeekOCR_Offline 文件夹")
    print("2. 压缩或复制到 U 盘")
    print("3. 在目标电脑上解压")
    print("4. 双击 '启动DeepSeekOCR.bat' 即可使用")
    print()
    
    return True


if __name__ == "__main__":
    import sys
    
    success = create_offline_package()
    
    if not success:
        input("\n按任意键退出...")
        sys.exit(1)
    
    input("\n按任意键退出...")
