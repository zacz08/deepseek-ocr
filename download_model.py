"""
DeepSeek OCR 模型下载工具
用于创建离线部署包
"""

from transformers import AutoModel, AutoTokenizer
import os
from pathlib import Path

def download_model():
    model_name = "deepseek-ai/DeepSeek-OCR"
    save_path = "./models/DeepSeek-OCR"
    
    print("="*60)
    print("DeepSeek OCR 模型下载工具")
    print("="*60)
    print()
    
    # 创建目录
    os.makedirs(save_path, exist_ok=True)
    
    print(f"模型: {model_name}")
    print(f"保存到: {save_path}")
    print()
    
    try:
        # 下载 tokenizer
        print("步骤 1/2: 下载 Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        tokenizer.save_pretrained(save_path)
        print("✓ Tokenizer 下载完成")
        print()
        
        # 下载模型
        print("步骤 2/2: 下载模型 (这可能需要较长时间，约 5-10 GB)...")
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_safetensors=True
        )
        model.save_pretrained(save_path)
        print("✓ 模型下载完成")
        print()
        
        # 验证文件
        print("验证下载的文件...")
        files = list(Path(save_path).glob("*"))
        print(f"共 {len(files)} 个文件:")
        for f in sorted(files):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  - {f.name:40s} ({size_mb:,.1f} MB)")
        
        print()
        print("="*60)
        print("✓ 模型下载成功！")
        print("="*60)
        print()
        print("下一步操作:")
        print("1. 将整个 'models' 文件夹复制到目标电脑")
        print("2. 在 GUI 中，模型路径设置为: ./models/DeepSeek-OCR")
        print("3. 或者直接使用绝对路径")
        print()
        
    except Exception as e:
        print()
        print("="*60)
        print("✗ 下载失败！")
        print("="*60)
        print(f"错误信息: {e}")
        print()
        print("可能的原因:")
        print("1. 网络连接问题")
        print("2. 磁盘空间不足")
        print("3. 权限问题")
        print()
        print("建议:")
        print("- 检查网络连接")
        print("- 确保有足够的磁盘空间 (至少 15GB)")
        print("- 尝试使用代理或镜像站")
        return False
    
    return True


if __name__ == "__main__":
    import sys
    
    success = download_model()
    
    if not success:
        sys.exit(1)
    
    input("\n按任意键退出...")
