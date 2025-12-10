#!/usr/bin/env python3
"""
Markdown 转换工具
将 DeepSeek-OCR 生成的 Markdown 文件转换为 PDF/Word/HTML
"""

import sys
import os
import subprocess
import argparse

def check_pandoc():
    """检查 pandoc 是否安装"""
    try:
        subprocess.run(['pandoc', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def remove_grounding_tags(content):
    """移除 grounding 标注，保留纯 Markdown"""
    import re
    # 移除 <|ref|>...<|/ref|><|det|>[[...]]<|/det|> 标签
    pattern = r'<\|ref\|>.*?<\|/ref\|><\|det\|>\[\[.*?\]\]<\|/det\>\n?'
    cleaned = re.sub(pattern, '', content)
    return cleaned

def convert_to_pdf(input_file, output_file, clean=True):
    """转换为 PDF"""
    # 读取并可选清理内容
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if clean:
        content = remove_grounding_tags(content)
        temp_file = input_file + '.temp.md'
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(content)
        input_to_use = temp_file
    else:
        input_to_use = input_file
    
    try:
        # 使用 xelatex 支持中文
        cmd = [
            'pandoc', input_to_use,
            '-o', output_file,
            '--pdf-engine=xelatex',
            '-V', 'CJKmainfont=SimSun',
            '-V', 'geometry:margin=1in'
        ]
        subprocess.run(cmd, check=True)
        print(f"✅ PDF 已生成: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"❌ 转换失败: {e}")
        print("提示: 请确保已安装 texlive-xetex")
    finally:
        if clean and os.path.exists(temp_file):
            os.remove(temp_file)

def convert_to_docx(input_file, output_file, clean=True):
    """转换为 Word"""
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if clean:
        content = remove_grounding_tags(content)
        temp_file = input_file + '.temp.md'
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(content)
        input_to_use = temp_file
    else:
        input_to_use = input_file
    
    try:
        cmd = ['pandoc', input_to_use, '-o', output_file]
        subprocess.run(cmd, check=True)
        print(f"✅ Word 文档已生成: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"❌ 转换失败: {e}")
    finally:
        if clean and os.path.exists(temp_file):
            os.remove(temp_file)

def convert_to_html(input_file, output_file, clean=True):
    """转换为 HTML"""
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if clean:
        content = remove_grounding_tags(content)
        temp_file = input_file + '.temp.md'
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(content)
        input_to_use = temp_file
    else:
        input_to_use = input_file
    
    try:
        cmd = [
            'pandoc', input_to_use,
            '-o', output_file,
            '--standalone',
            '--mathjax'  # 支持数学公式
        ]
        subprocess.run(cmd, check=True)
        print(f"✅ HTML 已生成: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"❌ 转换失败: {e}")
    finally:
        if clean and os.path.exists(temp_file):
            os.remove(temp_file)

def main():
    parser = argparse.ArgumentParser(
        description='将 Markdown 转换为 PDF/Word/HTML',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s result.md -f pdf              # 转换为 PDF
  %(prog)s result.md -f docx             # 转换为 Word
  %(prog)s result.md -f html             # 转换为 HTML
  %(prog)s result.md -f all              # 转换为所有格式
  %(prog)s result.md -f pdf --keep-tags  # 保留 grounding 标签
        """
    )
    
    parser.add_argument('input', help='输入的 Markdown 文件')
    parser.add_argument('-f', '--format', 
                        choices=['pdf', 'docx', 'html', 'all'],
                        default='pdf',
                        help='输出格式 (默认: pdf)')
    parser.add_argument('-o', '--output',
                        help='输出文件名（可选，默认自动生成）')
    parser.add_argument('--keep-tags', action='store_true',
                        help='保留 grounding 标签（默认移除）')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"❌ 文件不存在: {args.input}")
        sys.exit(1)
    
    # 检查 pandoc
    if not check_pandoc():
        print("❌ 未安装 pandoc")
        print("请安装: sudo apt-get install pandoc texlive-xetex")
        sys.exit(1)
    
    base_name = os.path.splitext(args.input)[0]
    clean = not args.keep_tags
    
    # 执行转换
    if args.format == 'pdf' or args.format == 'all':
        output = args.output or f"{base_name}.pdf"
        convert_to_pdf(args.input, output, clean)
    
    if args.format == 'docx' or args.format == 'all':
        output = args.output or f"{base_name}.docx"
        convert_to_docx(args.input, output, clean)
    
    if args.format == 'html' or args.format == 'all':
        output = args.output or f"{base_name}.html"
        convert_to_html(args.input, output, clean)

if __name__ == '__main__':
    main()
