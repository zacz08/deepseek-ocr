import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import threading
import os
import sys
import re
import subprocess
import logging
from pathlib import Path
from PIL import Image, ImageTk, ImageDraw, ImageFont
import torch

# ============ 重要：必须在导入 vLLM 之前设置环境变量 ============
os.environ['VLLM_USE_V1'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# CUDA 版本特定配置
if torch.version.cuda == '11.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"

# ============ 日志系统配置 ============
# 创建日志记录器（用于后台详细日志）
logger = logging.getLogger("DeepSeekOCR")
logger.setLevel(logging.DEBUG)

# 文件日志处理器（详细技术日志）
log_file = "deepseek_ocr_debug.log"
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# 控制台日志处理器（系统初始化时打印）
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(levelname)s: %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# ============ 硬件检测函数 ============
def detect_gpu_driver():
    """
    检测 NVIDIA GPU 驱动（用于 EXE 分发）
    注意：比检查 CUDA 版本更靠谱，因为驱动和 GPU 计算能力关系更直接
    """
    gpu_driver_info = {
        "driver_installed": False,
        "driver_version": "未知",
    }
    
    # 尝试查询 NVIDIA 驱动
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            driver_version = result.stdout.strip()
            if driver_version:
                gpu_driver_info["driver_installed"] = True
                gpu_driver_info["driver_version"] = driver_version
    except Exception as e:
        pass  # nvidia-smi 不可用
    
    return gpu_driver_info

def detect_hardware():
    """
    检测系统硬件配置（针对 EXE 离线分发优化）
    
    对于 EXE 分发：
    - 检查 GPU 硬件是否存在
    - 检查驱动而不是 CUDA 版本（驱动更稳定）
    - 不强制依赖特定 CUDA 版本
    """
    gpu_driver_info = detect_gpu_driver()
    
    hardware_info = {
        # GPU 硬件
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_name": "",
        "gpu_memory_gb": "0",
        
        # 驱动信息（用于 EXE 分发）
        "driver_available": gpu_driver_info["driver_installed"],
        "driver_version": gpu_driver_info["driver_version"],
        
        # CUDA/cuDNN 信息（可选，仅供参考）
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
    }
    
    if hardware_info["cuda_available"] and hardware_info["gpu_count"] > 0:
        try:
            hardware_info["gpu_name"] = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            hardware_info["gpu_memory_gb"] = f"{gpu_memory:.2f}"
        except:
            hardware_info["gpu_memory_gb"] = "未知"
    
    return hardware_info

# 检测硬件信息
HARDWARE_INFO = detect_hardware()

# ============ 推理引擎选择逻辑（针对 EXE 离线分发） ============
# 
# 策略说明：
# 1. 对于 CPU-Only 版本：使用 HuggingFace Transformers（推荐分发）
# 2. 对于 GPU 版本：检查驱动而不是 CUDA 版本
# 3. 仅在有 GPU 驱动时才尝试加载 vLLM

VLLM_AVAILABLE = False
HF_AVAILABLE = False

# ============ 决策逻辑 ============
# 默认使用 Transformers（CPU 友好）
# 仅当有 GPU 驱动时才考虑 vLLM

SHOULD_TRY_VLLM = HARDWARE_INFO["driver_available"] and HARDWARE_INFO["cuda_available"]

print("\n" + "="*70)
print("推理引擎初始化")
print("="*70)

# 尝试导入 vLLM（仅在有 GPU 时）
if SHOULD_TRY_VLLM:
    print(f"✓ 检测到 GPU 驱动: v{HARDWARE_INFO['driver_version']}")
    print("尝试加载 vLLM（高性能推理）...")
    
    try:
        # 环境变量已在文件开头设置
        from vllm import LLM, SamplingParams
        from vllm.model_executor.models.registry import ModelRegistry
        from deepseek_ocr import DeepseekOCRForCausalLM
        from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
        from process.image_process import DeepseekOCRProcessor
        
        ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)
        VLLM_AVAILABLE = True
        print("✓ vLLM 加载成功")
        
    except Exception as e:
        print(f"⚠ vLLM 加载失败: {e}")
        print("降级到 HuggingFace Transformers...")
else:
    if HARDWARE_INFO["cuda_available"]:
        print("✓ 检测到 GPU，但未检测到 NVIDIA 驱动")
        print("  建议：安装 NVIDIA 驱动以获得更好的性能")
        print("  使用 CPU 模式...")
    else:
        print("✓ 使用 CPU 模式（未检测到 GPU）")

# 如果 vLLM 不可用，使用 HuggingFace Transformers
if not VLLM_AVAILABLE:
    try:
        from transformers import AutoModel, AutoTokenizer
        HF_AVAILABLE = True
        print("✓ 使用 HuggingFace Transformers")
        if HARDWARE_INFO["cuda_available"]:
            print("  模式: GPU 加速")
        else:
            print("  模式: CPU 推理")
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        print("\n请确保已安装依赖包：")
        print("  pip install transformers")
        input("\n按任意键退出...")
        sys.exit(1)

if not VLLM_AVAILABLE and not HF_AVAILABLE:
    print("✗ 无法导入任何推理引擎！")
    input("\n按任意键退出...")
    sys.exit(1)

print("="*70 + "\n")


import fitz  # PyMuPDF
import io


class DeepSeekOCRVLLMGUI:
    def __init__(self, root):
        self.root = root
        engine_name = "vLLM" if VLLM_AVAILABLE else "HF Transformers"
        self.root.title(f"DeepSeek OCR GUI ({engine_name})")
        self.root.geometry("1200x800")
        
        # 硬件信息
        self.hardware_info = HARDWARE_INFO
        
        # 模型相关变量
        self.llm = None
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self.processing = False
        self.use_vllm = VLLM_AVAILABLE
        self.use_gpu = HARDWARE_INFO["cuda_available"]
        
        if VLLM_AVAILABLE:
            self.processor = DeepseekOCRProcessor()
        
        # 创建界面
        self.create_widgets()
        
        # 显示硬件和引擎信息
        self.show_hardware_info()
    
    def show_hardware_info(self):
        """显示硬件和推理引擎信息（针对 EXE 分发优化）"""
        self.log("\n" + "="*70, "info")
        self.log("💻 系统硬件检测", "info")
        self.log("="*70, "info")
        
        # GPU 驱动信息（最重要）
        self.log("\n【GPU 加速状态】", "info")
        if self.hardware_info["driver_available"]:
            self.log(f"✅ NVIDIA 驱动已安装", "info")
            self.log(f"   驱动版本: {self.hardware_info['driver_version']}", "info", show_in_gui=False)
        else:
            self.log("❌ NVIDIA 驱动未检测到", "info")
            if self.hardware_info["cuda_available"]:
                self.log("   💡 你的电脑有 GPU，但需要安装驱动来加速运算", "info")
                self.log("   📍 建议: 访问 https://www.nvidia.com/drivers 下载驱动", "info")
            else:
                self.log("   ℹ️  你的电脑没有 NVIDIA GPU，将使用 CPU（速度较慢）", "info")
        
        # GPU 硬件信息
        self.log("\n【硬件配置】", "info")
        if self.hardware_info["cuda_available"] and self.hardware_info["gpu_count"] > 0:
            self.log(f"✅ GPU: {self.hardware_info['gpu_name']}", "info")
            self.log(f"   显存: {self.hardware_info['gpu_memory_gb']} GB", "info", show_in_gui=False)
        else:
            self.log("ℹ️  使用 CPU 处理（建议配置: 8GB RAM 以上）", "info")
        
        # 推理引擎信息
        self.log("\n【推理引擎】", "info")
        if VLLM_AVAILABLE:
            self.log("⚡ vLLM（快速推理）", "info")
            self.log("   推理速度: 非常快 (~100+ tokens/秒)", "info")
            self.log("   适用场景: 日常使用，大批量处理", "info")
        elif HF_AVAILABLE:
            if self.use_gpu:
                self.log("🚀 GPU 加速推理", "info")
                self.log("   推理速度: 中等 (~20-50 tokens/秒)", "info")
            else:
                self.log("🐢 CPU 推理（较慢）", "info")
                self.log("   推理速度: 较慢 (~2-5 tokens/秒)", "info")
                self.log("   ⏱️  一张 A4 纸可能需要 1-3 分钟", "info")
        
        # 详细技术信息只输出到日志文件
        if self.hardware_info["cuda_available"]:
            logger.info(f"CUDA 版本: {self.hardware_info['cuda_version']}")
            if self.hardware_info['cudnn_version']:
                logger.info(f"cuDNN 版本: {self.hardware_info['cudnn_version']}")
        
        self.log("="*70 + "\n", "info")
        self.log("✨ 准备就绪！请点击\"加载模型\"开始使用\n", "info")
        
    def create_widgets(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置行列权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # 标题
        engine_name = "vLLM" if VLLM_AVAILABLE else "HF Transformers"
        title_label = ttk.Label(main_frame, text=f"DeepSeek OCR 图像/PDF识别工具 ({engine_name})", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=10)
        
        # 模型配置区域
        model_frame = ttk.LabelFrame(main_frame, text="模型配置", padding="10")
        model_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(model_frame, text="模型路径:").grid(row=0, column=0, sticky=tk.W, padx=5)
        
        # 检测本地模型
        local_model_path = "./models/DeepSeek-OCR"
        if os.path.exists(local_model_path):
            default_model = local_model_path
            self.local_model_detected = True
        else:
            default_model = "deepseek-ai/DeepSeek-OCR"
            self.local_model_detected = False
        
        self.model_path_var = tk.StringVar(value=default_model)
        model_entry = ttk.Entry(model_frame, textvariable=self.model_path_var, width=50)
        model_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        
        self.load_model_btn = ttk.Button(model_frame, text="加载模型", command=self.load_model)
        self.load_model_btn.grid(row=0, column=2, padx=5)
        
        ttk.Button(model_frame, text="📁", command=self.browse_model_path, width=3).grid(row=0, column=3, padx=2)
        
        self.model_status_label = ttk.Label(model_frame, text="状态: 未加载", foreground="red")
        self.model_status_label.grid(row=0, column=3, padx=5)
        
        # vLLM参数
        ttk.Label(model_frame, text="GPU内存利用率:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.gpu_util_var = tk.DoubleVar(value=0.75)
        gpu_util_spin = ttk.Spinbox(model_frame, from_=0.1, to=0.95, increment=0.05, 
                                     textvariable=self.gpu_util_var, width=10)
        gpu_util_spin.grid(row=1, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(model_frame, text="最大并发:").grid(row=1, column=2, sticky=tk.W, padx=5)
        self.max_concurrency_var = tk.IntVar(value=100)
        concurrency_spin = ttk.Spinbox(model_frame, from_=1, to=200, 
                                        textvariable=self.max_concurrency_var, width=10)
        concurrency_spin.grid(row=1, column=3, sticky=tk.W, padx=5)
        
        model_frame.columnconfigure(1, weight=1)
        
        # 输入配置区域
        input_frame = ttk.LabelFrame(main_frame, text="输入配置", padding="10")
        input_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # 文件类型选择
        ttk.Label(input_frame, text="输入类型:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.input_type_var = tk.StringVar(value="image")
        type_frame = ttk.Frame(input_frame)
        type_frame.grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Radiobutton(type_frame, text="图片", variable=self.input_type_var, 
                       value="image", command=self.on_input_type_change).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(type_frame, text="PDF", variable=self.input_type_var, 
                       value="pdf", command=self.on_input_type_change).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(type_frame, text="批量图片", variable=self.input_type_var, 
                       value="batch", command=self.on_input_type_change).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(input_frame, text="输入文件:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.input_path_var = tk.StringVar()
        input_entry = ttk.Entry(input_frame, textvariable=self.input_path_var, width=60)
        input_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5)
        
        self.select_input_btn = ttk.Button(input_frame, text="选择图片", command=self.select_input)
        self.select_input_btn.grid(row=1, column=2, padx=5)
        
        ttk.Label(input_frame, text="输出目录:").grid(row=2, column=0, sticky=tk.W, padx=5)
        self.output_path_var = tk.StringVar(value="./output")
        output_entry = ttk.Entry(input_frame, textvariable=self.output_path_var, width=60)
        output_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=5)
        
        ttk.Button(input_frame, text="选择目录", command=self.select_output_dir).grid(row=2, column=2, padx=5)
        
        ttk.Label(input_frame, text="提示词:").grid(row=3, column=0, sticky=tk.W, padx=5)
        self.prompt_var = tk.StringVar(value="<image>\n<|grounding|>Convert the document to markdown.")
        prompt_combo = ttk.Combobox(input_frame, textvariable=self.prompt_var, width=58)
        prompt_combo['values'] = (
            "<image>\n<|grounding|>Convert the document to markdown.",
            "<image>\nFree OCR.",
            "<image>\n<|grounding|>OCR this image.",
            "<image>\nParse the figure.",
            "<image>\nDescribe this image in detail."
        )
        prompt_combo.grid(row=3, column=1, columnspan=2, sticky=(tk.W, tk.E), padx=5)
        
        # 提示词帮助按钮
        ttk.Button(
            input_frame, 
            text="❓ 帮助", 
            command=self.show_prompt_help,
            width=6
        ).grid(row=3, column=3, sticky=tk.W, padx=2)
        
        input_frame.columnconfigure(1, weight=1)
        
        # 参数配置区域
        param_frame = ttk.LabelFrame(main_frame, text="OCR参数", padding="10")
        param_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # 第一行参数
        ttk.Label(param_frame, text="base_size:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.base_size_var = tk.IntVar(value=1024)
        base_size_combo = ttk.Combobox(param_frame, textvariable=self.base_size_var, 
                                       values=[512, 640, 1024, 1280], width=10, state="readonly")
        base_size_combo.grid(row=0, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(param_frame, text="image_size:").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.image_size_var = tk.IntVar(value=640)
        image_size_combo = ttk.Combobox(param_frame, textvariable=self.image_size_var, 
                                        values=[512, 640, 1024, 1280], width=10, state="readonly")
        image_size_combo.grid(row=0, column=3, sticky=tk.W, padx=5)
        
        ttk.Label(param_frame, text="PDF DPI:").grid(row=0, column=4, sticky=tk.W, padx=5)
        self.pdf_dpi_var = tk.IntVar(value=144)
        pdf_dpi_combo = ttk.Combobox(param_frame, textvariable=self.pdf_dpi_var, 
                                     values=[72, 96, 144, 200, 300], width=10, state="readonly")
        pdf_dpi_combo.grid(row=0, column=5, sticky=tk.W, padx=5)
        
        # 第二行参数
        self.crop_mode_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(param_frame, text="Crop Mode", variable=self.crop_mode_var).grid(
            row=1, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        self.save_results_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(param_frame, text="保存结果", variable=self.save_results_var).grid(
            row=1, column=2, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        self.save_visualization_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(param_frame, text="保存可视化", variable=self.save_visualization_var).grid(
            row=1, column=4, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        # 预设配置
        ttk.Label(param_frame, text="预设配置:").grid(row=2, column=0, sticky=tk.W, padx=5)
        preset_combo = ttk.Combobox(param_frame, values=["Gundam", "Tiny", "Small", "Base", "Large"], 
                                    width=10, state="readonly")
        preset_combo.grid(row=2, column=1, sticky=tk.W, padx=5)
        preset_combo.bind("<<ComboboxSelected>>", self.apply_preset)
        
        # 执行按钮
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=3, pady=10)
        
        self.run_btn = ttk.Button(button_frame, text="开始识别", command=self.run_ocr, 
                                  state=tk.DISABLED, style="Accent.TButton")
        self.run_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(button_frame, text="停止", command=self.stop_ocr, 
                                   state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="清空日志", command=self.clear_log).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="打开输出目录", command=self.open_output_dir).pack(side=tk.LEFT, padx=5)
        
        # 进度条
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # 日志输出区域
        log_frame = ttk.LabelFrame(main_frame, text="运行日志", padding="10")
        log_frame.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        main_frame.rowconfigure(6, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=20, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # 在日志创建后输出本地模型检测信息
        if hasattr(self, 'local_model_detected') and self.local_model_detected:
            self.log(f"✓ 检测到本地模型: ./models/DeepSeek-OCR")
        
    def log(self, message, level="info", show_in_gui=True):
        """
        双层日志系统：
        - GUI 界面显示用户友好的信息
        - 后台详细日志记录技术信息
        
        参数：
            message: 日志消息
            level: 日志级别 ("debug", "info", "warning", "error")
            show_in_gui: 是否在 GUI 中显示（False 时只记录到文件）
        """
        # 后台详细日志（技术人员查看）
        if level == "debug":
            logger.debug(message)
        elif level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)
        else:
            logger.info(message)
        
        # GUI 显示用户友好的信息
        if show_in_gui:
            self.log_text.insert(tk.END, f"{message}\n")
            self.log_text.see(tk.END)
            self.root.update_idletasks()
        
    def clear_log(self):
        """清空日志"""
        self.log_text.delete(1.0, tk.END)
    
    def browse_model_path(self):
        """浏览选择本地模型目录"""
        directory = filedialog.askdirectory(
            title="选择模型文件夹",
            initialdir="./models" if os.path.exists("./models") else "."
        )
        if directory:
            self.model_path_var.set(directory)
            self.log(f"已选择模型路径: {directory}")
            # 检查模型文件是否存在
            config_file = os.path.join(directory, "config.json")
            if os.path.exists(config_file):
                self.log("✓ 模型文件验证成功")
            else:
                self.log("⚠ 警告: 未找到 config.json，这可能不是有效的模型目录")
        
    def on_input_type_change(self):
        """输入类型改变时更新按钮文本"""
        input_type = self.input_type_var.get()
        if input_type == "image":
            self.select_input_btn.config(text="选择图片")
        elif input_type == "pdf":
            self.select_input_btn.config(text="选择PDF")
        else:  # batch
            self.select_input_btn.config(text="选择文件夹")
            
    def select_input(self):
        """选择输入文件"""
        input_type = self.input_type_var.get()
        if input_type == "image":
            filename = filedialog.askopenfilename(
                title="选择图像文件",
                filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("所有文件", "*.*")]
            )
            if filename:
                self.input_path_var.set(filename)
                self.log(f"已选择图像: {filename}")
        elif input_type == "pdf":
            filename = filedialog.askopenfilename(
                title="选择PDF文件",
                filetypes=[("PDF文件", "*.pdf"), ("所有文件", "*.*")]
            )
            if filename:
                self.input_path_var.set(filename)
                self.log(f"已选择PDF: {filename}")
        else:  # batch
            directory = filedialog.askdirectory(title="选择图片文件夹")
            if directory:
                self.input_path_var.set(directory)
                self.log(f"已选择文件夹: {directory}")
            
    def select_output_dir(self):
        """选择输出目录"""
        directory = filedialog.askdirectory(title="选择输出目录")
        if directory:
            self.output_path_var.set(directory)
            self.log(f"输出目录: {directory}")
    def open_output_dir(self):
        """打开输出目录（跨平台）"""
        output_path = self.output_path_var.get()
        if os.path.exists(output_path):
            try:
                if sys.platform == 'win32':
                    os.startfile(output_path)
                elif sys.platform == 'darwin':  # macOS
                    subprocess.Popen(['open', output_path])
                else:  # Linux
                    subprocess.Popen(['xdg-open', output_path])
            except Exception as e:
                self.log(f"无法打开目录: {e}")
                messagebox.showinfo("提示", f"输出目录: {output_path}")
        else:
            messagebox.showwarning("警告", f"输出目录不存在: {output_path}")
            messagebox.showwarning("警告", f"输出目录不存在: {output_path}")
            
    def apply_preset(self, event):
        """应用预设配置"""
        preset = event.widget.get()
        presets = {
            "Tiny": (512, 512, False),
            "Small": (640, 640, False),
            "Base": (1024, 1024, False),
            "Large": (1280, 1280, False),
            "Gundam": (1024, 640, True)
        }
        if preset in presets:
            base_size, image_size, crop_mode = presets[preset]
            self.base_size_var.set(base_size)
            self.image_size_var.set(image_size)
            self.crop_mode_var.set(crop_mode)
    
    def show_prompt_help(self):
        """显示提示词帮助信息"""
        help_text = """
【提示词 (Prompt) 说明】

提示词是告诉 AI 你想要什么的指令。以下是常用提示词：

1️⃣ 转换为 Markdown 格式（推荐）
   <image>
   <|grounding|>Convert the document to markdown.
   👉 用途：将文档转为 Markdown 格式
   
2️⃣ 自由 OCR 识别
   <image>
   Free OCR.
   👉 用途：提取文本内容
   
3️⃣ 详细 OCR 识别
   <image>
   <|grounding|>OCR this image.
   👉 用途：精确识别每个字符
   
4️⃣ 解析图表
   <image>
   Parse the figure.
   👉 用途：识别图表和数据
   
5️⃣ 描述图像
   <image>
   Describe this image in detail.
   👉 用途：生成详细的图像描述

📌 提示：
• <image> 是固定的，代表你要识别的图片
• 选择合适的提示词能获得更好的识别效果
• 默认推荐"Convert to markdown"
"""
        messagebox.showinfo("提示词帮助", help_text)
    
    def load_model(self):
        """加载模型（vLLM或HF）"""
        if self.model_loaded:
            self.log("模型已经加载")
            return
            
        def load_thread():
            try:
                self.load_model_btn.config(state=tk.DISABLED)
                model_path = self.model_path_var.get()
                
                # 检查模型路径是否存在（本地模型）
                if os.path.exists(model_path):
                    self.log(f"📂 使用本地模型: {model_path}", "info")
                    self.log("✅ 离线模式（无需网络连接）", "info")
                else:
                    self.log(f"🔄 模型路径: {model_path}", "info")
                    self.log("⚠️  在线模式（需要网络下载，可能较慢）", "info")
                
                # 详细日志输出到文件
                logger.debug(f"模型加载路径: {model_path}")
                logger.debug(f"CUDA 设备: {os.environ.get('CUDA_VISIBLE_DEVICES', '0')}")
                
                # 设置CUDA设备
                os.environ["CUDA_VISIBLE_DEVICES"] = '0'
                
                # 设置离线模式（如果是本地路径）
                if os.path.exists(model_path):
                    os.environ["HF_HUB_OFFLINE"] = "1"
                    os.environ["TRANSFORMERS_OFFLINE"] = "1"
                
                if self.use_vllm:
                    # 使用 vLLM
                    self.log("\n⏳ 正在加载模型（这可能需要几分钟）...", "info")
                    self.log("💡 初始化推理引擎...", "info", show_in_gui=False)
                    
                    # 检查 GPU 是否可用
                    if not self.use_gpu:
                        self.log("⚠️  警告: GPU 驱动未检测到", "warning")
                        self.log("       vLLM 可能无法运行，建议安装 NVIDIA 驱动", "warning")
                    
                    self.llm = LLM(
                        model=model_path,
                        hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
                        block_size=256,
                        enforce_eager=False,
                        trust_remote_code=True, 
                        max_model_len=8192,
                        swap_space=0,
                        max_num_seqs=self.max_concurrency_var.get(),
                        tensor_parallel_size=1,
                        gpu_memory_utilization=self.gpu_util_var.get(),
                        dtype="bfloat16",  # 明确指定数据类型，避免 float16/bfloat16 混用
                    )
                    
                    self.model_status_label.config(text="✅ 已加载 (vLLM)", foreground="green")
                    self.log("✅ 模型加载成功！", "info")
                    self.log(f"⚡ 推理模式: vLLM（高速）", "info")
                    
                    # 详细配置信息到日志文件
                    logger.info(f"GPU内存利用率: {self.gpu_util_var.get()}")
                    logger.info(f"最大并发数: {self.max_concurrency_var.get()}")
                    
                else:
                    # 使用 HuggingFace Transformers
                    self.log("\n⏳ 正在加载模型（这可能需要几分钟）...", "info")
                    self.log("💡 加载 tokenizer...", "info", show_in_gui=False)
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_path, 
                        trust_remote_code=True,
                        local_files_only=os.path.exists(model_path)
                    )
                    
                    self.log("💡 加载模型权重...", "info", show_in_gui=False)
                    self.model = AutoModel.from_pretrained(
                        model_path,
                        trust_remote_code=True, 
                        use_safetensors=True,
                        local_files_only=os.path.exists(model_path)
                    )
                    
                    # 检查CUDA是否可用
                    if self.use_gpu:
                        self.log("💡 正在加载到 GPU...", "info", show_in_gui=False)
                        try:
                            self.model = self.model.eval().cuda().to(torch.bfloat16)
                            self.log("✅ 模型加载成功！", "info")
                            self.log("⚡ 推理模式: GPU 加速", "info")
                            self.log(f"   GPU: {self.hardware_info['gpu_name']}", "info", show_in_gui=False)
                            logger.info("模型加载到 GPU，使用 bfloat16 精度")
                        except RuntimeError as gpu_err:
                            self.log(f"⚠️  GPU 加载失败，自动降级到 CPU", "warning")
                            self.log(f"     错误: {str(gpu_err)[:50]}...", "debug", show_in_gui=False)
                            self.model = self.model.eval()
                            self.use_gpu = False
                    else:
                        self.log("✅ 模型加载成功！", "info")
                        self.log("🐢 推理模式: CPU（较慢）", "info")
                        self.log("   提示: 建议安装 NVIDIA 驱动以获得更快速度", "info")
                        self.model = self.model.eval()
                    
                    self.model_status_label.config(text="✅ 已加载 (HF)", foreground="green")
                
                self.model_loaded = True
                self.run_btn.config(state=tk.NORMAL)
                self.log("\n✨ 现在可以开始处理文件了！", "info")
                
            except Exception as e:
                error_msg = str(e)
                self.log(f"\n❌ 加载失败", "error")
                
                # 检查是否是网络连接问题
                if "Connection" in error_msg or "timeout" in error_msg or "huggingface.co" in error_msg:
                    self.log("📡 原因: 网络连接失败", "error")
                    self.log("\n💡 解决方案:", "info")
                    self.log("1. 使用本地模型（推荐）:", "info")
                    self.log("   • 先在有网络的电脑下载模型", "info")
                    self.log("   • 将模型文件复制到 ./models 文件夹", "info")
                    self.log("   • 重新加载", "info")
                    self.log("2. 检查网络连接和防火墙", "info")
                    self.log("3. 如需在线下载，请稍后重试", "info")
                else:
                    self.log(f"📋 错误信息: {error_msg}", "error")
                    logger.error(f"模型加载失败: {error_msg}", exc_info=True)
                    self.log("💡 请查看程序目录下的 deepseek_ocr_debug.log 获取详细信息", "info")
                
                self.model_status_label.config(text="❌ 加载失败", foreground="red")
            finally:
                self.load_model_btn.config(state=tk.NORMAL)
        
        # 在新线程中启动模型加载
        thread = threading.Thread(target=load_thread, daemon=True)
        thread.start()
    
    def process_single_image(self, image_path, output_path):
        """处理单张图片"""
        self.log(f"加载图片: {image_path}")
        image = Image.open(image_path).convert('RGB')
        
        if self.use_vllm:
            # vLLM 处理
            self.log("处理图像特征...")
            image_features = self.processor.tokenize_with_images(
                images=[image], 
                bos=True, 
                eos=True, 
                cropping=self.crop_mode_var.get()
            )
            
            # 准备采样参数
            logits_processors = [NoRepeatNGramLogitsProcessor(
                ngram_size=30, 
                window_size=90, 
                whitelist_token_ids={128821, 128822}
            )]
            
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=8192,
                logits_processors=logits_processors,
                skip_special_tokens=False,
            )
            
            # 生成结果
            self.log("开始OCR识别...")
            request = {
                "prompt": self.prompt_var.get(),
                "multi_modal_data": {"image": image_features}
            }
            
            outputs = self.llm.generate(request, sampling_params)
            result = outputs[0].outputs[0].text
        else:
            # HuggingFace 处理
            self.log("开始OCR识别...")
            result = self.model.infer(
                self.tokenizer,
                prompt=self.prompt_var.get(),
                image_file=image_path,
                output_path=output_path,
                base_size=self.base_size_var.get(),
                image_size=self.image_size_var.get(),
                crop_mode=self.crop_mode_var.get(),
                save_results=False,  # 我们自己保存
                test_compress=False
            )
        
        # 保存结果
        if self.save_results_var.get():
            os.makedirs(output_path, exist_ok=True)
            os.makedirs(f'{output_path}/images', exist_ok=True)
            
            # 保存原始结果
            with open(f'{output_path}/result.md', 'w', encoding='utf-8') as f:
                f.write(result)
            
            self.log(f"结果已保存到: {output_path}/result.md")
        
        return result
    
    def pdf_to_images(self, pdf_path, dpi=144):
        """
        将PDF转换为高质量图像
        使用与 run_dpsk_ocr_pdf.py 相同的实现
        """
        import fitz
        import io
        
        self.log(f"打开PDF文件: {pdf_path}")
        
        images = []
        pdf_document = fitz.open(pdf_path)
        
        self.log(f"PDF共 {pdf_document.page_count} 页，开始转换 (DPI={dpi})...")
        
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            
            # 渲染为高质量图像
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            Image.MAX_IMAGE_PIXELS = None
            
            # 转换为PNG格式
            img_data = pixmap.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            images.append(img)
            
            if (page_num + 1) % 10 == 0:
                self.log(f"  已转换 {page_num + 1}/{pdf_document.page_count} 页")
        
        pdf_document.close()
        self.log(f"✓ PDF转换完成，共 {len(images)} 页")
        
        return images
        
    def process_pdf(self, pdf_path, output_path):
        """处理PDF文件"""
        # 转换PDF为图像
        images = self.pdf_to_images(pdf_path, self.pdf_dpi_var.get())
        
        os.makedirs(output_path, exist_ok=True)
        
        # 处理每一页
        all_results = []
        for page_num, image in enumerate(images, 1):
            self.log(f"\n{'='*50}")
            self.log(f"识别第 {page_num}/{len(images)} 页")
            self.log(f"{'='*50}")
            
            if self.use_vllm:
                # vLLM 处理
                image_features = self.processor.tokenize_with_images(
                    images=[image], 
                    bos=True, 
                    eos=True, 
                    cropping=self.crop_mode_var.get()
                )
                
                logits_processors = [NoRepeatNGramLogitsProcessor(
                    ngram_size=20, 
                    window_size=50, 
                    whitelist_token_ids={128821, 128822}
                )]
                
                sampling_params = SamplingParams(
                    temperature=0.0,
                    max_tokens=8192,
                    logits_processors=logits_processors,
                    skip_special_tokens=False,
                )
                
                request = {
                    "prompt": self.prompt_var.get(),
                    "multi_modal_data": {"image": image_features}
                }
                
                outputs = self.llm.generate(request, sampling_params)
                result = outputs[0].outputs[0].text
            else:
                # HuggingFace 处理 - 保存临时文件
                temp_image_path = f"{output_path}/temp_page_{page_num}.jpg"
                image.save(temp_image_path)
                
                result = self.model.infer(
                    self.tokenizer,
                    prompt=self.prompt_var.get(),
                    image_file=temp_image_path,
                    output_path=output_path,
                    base_size=self.base_size_var.get(),
                    image_size=self.image_size_var.get(),
                    crop_mode=self.crop_mode_var.get(),
                    save_results=False,
                    test_compress=False
                )
                
                # 删除临时文件
                try:
                    os.remove(temp_image_path)
                except:
                    pass
            
            all_results.append(f"# Page {page_num}\n\n{result}\n\n")
            
            # 保存单页结果
            if self.save_results_var.get():
                page_output = f"{output_path}/page_{page_num:03d}.md"
                with open(page_output, 'w', encoding='utf-8') as f:
                    f.write(result)
                self.log(f"第 {page_num} 页已保存")
        
        # 保存合并结果
        if self.save_results_var.get():
            combined_output = f"{output_path}/all_pages.md"
            with open(combined_output, 'w', encoding='utf-8') as f:
                f.writelines(all_results)
            self.log(f"\n所有页面已合并保存到: {combined_output}")
        
        return '\n'.join(all_results)
        
    def process_batch_images(self, folder_path, output_path):
        """批量处理图片文件夹"""
        # 获取所有图片文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(folder_path).glob(f'*{ext}'))
            image_files.extend(Path(folder_path).glob(f'*{ext.upper()}'))
        
        image_files = sorted(list(set(image_files)))
        total = len(image_files)
        
        if total == 0:
            self.log("未找到图片文件！")
            return
        
        self.log(f"找到 {total} 个图片文件")
        os.makedirs(output_path, exist_ok=True)
        
        # 处理每张图片
        for idx, image_file in enumerate(image_files, 1):
            self.log(f"\n{'='*50}")
            self.log(f"处理 {idx}/{total}: {image_file.name}")
            self.log(f"{'='*50}")
            
            try:
                result = self.process_single_image(str(image_file), output_path)
                
                # 保存结果
                if self.save_results_var.get():
                    output_file = f"{output_path}/{image_file.stem}.md"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(result)
                    self.log(f"结果已保存: {output_file}")
                    
            except Exception as e:
                self.log(f"处理失败: {str(e)}")
                continue
        
        self.log(f"\n批量处理完成！共处理 {total} 个文件")
        
    def run_ocr(self):
        """运行OCR识别"""
        if not self.model_loaded:
            messagebox.showwarning("警告", "请先加载模型！")
            return
            
        if not self.input_path_var.get():
            messagebox.showwarning("警告", "请先选择输入文件！")
            return
            
        if self.processing:
            messagebox.showwarning("警告", "正在处理中，请等待...")
            return
            
        def ocr_thread():
            try:
                self.processing = True
                self.run_btn.config(state=tk.DISABLED)
                self.stop_btn.config(state=tk.NORMAL)
                self.progress.start()
                
                self.log("="*70)
                self.log(f"开始OCR识别 - {self.input_type_var.get().upper()}模式")
                self.log("="*70)
                self.log(f"输入: {self.input_path_var.get()}")
                self.log(f"输出: {self.output_path_var.get()}")
                self.log(f"提示词: {self.prompt_var.get()}")
                
                input_type = self.input_type_var.get()
                input_path = self.input_path_var.get()
                output_path = self.output_path_var.get()
                
                if input_type == "image":
                    result = self.process_single_image(input_path, output_path)
                elif input_type == "pdf":
                    result = self.process_pdf(input_path, output_path)
                else:  # batch
                    self.process_batch_images(input_path, output_path)
                    result = "批量处理完成"
                
                self.log("\n" + "="*70)
                self.log("识别完成！")
                self.log("="*70)
                
                messagebox.showinfo("完成", "OCR识别完成！")
                
            except Exception as e:
                self.log(f"\n识别失败: {str(e)}")
                import traceback
                self.log(traceback.format_exc())
                messagebox.showerror("错误", f"OCR识别失败:\n{str(e)}")
            finally:
                self.processing = False
                self.run_btn.config(state=tk.NORMAL)
                self.stop_btn.config(state=tk.DISABLED)
                self.progress.stop()
        
        # 在新线程中运行OCR
        thread = threading.Thread(target=ocr_thread, daemon=True)
        thread.start()
        
    def stop_ocr(self):
        """停止OCR识别"""
        self.log("请求停止识别...")
        messagebox.showinfo("提示", "已发送停止请求，但可能需要等待当前任务完成")


def main():
    root = tk.Tk()
    app = DeepSeekOCRVLLMGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
