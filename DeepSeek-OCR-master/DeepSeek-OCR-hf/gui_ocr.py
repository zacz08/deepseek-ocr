import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import threading
import os
import sys
from PIL import Image, ImageTk
import torch
from transformers import AutoModel, AutoTokenizer


class DeepSeekOCRGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("DeepSeek OCR GUI")
        self.root.geometry("1000x700")
        
        # 模型相关变量
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self.processing = False
        
        # 创建界面
        self.create_widgets()
        
    def create_widgets(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置行列权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # 标题
        title_label = ttk.Label(main_frame, text="DeepSeek OCR 图像识别工具", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=10)
        
        # 模型配置区域
        model_frame = ttk.LabelFrame(main_frame, text="模型配置", padding="10")
        model_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(model_frame, text="模型名称:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.model_name_var = tk.StringVar(value="deepseek-ai/DeepSeek-OCR")
        model_entry = ttk.Entry(model_frame, textvariable=self.model_name_var, width=40)
        model_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        
        self.load_model_btn = ttk.Button(model_frame, text="加载模型", command=self.load_model)
        self.load_model_btn.grid(row=0, column=2, padx=5)
        
        self.model_status_label = ttk.Label(model_frame, text="状态: 未加载", foreground="red")
        self.model_status_label.grid(row=0, column=3, padx=5)
        
        model_frame.columnconfigure(1, weight=1)
        
        # 输入配置区域
        input_frame = ttk.LabelFrame(main_frame, text="输入配置", padding="10")
        input_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(input_frame, text="图像文件:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.image_path_var = tk.StringVar()
        image_entry = ttk.Entry(input_frame, textvariable=self.image_path_var, width=50)
        image_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        
        ttk.Button(input_frame, text="选择图像", command=self.select_image).grid(row=0, column=2, padx=5)
        
        ttk.Label(input_frame, text="输出目录:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.output_path_var = tk.StringVar(value="./output")
        output_entry = ttk.Entry(input_frame, textvariable=self.output_path_var, width=50)
        output_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5)
        
        ttk.Button(input_frame, text="选择目录", command=self.select_output_dir).grid(row=1, column=2, padx=5)
        
        ttk.Label(input_frame, text="提示词:").grid(row=2, column=0, sticky=tk.W, padx=5)
        self.prompt_var = tk.StringVar(value="<image>\n<|grounding|>Convert the document to markdown. ")
        prompt_entry = ttk.Entry(input_frame, textvariable=self.prompt_var, width=50)
        prompt_entry.grid(row=2, column=1, columnspan=2, sticky=(tk.W, tk.E), padx=5)
        
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
        
        # 第二行参数
        self.crop_mode_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(param_frame, text="Crop Mode", variable=self.crop_mode_var).grid(
            row=1, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        self.test_compress_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(param_frame, text="Test Compress", variable=self.test_compress_var).grid(
            row=1, column=2, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        self.save_results_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(param_frame, text="Save Results", variable=self.save_results_var).grid(
            row=2, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        # 预设配置
        ttk.Label(param_frame, text="预设配置:").grid(row=2, column=2, sticky=tk.W, padx=5)
        preset_combo = ttk.Combobox(param_frame, values=["Gundam", "Tiny", "Small", "Base", "Large"], 
                                    width=10, state="readonly")
        preset_combo.grid(row=2, column=3, sticky=tk.W, padx=5)
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
        
        # 进度条
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # 日志输出区域
        log_frame = ttk.LabelFrame(main_frame, text="运行日志", padding="10")
        log_frame.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        main_frame.rowconfigure(6, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
    def log(self, message):
        """添加日志信息"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
        
    def clear_log(self):
        """清空日志"""
        self.log_text.delete(1.0, tk.END)
        
    def select_image(self):
        """选择图像文件"""
        filename = filedialog.askopenfilename(
            title="选择图像文件",
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("所有文件", "*.*")]
        )
        if filename:
            self.image_path_var.set(filename)
            self.log(f"已选择图像: {filename}")
            
    def select_output_dir(self):
        """选择输出目录"""
        directory = filedialog.askdirectory(title="选择输出目录")
        if directory:
            self.output_path_var.set(directory)
            self.log(f"输出目录: {directory}")
            
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
            self.log(f"已应用预设: {preset} (base_size={base_size}, image_size={image_size}, crop_mode={crop_mode})")
            
    def load_model(self):
        """加载模型"""
        if self.model_loaded:
            self.log("模型已经加载")
            return
            
        def load_thread():
            try:
                self.load_model_btn.config(state=tk.DISABLED)
                self.log("开始加载模型...")
                self.log(f"模型名称: {self.model_name_var.get()}")
                
                # 设置CUDA设备
                os.environ["CUDA_VISIBLE_DEVICES"] = '0'
                
                # 加载tokenizer
                self.log("加载 tokenizer...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name_var.get(), 
                    trust_remote_code=True
                )
                
                # 加载模型
                self.log("加载模型 (这可能需要几分钟)...")
                self.model = AutoModel.from_pretrained(
                    self.model_name_var.get(),
                    trust_remote_code=True, 
                    use_safetensors=True
                )
                
                # 检查CUDA是否可用
                if torch.cuda.is_available():
                    self.log("CUDA 可用，将模型加载到 GPU...")
                    self.model = self.model.eval().cuda().to(torch.bfloat16)
                else:
                    self.log("CUDA 不可用，使用 CPU (速度会较慢)...")
                    self.model = self.model.eval()
                
                self.model_loaded = True
                self.model_status_label.config(text="状态: 已加载", foreground="green")
                self.run_btn.config(state=tk.NORMAL)
                self.log("模型加载成功！")
                
            except Exception as e:
                self.log(f"加载模型失败: {str(e)}")
                self.model_status_label.config(text="状态: 加载失败", foreground="red")
                messagebox.showerror("错误", f"加载模型失败:\n{str(e)}")
            finally:
                self.load_model_btn.config(state=tk.NORMAL)
        
        # 在新线程中加载模型
        thread = threading.Thread(target=load_thread, daemon=True)
        thread.start()
        
    def run_ocr(self):
        """运行OCR识别"""
        if not self.model_loaded:
            messagebox.showwarning("警告", "请先加载模型！")
            return
            
        if not self.image_path_var.get():
            messagebox.showwarning("警告", "请先选择图像文件！")
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
                
                self.log("="*50)
                self.log("开始OCR识别...")
                self.log(f"图像文件: {self.image_path_var.get()}")
                self.log(f"输出目录: {self.output_path_var.get()}")
                self.log(f"提示词: {self.prompt_var.get()}")
                self.log(f"参数: base_size={self.base_size_var.get()}, image_size={self.image_size_var.get()}, crop_mode={self.crop_mode_var.get()}")
                
                # 执行OCR
                res = self.model.infer(
                    self.tokenizer, 
                    prompt=self.prompt_var.get(), 
                    image_file=self.image_path_var.get(), 
                    output_path=self.output_path_var.get(), 
                    base_size=self.base_size_var.get(), 
                    image_size=self.image_size_var.get(), 
                    crop_mode=self.crop_mode_var.get(), 
                    save_results=self.save_results_var.get(), 
                    test_compress=self.test_compress_var.get()
                )
                
                self.log("="*50)
                self.log("识别结果:")
                self.log(res)
                self.log("="*50)
                self.log("OCR识别完成！")
                
                if self.save_results_var.get():
                    self.log(f"结果已保存到: {self.output_path_var.get()}")
                
                messagebox.showinfo("完成", "OCR识别完成！")
                
            except Exception as e:
                self.log(f"识别失败: {str(e)}")
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
        # 注意: 实际停止推理比较复杂，这里只是更新UI状态
        self.log("请求停止识别...")
        messagebox.showinfo("提示", "已发送停止请求，但可能需要等待当前任务完成")


def main():
    root = tk.Tk()
    app = DeepSeekOCRGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
