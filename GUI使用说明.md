# DeepSeek OCR GUI 使用说明

## 功能介绍

这是一个基于 tkinter 的图形界面应用程序，用于执行 DeepSeek-OCR 模型的图像文字识别功能。

## 启动方式

### 方式1：使用批处理文件（推荐）
双击 `run_gui.bat` 文件即可启动

### 方式2：命令行启动
```bash
python DeepSeek-OCR-master\DeepSeek-OCR-hf\gui_ocr.py
```

## 使用步骤

### 1. 加载模型
- 首次使用时，点击"加载模型"按钮
- 默认使用模型: `deepseek-ai/DeepSeek-OCR`
- 加载过程可能需要几分钟，请耐心等待
- 加载成功后，状态会显示为"已加载"（绿色）

### 2. 配置输入
- **图像文件**: 点击"选择图像"按钮选择要识别的图片
- **输出目录**: 指定结果保存的目录（默认为 `./output`）
- **提示词**: 根据需要修改提示词
  - 默认: `<image>\n<|grounding|>Convert the document to markdown. `
  - 简单OCR: `<image>\nFree OCR. `

### 3. 设置参数
可以使用预设配置或手动调整：

#### 预设配置
- **Gundam** (推荐): base_size=1024, image_size=640, crop_mode=True
- **Tiny**: base_size=512, image_size=512, crop_mode=False
- **Small**: base_size=640, image_size=640, crop_mode=False
- **Base**: base_size=1024, image_size=1024, crop_mode=False
- **Large**: base_size=1280, image_size=1280, crop_mode=False

#### 手动参数
- **base_size**: 基础图像大小（512/640/1024/1280）
- **image_size**: 处理图像大小（512/640/1024/1280）
- **Crop Mode**: 是否启用裁剪模式
- **Test Compress**: 是否测试压缩
- **Save Results**: 是否保存结果文件

### 4. 开始识别
- 点击"开始识别"按钮
- 等待识别完成
- 查看运行日志区域的输出结果
- 如果启用了"Save Results"，结果会保存到指定的输出目录

## 界面功能说明

### 按钮
- **加载模型**: 加载 DeepSeek-OCR 模型到内存
- **选择图像**: 浏览并选择要识别的图片文件
- **选择目录**: 选择结果保存的目录
- **开始识别**: 开始执行OCR识别
- **停止**: 尝试停止当前识别任务
- **清空日志**: 清空运行日志显示区域

### 日志区域
显示程序运行的详细信息，包括：
- 模型加载进度
- 识别参数配置
- OCR识别结果
- 错误信息（如有）

## 注意事项

1. **首次使用**: 模型会自动从 HuggingFace 下载，需要良好的网络连接
2. **硬件要求**: 
   - 推荐使用 NVIDIA GPU（支持 CUDA）
   - 如果没有 GPU，会自动使用 CPU（速度较慢）
3. **内存要求**: 至少需要 8GB RAM，推荐 16GB 或更多
4. **图像格式**: 支持 jpg, jpeg, png, bmp, tiff 等常见格式

## 故障排除

### 模型加载失败
- 检查网络连接是否正常
- 确认是否有足够的磁盘空间
- 查看日志中的详细错误信息

### 识别速度慢
- 检查是否正确使用了 GPU
- 尝试使用较小的 base_size 和 image_size
- 关闭不必要的其他程序

### CUDA 相关错误
- 确认已安装 PyTorch 的 CUDA 版本
- 检查 CUDA 驱动是否正确安装
- 尝试重新安装 PyTorch

## 技术支持

如遇到问题，请查看日志输出获取详细错误信息。
