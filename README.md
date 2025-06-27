# MultiSaveImage Node for ComfyUI

[English](#english) | [中文](#中文)

## English

### Overview

MultiSaveImage is a custom node for ComfyUI that allows you to save multiple image inputs simultaneously with advanced features like batch processing, resizing, and flexible naming conventions.

### Features

- **Multiple Image Inputs**: Support up to 8 different image inputs in a single node
- **Batch Processing**: Handle multiple images in each input slot
- **Automatic Resizing**: Option to resize all images to the same dimensions
- **Flexible Naming**: Customizable filename prefix with automatic numbering
- **Format Support**: Supports RGB, RGBA, and grayscale images
- **Collision Prevention**: Automatic filename collision detection and resolution

### Installation

1. Clone or download this repository to your ComfyUI custom nodes directory:
 ```bash
 cd ComfyUI/custom_nodes/
 git clone https://github.com/yourusername/comfyui-multi-save-image.git
 ```

2. Restart ComfyUI

3. The MultiSaveImage node will appear in the "image" category

### Usage

#### Input Parameters

**Required:**
- `images_1`: The primary image input (required)
- `filename_prefix`: Base name for saved files (default: "ComfyUI")

**Optional:**
- `images_2` to `images_8`: Additional image inputs
- `save_individually`: Save each image separately (default: False)
- `resize_to_same`: Resize all images to the same dimensions (default: False)
- `target_width`: Target width for resizing (default: 512, range: 64-8192)
- `target_height`: Target height for resizing (default: 512, range: 64-8192)

#### Example Workflow

1. Connect your image sources to the `images_1` through `images_8` inputs
2. Set your desired `filename_prefix`
3. Enable `resize_to_same` if you want all images to have uniform dimensions
4. Set `target_width` and `target_height` if resizing is enabled
5. Execute the node to save all images

### Output

- Images are saved to the ComfyUI output directory
- Filenames follow the pattern: `{prefix}_{sequential_number:03d}_{unique_id:05d}.png`
- The node returns UI information about saved images

### Technical Details

- **Image Format**: PNG with compression level 4
- **Color Spaces**: RGB, RGBA, and grayscale support
- **Tensor Handling**: Automatic conversion between PyTorch tensors and PIL Images
- **Memory Efficient**: Processes images individually to minimize memory usage
- **Resizing Algorithm**: Uses Lanczos interpolation for high-quality resizing

### Requirements

- ComfyUI
- PyTorch
- PIL (Pillow)
- NumPy

### License

This project is licensed under the MIT License - see the LICENSE file for details.

### Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Issues

If you encounter any issues, please report them on the [GitHub Issues](https://github.com/yourusername/comfyui-multi-save-image/issues) page.

---

## 中文

### 概述

MultiSaveImage 是 ComfyUI 的自定义节点，允许您同时保存多个图像输入，具有批处理、调整大小和灵活命名约定等高级功能。

### 功能特性

- **多图像输入**：单个节点支持多达8个不同的图像输入
- **批处理**：每个输入槽可处理多张图像
- **自动调整大小**：可选择将所有图像调整为相同尺寸
- **灵活命名**：可自定义文件名前缀，自动编号
- **格式支持**：支持RGB、RGBA和灰度图像
- **冲突防护**：自动检测和解决文件名冲突

### 安装

1. 将此仓库克隆或下载到您的 ComfyUI 自定义节点目录：
 ```bash
 cd ComfyUI/custom_nodes/
 git clone https://github.com/yourusername/comfyui-multi-save-image.git
 ```

2. 重启 ComfyUI

3. MultiSaveImage 节点将出现在"image"类别中

### 使用方法

#### 输入参数

**必需参数：**
- `images_1`：主要图像输入（必需）
- `filename_prefix`：保存文件的基础名称（默认："ComfyUI"）

**可选参数：**
- `images_2` 到 `images_8`：额外的图像输入
- `save_individually`：单独保存每张图像（默认：False）
- `resize_to_same`：将所有图像调整为相同尺寸（默认：False）
- `target_width`：调整大小的目标宽度（默认：512，范围：64-8192）
- `target_height`：调整大小的目标高度（默认：512，范围：64-8192）

#### 示例工作流

1. 将图像源连接到 `images_1` 到 `images_8` 输入
2. 设置所需的 `filename_prefix`
3. 如果希望所有图像具有统一尺寸，启用 `resize_to_same`
4. 如果启用了调整大小，设置 `target_width` 和 `target_height`
5. 执行节点以保存所有图像

### 输出

- 图像保存到 ComfyUI 输出目录
- 文件名遵循模式：`{前缀}_{序号:03d}_{唯一ID:05d}.png`
- 节点返回已保存图像的UI信息

### 技术细节

- **图像格式**：PNG，压缩级别4
- **色彩空间**：支持RGB、RGBA和灰度
- **张量处理**：PyTorch张量和PIL图像之间的自动转换
- **内存高效**：逐个处理图像以最小化内存使用
- **调整大小算法**：使用Lanczos插值进行高质量调整

### 依赖要求

- ComfyUI
- PyTorch
- PIL (Pillow)
- NumPy

### 许可证

本项目采用 MIT 许可证 - 详情请参阅 LICENSE 文件。

### 贡献

欢迎贡献！请随时提交 Pull Request。

### 问题反馈

如果遇到任何问题，请在 [GitHub Issues](https://github.com/yourusername/comfyui-multi-save-image/issues) 页面报告。
