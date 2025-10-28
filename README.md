# ComfyUI Multi Save Toolkit

[English](#english) | [中文](#中文)

---

## English

### Overview

ComfyUI Multi Save Toolkit is a collection of custom nodes that streamline common workflow chores in ComfyUI.  
Current version: **1.1.0**

- `MultiSaveImage`: save up to sixteen image batches with optional resizing.
- `AdvancedImageCrop`: crop images by pixels or percentage with optional grid alignment.
- `SimpleLLMNode`: call an external chat-completions style LLM API directly inside a workflow.

The package now uses a modular `nodes/` directory so each node is easy to maintain and extend.

### Changelog

- **v1.1.0**
  - Restructured package into modular node files under `nodes/`.
  - Added `AdvancedImageCrop` and `SimpleLLMNode` registrations to the public export.
  - Updated documentation with usage guides for every node.
- **v1.0.0**
  - Initial release with the `MultiSaveImage` node.

### Installation / Update

1. Clone or download into your ComfyUI custom node folder:

   ```bash
   cd ComfyUI/custom_nodes/
   git clone https://github.com/yourusername/comfyui-multi-save-image.git
   ```

2. For updates, pull the latest changes:

   ```bash
   cd ComfyUI/custom_nodes/comfyui-multi-save-image
   git pull
   ```

3. Restart ComfyUI. The nodes appear under the `image`, `Image/Transform`, and `LLM` categories.

### Usage Tutorial

#### MultiSaveImage

1. Connect one or more `IMAGE` tensors to `images_1` … `images_16`. Batches are split automatically.
2. Set `filename_prefix` (unsafe characters are cleaned automatically).
3. Optional toggles:
   - `save_individually`: keep the sequential counter for each input image.
   - `resize_to_same`: rescale every image to `target_width` × `target_height` using Lanczos.
4. Run the node. Images are written to the ComfyUI output directory as  
   `prefix_{slotIndex:03d}_{counter:05d}.png`, and UI metadata is returned for gallery preview.

#### AdvancedImageCrop

1. Feed an `IMAGE` tensor into `image`.
2. Choose `measurement` mode:
   - `Pixels`: specify absolute crop margins.
   - `Percentage`: margins are interpreted relative to current width/height.
3. Optionally snap the crop window by setting `align_to` to `8` or `16`.
4. Execute to obtain the cropped tensor (with console logging of the crop result).

#### SimpleLLMNode

1. Provide your API key and endpoint (`api_url` defaults to siliconflow chat completions).
2. Set the `model` identifier supported by your provider.
3. Enter the `user_prompt`; optionally supply a `system_prompt` and sampling parameters (`temperature`, `top_p`, `top_k`, penalties).
4. Run the node. The main output is the assistant message, along with the full JSON payload and token usage.

### Folder Structure

```
Comfyui_MultiSaveImage/
├── __init__.py          # Registers all exposed nodes
└── nodes/
    ├── advanced_image_crop.py
    ├── multi_save_image.py
    └── simple_llm_node.py
```

### Requirements

- ComfyUI
- PyTorch
- Pillow
- NumPy
- Requests (for `SimpleLLMNode`)

### License & Support

Licensed under the MIT License.  
Issues and feature requests: [GitHub Issues](https://github.com/yourusername/comfyui-multi-save-image/issues).

---

## 中文

### 概述

ComfyUI Multi Save Toolkit 是一组帮助简化 ComfyUI 工作流的自定义节点。  
当前版本：**1.1.0**

- `MultiSaveImage`：一次保存最多 16 组图像，支持可选统一尺寸。
- `AdvancedImageCrop`：按像素或百分比裁剪，可选择 8/16 像素对齐。
- `SimpleLLMNode`：在工作流中调用外部 LLM Chat Completion 接口。

项目已改用模块化的 `nodes/` 目录，便于后续维护与扩展。

### 更新日志

- **v1.1.0**
  - 重构包结构至 `nodes/` 子目录。
  - 注册 `AdvancedImageCrop` 与 `SimpleLLMNode` 节点。
  - 文档新增全部节点的使用教程。
- **v1.0.0**
  - 发布 `MultiSaveImage` 节点初版。

### 安装 / 更新

1. 克隆或下载到 ComfyUI 自定义节点目录：

   ```bash
   cd ComfyUI/custom_nodes/
   git clone https://github.com/yourusername/comfyui-multi-save-image.git
   ```

2. 更新时执行：

   ```bash
   cd ComfyUI/custom_nodes/comfyui-multi-save-image
   git pull
   ```

3. 重启 ComfyUI，节点将出现在 `image`、`Image/Transform` 与 `LLM` 分类下。

### 使用教程

#### MultiSaveImage

1. 将一张或多张 `IMAGE` 张量连接到 `images_1` … `images_16`，批次会自动拆分。
2. 设置 `filename_prefix`（系统会自动清理危险字符）。
3. 可选项：
   - `save_individually`：为每张图单独编号。
   - `resize_to_same`：按 `target_width` × `target_height` 使用 Lanczos 算法统一尺寸。
4. 运行后，图像保存到 ComfyUI 输出目录，命名格式  
   `前缀_{输入序号:03d}_{计数:05d}.png`，同时返回 UI 预览信息。

#### AdvancedImageCrop

1. 将 `IMAGE` 张量接入 `image`。
2. 选择 `measurement` 模式：
   - `Pixels`：输入像素裁剪边距。
   - `Percentage`：边距按当前宽/高的百分比计算。
3. 如需像素对齐，将 `align_to` 设为 `8` 或 `16`。
4. 运行节点即可得到裁剪后的图像，并在控制台查看裁剪日志。

#### SimpleLLMNode

1. 输入可用的 API Key 和接口地址（默认指向 siliconflow chat completions）。
2. 设置服务商支持的 `model` 名称。
3. 填写 `user_prompt`；根据需要添加 `system_prompt` 和采样参数（`temperature`、`top_p`、`top_k`、惩罚项）。
4. 执行节点将返回：
   - 主输出：模型回复文本；
   - 附加输出：完整 JSON 响应与消耗的 token 数。

### 目录结构

```
Comfyui_MultiSaveImage/
├── __init__.py          # 节点入口注册
└── nodes/
    ├── advanced_image_crop.py
    ├── multi_save_image.py
    └── simple_llm_node.py
```

### 依赖

- ComfyUI
- PyTorch
- Pillow
- NumPy
- Requests（供 `SimpleLLMNode` 调用 HTTP 接口）

### 许可证与支持

采用 MIT 许可证。  
问题反馈与功能建议： [GitHub Issues](https://github.com/yourusername/comfyui-multi-save-image/issues)。
