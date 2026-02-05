# Baidu Meux ComfyTools

[English](#english) | [中文](#中文)

---

## English

### Overview

Baidu Meux ComfyTools is a collection of custom nodes that streamline common workflow chores in ComfyUI for the Baidu Meux asset platform.  
Current version: **1.5.0**

- `MeuxMultiSaveImage`: save up to sixteen image batches with optional resizing.
- `MeuxAdvancedImageCrop`: crop images by pixels or percentage with optional grid alignment.
- `MeuxSimpleLLMNode`: call an external chat-completions style LLM API directly inside a workflow.
- `MeuxImageLoader`: drop-in replacement for ComfyUI's Load Image node with optional HTTP/HTTPS downloading and persistence to the input folder.
- `MeuxSmartEmptyLatent`: generate a safe-sized empty latent tensor based on target size alignment.
- `MeuxSizePresetSafe`: compute safe generation size and return size metadata for downstream nodes.
- `MeuxOutpaintSizePresetSafe`: compute safe per-side outpaint expansion aligned to 8/64.
- `MeuxSmartExactResize`: smart crop/pad to exact target size with auto mode and padding options.
- `MeuxRealESRGANUpscale` (displayed as “Meux ESRGAN Upscale”): local RealESRGAN upscale node using weights from `ComfyUI/models/upscale`.

The package now uses a modular `nodes/` directory so each node is easy to maintain and extend.

### Changelog

- **v1.5.0**
  - Added `MeuxRealESRGANUpscale` for local RealESRGAN image upscaling.
- **v1.4.0**
  - Added `MeuxOutpaintSizePresetSafe` for safe outpaint expansion sizes.
  - Capped size inputs at 4096 for size-related nodes.
- **v1.3.0**
  - Added `MeuxSmartEmptyLatent`, `MeuxSizePresetSafe`, and `MeuxSmartExactResize`.
- **v1.2.0**
  - Introduced `MeuxImageLoader`, a drop-in replacement for ComfyUI's Load Image with URL downloading.
- **v1.1.0**
  - Restructured package into modular node files under `nodes/`.
  - Added `MeuxAdvancedImageCrop` and `MeuxSimpleLLMNode` registrations to the public export.
  - Updated documentation with usage guides for every node.
- **v1.0.0**
  - Initial release with the `MeuxMultiSaveImage` node.

### Installation / Update

1. Clone or download into your ComfyUI custom node folder:

   ```bash
   cd ComfyUI/custom_nodes/
   git clone https://github.com/yourusername/Baidu_Meux_ComfyTools.git
   ```

2. For updates, pull the latest changes:

   ```bash
   cd ComfyUI/custom_nodes/Baidu_Meux_ComfyTools
   git pull
   ```

3. Restart ComfyUI. The nodes appear under the `image`, `Image/Transform`, `LLM`, and `utils/size` categories.

### Usage Tutorial

#### MeuxImageLoader

1. Set `source_type`:
   - `local`: behaves exactly like the stock Load Image node; pick any file under `ComfyUI/input`.
   - `url`: paste an HTTP/HTTPS link to download on the fly.
2. Optional controls:
   - `filename_hint`: suggest the persisted filename; falls back to a hash of the URL.
   - `persist_to_input`: keep a copy in `ComfyUI/input` so other nodes can reference it by name.
   - `overwrite_existing`: replace an existing file instead of auto-appending numeric suffixes.
   - `download_timeout`, `max_download_mb`, `verify_ssl`: guardrails for network transfers.
3. Run the node to receive the image tensor plus the derived `MASK` (alpha or grayscale channel when available), making it a drop-in replacement for ComfyUI's default loader.

#### MeuxMultiSaveImage

1. Connect one or more `IMAGE` tensors to `images_1` … `images_16`. Batches are split automatically.
2. Set `filename_prefix` (unsafe characters are cleaned automatically).
3. Optional toggles:
   - `save_individually`: keep the sequential counter for each input image.
   - `resize_to_same`: rescale every image to `target_width` × `target_height` using Lanczos.
4. Run the node. Images are written to the ComfyUI output directory as  
   `prefix_{slotIndex:03d}_{counter:05d}.png`, and UI metadata is returned for gallery preview.

#### MeuxAdvancedImageCrop

1. Feed an `IMAGE` tensor into `image`.
2. Choose `measurement` mode:
   - `Pixels`: specify absolute crop margins.
   - `Percentage`: margins are interpreted relative to current width/height.
3. Optionally snap the crop window by setting `align_to` to `8` or `16`.
4. Execute to obtain the cropped tensor (with console logging of the crop result).

#### MeuxSimpleLLMNode

1. Provide your API key and endpoint (`api_url` defaults to siliconflow chat completions).
2. Set the `model` identifier supported by your provider.
3. Enter the `user_prompt`; optionally supply a `system_prompt` and sampling parameters (`temperature`, `top_p`, `top_k`, penalties).
4. Run the node. The main output is the assistant message, along with the full JSON payload and token usage.

#### MeuxSmartEmptyLatent

1. Set `target_width`, `target_height`, and `batch_size`.
2. Choose `align` (8 or 64). The node rounds up to a safe generation size.
3. Run the node to output an empty `LATENT` plus `gen_width` and `gen_height`.

#### MeuxSizePresetSafe

1. Set `target_width`, `target_height`, and `batch_size`.
2. Choose `align` (8 or 64). The node rounds up to a safe generation size.
3. Run the node to output `gen_width`, `gen_height`, and the input size metadata.

#### MeuxOutpaintSizePresetSafe

1. Set `expand_left`, `expand_right`, `expand_top`, and `expand_bottom`.
2. Choose `align` (8 or 64). Each side rounds up to a safe expansion size.
3. Run the node to output `safe_*` values plus the original `target_*` values.

#### MeuxSmartExactResize

1. Feed an `IMAGE` tensor into `image`.
2. Set `target_width` and `target_height`.
3. Choose `mode`: `auto`, `crop_only`, or `pad_only`.
4. Optional: `anchor` for cropping, `safe_margin_percent` for auto crop safety, and `padding_mode` for padding style.
5. Run the node to output a resized image with exact target dimensions.

#### Meux ESRGAN Upscale

1. Download the model weight `RealESRGAN_x4plus.pth` from the official release:

   ```bash
   wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
   ```

2. Install it under `ComfyUI/models/upscale/`:

   ```bash
   mkdir -p ComfyUI/models/upscale
   mv RealESRGAN_x4plus.pth ComfyUI/models/upscale/
   ```

2. Feed an `IMAGE` tensor into `image`.
3. Choose `scale_mode` (`2x/3x/4x/6x/8x/custom`) and set `custom_scale` when using `custom`.
   Note: the bundled model is a 4x model. Non-4x modes are produced by resizing after 4x inference.
4. Set `model_name` (default `RealESRGAN_x4plus.pth`).
5. Optional: set `model_path` to an absolute path to override the lookup.
6. Optional: enable `free_gpu_after` to release GPU memory after each run (slower but safer for long sessions).
7. Run the node to output the upscaled image tensor.

### Folder Structure

```
Baidu_Meux_ComfyTools/
├── __init__.py          # Registers all exposed nodes
└── nodes/
    ├── image_loader.py
    ├── advanced_image_crop.py
    ├── multi_save_image.py
    ├── simple_llm_node.py
    ├── smart_empty_latent.py
    ├── size_preset_safe.py
    ├── outpaint_size_preset_safe.py
    ├── smart_exact_resize.py
    └── realesrgan_upscale.py
```

### Requirements

- ComfyUI
- PyTorch
- Pillow
- NumPy
- Requests (required by `MeuxSimpleLLMNode` and `MeuxImageLoader`)
- RealESRGAN (required by `MeuxRealESRGANUpscale`)

### License & Support

Licensed under the MIT License.  
Issues and feature requests: [GitHub Issues](https://github.com/yourusername/Baidu_Meux_ComfyTools/issues).

---

## 中文

### 概述

Baidu Meux ComfyTools 是一组面向百度 Meux 资产平台、帮助简化 ComfyUI 工作流的自定义节点。  
当前版本：**1.5.0**

- `MeuxMultiSaveImage`：一次保存最多 16 组图像，支持可选统一尺寸。
- `MeuxAdvancedImageCrop`：按像素或百分比裁剪，可选择 8/16 像素对齐。
- `MeuxSimpleLLMNode`：在工作流中调用外部 LLM Chat Completion 接口。
- `MeuxImageLoader`：兼容本地/URL 两种来源，可选写入 input 目录，完全替换原生 Load Image 节点。
- `MeuxSmartEmptyLatent`：根据目标尺寸对齐规则，生成安全尺寸的空白 latent。
- `MeuxSizePresetSafe`：计算安全生成尺寸并输出尺寸元信息，供下游节点使用。
- `MeuxOutpaintSizePresetSafe`：计算外扩的安全尺寸（按 8/64 对齐）。
- `MeuxSmartExactResize`：智能裁剪/补边到精确尺寸，支持自动模式与多种补边方式。
- `MeuxRealESRGANUpscale`（显示为“Meux ESRGAN Upscale”）：本地 RealESRGAN 放大节点，默认从 `ComfyUI/models/upscale` 读取权重。

项目已改用模块化的 `nodes/` 目录，便于后续维护与扩展。

### 更新日志

- **v1.5.0**
  - 新增 `MeuxRealESRGANUpscale`，用于本地 RealESRGAN 放大。
- **v1.4.0**
  - 新增 `MeuxOutpaintSizePresetSafe`，用于安全外扩尺寸计算。
  - 尺寸相关输入上限统一为 4096。
- **v1.3.0**
  - 新增 `MeuxSmartEmptyLatent`、`MeuxSizePresetSafe`、`MeuxSmartExactResize`。
- **v1.2.0**
  - 新增 `MeuxImageLoader`，支持 URL 下载、可完全替换原生 Load Image。
- **v1.1.0**
  - 重构包结构至 `nodes/` 子目录。
  - 注册 `MeuxAdvancedImageCrop` 与 `MeuxSimpleLLMNode` 节点。
  - 文档新增全部节点的使用教程。
- **v1.0.0**
  - 发布 `MeuxMultiSaveImage` 节点初版。

### 安装 / 更新

1. 克隆或下载到 ComfyUI 自定义节点目录：

   ```bash
   cd ComfyUI/custom_nodes/
   git clone https://github.com/yourusername/Baidu_Meux_ComfyTools.git
   ```

2. 更新时执行：

   ```bash
   cd ComfyUI/custom_nodes/Baidu_Meux_ComfyTools
   git pull
   ```

3. 重启 ComfyUI，节点将出现在 `image`、`Image/Transform`、`LLM` 与 `utils/size` 分类下。

### 使用教程

#### MeuxImageLoader

1. 选择 `source_type`：
   - `local`：与原生 Load Image 完全一致，从 `ComfyUI/input` 选择文件。
   - `url`：粘贴 HTTP/HTTPS 图片地址，节点会在运行时自动下载。
2. 可选参数：
   - `filename_hint`：为持久化的文件提供自定义名称，默认使用 URL 哈希。
   - `persist_to_input`：勾选后会把下载结果写入 `ComfyUI/input` 以供其他节点复用。
   - `overwrite_existing`：允许覆盖同名文件，否则自动追加序号。
   - `download_timeout`、`max_download_mb`、`verify_ssl`：用于限制下载时间、大小与验证方式。
3. 运行后输出图片张量与 `MASK`（若原图包含 Alpha 或灰度通道），可直接替换 ComfyUI 自带的 Load Image 节点。

#### MeuxMultiSaveImage

1. 将一张或多张 `IMAGE` 张量连接到 `images_1` … `images_16`，批次会自动拆分。
2. 设置 `filename_prefix`（系统会自动清理危险字符）。
3. 可选项：
   - `save_individually`：为每张图单独编号。
   - `resize_to_same`：按 `target_width` × `target_height` 使用 Lanczos 算法统一尺寸。
4. 运行后，图像保存到 ComfyUI 输出目录，命名格式  
   `前缀_{输入序号:03d}_{计数:05d}.png`，同时返回 UI 预览信息。

#### MeuxAdvancedImageCrop

1. 将 `IMAGE` 张量接入 `image`。
2. 选择 `measurement` 模式：
   - `Pixels`：输入像素裁剪边距。
   - `Percentage`：边距按当前宽/高的百分比计算。
3. 如需像素对齐，将 `align_to` 设为 `8` 或 `16`。
4. 运行节点即可得到裁剪后的图像，并在控制台查看裁剪日志。

#### MeuxSimpleLLMNode

1. 输入可用的 API Key 和接口地址（默认指向 siliconflow chat completions）。
2. 设置服务商支持的 `model` 名称。
3. 填写 `user_prompt`；根据需要添加 `system_prompt` 和采样参数（`temperature`、`top_p`、`top_k`、惩罚项）。
4. 执行节点将返回：
   - 主输出：模型回复文本；
   - 附加输出：完整 JSON 响应与消耗的 token 数。

#### MeuxSmartEmptyLatent

1. 设置 `target_width`、`target_height` 与 `batch_size`。
2. 选择对齐方式 `align`（8 或 64），节点会向上取整为安全生成尺寸。
3. 运行后输出空白 `LATENT`，以及 `gen_width`、`gen_height`。

#### MeuxSizePresetSafe

1. 设置 `target_width`、`target_height` 与 `batch_size`。
2. 选择对齐方式 `align`（8 或 64），节点会向上取整为安全生成尺寸。
3. 运行后输出 `gen_width`、`gen_height` 以及输入尺寸元信息。

#### MeuxOutpaintSizePresetSafe

1. 设置 `expand_left`、`expand_right`、`expand_top`、`expand_bottom`。
2. 选择 `align`（8 或 64），各方向向上取整为安全外扩尺寸。
3. 运行后输出 `safe_*` 以及原始 `target_*` 数值。

#### MeuxSmartExactResize

1. 将 `IMAGE` 张量接入 `image`。
2. 设置 `target_width` 与 `target_height`。
3. 选择 `mode`：`auto`、`crop_only` 或 `pad_only`。
4. 可选：裁剪锚点 `anchor`、自动裁剪安全边距 `safe_margin_percent`、补边方式 `padding_mode`。
5. 运行后输出精确尺寸的图像张量。

#### Meux ESRGAN Upscale

1. 从官方 Release 下载 `RealESRGAN_x4plus.pth`：

   ```bash
   wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
   ```

2. 安装路径为 `ComfyUI/models/upscale/`：

   ```bash
   mkdir -p ComfyUI/models/upscale
   mv RealESRGAN_x4plus.pth ComfyUI/models/upscale/
   ```

2. 将 `IMAGE` 张量接入 `image`。
3. 选择 `scale_mode`（`2x/3x/4x/6x/8x/custom`），使用 `custom` 时设置 `custom_scale`。
   注意：内置模型为 4x，非 4x 模式是在 4x 推理后再缩放得到。
4. 设置 `model_name`（默认 `RealESRGAN_x4plus.pth`）。
5. 可选：设置 `model_path` 为绝对路径以覆盖默认查找。
6. 可选：开启 `free_gpu_after`，每次运行后释放显存（速度更慢，但适合长时间运行）。
7. 运行后输出放大后的图像张量。

### 目录结构

```
Baidu_Meux_ComfyTools/
├── __init__.py          # 节点入口注册
└── nodes/
    ├── image_loader.py
    ├── advanced_image_crop.py
    ├── multi_save_image.py
    ├── simple_llm_node.py
    ├── smart_empty_latent.py
    ├── size_preset_safe.py
    ├── outpaint_size_preset_safe.py
    ├── smart_exact_resize.py
    └── realesrgan_upscale.py
```

### 依赖

- ComfyUI
- PyTorch
- Pillow
- NumPy
- Requests（供 `MeuxSimpleLLMNode` 与 `MeuxImageLoader` 访问 HTTP 接口）
- RealESRGAN（供 `MeuxRealESRGANUpscale` 使用）

### 许可证与支持

采用 MIT 许可证。  
问题反馈与功能建议： [GitHub Issues](https://github.com/yourusername/Baidu_Meux_ComfyTools/issues)。
