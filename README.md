# Baidu Meux ComfyTools

[English](#english) | [中文](#中文)

---

## English

### Overview

Baidu Meux ComfyTools is a collection of custom nodes for ComfyUI used by the Baidu Meux asset platform.  
Current version: **1.7.1**

Main nodes:

- `MeuxImageLoader`: load an image from `ComfyUI/input` or an HTTP/HTTPS URL.
- `MeuxMaskLoader`: load a mask from `ComfyUI/input` or an HTTP/HTTPS URL and align it to a reference image.
- `MeuxMaskFillHoles`: fill enclosed holes in a mask.
- `MeuxMaskBlurPlus`: blur or feather a mask.
- `MeuxSeed`: workflow-compatible seed passthrough node.
- `MeuxMultiSaveImage`: save up to sixteen image batches with optional resizing.
- `MeuxAdvancedImageCrop`: crop images by pixels or percentage with optional alignment.
- `MeuxSimpleLLMNode`: call a chat-completions style LLM API inside a workflow.
- `MeuxTextAreaInput`: multi-line text input helper.
- `MeuxArtisticTextPreview`: advanced transparent text preview with fill, outline, spacing, shadow, glow, and mask output.
- `MeuxSmartEmptyLatent`: generate a safe-sized empty latent tensor.
- `MeuxSizePresetSafe`: compute aligned generation sizes.
- `MeuxOutpaintSizePresetSafe`: compute safe per-side outpaint expansion sizes.
- `MeuxSmartExactResize`: crop or pad to an exact target size.
- `MeuxRMBG`: BiRefNet-based background removal.
- `MeuxRealESRGANUpscale`: local RealESRGAN upscaling.

Workflow-compatible aliases:

- `Mask Fill Holes`
- `MaskBlur+`
- `Seed`

### Changelog

- **v1.7.1**
  - Updated URL persistence so downloaded files are stored under `ComfyUI/input/meux_downloads`.
  - Added `text_align` and `vertical_align` controls to `MeuxArtisticTextPreview`.
- **v1.7.0**
  - Added `MeuxArtisticTextPreview` for transparent text preview rendering with fill, outline, spacing, mask output, and preset-based effects.
  - Added a project-level `fonts/` directory and made it the first font lookup location for the preview node.
  - Added node-specific parameter documentation for `MeuxArtisticTextPreview`.
- **v1.6.0**
  - Added `MeuxMaskLoader` as a dedicated mask-loading node.
  - Added workflow-compatible mask utility nodes: `MeuxMaskFillHoles`, `MeuxMaskBlurPlus`, `MeuxSeed`.
  - Registered compatibility aliases: `Mask Fill Holes`, `MaskBlur+`, `Seed`.
  - Added `scipy` dependency for hole-filling behavior aligned with upstream workflows.
- **v1.5.0**
  - Added `MeuxRMBG` for local/remote BiRefNet background removal.
  - Added `MeuxRealESRGANUpscale` for local RealESRGAN upscaling.
- **v1.4.0**
  - Added `MeuxOutpaintSizePresetSafe`.
  - Capped size-related inputs at 4096.
- **v1.3.0**
  - Added `MeuxSmartEmptyLatent`, `MeuxSizePresetSafe`, and `MeuxSmartExactResize`.
- **v1.2.0**
  - Added `MeuxImageLoader` with URL downloading support.
- **v1.1.0**
  - Restructured the package into modular files under `nodes/`.
- **v1.0.0**
  - Initial release with `MeuxMultiSaveImage`.

### Installation / Update

1. Clone into `ComfyUI/custom_nodes/`:

   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/fchangjun/Baidu_Meux_ComfyTools.git
   ```

2. Update when needed:

   ```bash
   cd ComfyUI/custom_nodes/Baidu_Meux_ComfyTools
   git pull
   ```

3. Install dependencies:

   ```bash
   cd ComfyUI/custom_nodes/Baidu_Meux_ComfyTools
   pip install -r requirements.txt
   ```

4. Restart ComfyUI.

### Quick Usage

#### MeuxImageLoader

- `source_type = local`: select a file from `ComfyUI/input`
- `source_type = url`: download from an HTTP/HTTPS URL
- Downloaded URL files are persisted under `ComfyUI/input/meux_downloads` when `persist_to_input = true`
- Outputs `IMAGE` and the derived mask from alpha or grayscale when present

#### MeuxMaskLoader

- `reference_image` is required so the mask can match the image size
- `source_type = local | url`
- Supports `mask_channel`, `mask_invert`, and `resize_to_reference`
- Outputs native ComfyUI-style `MASK`

Recommended pairing:

1. `MeuxImageLoader.image` -> downstream image pipeline
2. `MeuxImageLoader.image` -> `MeuxMaskLoader.reference_image`
3. `MeuxMaskLoader.mask` -> downstream mask pipeline

#### Workflow-Compatible Mask Nodes

- `MeuxMaskFillHoles` / `Mask Fill Holes`: fill enclosed regions inside a mask
- `MeuxMaskBlurPlus` / `MaskBlur+`: feather a mask
- `MeuxSeed` / `Seed`: expose a workflow-compatible seed value

#### Other Utility Nodes

- `MeuxMultiSaveImage`: batch image saving
- `MeuxAdvancedImageCrop`: crop by pixels or percentage
- `MeuxSimpleLLMNode`: direct LLM API calls
- `MeuxTextAreaInput`: long prompt input
- `MeuxArtisticTextPreview`: advanced transparent text preview with optional gradient fill, outline, shadow, inner shadow, glow, and mask output
- `MeuxSmartEmptyLatent`: safe latent initialization
- `MeuxSizePresetSafe`: safe size planning
- `MeuxOutpaintSizePresetSafe`: safe outpaint size planning
- `MeuxSmartExactResize`: exact resize with crop/pad logic
- `MeuxRMBG`: local background removal
- `MeuxRealESRGANUpscale`: local image upscaling

#### MeuxArtisticTextPreview

- Outputs a transparent `IMAGE` preview and a matching `MASK`
- Reads fonts from `Baidu_Meux_ComfyTools/fonts` first, then falls back to bundled/system fonts
- Supports `fixed` and `fit` font sizing modes
- Supports `font_name`, `char_spacing`, `bold`, and `italic`
- Supports solid or gradient fill
- Supports solid or gradient outline
- Supports shadow, inner shadow, and outer glow
- Supports transparent or colored preview backgrounds

Recommended first test:

1. Add `MeuxArtisticTextPreview`
2. Set `text = Hello`
3. Set `font_size_mode = fixed`
4. Set `font_size = 128`
5. Set `outline_type = solid`
6. Connect the `image` output to a preview or save node

Parameters:

- Basic
  - `text`: preview text, supports multiple lines
  - `font_name`: font dropdown, reads project `fonts/` first
  - `width` / `height`: output canvas size
  - `font_size_mode`: `fixed` uses the exact font size, `fit` scales text to the canvas
  - `font_size`: base font size or max fitting size
  - `char_spacing`: character spacing in pixels
  - `line_spacing`: line spacing in pixels; `0` keeps the automatic spacing rule, positive values increase spacing, negative values pull lines closer together
  - `bold`: prefer bold font variants, otherwise use a fallback bold draw mode
  - `italic`: prefer italic font variants, otherwise use a fallback italic transform
  - `background_color`: preview background color
  - `background_opacity`: preview background opacity, `0` means fully transparent
  - `padding_percent`: safe area margin inside the canvas

- Fill
  - `fill_type`: `solid`, `gradient`, or `none`
  - `fill_color_1`: solid fill color, or the start color of a gradient
  - `fill_color_2`: end color of a gradient
  - `fill_direction`: fill gradient direction

- Outline
  - `outline_type`: `none`, `solid`, or `gradient`
  - `outline_width`: outline thickness
  - `outline_opacity`: outline alpha
  - `outline_color_1`: solid outline color, or the start color of an outline gradient
  - `outline_color_2`: end color of an outline gradient
  - `outline_direction`: outline gradient direction

- Effects
  - `effect_preset`: one-click effect preset
  - `effect_strength`: preset intensity multiplier
  - `shadow_mode`: `preset` uses preset shadow settings, `custom` ignores preset shadow placement/blur and uses manual values below
  - `shadow_color`: shadow color; still works in preset mode
  - `shadow_offset_x`: custom shadow horizontal offset
  - `shadow_offset_y`: custom shadow vertical offset
  - `shadow_opacity`: custom shadow alpha
  - `shadow_blur`: custom shadow blur radius; `0` means a solid hard-edged shadow, `> 0` means a blurred shadow
  - Presets:
    - `none`: no extra effect
    - `soft_shadow`: light shadow
    - `heavy_shadow`: heavier shadow
    - `soft_glow`: soft glow
    - `neon_glow`: stronger glow
    - `engraved`: inner-shadow style engraved look
    - `shadow_glow`: combines shadow and glow

- Outputs
  - `image`: rendered preview image
  - `mask`: alpha mask derived from the text result

### Folder Structure

```text
Baidu_Meux_ComfyTools/
├── __init__.py
├── fonts/
│   └── README.md
├── pyproject.toml
├── requirements.txt
└── nodes/
    ├── advanced_image_crop.py
    ├── image_loader.py
    ├── mask_blur_plus.py
    ├── mask_fill_holes.py
    ├── mask_loader.py
    ├── artistic_text_preview.py
    ├── multi_save_image.py
    ├── outpaint_size_preset_safe.py
    ├── realesrgan_upscale.py
    ├── rmbg_birefnet.py
    ├── seed_node.py
    ├── simple_llm_node.py
    ├── size_preset_safe.py
    ├── smart_empty_latent.py
    ├── smart_exact_resize.py
    └── text_area_input.py
```

### Requirements

- ComfyUI
- PyTorch
- Pillow
- NumPy
- Requests
- transformers
- torchvision
- realesrgan
- scipy

### Support

- Repository: [fchangjun/Baidu_Meux_ComfyTools](https://github.com/fchangjun/Baidu_Meux_ComfyTools)
- Issues: [GitHub Issues](https://github.com/fchangjun/Baidu_Meux_ComfyTools/issues)

---

## 中文

### 概述

Baidu Meux ComfyTools 是一组用于 ComfyUI 的自定义节点，服务于百度 Meux 资产平台相关工作流。  
当前版本：**1.7.1**

主要节点：

- `MeuxImageLoader`：从 `ComfyUI/input` 或 HTTP/HTTPS URL 加载图片
- `MeuxMaskLoader`：从 `ComfyUI/input` 或 HTTP/HTTPS URL 加载遮罩，并对齐到参考图尺寸
- `MeuxMaskFillHoles`：填充遮罩内部封闭空洞
- `MeuxMaskBlurPlus`：对遮罩做模糊/羽化
- `MeuxSeed`：兼容工作流的种子透传节点
- `MeuxMultiSaveImage`：一次保存最多 16 组图像，可选统一尺寸
- `MeuxAdvancedImageCrop`：按像素或百分比裁剪图像
- `MeuxSimpleLLMNode`：在工作流中调用聊天式 LLM 接口
- `MeuxTextAreaInput`：多行文本输入辅助节点
- `MeuxArtisticTextPreview`：在透明画布上渲染高级文字预览，支持填充、描边、字距、阴影、内阴影、外发光，并输出遮罩
- `MeuxSmartEmptyLatent`：生成安全尺寸的空白 latent
- `MeuxSizePresetSafe`：计算安全生成尺寸
- `MeuxOutpaintSizePresetSafe`：计算安全外扩尺寸
- `MeuxSmartExactResize`：精确尺寸裁剪/补边
- `MeuxRMBG`：基于 BiRefNet 的抠图节点
- `MeuxRealESRGANUpscale`：本地 RealESRGAN 放大节点

工作流兼容别名：

- `Mask Fill Holes`
- `MaskBlur+`
- `Seed`

### 更新日志

- **v1.7.1**
  - 调整 URL 落盘位置，下载文件现在保存到 `ComfyUI/input/meux_downloads`
  - 为 `MeuxArtisticTextPreview` 新增 `text_align` 和 `vertical_align` 对齐控制
- **v1.7.0**
  - 新增 `MeuxArtisticTextPreview`，用于透明背景文字预览，支持填充、描边、字距、遮罩输出和预设效果。
  - 新增项目级 `fonts/` 目录，并作为预览节点的第一字体读取位置。
  - 补充 `MeuxArtisticTextPreview` 的节点参数说明文档。
- **v1.6.0**
  - 新增独立遮罩加载节点 `MeuxMaskLoader`
  - 新增工作流兼容节点 `MeuxMaskFillHoles`、`MeuxMaskBlurPlus`、`MeuxSeed`
  - 注册兼容别名 `Mask Fill Holes`、`MaskBlur+`、`Seed`
  - 新增 `scipy` 依赖，用于按上游逻辑填充遮罩空洞
- **v1.5.0**
  - 新增 `MeuxRMBG`，支持本地/远端 BiRefNet 抠图
  - 新增 `MeuxRealESRGANUpscale`，支持本地 RealESRGAN 放大
- **v1.4.0**
  - 新增 `MeuxOutpaintSizePresetSafe`
  - 尺寸相关输入上限统一为 4096
- **v1.3.0**
  - 新增 `MeuxSmartEmptyLatent`、`MeuxSizePresetSafe`、`MeuxSmartExactResize`
- **v1.2.0**
  - 新增 `MeuxImageLoader`，支持 URL 下载
- **v1.1.0**
  - 重构为 `nodes/` 模块化结构
- **v1.0.0**
  - 发布 `MeuxMultiSaveImage` 初版

### 安装 / 更新

1. 克隆到 `ComfyUI/custom_nodes/`：

   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/fchangjun/Baidu_Meux_ComfyTools.git
   ```

2. 更新代码：

   ```bash
   cd ComfyUI/custom_nodes/Baidu_Meux_ComfyTools
   git pull
   ```

3. 安装依赖：

   ```bash
   cd ComfyUI/custom_nodes/Baidu_Meux_ComfyTools
   pip install -r requirements.txt
   ```

4. 重启 ComfyUI。

### 快速使用

#### MeuxImageLoader

- `source_type = local`：从 `ComfyUI/input` 选择本地文件
- `source_type = url`：运行时下载 HTTP/HTTPS 图片
- `persist_to_input = true` 时，URL 下载文件会保存到 `ComfyUI/input/meux_downloads`
- 输出 `IMAGE`，以及图片自带 alpha/灰度派生的 `MASK`

#### MeuxMaskLoader

- 必须连接 `reference_image`，用于保证遮罩尺寸和原图一致
- `source_type = local | url`
- 支持 `mask_channel`、`mask_invert`、`resize_to_reference`
- 输出原生兼容的 `MASK`

推荐连法：

1. `MeuxImageLoader.image` 接图像处理链路
2. `MeuxImageLoader.image` 接 `MeuxMaskLoader.reference_image`
3. `MeuxMaskLoader.mask` 接遮罩处理链路

#### 工作流兼容遮罩节点

- `MeuxMaskFillHoles` / `Mask Fill Holes`：填充遮罩内部封闭区域
- `MeuxMaskBlurPlus` / `MaskBlur+`：做遮罩羽化
- `MeuxSeed` / `Seed`：输出兼容工作流的 seed

#### 其他工具节点

- `MeuxMultiSaveImage`：批量保存图片
- `MeuxAdvancedImageCrop`：按像素或百分比裁剪
- `MeuxSimpleLLMNode`：直接调用 LLM API
- `MeuxTextAreaInput`：长文本输入
- `MeuxArtisticTextPreview`：高级透明文字预览节点，支持渐变填充、描边、字距、阴影、内阴影、外发光和遮罩输出
- `MeuxSmartEmptyLatent`：安全 latent 初始化
- `MeuxSizePresetSafe`：安全尺寸规划
- `MeuxOutpaintSizePresetSafe`：安全外扩尺寸规划
- `MeuxSmartExactResize`：精确尺寸裁剪/补边
- `MeuxRMBG`：本地抠图
- `MeuxRealESRGANUpscale`：本地放大

#### MeuxArtisticTextPreview

- 输出透明背景 `IMAGE` 预览图，以及对应的 `MASK`
- 优先读取 `Baidu_Meux_ComfyTools/fonts` 目录中的字体，再回退到内置字体和系统字体
- 支持 `fixed` 和 `fit` 两种字号模式
- 支持 `font_name`、`char_spacing`、`bold`、`italic`
- 支持纯色或渐变填充
- 支持纯色或渐变描边
- 支持阴影、内阴影、外发光
- 支持透明或纯色背景预览

建议先用下面这组参数测试：

1. 添加 `MeuxArtisticTextPreview`
2. 设置 `text = Hello`
3. 设置 `font_size_mode = fixed`
4. 设置 `font_size = 128`
5. 设置 `outline_type = solid`
6. 将 `image` 输出连接到预览或保存节点

参数说明：

- 基础参数
  - `text`：预览文字，支持多行
  - `font_name`：字体下拉，优先读取项目根目录 `fonts/`
  - `width` / `height`：输出画布尺寸
  - `font_size_mode`：`fixed` 使用固定字号，`fit` 自动适配画布
  - `font_size`：基础字号，或自动适配时的最大字号
  - `char_spacing`：字符间距，单位为像素
  - `line_spacing`：行间距，单位为像素；`0` 表示继续使用自动行间距规则，正数拉开行距，负数压缩行距
  - `bold`：优先匹配粗体字体，没有时走伪粗体回退
  - `italic`：优先匹配斜体字体，没有时走伪斜体回退
  - `background_color`：预览背景颜色
  - `background_opacity`：背景透明度，`0` 表示完全透明
  - `padding_percent`：文字与画布边缘的安全边距比例

- 填充参数
  - `fill_type`：`solid` 纯色，`gradient` 渐变，`none` 无填充
  - `fill_color_1`：纯色填充时使用它；渐变时是起始色
  - `fill_color_2`：渐变结束色
  - `fill_direction`：填充渐变方向

- 描边参数
  - `outline_type`：`none` 无描边，`solid` 纯色描边，`gradient` 渐变描边
  - `outline_width`：描边宽度
  - `outline_opacity`：描边透明度
  - `outline_color_1`：纯色描边时使用它；渐变时是起始色
  - `outline_color_2`：渐变结束色
  - `outline_direction`：描边渐变方向

- 效果参数
  - `effect_preset`：一键效果预设
  - `effect_strength`：预设效果强度倍率
  - `shadow_mode`：`preset` 使用预设阴影参数，`custom` 使用下面的自定义阴影参数
  - `shadow_color`：阴影颜色；在预设模式下也会生效
  - `shadow_offset_x`：自定义阴影水平偏移
  - `shadow_offset_y`：自定义阴影垂直偏移
  - `shadow_opacity`：自定义阴影透明度
  - `shadow_blur`：自定义阴影模糊半径；`0` 表示实心硬边阴影，`> 0` 表示模糊阴影
  - 预设说明：
    - `none`：不加额外效果
    - `soft_shadow`：轻阴影
    - `heavy_shadow`：重阴影
    - `soft_glow`：柔和发光
    - `neon_glow`：更强的霓虹发光
    - `engraved`：偏内阴影的雕刻感
    - `shadow_glow`：阴影和发光组合

- 输出
  - `image`：渲染后的预览图
  - `mask`：由文字 alpha 生成的遮罩

### 目录结构

```text
Baidu_Meux_ComfyTools/
├── __init__.py
├── fonts/
│   └── README.md
├── pyproject.toml
├── requirements.txt
└── nodes/
    ├── advanced_image_crop.py
    ├── image_loader.py
    ├── mask_blur_plus.py
    ├── mask_fill_holes.py
    ├── mask_loader.py
    ├── artistic_text_preview.py
    ├── multi_save_image.py
    ├── outpaint_size_preset_safe.py
    ├── realesrgan_upscale.py
    ├── rmbg_birefnet.py
    ├── seed_node.py
    ├── simple_llm_node.py
    ├── size_preset_safe.py
    ├── smart_empty_latent.py
    ├── smart_exact_resize.py
    └── text_area_input.py
```

### 依赖

- ComfyUI
- PyTorch
- Pillow
- NumPy
- Requests
- transformers
- torchvision
- realesrgan
- scipy

### 支持

- 仓库地址：[fchangjun/Baidu_Meux_ComfyTools](https://github.com/fchangjun/Baidu_Meux_ComfyTools)
- 问题反馈：[GitHub Issues](https://github.com/fchangjun/Baidu_Meux_ComfyTools/issues)
