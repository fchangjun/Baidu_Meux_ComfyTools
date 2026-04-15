# 从 nodes 模块导入节点类
from .nodes.multi_save_image import MultiSaveImage
from .nodes.simple_llm_node import SimpleLLMNode
from .nodes.advanced_image_crop import AdvancedImageCrop
from .nodes.image_loader import ImageLoader
from .nodes.mask_loader import MaskLoader
from .nodes.text_area_input import TextAreaInput
from .nodes.smart_empty_latent import SmartEmptyLatent
from .nodes.size_preset_safe import SizePresetSafe
from .nodes.smart_exact_resize import SmartExactResize 
from .nodes.outpaint_size_preset_safe import OutpaintSizePresetSafe 
from .nodes.rmbg_birefnet import MeuxRMBG
from .nodes.realesrgan_upscale import MeuxRealESRGANUpscale
from .nodes.mask_fill_holes import MeuxMaskFillHoles
from .nodes.mask_blur_plus import MeuxMaskBlurPlus
from .nodes.seed_node import MeuxSeed
from .nodes.artistic_text_preview import MeuxArtisticTextPreview

NODE_CLASS_MAPPINGS = {
  "MeuxImageLoader": ImageLoader,
  "MeuxMultiSaveImage": MultiSaveImage,
  "MeuxSimpleLLMNode": SimpleLLMNode,
  "MeuxMaskLoader": MaskLoader,
  "MeuxAdvancedImageCrop": AdvancedImageCrop,
  "MeuxTextAreaInput": TextAreaInput,
  "MeuxSmartEmptyLatent": SmartEmptyLatent,
  "MeuxSizePresetSafe": SizePresetSafe,
  "MeuxSmartExactResize": SmartExactResize,
  "MeuxOutpaintSizePresetSafe": OutpaintSizePresetSafe,
  "MeuxRMBG": MeuxRMBG,
  "MeuxRealESRGANUpscale": MeuxRealESRGANUpscale,
  "MeuxMaskFillHoles": MeuxMaskFillHoles,
  "MeuxMaskBlurPlus": MeuxMaskBlurPlus,
  "MeuxSeed": MeuxSeed,
  "MeuxArtisticTextPreview": MeuxArtisticTextPreview,
  "Mask Fill Holes": MeuxMaskFillHoles,
  "MaskBlur+": MeuxMaskBlurPlus,
  "Seed": MeuxSeed,
}

NODE_DISPLAY_NAME_MAPPINGS = {
  "MeuxImageLoader": "Meux Image Loader",
  "MeuxMultiSaveImage": "Meux Multi Save Image",
  "MeuxSimpleLLMNode": "Meux LLM API Call",
  "MeuxMaskLoader": "Meux Mask Loader",
  "MeuxAdvancedImageCrop": "Meux Advanced Image Crop",
  "MeuxTextAreaInput": "Meux Text Area",
  "MeuxSmartEmptyLatent": "Meux Smart Empty Latent",
  "MeuxSizePresetSafe": "Meux Size Preset Node",    
  "MeuxSmartExactResize": "Meux Smart Exact Resize",
  "MeuxOutpaintSizePresetSafe": "Meux Outpaint Size Preset Node",
  "MeuxRMBG": "Meux RMBG (BiRefNet)",
  "MeuxRealESRGANUpscale": "Meux ESRGAN Upscale",
  "MeuxMaskFillHoles": "Meux Mask Fill Holes",
  "MeuxMaskBlurPlus": "Meux Mask Blur+",
  "MeuxSeed": "Meux Seed",
  "MeuxArtisticTextPreview": "Meux Artistic Text Preview",
  "Mask Fill Holes": "Mask Fill Holes",
  "MaskBlur+": "MaskBlur+",
  "Seed": "Seed",

}

# 可选：添加版本和作者信息
__version__ = "1.7.1"
__author__ = "fangchangjun"

# 调试信息 - 可以帮助确认导入是否成功
print(f"[INFO] 成功加载自定义节点: {list(NODE_CLASS_MAPPINGS.keys())}")
