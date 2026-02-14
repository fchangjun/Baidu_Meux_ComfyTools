# 从 nodes 模块导入节点类
from .nodes.multi_save_image import MultiSaveImage
from .nodes.simple_llm_node import SimpleLLMNode
from .nodes.advanced_image_crop import AdvancedImageCrop
from .nodes.image_loader import ImageLoader
from .nodes.text_area_input import TextAreaInput
from .nodes.smart_empty_latent import SmartEmptyLatent
from .nodes.size_preset_safe import SizePresetSafe
from .nodes.smart_exact_resize import SmartExactResize 
from .nodes.outpaint_size_preset_safe import OutpaintSizePresetSafe 
from .nodes.rmbg_birefnet import MeuxRMBG
from .nodes.realesrgan_upscale import MeuxRealESRGANUpscale

NODE_CLASS_MAPPINGS = {
  "MeuxImageLoader": ImageLoader,
  "MeuxMultiSaveImage": MultiSaveImage,
  "MeuxSimpleLLMNode": SimpleLLMNode,
  "MeuxAdvancedImageCrop": AdvancedImageCrop,
  "MeuxTextAreaInput": TextAreaInput,
  "MeuxSmartEmptyLatent": SmartEmptyLatent,
  "MeuxSizePresetSafe": SizePresetSafe,
  "MeuxSmartExactResize": SmartExactResize,
  "MeuxOutpaintSizePresetSafe": OutpaintSizePresetSafe,
  "MeuxRMBG": MeuxRMBG,
  "MeuxRealESRGANUpscale": MeuxRealESRGANUpscale
}

NODE_DISPLAY_NAME_MAPPINGS = {
  "MeuxImageLoader": "Meux Image Loader",
  "MeuxMultiSaveImage": "Meux Multi Save Image",
  "MeuxSimpleLLMNode": "Meux LLM API Call",
  "MeuxAdvancedImageCrop": "Meux Advanced Image Crop",
  "MeuxTextAreaInput": "Meux Text Area",
  "MeuxSmartEmptyLatent": "Meux Smart Empty Latent",
  "MeuxSizePresetSafe": "Meux Size Preset Node",    
  "MeuxSmartExactResize": "Meux Smart Exact Resize",
  "MeuxOutpaintSizePresetSafe": "Meux Outpaint Size Preset Node",
  "MeuxRMBG": "Meux RMBG (BiRefNet)",
  "MeuxRealESRGANUpscale": "Meux ESRGAN Upscale"

}

# 可选：添加版本和作者信息
__version__ = "1.5.0"
__author__ = "fangchangjun"

# 调试信息 - 可以帮助确认导入是否成功
print(f"[INFO] 成功加载自定义节点: {list(NODE_CLASS_MAPPINGS.keys())}")
