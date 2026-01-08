# 从 nodes 模块导入节点类
from .nodes.multi_save_image import MultiSaveImage
from .nodes.simple_llm_node import SimpleLLMNode
from .nodes.advanced_image_crop import AdvancedImageCrop
from .nodes.image_loader import ImageLoader
from .nodes.text_area_input import TextAreaInput
from .nodes.smart_empty_latent import SmartEmptyLatent
from .nodes.size_preset_safe import SizePresetSafe

NODE_CLASS_MAPPINGS = {
  "MeuxImageLoader": ImageLoader,
  "MeuxMultiSaveImage": MultiSaveImage,
  "MeuxSimpleLLMNode": SimpleLLMNode,
  "MeuxAdvancedImageCrop": AdvancedImageCrop,
  "MeuxTextAreaInput": TextAreaInput,
  "MeuxSmartEmptyLatent": SmartEmptyLatent,
  "MeuxSizePresetSafe": SizePresetSafe
}

NODE_DISPLAY_NAME_MAPPINGS = {
  "MeuxImageLoader": "Meux Image Loader",
  "MeuxMultiSaveImage": "Meux Multi Save Image",
  "MeuxSimpleLLMNode": "Meux LLM API Call",
  "MeuxAdvancedImageCrop": "Meux Advanced Image Crop",
  "MeuxTextAreaInput": "Meux Text Area",
  "MeuxSmartEmptyLatent": " meux latent size node",
  "MeuxSizePresetSafe": " meux size Preset node"

}

# 可选：添加版本和作者信息
__version__ = "1.2.0"
__author__ = "fangchangjun"

# 调试信息 - 可以帮助确认导入是否成功
print(f"[INFO] 成功加载自定义节点: {list(NODE_CLASS_MAPPINGS.keys())}")
