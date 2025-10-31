# 从 nodes 模块导入节点类
from .nodes.multi_save_image import MultiSaveImage
from .nodes.simple_llm_node import SimpleLLMNode
from .nodes.advanced_image_crop import AdvancedImageCrop


NODE_CLASS_MAPPINGS = {
  "MeuxMultiSaveImage": MultiSaveImage,
  "MeuxSimpleLLMNode": SimpleLLMNode,
  "MeuxAdvancedImageCrop": AdvancedImageCrop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
  "MeuxMultiSaveImage": "Meux Multi Save Image",
  "MeuxSimpleLLMNode": "Meux LLM API Call",
  "MeuxAdvancedImageCrop": "Meux Advanced Image Crop",
}

# 可选：添加版本和作者信息
__version__ = "1.1.0"
__author__ = "fangchangjun"

# 调试信息 - 可以帮助确认导入是否成功
print(f"[INFO] 成功加载自定义节点: {list(NODE_CLASS_MAPPINGS.keys())}")

