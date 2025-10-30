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
  "MeuxSimpleLLMNode": "Meux LLM API调用",
  "MeuxAdvancedImageCrop": "Meux Advanced Image Crop",
}
# NODE_CLASS_MAPPINGS = {
#   "PracticalBatchImageCollector": PracticalBatchImageCollector,
#   "FlexibleBatchImageCollector": FlexibleBatchImageCollector,
# }

# NODE_DISPLAY_NAME_MAPPINGS = {
#   "PracticalBatchImageCollector": "实用批量图片收集器",
#   "FlexibleBatchImageCollector": "灵活批量图片收集器",
# }
# 可选：添加版本和作者信息
__version__ = "1.1.0"
__author__ = "Your Name"

# 调试信息 - 可以帮助确认导入是否成功
print(f"[INFO] 成功加载自定义节点: {list(NODE_CLASS_MAPPINGS.keys())}")

# 正常的comfyui提供一个saveimg节点 可以将传入的图像进行保存 
# 现在我有这样一个需求 有不同的多个工作流产生不同的图,我希望有这么一个节点,可以在最后一个输入完成后 统一保存多张图片
# 生成的图片尺寸可能不一样
# 希望输出的方式和saveimg 一样,这样我调用接口就不用修改其他东西了
# 这个节点可以单独作为输出结束节点
