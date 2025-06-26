import os
import json
import numpy as np
from PIL import Image
import torch
import folder_paths
from comfy.utils import common_upscale
import comfy.model_management

class MultiSaveImage:
  def __init__(self):
      self.output_dir = folder_paths.get_output_directory()
      self.type = "output"
      self.prefix_append = ""
      self.compress_level = 4

  @classmethod
  def INPUT_TYPES(s):
      return {
          "required": {
              "images_1": ("IMAGE",),
              "filename_prefix": ("STRING", {"default": "ComfyUI"}),
          },
          "optional": {
              "images_2": ("IMAGE",),
              "images_3": ("IMAGE",),
              "images_4": ("IMAGE",),
              "images_5": ("IMAGE",),
              "images_6": ("IMAGE",),
              "images_7": ("IMAGE",),
              "images_8": ("IMAGE",),
              "save_individually": ("BOOLEAN", {"default": False}),
              "resize_to_same": ("BOOLEAN", {"default": False}),
              "target_width": ("INT", {"default": 512, "min": 64, "max": 8192}),
              "target_height": ("INT", {"default": 512, "min": 64, "max": 8192}),
          }
      }

#   RETURN_TYPES = ("IMAGE",)
#   RETURN_NAMES = ("images",)
  RETURN_TYPES = ()
  RETURN_NAMES = ()
  FUNCTION = "save_images"
  OUTPUT_NODE = True
  CATEGORY = "image"

  def save_images(self, images_1, filename_prefix="ComfyUI", images_2=None, images_3=None, 
                 images_4=None, images_5=None, images_6=None, images_7=None, images_8=None,
                 save_individually=False, resize_to_same=False, target_width=512, target_height=512):
      
      # 收集所有非空的图像输入
      all_images = []
      image_inputs = [images_1, images_2, images_3, images_4, images_5, images_6, images_7, images_8]
      
      for img_batch in image_inputs:
          if img_batch is not None:
              # 如果是批量图像，逐个添加
              if len(img_batch.shape) == 4:  # 批量图像 [B, H, W, C]
                  for i in range(img_batch.shape[0]):
                      all_images.append(img_batch[i])
              else:  # 单张图像 [H, W, C]
                  all_images.append(img_batch)
      
      if not all_images:
          raise ValueError("至少需要提供一个图像输入")
      
      # 如果需要调整到相同尺寸
      if resize_to_same and len(all_images) > 1:
          resized_images = []
          for img in all_images:
              # 添加batch维度进行resize
              img_batch = img.unsqueeze(0)  # [1, H, W, C]
              # 使用ComfyUI的upscale函数进行尺寸调整
              resized = common_upscale(img_batch, target_width, target_height, "lanczos", "center")
              resized_images.append(resized.squeeze(0))  # 移除batch维度 [H, W, C]
          all_images = resized_images
      
      # 保存图像的结果信息
      results = []
      
      # 保存所有图像
      for i, img in enumerate(all_images):
          filename = f"{filename_prefix}_{i+1:03d}"
          result = self._save_single_image(img, filename)
          results.append(result)
      
      # 准备返回的图像数据
      # 如果所有图像尺寸相同，可以stack；否则返回第一张图像
      try:
          # 检查所有图像是否具有相同尺寸
          shapes = [img.shape for img in all_images]
          if all(shape == shapes[0] for shape in shapes):
              # 所有图像尺寸相同，可以stack
              output_batch = torch.stack(all_images, dim=0)
          else:
              # 尺寸不同，返回第一张图像（保持与SaveImage兼容）
              output_batch = all_images[0].unsqueeze(0)
      except:
          # 如果出现任何问题，返回第一张图像
          output_batch = all_images[0].unsqueeze(0)
      
    #   return {"ui": {"images": results}, "result": (output_batch,)}
      return {"ui": {"images": results}}

  def _save_single_image(self, image, filename_prefix):
      """保存单张图像"""
      # 确保图像格式正确 [H, W, C]
      if len(image.shape) == 4:
          image = image.squeeze(0)
      
      # 确保值在0-1范围内
      image = torch.clamp(image, 0.0, 1.0)
      
      # 转换为numpy并调整到0-255范围
      img_array = (image.cpu().numpy() * 255).astype(np.uint8)
      
      # 创建PIL Image
      if img_array.shape[2] == 4:
          pil_image = Image.fromarray(img_array, 'RGBA')
      elif img_array.shape[2] == 3:
          pil_image = Image.fromarray(img_array, 'RGB')
      else:
          # 处理灰度图像
          if img_array.shape[2] == 1:
              img_array = img_array.squeeze(2)
          pil_image = Image.fromarray(img_array, 'L')
      
      # 生成唯一文件名
      counter = 1
      while True:
          filename = f"{filename_prefix}_{counter:05d}.png"
          filepath = os.path.join(self.output_dir, filename)
          if not os.path.exists(filepath):
              break
          counter += 1
      
      # 保存图像
      pil_image.save(filepath, compress_level=self.compress_level)
      
      return {
          "filename": filename,
          "subfolder": "",
          "type": self.type
      }

# # 节点映射
# NODE_CLASS_MAPPINGS = {
#   "MultiSaveImage": MultiSaveImage
# }

# NODE_DISPLAY_NAME_MAPPINGS = {
#   "MultiSaveImage": "Multi Save Image"
# }