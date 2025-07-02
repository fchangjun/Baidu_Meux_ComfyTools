import os
import json
import numpy as np
from PIL import Image
import requests
import time
from typing import Tuple
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

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "image"

    def sanitize_filename(self, filename):
        """
        消毒文件名，防止路径遍历攻击
        """
        # 移除或替换危险字符
        dangerous_chars = ['..', '/', '\\', ':', '*', '?', '"', '<', '>', '|']
        sanitized = filename
        
        # 移除路径分隔符和相对路径
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '_')
        
        # 移除开头和结尾的空格和点
        sanitized = sanitized.strip(' .')
        
        # 确保不为空
        if not sanitized or sanitized.isspace():
            sanitized = "ComfyUI"
        
        # 限制长度（可选，防止过长的文件名）
        max_length = 100
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized

    def validate_output_path(self, filepath):
        """
        验证输出路径是否在允许的目录内
        """
        # 获取规范化的绝对路径
        abs_filepath = os.path.abspath(filepath)
        abs_output_dir = os.path.abspath(self.output_dir)
        
        # 检查文件路径是否在输出目录内
        return abs_filepath.startswith(abs_output_dir + os.sep) or abs_filepath == abs_output_dir

    def save_images(self, images_1, filename_prefix="ComfyUI", images_2=None, images_3=None, 
                   images_4=None, images_5=None, images_6=None, images_7=None, images_8=None,
                   save_individually=False, resize_to_same=False, target_width=512, target_height=512):
        
        # 消毒文件名前缀
        sanitized_prefix = self.sanitize_filename(filename_prefix)
        
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
            filename = f"{sanitized_prefix}_{i+1:03d}"
            result = self._save_single_image(img, filename)
            results.append(result)
        
        # 准备返回的图像数据
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
            
            # 验证路径安全性
            if not self.validate_output_path(filepath):
                raise ValueError(f"不安全的文件路径: {filepath}")
            
            if not os.path.exists(filepath):
                break
            counter += 1
        
        # 保存图像
        try:
            pil_image.save(filepath, compress_level=self.compress_level)
        except Exception as e:
            raise ValueError(f"保存图像失败: {str(e)}")
        
        return {
            "filename": filename,
            "subfolder": "",
            "type": self.type
        }




class SimpleLLMNode:
  """
  简化的ComfyUI LLM节点，用于单轮对话
  用户可以自定义API密钥、模型、温度等关键参数
  """
  
  def __init__(self):
      pass
  
  @classmethod
  def INPUT_TYPES(cls):
      return {
          "required": {
              "api_key": ("STRING", {
                  "default": "",
                  "multiline": False,
                  "placeholder": "输入你的API密钥"
              }),
              "model": ("STRING", {
                  "default": "Qwen/Qwen2.5-72B-Instruct",
                  "multiline": False,
                  "placeholder": "例如: Qwen/Qwen2.5-72B-Instruct"
              }),
              "user_prompt": ("STRING", {
                #   "default": "Hello, how are you?",
                  "multiline": True,
                  "placeholder": "输入你的问题或提示"
              }),
              "temperature": ("FLOAT", {
                  "default": 0.7,
                  "min": 0.0,
                  "max": 2.0,
                  "step": 0.1,
                  "display": "slider"
              }),
              "max_tokens": ("INT", {
                  "default": 1024,
                  "min": 1,
                  "max": 4096,
                  "step": 1
              }),
          },
          "optional": {
              "system_prompt": ("STRING", {
                #   "default": "You are a helpful assistant.",
                  "multiline": True,
                  "placeholder": "输入系统提示词（可选）"
              }),
              "api_url": ("STRING", {
                  "default": "https://api.siliconflow.cn/v1/chat/completions",
                  "multiline": False,
                  "placeholder": "API端点URL"
              }),
              "top_p": ("FLOAT", {
                  "default": 0.9,
                  "min": 0.0,
                  "max": 1.0,
                  "step": 0.1,
                  "display": "slider"
              }),
              "top_k": ("INT", {
                  "default": 50,
                  "min": 1,
                  "max": 100,
                  "step": 1
              }),
              "frequency_penalty": ("FLOAT", {
                  "default": 0.0,
                  "min": 0.0,
                  "max": 2.0,
                  "step": 0.1,
                  "display": "slider"
              }),
              "presence_penalty": ("FLOAT", {
                  "default": 0.0,
                  "min": 0.0,
                  "max": 2.0,
                  "step": 0.1,
                  "display": "slider"
              }),
          }
      }
  
  RETURN_TYPES = ("STRING", "STRING", "INT")
  RETURN_NAMES = ("response", "full_json", "tokens_used")
  FUNCTION = "call_llm"
  CATEGORY = "LLM"
  DESCRIPTION = "调用LLM API进行单轮对话"
  
  def call_llm(self, api_key: str, model: str, user_prompt: str, temperature: float, 
               max_tokens: int, system_prompt: str = "", 
               api_url: str = "https://api.siliconflow.cn/v1/chat/completions",
               top_p: float = 0.9, top_k: int = 50, 
               frequency_penalty: float = 0.0, presence_penalty: float = 0.0) -> Tuple[str, str, int]:
      """
      调用LLM API进行单轮对话
      """
      
      # 验证必要参数
      if not api_key.strip():
          return ("错误：请输入API密钥", json.dumps({"error": "API密钥为空"}), 0)
      
      if not user_prompt.strip():
          return ("错误：请输入用户提示", json.dumps({"error": "用户提示为空"}), 0)
      
      try:
          # 构建消息
          messages = []
          if system_prompt.strip():
              messages.append({
                  "role": "system", 
                  "content": system_prompt.strip()
              })
          
          messages.append({
              "role": "user", 
              "content": user_prompt.strip()
          })
          
          # 构建请求数据
          request_data = {
              "model": model,
              "messages": messages,
              "temperature": temperature,
              "max_tokens": max_tokens,
              "top_p": top_p,
              "top_k": top_k,
              "frequency_penalty": frequency_penalty,
              "presence_penalty": presence_penalty,
              "stream": False
          }
          
          # 设置请求头
          headers = {
              "Authorization": f"Bearer {api_key}",
              "Content-Type": "application/json"
          }
          
          print(f"正在调用LLM API: {model}")
          print(f"用户提示: {user_prompt[:100]}{'...' if len(user_prompt) > 100 else ''}")
          
          # 发送请求
          start_time = time.time()
          response = requests.post(
              api_url, 
              headers=headers, 
              json=request_data, 
              timeout=120  # 2分钟超时
          )
          
          response_time = time.time() - start_time
          
          # 检查HTTP状态
          if response.status_code != 200:
              error_msg = f"API调用失败 (状态码: {response.status_code}): {response.text}"
              print(error_msg)
              return (error_msg, json.dumps({"error": error_msg, "status_code": response.status_code}), 0)
          
          # 解析响应
          try:
              response_json = response.json()
          except json.JSONDecodeError as e:
              error_msg = f"响应JSON解析失败: {str(e)}"
              print(error_msg)
              return (error_msg, json.dumps({"error": error_msg}), 0)
          
          # 提取响应内容
          if "choices" in response_json and len(response_json["choices"]) > 0:
              llm_response = response_json["choices"][0]["message"]["content"]
          else:
              error_msg = "API响应中没有找到有效内容"
              print(error_msg)
              return (error_msg, json.dumps(response_json), 0)
          
          # 获取token使用情况
          tokens_used = 0
          if "usage" in response_json:
              tokens_used = response_json["usage"].get("total_tokens", 0)
          
          # 格式化完整响应
          full_response = json.dumps(response_json, indent=2, ensure_ascii=False)
          
          print(f"✅ API调用成功！")
          print(f"📊 响应时间: {response_time:.2f}秒")
          print(f"🔢 Token使用: {tokens_used}")
          print(f"📝 响应长度: {len(llm_response)}字符")
          
          return (llm_response, full_response, tokens_used)
          
      except requests.exceptions.Timeout:
          error_msg = "请求超时，请稍后重试"
          print(error_msg)
          return (error_msg, json.dumps({"error": error_msg}), 0)
          
      except requests.exceptions.ConnectionError:
          error_msg = "网络连接错误，请检查网络连接"
          print(error_msg)
          return (error_msg, json.dumps({"error": error_msg}), 0)
          
      except requests.exceptions.RequestException as e:
          error_msg = f"请求错误: {str(e)}"
          print(error_msg)
          return (error_msg, json.dumps({"error": error_msg}), 0)
          
      except Exception as e:
          error_msg = f"未知错误: {str(e)}"
          print(error_msg)
          return (error_msg, json.dumps({"error": error_msg}), 0)

# # 节点注册
# NODE_CLASS_MAPPINGS = {
#   "SimpleLLMNode": SimpleLLMNode,
# }

# NODE_DISPLAY_NAME_MAPPINGS = {
#   "SimpleLLMNode": "🤖 LLM API调用",
# }

