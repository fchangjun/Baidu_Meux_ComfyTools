import os
import numpy as np
import torch
from PIL import Image

import folder_paths
from comfy.utils import common_upscale


class MultiSaveImage:
    """
    Output node that saves up to sixteen image batches to the configured ComfyUI output directory.
    Supports optional resizing to a shared resolution and filename sanitisation.
    """

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
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
                "images_9": ("IMAGE",),
                "images_10": ("IMAGE",),
                "images_11": ("IMAGE",),
                "images_12": ("IMAGE",),
                "images_13": ("IMAGE",),
                "images_14": ("IMAGE",),
                "images_15": ("IMAGE",),
                "images_16": ("IMAGE",),
                "save_individually": ("BOOLEAN", {"default": False}),
                "resize_to_same": ("BOOLEAN", {"default": False}),
                "target_width": ("INT", {"default": 512, "min": 64, "max": 8192}),
                "target_height": ("INT", {"default": 512, "min": 64, "max": 8192}),
            },
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "image"

    def sanitize_filename(self, filename: str) -> str:
        """
        Remove dangerous characters to prevent path traversal when saving files.
        """
        dangerous_chars = ["..", "/", "\\", ":", "*", "?", '"', "<", ">", "|"]
        sanitized = filename

        for char in dangerous_chars:
            sanitized = sanitized.replace(char, "_")

        sanitized = sanitized.strip(" .")

        if not sanitized or sanitized.isspace():
            sanitized = "ComfyUI"

        max_length = 100
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]

        return sanitized

    def validate_output_path(self, filepath: str) -> bool:
        """
        Ensure the output path stays inside the configured output directory.
        """
        abs_filepath = os.path.abspath(filepath)
        abs_output_dir = os.path.abspath(self.output_dir)
        return abs_filepath.startswith(abs_output_dir + os.sep) or abs_filepath == abs_output_dir

    def save_images(
        self,
        images_1,
        filename_prefix="ComfyUI",
        images_2=None,
        images_3=None,
        images_4=None,
        images_5=None,
        images_6=None,
        images_7=None,
        images_8=None,
        images_9=None,
        images_10=None,
        images_11=None,
        images_12=None,
        images_13=None,
        images_14=None,
        images_15=None,
        images_16=None,
        save_individually=False,
        resize_to_same=False,
        target_width=512,
        target_height=512,
    ):
        sanitized_prefix = self.sanitize_filename(filename_prefix)

        all_images = []
        image_inputs = [
            images_1,
            images_2,
            images_3,
            images_4,
            images_5,
            images_6,
            images_7,
            images_8,
            images_9,
            images_10,
            images_11,
            images_12,
            images_13,
            images_14,
            images_15,
            images_16,
        ]

        for img_batch in image_inputs:
            if img_batch is None:
                continue

            if len(img_batch.shape) == 4:
                for i in range(img_batch.shape[0]):
                    all_images.append(img_batch[i])
            else:
                all_images.append(img_batch)

        if not all_images:
            raise ValueError("至少需要提供一个图像输入")

        if resize_to_same and len(all_images) > 1:
            resized_images = []
            for img in all_images:
                img_batch = img.unsqueeze(0)
                resized = common_upscale(img_batch, target_width, target_height, "lanczos", "center")
                resized_images.append(resized.squeeze(0))
            all_images = resized_images

        results = []

        for i, img in enumerate(all_images):
            filename = f"{sanitized_prefix}_{i + 1:03d}"
            result = self._save_single_image(img, filename)
            results.append(result)

        try:
            shapes = [img.shape for img in all_images]
            if all(shape == shapes[0] for shape in shapes):
                torch.stack(all_images, dim=0)
            else:
                all_images[0].unsqueeze(0)
        except Exception:
            all_images[0].unsqueeze(0)

        return {"ui": {"images": results}}

    def _save_single_image(self, image, filename_prefix: str) -> dict:
        if len(image.shape) == 4:
            image = image.squeeze(0)

        image = torch.clamp(image, 0.0, 1.0)
        img_array = (image.cpu().numpy() * 255).astype(np.uint8)

        if img_array.shape[2] == 4:
            pil_image = Image.fromarray(img_array, "RGBA")
        elif img_array.shape[2] == 3:
            pil_image = Image.fromarray(img_array, "RGB")
        else:
            if img_array.shape[2] == 1:
                img_array = img_array.squeeze(2)
            pil_image = Image.fromarray(img_array, "L")

        counter = 1
        while True:
            filename = f"{filename_prefix}_{counter:05d}.png"
            filepath = os.path.join(self.output_dir, filename)

            if not self.validate_output_path(filepath):
                raise ValueError(f"不安全的文件路径: {filepath}")

            if not os.path.exists(filepath):
                break
            counter += 1

        pil_image.save(filepath, compress_level=self.compress_level)

        return {
            "filename": filename,
            "subfolder": "",
            "type": self.type,
        }


__all__ = ["MultiSaveImage"]
