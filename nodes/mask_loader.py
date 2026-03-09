import os

import numpy as np
import torch
from PIL import Image
import folder_paths

from .image_loader import ImageLoader


class MaskLoader(ImageLoader):
    """
    Mask-only loader that follows MeuxImageLoader's local/url semantics.
    """

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "load_mask"
    CATEGORY = "image"

    @classmethod
    def INPUT_TYPES(cls):
        try:
            input_dir = folder_paths.get_input_directory()
            files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
            if hasattr(folder_paths, "filter_files_content_types"):
                files = folder_paths.filter_files_content_types(files, ["image"])
            input_files = sorted(files) or [""]
        except Exception:
            print("[MeuxMaskLoader] WARN: 无法访问 input 目录，本地文件选择已禁用。")
            input_files = [""]

        return {
            "required": {
                "reference_image": ("IMAGE",),
                "source_type": (["local", "url"], {"default": "local"}),
                "mask_image": (input_files, {"default": "", "image_upload": True}),
            },
            "optional": {
                "mask_url": ("STRING", {"default": ""}),
                "filename_hint": ("STRING", {"default": ""}),
                "persist_to_input": ("BOOLEAN", {"default": True}),
                "overwrite_existing": ("BOOLEAN", {"default": False}),
                "download_timeout": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 120.0}),
                "max_download_mb": ("FLOAT", {"default": 20.0, "min": 1.0, "max": 200.0}),
                "verify_ssl": ("BOOLEAN", {"default": True}),
                "mask_channel": (
                    ["alpha_or_luminance", "luminance", "alpha", "red", "green", "blue"],
                    {"default": "alpha_or_luminance"},
                ),
                "mask_invert": ("BOOLEAN", {"default": False}),
                "resize_to_reference": ("BOOLEAN", {"default": True}),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, source_type, mask_image=None, mask_url="", **kwargs):
        if source_type == "local":
            if not mask_image or str(mask_image).strip() == "":
                return "请选择本地遮罩文件。"
        elif source_type == "url":
            if not str(mask_url).strip():
                return "请输入有效的遮罩 URL。"
        return True

    def load_mask(
        self,
        reference_image,
        source_type="local",
        mask_image=None,
        mask_url="",
        filename_hint="",
        persist_to_input=True,
        overwrite_existing=False,
        download_timeout=10.0,
        max_download_mb=20.0,
        verify_ssl=True,
        mask_channel="alpha_or_luminance",
        mask_invert=False,
        resize_to_reference=True,
    ):
        if reference_image is None:
            raise ValueError("MeuxMaskLoader 需要连接 reference_image。")

        if source_type == "local":
            pil_image = self._load_local_image(mask_image)
        elif source_type == "url":
            pil_image = self._load_url_image(
                image_url=mask_url,
                timeout=download_timeout,
                max_bytes=int(max_download_mb * 1024 * 1024),
                verify_ssl=verify_ssl,
            )
            if persist_to_input:
                self._persist_image(
                    pil_image,
                    filename_hint=filename_hint,
                    url=mask_url,
                    overwrite=overwrite_existing,
                )
        else:
            raise ValueError(f"未知 source_type: {source_type}")

        mask_tensor = self._mask_from_image(pil_image, mask_channel)

        mask_height, mask_width = mask_tensor.shape[1:3]
        ref_height, ref_width = reference_image.shape[1:3]
        if (mask_height, mask_width) != (ref_height, ref_width):
            if resize_to_reference:
                mask_tensor = self._resize_mask(mask_tensor, ref_width, ref_height)
            else:
                raise ValueError(
                    f"遮罩尺寸 {(mask_height, mask_width)} 与 reference_image 尺寸 {(ref_height, ref_width)} 不一致。"
                )

        if mask_invert:
            mask_tensor = 1.0 - mask_tensor

        return (mask_tensor.clamp(0.0, 1.0),)

    def _mask_from_image(self, image: Image.Image, channel: str) -> torch.Tensor:
        original = image
        if image.mode == "P":
            image = image.convert("RGBA")
        elif image.mode in {"I", "F"}:
            image = image.convert("RGB")

        if channel == "alpha_or_luminance":
            if "A" in image.getbands():
                return self._mask_from_alpha(image.getchannel("A"))
            return self._mask_from_luminance(image)
        if channel == "alpha":
            if "A" not in image.getbands():
                raise ValueError(f"遮罩图不包含 alpha 通道，当前模式={original.mode}")
            return self._mask_from_alpha(image.getchannel("A"))
        if channel == "luminance":
            return self._mask_from_luminance(image)
        if channel == "red":
            return self._mask_from_standard_channel(image, "R")
        if channel == "green":
            return self._mask_from_standard_channel(image, "G")
        if channel == "blue":
            return self._mask_from_standard_channel(image, "B")
        raise ValueError(f"不支持的遮罩通道模式: {channel}")

    def _mask_from_alpha(self, channel: Image.Image) -> torch.Tensor:
        array = np.array(channel).astype(np.float32) / 255.0
        return 1.0 - torch.from_numpy(array).unsqueeze(0)

    def _mask_from_luminance(self, image: Image.Image) -> torch.Tensor:
        array = np.array(image.convert("L")).astype(np.float32) / 255.0
        return torch.from_numpy(array).unsqueeze(0)

    def _mask_from_standard_channel(self, image: Image.Image, channel_name: str) -> torch.Tensor:
        if image.mode not in {"RGB", "RGBA"}:
            image = image.convert("RGBA" if "A" in image.getbands() else "RGB")
        if channel_name not in image.getbands():
            raise ValueError(f"遮罩图不包含 {channel_name} 通道。")
        array = np.array(image.getchannel(channel_name)).astype(np.float32) / 255.0
        return torch.from_numpy(array).unsqueeze(0)

    def _resize_mask(self, mask_tensor: torch.Tensor, target_width: int, target_height: int) -> torch.Tensor:
        mask = mask_tensor.unsqueeze(1)
        mask = torch.nn.functional.interpolate(mask, size=(target_height, target_width), mode="nearest")
        return mask.squeeze(1)


__all__ = ["MaskLoader"]
