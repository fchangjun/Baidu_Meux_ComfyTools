import os

import numpy as np
import torch
from PIL import Image
import folder_paths

from .image_loader import ImageLoader


class ImageMaskLoader(ImageLoader):
    """
    Prompt-only friendly loader for an image and a separate mask.

    Source types:
    - local: file name under ComfyUI input/
    - url: remote HTTP/HTTPS URL
    - embedded: use alpha/luminance extracted from the image itself (mask only)
    - empty: produce an empty mask (mask only)
    """

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "load_image_mask"
    CATEGORY = "image"

    @classmethod
    def INPUT_TYPES(cls):
        input_files = cls._list_input_images()
        return {
            "required": {
                "image_source_type": (["local", "url"], {"default": "local"}),
                "image": (input_files, {"default": "", "image_upload": True}),
                "mask_source_type": (["embedded", "url", "local", "empty"], {"default": "embedded"}),
                "mask_image": (input_files, {"default": "", "image_upload": True}),
            },
            "optional": {
                "image_url": ("STRING", {"default": ""}),
                "mask_url": ("STRING", {"default": ""}),
                "image_filename_hint": ("STRING", {"default": ""}),
                "mask_filename_hint": ("STRING", {"default": ""}),
                "persist_image_to_input": ("BOOLEAN", {"default": True}),
                "persist_mask_to_input": ("BOOLEAN", {"default": False}),
                "overwrite_existing": ("BOOLEAN", {"default": False}),
                "download_timeout": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 120.0}),
                "max_download_mb": ("FLOAT", {"default": 20.0, "min": 1.0, "max": 200.0}),
                "verify_ssl": ("BOOLEAN", {"default": True}),
                "mask_channel": (
                    ["alpha_or_luminance", "luminance", "alpha", "red", "green", "blue"],
                    {"default": "alpha_or_luminance"},
                ),
                "mask_invert": ("BOOLEAN", {"default": False}),
                "resize_mask_to_image": ("BOOLEAN", {"default": True}),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(
        cls,
        image_source_type,
        image=None,
        image_url="",
        mask_source_type="embedded",
        mask_image=None,
        mask_url="",
        **kwargs,
    ):
        image_check = cls._validate_source(
            source_type=image_source_type,
            local_value=image,
            url_value=image_url,
            label="原图",
            allow_embedded=False,
            allow_empty=False,
        )
        if image_check is not True:
            return image_check

        mask_check = cls._validate_source(
            source_type=mask_source_type,
            local_value=mask_image,
            url_value=mask_url,
            label="遮罩",
            allow_embedded=True,
            allow_empty=True,
        )
        if mask_check is not True:
            return mask_check

        return True

    def load_image_mask(
        self,
        image_source_type="local",
        image=None,
        mask_source_type="embedded",
        mask_image=None,
        image_url="",
        mask_url="",
        image_filename_hint="",
        mask_filename_hint="",
        persist_image_to_input=True,
        persist_mask_to_input=False,
        overwrite_existing=False,
        download_timeout=10.0,
        max_download_mb=20.0,
        verify_ssl=True,
        mask_channel="alpha_or_luminance",
        mask_invert=False,
        resize_mask_to_image=True,
    ):
        image_pil = self._load_any_image(
            source_type=image_source_type,
            local_name=image,
            url=image_url,
            filename_hint=image_filename_hint,
            persist_to_input=persist_image_to_input,
            overwrite_existing=overwrite_existing,
            download_timeout=download_timeout,
            max_download_mb=max_download_mb,
            verify_ssl=verify_ssl,
        )
        image_tensor, embedded_mask = self._pil_to_tensors(image_pil)

        if mask_source_type == "embedded":
            mask_tensor = embedded_mask
        elif mask_source_type == "empty":
            mask_tensor = torch.zeros_like(embedded_mask)
        else:
            mask_pil = self._load_any_image(
                source_type=mask_source_type,
                local_name=mask_image,
                url=mask_url,
                filename_hint=mask_filename_hint,
                persist_to_input=persist_mask_to_input,
                overwrite_existing=overwrite_existing,
                download_timeout=download_timeout,
                max_download_mb=max_download_mb,
                verify_ssl=verify_ssl,
            )
            mask_tensor = self._mask_from_image(mask_pil, mask_channel)

            if resize_mask_to_image and tuple(mask_tensor.shape[1:3]) != tuple(image_tensor.shape[1:3]):
                mask_tensor = self._resize_mask(mask_tensor, image_tensor.shape[2], image_tensor.shape[1])
            elif tuple(mask_tensor.shape[1:3]) != tuple(image_tensor.shape[1:3]):
                raise ValueError(
                    f"遮罩尺寸 {tuple(mask_tensor.shape[1:3])} 与原图尺寸 {tuple(image_tensor.shape[1:3])} 不一致。"
                )

        if mask_invert:
            mask_tensor = 1.0 - mask_tensor

        return (image_tensor, mask_tensor.clamp(0.0, 1.0))

    @classmethod
    def _list_input_images(cls):
        try:
            input_dir = folder_paths.get_input_directory()
            files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
            if hasattr(folder_paths, "filter_files_content_types"):
                files = folder_paths.filter_files_content_types(files, ["image"])
            return sorted(files) or [""]
        except Exception:
            print("[MeuxImageMaskLoader] WARN: 无法访问 input 目录，本地文件选择已禁用。")
            return [""]

    @classmethod
    def _validate_source(
        cls,
        source_type,
        local_value,
        url_value,
        label,
        allow_embedded,
        allow_empty,
    ):
        if source_type == "local":
            if not local_value or str(local_value).strip() == "":
                return f"请选择{label}文件。"
        elif source_type == "url":
            if not str(url_value).strip():
                return f"请输入{label} URL。"
        elif source_type == "embedded":
            if not allow_embedded:
                return f"{label}不支持 embedded 模式。"
        elif source_type == "empty":
            if not allow_empty:
                return f"{label}不支持 empty 模式。"
        else:
            return f"未知的{label}来源类型: {source_type}"
        return True

    def _load_any_image(
        self,
        source_type,
        local_name,
        url,
        filename_hint,
        persist_to_input,
        overwrite_existing,
        download_timeout,
        max_download_mb,
        verify_ssl,
    ):
        if source_type == "local":
            return self._load_local_image(local_name)
        if source_type == "url":
            image = self._load_url_image(
                image_url=url,
                timeout=download_timeout,
                max_bytes=int(max_download_mb * 1024 * 1024),
                verify_ssl=verify_ssl,
            )
            if persist_to_input:
                self._persist_image(
                    image,
                    filename_hint=filename_hint,
                    url=url,
                    overwrite=overwrite_existing,
                )
            return image
        raise ValueError(f"未知来源类型: {source_type}")

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
        return torch.from_numpy(array)[None, ..., None]

    def _mask_from_luminance(self, image: Image.Image) -> torch.Tensor:
        array = np.array(image.convert("L")).astype(np.float32) / 255.0
        return torch.from_numpy(array)[None, ..., None]

    def _mask_from_standard_channel(self, image: Image.Image, channel_name: str) -> torch.Tensor:
        if image.mode not in {"RGB", "RGBA"}:
            image = image.convert("RGBA" if "A" in image.getbands() else "RGB")
        if channel_name not in image.getbands():
            raise ValueError(f"遮罩图不包含 {channel_name} 通道。")
        array = np.array(image.getchannel(channel_name)).astype(np.float32) / 255.0
        return torch.from_numpy(array)[None, ..., None]

    def _resize_mask(self, mask_tensor: torch.Tensor, target_width: int, target_height: int) -> torch.Tensor:
        mask = mask_tensor.permute(0, 3, 1, 2)
        mask = torch.nn.functional.interpolate(mask, size=(target_height, target_width), mode="nearest")
        return mask.permute(0, 2, 3, 1)


__all__ = ["ImageMaskLoader"]
