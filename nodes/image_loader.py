import hashlib
import io
import os
from typing import Optional
from urllib.parse import urlparse

import numpy as np
import requests
import torch
from PIL import Image, ImageOps

import folder_paths


class ImageLoader:
    """
    Hybrid image loader that mirrors ComfyUI's Load Image node while adding URL download support.
    """

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "load_image"
    CATEGORY = "image"

    @classmethod
    def INPUT_TYPES(cls):
        # Gracefully handle cases where ComfyUI hasn't registered the input folder (avoids KeyError).
        try:
            input_files = folder_paths.get_filename_list("input")
        except KeyError:
            print("[MeuxImageLoader] WARN: 'input' folder not registered in folder_paths; local picker disabled.")
            input_files = []
        return {
            "required": {
                "source_type": (["local", "url"], {"default": "local"}),
                "image": (input_files, {"default": ""}),
            },
            "optional": {
                "image_url": ("STRING", {"default": ""}),
                "filename_hint": ("STRING", {"default": ""}),
                "persist_to_input": ("BOOLEAN", {"default": True}),
                "overwrite_existing": ("BOOLEAN", {"default": False}),
                "download_timeout": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 120.0}),
                "max_download_mb": ("FLOAT", {"default": 20.0, "min": 1.0, "max": 200.0}),
                "verify_ssl": ("BOOLEAN", {"default": True}),
            },
        }

    def load_image(
        self,
        source_type="local",
        image=None,
        image_url="",
        filename_hint="",
        persist_to_input=True,
        overwrite_existing=False,
        download_timeout=10.0,
        max_download_mb=20.0,
        verify_ssl=True,
    ):
        if source_type == "local":
            pil_image = self._load_local_image(image)
        elif source_type == "url":
            pil_image = self._load_url_image(
                image_url=image_url,
                timeout=download_timeout,
                max_bytes=int(max_download_mb * 1024 * 1024),
                verify_ssl=verify_ssl,
            )
            if persist_to_input:
                self._persist_image(
                    pil_image,
                    filename_hint=filename_hint,
                    url=image_url,
                    overwrite=overwrite_existing,
                )
        else:
            raise ValueError(f"未知 source_type: {source_type}")

        image_tensor, mask_tensor = self._pil_to_tensors(pil_image)
        return (image_tensor, mask_tensor)

    def _load_local_image(self, image_name: Optional[str]) -> Image.Image:
        if not image_name:
            raise ValueError("未指定要加载的本地图片文件。")

        try:
            input_dir = folder_paths.get_input_directory()
        except KeyError:
            raise ValueError("未检测到 ComfyUI 的 input 目录，请在根目录创建 input 文件夹后重启。")
        full_path = os.path.abspath(os.path.join(input_dir, image_name))
        if not (full_path.startswith(os.path.abspath(input_dir) + os.sep) or full_path == os.path.abspath(input_dir)):
            raise ValueError("无效的图片路径。")

        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"找不到图片：{image_name}")

        image = Image.open(full_path)
        image = ImageOps.exif_transpose(image)
        return image

    def _load_url_image(self, image_url: str, timeout: float, max_bytes: int, verify_ssl: bool) -> Image.Image:
        if not image_url:
            raise ValueError("请提供有效的图片 URL。")

        url = image_url.strip()
        if not url.lower().startswith(("http://", "https://")):
            raise ValueError("只支持 HTTP/HTTPS 图片 URL。")

        buffer = io.BytesIO()
        with requests.get(url, timeout=timeout, stream=True, verify=verify_ssl) as response:
            response.raise_for_status()

            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                buffer.write(chunk)
                if buffer.tell() > max_bytes:
                    raise ValueError(f"图片超过限制：{max_bytes / (1024 * 1024):.1f} MB")

        buffer.seek(0)
        image = Image.open(buffer)
        image = ImageOps.exif_transpose(image)
        return image

    def _pil_to_tensors(self, image: Image.Image):
        original_mode = image.mode

        if image.mode in {"I", "F"}:
            image = image.convert("RGB")
        elif image.mode == "P":
            image = image.convert("RGBA")

        mask_tensor = None

        if image.mode == "RGBA":
            mask_tensor = self._mask_from_channel(image.getchannel("A"))
            image = image.convert("RGB")
        elif image.mode == "LA":
            mask_tensor = self._mask_from_channel(image.getchannel("A"))
            image = image.convert("RGB")
        elif image.mode == "L":
            mask_tensor = self._mask_from_channel(image)
            image = image.convert("RGB")
        else:
            if image.mode != "RGB":
                image = image.convert("RGB")

        np_image = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(np_image)[None, ...]

        if mask_tensor is None:
            mask_tensor = torch.zeros((1, image_tensor.shape[1], image_tensor.shape[2], 1), dtype=torch.float32)

        print(f"[MeuxImageLoader] mode={original_mode} -> tensor shape={tuple(image_tensor.shape)}")
        return image_tensor, mask_tensor

    def _mask_from_channel(self, channel: Image.Image) -> torch.Tensor:
        array = np.array(channel).astype(np.float32) / 255.0
        mask = torch.from_numpy(array)[None, ..., None]
        return mask

    def _persist_image(self, image: Image.Image, filename_hint: str, url: str, overwrite: bool):
        try:
            input_dir = folder_paths.get_input_directory()
        except KeyError:
            raise ValueError("未检测到 ComfyUI 的 input 目录，请在根目录创建 input 文件夹后重启。")

        extension = self._infer_extension(image, url)
        filename = self._build_filename(filename_hint or os.path.basename(urlparse(url).path), extension, url)
        full_path = os.path.abspath(os.path.join(input_dir, filename))
        input_dir_abs = os.path.abspath(input_dir)
        if not (full_path.startswith(input_dir_abs + os.sep) or full_path == input_dir_abs):
            raise ValueError("无法写入到 input 目录之外。")

        if os.path.exists(full_path) and not overwrite:
            stem, ext = os.path.splitext(filename)
            counter = 1
            while True:
                candidate = f"{stem}_{counter}{ext}"
                candidate_path = os.path.join(input_dir, candidate)
                if not os.path.exists(candidate_path):
                    full_path = candidate_path
                    break
                counter += 1

        image.save(full_path)
        print(f"[MeuxImageLoader] 已保存到 {os.path.relpath(full_path, input_dir)}")

    def _build_filename(self, hint: str, extension: str, url: str) -> str:
        sanitized = self._sanitize_filename(hint) if hint else ""
        if sanitized.endswith(extension):
            base = sanitized[: -len(extension)]
        else:
            base = sanitized or ""

        if not base:
            digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
            base = f"meux_url_{digest}"

        return f"{base}{extension}"

    def _infer_extension(self, image: Image.Image, url: str) -> str:
        parsed = urlparse(url)
        path = parsed.path or ""
        ext = os.path.splitext(path)[1].lower()
        if ext in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            return ext

        if image.format:
            format_map = {
                "PNG": ".png",
                "JPEG": ".jpg",
                "JPG": ".jpg",
                "WEBP": ".webp",
                "BMP": ".bmp",
            }
            if image.format.upper() in format_map:
                return format_map[image.format.upper()]

        return ".png"

    def _sanitize_filename(self, name: str) -> str:
        dangerous_chars = ["..", "/", "\\", ":"]
        sanitized = name
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, "_")

        sanitized = sanitized.strip().strip(".")
        return sanitized[:100] if sanitized else ""


__all__ = ["ImageLoader"]
