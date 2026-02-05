import os
from typing import Dict, Tuple

import numpy as np
import torch
from PIL import Image

try:
    import folder_paths
except Exception:
    folder_paths = None

try:
    from RealESRGAN import RealESRGAN
    _IMPORT_ERROR = None
except Exception as e:
    RealESRGAN = None
    _IMPORT_ERROR = e


_UPSCALER_CACHE: Dict[Tuple[str, str], RealESRGAN] = {}


def _require_deps():
    if _IMPORT_ERROR is not None:
        raise RuntimeError(
            "MeuxRealESRGANUpscale 依赖未安装，请先安装 RealESRGAN。"
        ) from _IMPORT_ERROR


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _candidate_model_dirs():
    candidates = []
    if folder_paths is not None:
        try:
            models_dir = folder_paths.models_dir
        except Exception:
            models_dir = None
        if models_dir:
            candidates.append(os.path.join(models_dir, "upscale"))
            candidates.append(os.path.join(models_dir, "upscale_models"))
            candidates.append(os.path.join(models_dir, "upscaler"))
    return [c for c in candidates if c and os.path.isdir(c)]


def _resolve_model_path(model_name: str, model_path: str) -> str:
    if model_path and os.path.isfile(model_path):
        return model_path

    if model_name and os.path.isabs(model_name) and os.path.isfile(model_name):
        return model_name

    for base_dir in _candidate_model_dirs():
        path = os.path.join(base_dir, model_name)
        if os.path.isfile(path):
            return path

    raise FileNotFoundError(
        "未找到 RealESRGAN 权重文件。请确认文件位于 ComfyUI/models/upscale/"
        f" 或指定绝对路径。当前查找名: {model_name}"
    )


def _get_upscaler(model_path: str) -> RealESRGAN:
    _require_deps()
    device = _get_device()
    cache_key = (model_path, device)
    if cache_key in _UPSCALER_CACHE:
        return _UPSCALER_CACHE[cache_key]

    upscaler = RealESRGAN(device=device, scale=4)
    upscaler.load_weights(model_path, download=False)
    _UPSCALER_CACHE[cache_key] = upscaler
    return upscaler


class MeuxRealESRGANUpscale:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "scale_mode": (
                    ["2x", "3x", "4x", "6x", "8x", "custom"],
                    {"default": "2x"}
                ),
                "custom_scale": (
                    "FLOAT",
                    {"default": 2.0, "min": 0.1, "max": 8.0, "step": 0.1}
                ),
                "model_name": (
                    "STRING",
                    {"default": "RealESRGAN_x4plus.pth"}
                ),
            },
            "optional": {
                "model_path": ("STRING", {"default": ""}),
                "free_gpu_after": ("BOOLEAN", {"default": False})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "image/upscale"

    def process(self, image, scale_mode, custom_scale, model_name, model_path="", free_gpu_after=False):
        model_path = _resolve_model_path(model_name, model_path)
        upscaler = _get_upscaler(model_path)

        output_images = []
        batch_size = image.shape[0]

        if scale_mode == "custom":
            scale_size = float(custom_scale)
        else:
            scale_size = float(scale_mode.replace("x", ""))

        for i in range(batch_size):
            img = image[i].cpu().numpy()
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            pil = Image.fromarray(img)
            if pil.mode != "RGB":
                pil = pil.convert("RGB")

            ow, oh = pil.size
            upscaled = upscaler.predict(pil)

            target_w = max(1, int(round(ow * float(scale_size))))
            target_h = max(1, int(round(oh * float(scale_size))))
            if upscaled.size != (target_w, target_h):
                upscaled = upscaled.resize((target_w, target_h), Image.LANCZOS)

            if upscaled.mode != "RGB":
                upscaled = upscaled.convert("RGB")
            out = torch.from_numpy(np.array(upscaled)).float() / 255.0
            output_images.append(out)

        if free_gpu_after and _get_device() == "cuda":
            try:
                upscaler.model.to(device="cpu")
            except Exception:
                pass
            _UPSCALER_CACHE.pop((model_path, "cuda"), None)
            torch.cuda.empty_cache()

        return (torch.stack(output_images),)
