import os
import sys
import types
from typing import Any, Dict, Tuple

import numpy as np
import torch
from PIL import Image

try:
    import folder_paths
except Exception:
    folder_paths = None

def _ensure_torchvision_shim():
    """
    basicsr (realesrgan dependency) expects torchvision.transforms.functional_tensor.rgb_to_grayscale
    which is removed in newer torchvision. Provide a minimal shim so imports succeed.
    """
    try:
        import torchvision  # noqa: F401
    except Exception:
        return

    module_name = "torchvision.transforms.functional_tensor"
    if module_name in sys.modules:
        return

    try:
        import torchvision.transforms.functional as F
    except Exception:
        return

    def _rgb_to_grayscale(img, num_output_channels=1):
        if not torch.is_tensor(img):
            return F.rgb_to_grayscale(img, num_output_channels=num_output_channels)

        if img.dim() == 3:
            r, g, b = img[0:1], img[1:2], img[2:3]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        elif img.dim() >= 4:
            r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        else:
            return img

        if num_output_channels == 3:
            gray = gray.repeat(1, 3, 1, 1) if gray.dim() >= 4 else gray.repeat(3, 1, 1)
        return gray

    shim = types.ModuleType(module_name)
    shim.rgb_to_grayscale = _rgb_to_grayscale
    sys.modules[module_name] = shim


_BACKEND = None
RealESRGAN = None
RealESRGANer = None
RRDBNet = None

try:
    _ensure_torchvision_shim()
    from RealESRGAN import RealESRGAN  # type: ignore
    _BACKEND = "class"
    _IMPORT_ERROR = None
except Exception:
    try:
        _ensure_torchvision_shim()
        from realesrgan import RealESRGANer  # type: ignore
        from basicsr.archs.rrdbnet_arch import RRDBNet  # type: ignore
        _BACKEND = "er"
        _IMPORT_ERROR = None
    except Exception as e:
        _BACKEND = None
        _IMPORT_ERROR = e


_UPSCALER_CACHE: Dict[Tuple[str, str], Any] = {}
_MODEL_CHOICES_CACHE = None


def _require_deps():
    if _IMPORT_ERROR is not None:
        raise RuntimeError(
            "MeuxRealESRGANUpscale 依赖未安装，请先安装 RealESRGAN。"
        ) from _IMPORT_ERROR


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _is_oom_error(exc: Exception) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    tokens = [
        "out of memory",
        "allocation on device",
        "cuda error: out of memory",
        "cublas_status_alloc_failed",
    ]
    return any(token in msg for token in tokens)


def _candidate_model_dirs():
    candidates = []
    if folder_paths is not None:
        try:
            candidates.extend(folder_paths.get_folder_paths("upscale_models"))
        except Exception:
            pass
        try:
            models_dir = folder_paths.models_dir
        except Exception:
            models_dir = None
        if models_dir:
            candidates.append(os.path.join(models_dir, "upscale_models"))
            candidates.append(os.path.join(models_dir, "upscaler"))
    return [c for c in candidates if c and os.path.isdir(c)]


def _list_model_files():
    global _MODEL_CHOICES_CACHE
    if _MODEL_CHOICES_CACHE is not None:
        return _MODEL_CHOICES_CACHE

    files = []
    for base_dir in _candidate_model_dirs():
        try:
            for name in os.listdir(base_dir):
                if name.endswith(".pth") or name.endswith(".pt") or name.endswith(".safetensors"):
                    files.append(name)
        except Exception:
            continue

    unique = sorted(set(files))
    if not unique:
        unique = [""]

    _MODEL_CHOICES_CACHE = unique
    return unique


def _invalidate_model_cache():
    global _MODEL_CHOICES_CACHE
    _MODEL_CHOICES_CACHE = None


def _resolve_model_path(model_name: str, model_path: str) -> str:
    if model_path and os.path.isfile(model_path):
        return model_path

    if not model_name:
        raise FileNotFoundError(
            "未找到可用模型。请在下拉中选择模型，或填写绝对路径。"
        )

    if model_name and os.path.isabs(model_name) and os.path.isfile(model_name):
        return model_name

    for base_dir in _candidate_model_dirs():
        path = os.path.join(base_dir, model_name)
        if os.path.isfile(path):
            return path

    raise FileNotFoundError(
        "未找到 RealESRGAN 权重文件。请确认文件位于 ComfyUI/models/upscale_models/"
        f" 或指定绝对路径。当前查找名: {model_name}"
    )


def _get_upscaler(model_path: str) -> RealESRGAN:
    _require_deps()
    device = _get_device()
    cache_key = (model_path, device)
    if cache_key in _UPSCALER_CACHE:
        return _UPSCALER_CACHE[cache_key]

    if _BACKEND == "class":
        upscaler = RealESRGAN(device=device, scale=4)
        upscaler.load_weights(model_path, download=False)
    elif _BACKEND == "er":
        class _RealESRGANerWrapper:
            _TILE_SCHEDULE = (0, 1024, 768, 512, 384, 256, 192, 128, 96, 64)

            def __init__(self, model_path: str, device: str):
                self.model_path = model_path
                self.device = device
                self.upsampler = self._build_upsampler(device=device, tile=0, half=(device == "cuda"))
                self._cpu_upsampler = None

            def _build_model(self):
                return RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=23,
                    num_grow_ch=32,
                    scale=4,
                )

            def _build_upsampler(self, device: str, tile: int, half: bool):
                model = self._build_model()
                return RealESRGANer(
                    scale=4,
                    model_path=self.model_path,
                    model=model,
                    tile=tile,
                    tile_pad=10,
                    pre_pad=0,
                    half=(half and device == "cuda"),
                    device=device,
                )

            def _run_with_tile(self, img, tile: int):
                self.upsampler.tile = tile
                output, _ = self.upsampler.enhance(img, outscale=1)
                return output

            def release(self):
                try:
                    self.upsampler.model.to(device="cpu")
                except Exception:
                    pass
                self._cpu_upsampler = None

            def predict(self, pil_image: Image.Image) -> Image.Image:
                import cv2
                img = np.array(pil_image)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                max_side = max(img.shape[0], img.shape[1])
                tile_schedule = []
                for tile in self._TILE_SCHEDULE:
                    if tile == 0:
                        tile_schedule.append(0)
                        continue
                    tile_eff = min(tile, max_side)
                    if tile_eff >= 64:
                        tile_schedule.append(tile_eff)

                tile_schedule = list(dict.fromkeys(tile_schedule))
                last_oom = None
                for tile in tile_schedule:
                    try:
                        output = self._run_with_tile(img, tile=tile)
                        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                        return Image.fromarray(output)
                    except Exception as e:
                        if not _is_oom_error(e):
                            raise
                        last_oom = e
                        if self.device == "cuda":
                            torch.cuda.empty_cache()
                        continue

                if self.device == "cuda":
                    try:
                        if self._cpu_upsampler is None:
                            self._cpu_upsampler = self._build_upsampler(device="cpu", tile=256, half=False)
                        output, _ = self._cpu_upsampler.enhance(img, outscale=1)
                        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                        return Image.fromarray(output)
                    except Exception as cpu_error:
                        if last_oom is not None:
                            raise RuntimeError(
                                "RealESRGAN 显存不足：已尝试分块推理与 CPU 回退，仍失败。"
                            ) from cpu_error
                        raise

                if last_oom is not None:
                    raise RuntimeError(
                        "RealESRGAN 显存不足：请减小输入分辨率或切换到 CPU。"
                    ) from last_oom

                raise RuntimeError("RealESRGAN 推理失败，未返回输出。")

        upscaler = _RealESRGANerWrapper(model_path, device)
    else:
        raise RuntimeError("RealESRGAN backend not available")

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
                "model_name": (_list_model_files(), {"default": ""}),
            },
            "optional": {
                "model_path": ("STRING", {"default": ""}),
                "free_gpu_after": ("BOOLEAN", {"default": False}),
                "refresh_model_list": ("BOOLEAN", {"default": False})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "image/upscale"

    def process(
        self,
        image,
        scale_mode,
        custom_scale,
        model_name,
        model_path="",
        free_gpu_after=False,
        refresh_model_list=False
    ):
        if refresh_model_list:
            _invalidate_model_cache()
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
                if hasattr(upscaler, "release"):
                    upscaler.release()
                else:
                    upscaler.model.to(device="cpu")
            except Exception:
                pass
            _UPSCALER_CACHE.pop((model_path, "cuda"), None)
            torch.cuda.empty_cache()

        return (torch.stack(output_images),)
