import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import folder_paths
except Exception:
    folder_paths = None

try:
    from torchvision import transforms
    from transformers import AutoModelForImageSegmentation
    _IMPORT_ERROR = None
except Exception as e:
    transforms = None
    AutoModelForImageSegmentation = None
    _IMPORT_ERROR = e


MODEL_CACHE = {}
MODEL_PRESET_MAP = {
    "人像抠图 (BiRefNet-portrait)": "BiRefNet-portrait",
    "通用抠图 (BiRefNet)": "BiRefNet",
}


def _require_deps():
    if _IMPORT_ERROR is not None:
        raise RuntimeError(
            "MeuxRMBG 依赖未安装，请先安装 transformers 与 torchvision。"
        ) from _IMPORT_ERROR


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_model_source(model_name: str) -> Tuple[str, str, bool]:
    """
    Resolve model source.
    Returns (repo_id, local_dir, local_only).
    """
    model_name = (model_name or "").strip() or "portrait"
    model_name = MODEL_PRESET_MAP.get(model_name, model_name)

    if os.path.isabs(model_name) or os.path.isdir(model_name):
        return "", model_name, True

    local_dir = resolve_local_model_dir(model_name)
    if local_dir:
        return "", local_dir, True

    if "/" in model_name:
        return model_name, "", os.getenv("LOCAL_ONLY") == "1"

    repo_id = os.getenv("MODEL_REPO", "ZhengPeng7/BiRefNet-portrait")
    return repo_id, "", os.getenv("LOCAL_ONLY") == "1"


def get_candidate_base_dirs() -> List[str]:
    candidates: List[str] = []

    env_dir = os.getenv("MODEL_BASE_DIR", "").strip()
    if env_dir:
        candidates.append(env_dir)

    if folder_paths is not None:
        try:
            candidates.extend(folder_paths.get_folder_paths("birefnet"))
        except Exception:
            pass
        try:
            candidates.extend(folder_paths.get_folder_paths("BiRefNet"))
        except Exception:
            pass
        try:
            models_dir = folder_paths.models_dir
        except Exception:
            models_dir = None
        if models_dir:
            candidates.append(os.path.join(models_dir, "BiRefNet"))
            candidates.append(os.path.join(models_dir, "birefnet"))

    seen = set()
    filtered: List[str] = []
    for path in candidates:
        if not path:
            continue
        norm = os.path.normpath(os.path.expanduser(path))
        if norm in seen:
            continue
        seen.add(norm)
        if os.path.isdir(norm):
            filtered.append(norm)

    return filtered


def resolve_local_model_dir(model_name: str) -> str:
    for base_dir in get_candidate_base_dirs():
        local_dir = os.path.join(base_dir, model_name)
        if os.path.isdir(local_dir):
            return local_dir
    return ""


def list_local_models():
    available = set()
    for base_dir in get_candidate_base_dirs():
        for name in os.listdir(base_dir):
            full = os.path.join(base_dir, name)
            if os.path.isdir(full):
                available.add(name)

    if not available:
        return []

    choices = []
    for friendly, real in MODEL_PRESET_MAP.items():
        if real in available:
            choices.append(friendly)

    remaining = sorted(name for name in available if name not in MODEL_PRESET_MAP.values())
    choices.extend(remaining)
    return choices


def load_model(model_name: str):
    _require_deps()
    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]

    repo_id, local_dir, local_only = resolve_model_source(model_name)
    if local_dir:
        if not os.path.isdir(local_dir):
            raise FileNotFoundError(
                f"MODEL_BASE_DIR 已设置，但找不到模型目录: {local_dir}"
            )
        model = AutoModelForImageSegmentation.from_pretrained(
            local_dir,
            local_files_only=True,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForImageSegmentation.from_pretrained(
            repo_id,
            trust_remote_code=True,
            local_files_only=local_only,
        )

    device = get_device()
    model.to(device)
    model.eval()
    MODEL_CACHE[model_name] = model
    return model


def preprocess(image: Image.Image, size: Tuple[int, int]) -> torch.Tensor:
    _require_deps()
    transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return transform(image)


def birefnet_mask(image: Image.Image, model_name: str, input_size: int) -> torch.Tensor:
    device = get_device()
    model = load_model(model_name)

    image = image.convert("RGB")
    image_tensor = preprocess(image, (input_size, input_size)).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(image_tensor)[-1].sigmoid().cpu()

    w, h = image.size
    result = torch.squeeze(F.interpolate(preds, size=(h, w), mode="bilinear", align_corners=False))
    ma = torch.max(result)
    mi = torch.min(result)
    denom = (ma - mi) if (ma - mi) > 0 else torch.tensor(1.0)
    result = (result - mi) / denom
    return result.clamp(0, 1)


class MeuxRMBG:
    """
    Background removal using BiRefNet models.

    Env vars:
    - MODEL_BASE_DIR: local model root, e.g. /path/to/models
    - MODEL_REPO: HF repo id, default ZhengPeng7/BiRefNet-portrait
    - LOCAL_ONLY=1: only load local cache
    """

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "process"
    CATEGORY = "image/segmentation"

    @classmethod
    def INPUT_TYPES(cls):
        local_models = list_local_models()
        model_choices = local_models or ["portrait"]
        return {
            "required": {
                "image": ("IMAGE",),
                "model_preset": (model_choices, {"default": model_choices[0]}),
                "input_size": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "apply_mask": ("BOOLEAN", {"default": True}),
                "invert_mask": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "model_override": ("STRING", {"default": ""}),
            },
        }

    def process(self, image, model_preset, input_size, apply_mask, invert_mask, model_override=""):
        output_images = []
        output_masks = []

        model = model_override.strip() or model_preset
        model = MODEL_PRESET_MAP.get(model, model)
        batch_size = image.shape[0]
        for i in range(batch_size):
            img_tensor = image[i].cpu()
            np_image = (img_tensor.numpy() * 255.0).clip(0, 255).astype(np.uint8)
            pil = Image.fromarray(np_image)
            mask = birefnet_mask(pil, model, input_size)

            if invert_mask:
                mask = 1.0 - mask

            mask_tensor = mask.unsqueeze(-1)
            output_masks.append(mask_tensor)

            if apply_mask:
                masked = img_tensor * mask_tensor
                output_images.append(masked)
            else:
                output_images.append(img_tensor)

        return (torch.stack(output_images), torch.stack(output_masks))
