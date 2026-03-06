import comfy.model_management
import torch
import torchvision.transforms.v2 as T


class MeuxMaskBlurPlus:
    """
    Upstream logic aligned to ComfyUI Essentials' "MaskBlur+".
    """

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = "mask/Meux"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "amount": ("INT", {"default": 6, "min": 0, "max": 256, "step": 1}),
                "device": (["auto", "cpu", "gpu"], {"default": "auto"}),
            }
        }

    def execute(self, mask, amount, device):
        original_has_channel_dim = mask.ndim == 4 and mask.shape[1] == 1
        if original_has_channel_dim:
            mask = mask.squeeze(1)

        if amount <= 0:
            if original_has_channel_dim:
                mask = mask.unsqueeze(1)
            return (mask,)

        if device == "gpu":
            mask = mask.to(comfy.model_management.get_torch_device())
        elif device == "cpu":
            mask = mask.to("cpu")

        if amount % 2 == 0:
            amount += 1

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        mask = T.functional.gaussian_blur(mask.unsqueeze(1), amount).squeeze(1)

        if device in {"gpu", "cpu"}:
            mask = mask.to(comfy.model_management.intermediate_device())

        if original_has_channel_dim:
            mask = mask.unsqueeze(1)

        return (mask,)


__all__ = ["MeuxMaskBlurPlus"]
