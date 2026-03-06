import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_fill_holes


class MeuxMaskFillHoles:
    """
    Upstream logic aligned to WAS Node Suite's "Mask Fill Holes".
    """

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASKS",)
    FUNCTION = "fill_region"
    CATEGORY = "mask/Meux"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
            }
        }

    def fill_region(self, masks):
        # WAS emits [B,1,H,W] for batched outputs. Accept that layout on input too.
        if masks.ndim > 3 and masks.shape[1] == 1:
            masks = masks.squeeze(1)

        if masks.ndim > 3:
            regions = []
            for mask in masks:
                region_tensor = self._fill_single(mask).unsqueeze(0).unsqueeze(1)
                regions.append(region_tensor)
            return (torch.cat(regions, dim=0),)

        return (self._fill_single(masks).unsqueeze(0).unsqueeze(1),)

    def _fill_single(self, mask):
        mask_np = np.clip(255.0 * mask.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        image = Image.fromarray(mask_np, mode="L").convert("L")
        binary_mask = np.array(image) > 0
        filled_mask = binary_fill_holes(binary_mask)
        filled_image = Image.fromarray(filled_mask.astype(np.uint8) * 255, mode="L")

        image_np = np.array(filled_image.convert("L")).astype(np.float32) / 255.0
        return 1.0 - torch.from_numpy(image_np)


__all__ = ["MeuxMaskFillHoles"]
