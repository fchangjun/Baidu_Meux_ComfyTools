from typing import Tuple

try:
    from nodes import MAX_RESOLUTION
except Exception:
    MAX_RESOLUTION = 8192


def _compute_bounds(width: int, height: int, left: int, right: int, top: int, bottom: int) -> Tuple[int, int, int, int]:
    left_inset = left
    right_inset = width - right
    top_inset = top
    bottom_inset = height - bottom
    return left_inset, right_inset, top_inset, bottom_inset


class AdvancedImageCrop:
    """
    Image crop node that supports pixel or percentage inputs with optional alignment to 8/16px grids.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "measurement": (["Pixels", "Percentage"], {"default": "Pixels"}),
                "align_to": (["0", "8", "16"], {"default": "8"}),
                "left": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "right": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "top": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "bottom": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crop"
    CATEGORY = "Image/Transform"
    NAME = "Advanced Image Crop"

    def crop(self, image=None, measurement="Pixels", align_to="8", left=0, right=0, top=0, bottom=0):
        if image is None:
            raise ValueError("No image tensor provided for cropping.")

        _, height, width, _ = image.shape
        align = int(align_to)

        if measurement == "Percentage":
            left = int(width * left / 100.0)
            right = int(width * right / 100.0)
            top = int(height * top / 100.0)
            bottom = int(height * bottom / 100.0)

        if align > 0:
            left = (left // align) * align
            right = (right // align) * align
            top = (top // align) * align
            bottom = (bottom // align) * align

        if left == 0 and right == 0 and top == 0 and bottom == 0:
            return (image,)

        left_inset, right_inset, top_inset, bottom_inset = _compute_bounds(width, height, left, right, top, bottom)

        if left_inset >= right_inset:
            raise ValueError(f"Invalid crop: left ({left_inset}) >= right ({right_inset})")
        if top_inset >= bottom_inset:
            raise ValueError(f"Invalid crop: top ({top_inset}) >= bottom ({bottom_inset})")

        cropped = image[:, top_inset:bottom_inset, left_inset:right_inset, :]

        print(
            f"[AdvancedImageCrop] Original: {width}x{height}, Cropped: {right_inset - left_inset}x{bottom_inset - top_inset}, align_to={align}"
        )

        return (cropped,)


__all__ = ["AdvancedImageCrop"]
