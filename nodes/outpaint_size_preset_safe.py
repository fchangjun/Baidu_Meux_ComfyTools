import math


class OutpaintSizePresetSafe:
    """
    Outpaint Size Preset (Safe Expand)

    - Compute safe expand size for outpainting
    - Each direction rounds UP to align (8 / 64)
    - No image / latent processing
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "expand_left": (
                    "INT",
                    {"default": 0, "min": 0, "max": 4096, "step": 1}
                ),
                "expand_right": (
                    "INT",
                    {"default": 0, "min": 0, "max": 4096, "step": 1}
                ),
                "expand_top": (
                    "INT",
                    {"default": 0, "min": 0, "max": 4096, "step": 1}
                ),
                "expand_bottom": (
                    "INT",
                    {"default": 0, "min": 0, "max": 4096, "step": 1}
                ),
                "align": (
                    ["8", "64"],
                    {"default": "64"}
                ),
            }
        }

    RETURN_TYPES = (
        "INT", "INT", "INT", "INT",
        "INT", "INT", "INT", "INT"
    )
    RETURN_NAMES = (
        "safe_left",
        "safe_right",
        "safe_top",
        "safe_bottom",
        "target_left",
        "target_right",
        "target_top",
        "target_bottom",
    )
    FUNCTION = "compute"
    CATEGORY = "utils/size"

    def compute(
        self,
        expand_left,
        expand_right,
        expand_top,
        expand_bottom,
        align
    ):
        align_value = int(align)

        safe_left = (
            math.ceil(expand_left / align_value) * align_value
            if expand_left > 0 else 0
        )
        safe_right = (
            math.ceil(expand_right / align_value) * align_value
            if expand_right > 0 else 0
        )
        safe_top = (
            math.ceil(expand_top / align_value) * align_value
            if expand_top > 0 else 0
        )
        safe_bottom = (
            math.ceil(expand_bottom / align_value) * align_value
            if expand_bottom > 0 else 0
        )

        return (
            safe_left,
            safe_right,
            safe_top,
            safe_bottom,
            expand_left,
            expand_right,
            expand_top,
            expand_bottom,
        )
