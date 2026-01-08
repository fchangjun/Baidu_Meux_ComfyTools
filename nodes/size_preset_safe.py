import math


class SizePresetSafe:
    """
    Size Preset (Safe Generate Size)

    - Compute safe generation size from target size
    - Always round UP
    - No latent, no image processing
    - Acts as single source of truth for size
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_width": (
                    "INT",
                    {"default": 1000, "min": 1, "step": 1}
                ),
                "target_height": (
                    "INT",
                    {"default": 600, "min": 1, "step": 1}
                ),
                "batch_size": (
                    "INT",
                    {"default": 1, "min": 1, "step": 1}
                ),
                "align": (
                    ["8", "64"],
                    {"default": "64"}
                ),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = (
        "gen_width",
        "gen_height",
        "batch_size",
        "target_width",
        "target_height",
    )
    FUNCTION = "compute"
    CATEGORY = "utils/size"

    def compute(
        self,
        target_width,
        target_height,
        batch_size,
        align
    ):
        align_value = int(align)

        # 👇 你的核心逻辑：向上取整
        gen_width = math.ceil(target_width / align_value) * align_value
        gen_height = math.ceil(target_height / align_value) * align_value

        return (
            gen_width,
            gen_height,
            batch_size,
            target_width,
            target_height,
        )
