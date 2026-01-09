import math
import torch


class SmartEmptyLatent:
    """
    Smart Empty Latent
    - Compute safe latent size from target size
    - Always round UP
    - Supports batch generation
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
                )
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("latent", "gen_width", "gen_height")
    FUNCTION = "generate"
    CATEGORY = "latent"

    def generate(
        self,
        target_width,
        target_height,
        batch_size,
        align
    ):
        align_value = int(align)

        # Step 2️⃣ 安全生成尺寸（向上取整）
        gen_width = math.ceil(target_width / align_value) * align_value
        gen_height = math.ceil(target_height / align_value) * align_value

        # Latent 尺寸（SD latent = image / 8）
        latent_w = gen_width // 8
        latent_h = gen_height // 8

        # Create empty latent
        latent = torch.zeros(
            [batch_size, 4, latent_h, latent_w],
            dtype=torch.float32
        )

        return (
            {"samples": latent},
            gen_width,
            gen_height
        )
