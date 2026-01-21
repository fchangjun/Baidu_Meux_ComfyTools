from PIL import Image, ImageFilter
import torch
import numpy as np

class SmartExactResize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),

                "target_width": (
                    "INT",
                    {"default": 1000, "min": 1,"max": 4096, "step": 1}
                ),
                "target_height": (
                    "INT",
                    {"default": 600, "min": 1,"max": 4096, "step": 1}
                ),

                "mode": (
                    ["auto", "crop_only", "pad_only"],
                    {"default": "auto"}
                ),

                "anchor": (
                    ["center", "top", "bottom", "left", "right"],
                    {"default": "center"}
                ),

                "safe_margin_percent": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": 0.0,
                        "max": 0.4,
                        "step": 0.01
                    }
                ),

                "padding_mode": (
                    ["solid_color", "edge_extend", "blur_extend"],
                    {"default": "blur_extend"}
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "image/resize"

    # ----------------------------------------------------
    # Core
    # ----------------------------------------------------

    def process(
        self,
        image,
        target_width,
        target_height,
        mode,
        anchor,
        safe_margin_percent,
        padding_mode
    ):
        output_images = []

        batch_size = image.shape[0]

        for i in range(batch_size):
            pil = Image.fromarray(
                (image[i].cpu().numpy() * 255).astype(np.uint8)
            )
            ow, oh = pil.size

            current_mode = mode

            # AUTO decision
            if current_mode == "auto":
                if ow >= target_width and oh >= target_height:
                    if self._can_crop_safely(
                        ow, oh,
                        target_width, target_height,
                        safe_margin_percent
                    ):
                        current_mode = "crop_only"
                    else:
                        current_mode = "pad_only"
                else:
                    current_mode = "pad_only"

            # Crop
            if current_mode == "crop_only":
                pil = self._safe_crop(
                    pil,
                    target_width,
                    target_height,
                    anchor
                )

            # Pad
            if current_mode == "pad_only":
                pil = self._pad(
                    pil,
                    target_width,
                    target_height,
                    padding_mode
                )

            out = torch.from_numpy(
                np.array(pil)
            ).float() / 255.0

            output_images.append(out)

        # Stack back to batch
        return (torch.stack(output_images),)

    # ----------------------------------------------------
    # Safety check
    # ----------------------------------------------------

    def _can_crop_safely(
        self,
        ow, oh,
        tw, th,
        safe_margin
    ):
        safe_w = ow * (1 - 2 * safe_margin)
        safe_h = oh * (1 - 2 * safe_margin)
        return tw >= safe_w and th >= safe_h

    # ----------------------------------------------------
    # Crop logic
    # ----------------------------------------------------

    def _safe_crop(
        self,
        image,
        tw,
        th,
        anchor
    ):
        ow, oh = image.size

        if anchor == "center":
            x = (ow - tw) // 2
            y = (oh - th) // 2
        elif anchor == "top":
            x = (ow - tw) // 2
            y = 0
        elif anchor == "bottom":
            x = (ow - tw) // 2
            y = oh - th
        elif anchor == "left":
            x = 0
            y = (oh - th) // 2
        elif anchor == "right":
            x = ow - tw
            y = (oh - th) // 2
        else:
            x = (ow - tw) // 2
            y = (oh - th) // 2

        x = max(x, 0)
        y = max(y, 0)

        return image.crop((x, y, x + tw, y + th))

    # ----------------------------------------------------
    # Padding logic
    # ----------------------------------------------------

    def _pad(
        self,
        image,
        tw,
        th,
        padding_mode
    ):
        ow, oh = image.size

        if padding_mode == "blur_extend":
            bg = image.resize((tw, th))
            bg = bg.filter(ImageFilter.GaussianBlur(30))

        elif padding_mode == "edge_extend":
            bg = Image.new("RGB", (tw, th))
        else:
            bg = Image.new("RGB", (tw, th), (0, 0, 0))

        offset_x = (tw - ow) // 2
        offset_y = (th - oh) // 2
        bg.paste(image, (offset_x, offset_y))

        return bg