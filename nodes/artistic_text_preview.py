import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont


class MeuxArtisticTextPreview:
    """
    Preview node that renders styled text on a transparent canvas.
    Supports font size, color overrides, character spacing, bold/italic fallback,
    gradient fill/outline, shadow, inner shadow and glow.
    """

    CATEGORY = "image/text"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "process"
    _FONT_MAP_CACHE = None
    _EFFECT_PRESETS = [
        "none",
        "soft_shadow",
        "heavy_shadow",
        "soft_glow",
        "neon_glow",
        "engraved",
        "shadow_glow",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        fonts = cls._font_names_for_ui()
        return {
            "required": {
                "text": ("STRING", {"default": "Hello\nArt Text", "multiline": True}),
                "font_name": (fonts, {"default": fonts[0] if fonts else "Default"}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                "font_size_mode": (["fixed", "fit"], {"default": "fixed"}),
                "font_size": ("INT", {"default": 128, "min": 8, "max": 1024, "step": 1}),
                "char_spacing": ("INT", {"default": 0, "min": -20, "max": 200, "step": 1}),
                "line_spacing": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                "text_align": (["left", "center", "right"], {"default": "center"}),
                "vertical_align": (["top", "middle", "bottom"], {"default": "middle"}),
                "bold": ("BOOLEAN", {"default": False}),
                "italic": ("BOOLEAN", {"default": False}),
                "background_color": ("STRING", {"default": "#000000"}),
                "background_opacity": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "padding_percent": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.45, "step": 0.01}),
                "fill_type": (["solid", "gradient", "none"], {"default": "solid"}),
                "fill_color_1": ("STRING", {"default": "#ff3b30"}),
                "fill_color_2": ("STRING", {"default": "#ffd84d"}),
                "fill_direction": (
                    ["left_right", "right_left", "top_bottom", "bottom_top", "diagonal", "diagonal_reverse"],
                    {"default": "top_bottom"},
                ),
                "outline_type": (["none", "solid", "gradient"], {"default": "none"}),
                "outline_width": ("INT", {"default": 6, "min": 0, "max": 128, "step": 1}),
                "outline_opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "outline_color_1": ("STRING", {"default": "#ffffff"}),
                "outline_color_2": ("STRING", {"default": "#00d4ff"}),
                "outline_direction": (
                    ["left_right", "right_left", "top_bottom", "bottom_top", "diagonal", "diagonal_reverse"],
                    {"default": "left_right"},
                ),
                "effect_preset": (cls._EFFECT_PRESETS, {"default": "none"}),
                "effect_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "shadow_mode": (["preset", "custom"], {"default": "preset"}),
                "shadow_color": ("STRING", {"default": "#000000"}),
                "shadow_offset_x": ("INT", {"default": 4, "min": -512, "max": 512, "step": 1}),
                "shadow_offset_y": ("INT", {"default": 4, "min": -512, "max": 512, "step": 1}),
                "shadow_opacity": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "shadow_blur": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 128.0, "step": 0.1}),
            },
        }

    def process(
        self,
        text,
        font_name,
        width,
        height,
        font_size_mode,
        font_size,
        char_spacing,
        line_spacing,
        text_align,
        vertical_align,
        bold,
        italic,
        background_color,
        background_opacity,
        padding_percent,
        fill_type="solid",
        fill_color_1="#ff3b30",
        fill_color_2="#ffd84d",
        fill_direction="top_bottom",
        outline_type="none",
        outline_width=6,
        outline_opacity=1.0,
        outline_color_1="#ffffff",
        outline_color_2="#00d4ff",
        outline_direction="left_right",
        effect_preset="none",
        effect_strength=1.0,
        shadow_mode="preset",
        shadow_color="#000000",
        shadow_offset_x=4,
        shadow_offset_y=4,
        shadow_opacity=0.35,
        shadow_blur=6.0,
    ):
        safe_area = self._safe_area(width, height, padding_percent)
        font_path, fake_bold, fake_italic = self._choose_font_file(font_name, bold, italic)
        font = self._load_font(font_path, font_size)
        effect_settings = self._effect_settings(effect_preset, effect_strength)
        effect_settings = self._resolve_shadow_settings(
            effect_settings=effect_settings,
            shadow_mode=shadow_mode,
            shadow_color=shadow_color,
            shadow_offset_x=shadow_offset_x,
            shadow_offset_y=shadow_offset_y,
            shadow_opacity=shadow_opacity,
            shadow_blur=shadow_blur,
        )

        if font_size_mode == "fit":
            fitted_size = self._find_optimal_font_size(text, font_path, safe_area, char_spacing, line_spacing, font_size)
            font = self._load_font(font_path, fitted_size)

        text_mask = self._render_text_mask(
            text=text,
            width=width,
            height=height,
            font=font,
            char_spacing=char_spacing,
            line_spacing=line_spacing,
            text_align=text_align,
            vertical_align=vertical_align,
            safe_area=safe_area,
            fake_bold=fake_bold,
        )

        if fake_italic:
            text_mask = self._apply_fake_italic(text_mask)

        text_image = self._apply_effects(
            text_mask=text_mask,
            width=width,
            height=height,
            fill_type=fill_type,
            fill_color_1=fill_color_1,
            fill_color_2=fill_color_2,
            fill_direction=fill_direction,
            outline_width=outline_width,
            outline_opacity=outline_opacity,
            outline_type=outline_type,
            outline_color_1=outline_color_1,
            outline_color_2=outline_color_2,
            outline_direction=outline_direction,
            effect_settings=effect_settings,
        )

        background = Image.new("RGBA", (width, height), self._hex_to_rgba(background_color, int(background_opacity * 255)))
        final = Image.alpha_composite(background, text_image)
        return (self._pil_to_tensor(final), self._alpha_to_mask(text_image))

    @classmethod
    def _font_names_for_ui(cls):
        fonts = cls._font_map()
        return sorted(fonts.keys()) or ["Default"]

    @classmethod
    def _font_map(cls):
        if cls._FONT_MAP_CACHE is not None:
            return cls._FONT_MAP_CACHE

        font_paths = {}
        for directory in cls._font_search_dirs():
            if not directory.exists():
                continue
            try:
                for path in directory.iterdir():
                    if not path.is_file():
                        continue
                    if path.suffix.lower() not in {".ttf", ".otf", ".ttc"}:
                        continue
                    font_paths.setdefault(path.stem, path)
            except Exception:
                continue
        cls._FONT_MAP_CACHE = font_paths
        return font_paths

    @classmethod
    def _font_search_dirs(cls):
        root = Path(__file__).resolve().parent.parent
        dirs = [
            root / "fonts",
            root / "PIP_ArtisticWords" / "fonts",
            Path("/System/Library/Fonts"),
            Path("/Library/Fonts"),
            Path.home() / "Library" / "Fonts",
        ]
        windir = os.environ.get("WINDIR")
        if windir:
            dirs.append(Path(windir) / "Fonts")
        return dirs

    def _choose_font_file(self, font_name, bold, italic):
        font_map = self._font_map()
        font_path = font_map.get(font_name)

        if font_path is None and font_map:
            for name, path in font_map.items():
                if font_name.lower() in name.lower():
                    font_path = path
                    break

        if font_path is None:
            return (None, bold, italic)

        fake_bold = bold
        fake_italic = italic
        if bold or italic:
            family_key = font_path.stem.lower().split("-")[0]
            keywords = []
            if bold:
                keywords.extend(["bold", "black", "heavy"])
            if italic:
                keywords.extend(["italic", "oblique"])

            for candidate_name, candidate_path in font_map.items():
                lower_name = candidate_name.lower()
                if family_key not in lower_name:
                    continue
                if all(keyword in lower_name for keyword in keywords):
                    font_path = candidate_path
                    fake_bold = False
                    fake_italic = False
                    break

        return (font_path, fake_bold, fake_italic)

    def _load_font(self, font_path, font_size):
        if font_path is None:
            return ImageFont.load_default()
        try:
            return ImageFont.truetype(str(font_path), font_size)
        except Exception:
            return ImageFont.load_default()

    def _safe_area(self, width, height, padding_percent):
        pad_x = int(width * padding_percent)
        pad_y = int(height * padding_percent)
        return (pad_x, pad_y, width - pad_x, height - pad_y)

    def _find_optimal_font_size(self, text, font_path, safe_area, char_spacing, line_spacing, max_size):
        left, top, right, bottom = safe_area
        max_width = max(1, right - left)
        max_height = max(1, bottom - top)
        low = 8
        high = max(8, max_size)
        best = low

        while low <= high:
            mid = (low + high) // 2
            font = self._load_font(font_path, mid)
            resolved_line_spacing = self._resolve_line_spacing(font, line_spacing)
            width, height = self._text_block_size(font, text, char_spacing, resolved_line_spacing)
            if width <= max_width and height <= max_height:
                best = mid
                low = mid + 1
            else:
                high = mid - 1

        return best

    def _char_advance(self, font, char):
        if hasattr(font, "getlength"):
            return float(font.getlength(char))
        bbox = font.getbbox(char)
        return float(bbox[2] - bbox[0])

    def _line_width(self, font, text, char_spacing):
        if not text:
            return 0
        total = 0.0
        for index, char in enumerate(text):
            total += self._char_advance(font, char)
            if index < len(text) - 1:
                total += char_spacing
        return int(round(total))

    def _text_block_size(self, font, text, char_spacing, line_spacing):
        lines = text.splitlines() or [text]
        ascent, descent = font.getmetrics()
        line_height = ascent + descent
        width = max(self._line_width(font, line, char_spacing) for line in lines) if lines else 0
        height = len(lines) * line_height + max(0, len(lines) - 1) * line_spacing
        return width, height

    def _resolve_line_spacing(self, font, line_spacing):
        if int(line_spacing) != 0:
            return int(line_spacing)
        return max(6, font.size // 5 if hasattr(font, "size") else 6)

    def _render_text_mask(
        self, text, width, height, font, char_spacing, line_spacing, text_align, vertical_align, safe_area, fake_bold
    ):
        mask_image = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask_image)

        line_spacing = self._resolve_line_spacing(font, line_spacing)
        _, block_height = self._text_block_size(font, text, char_spacing, line_spacing)

        left, top, right, bottom = safe_area
        start_y = self._block_start_y(block_height, vertical_align, top, bottom)

        offsets = ((0, 0), (1, 0), (0, 1), (1, 1)) if fake_bold else ((0, 0),)
        lines = text.splitlines() or [text]
        ascent, descent = font.getmetrics()
        line_height = ascent + descent
        current_y = start_y

        for line in lines:
            line_width = self._line_width(font, line, char_spacing)
            current_x = self._line_start_x(line_width, text_align, left, right)
            for index, char in enumerate(line):
                for dx, dy in offsets:
                    draw.text((current_x + dx, current_y + dy), char, fill=255, font=font)
                current_x += self._char_advance(font, char)
                if index < len(line) - 1:
                    current_x += char_spacing
            current_y += line_height + line_spacing

        return mask_image

    def _line_start_x(self, line_width, text_align, left, right):
        safe_width = right - left
        if text_align == "left":
            return left
        if text_align == "right":
            return right - line_width
        return left + (safe_width - line_width) / 2

    def _block_start_y(self, block_height, vertical_align, top, bottom):
        safe_height = bottom - top
        if vertical_align == "top":
            return top
        if vertical_align == "bottom":
            return bottom - block_height
        return top + (safe_height - block_height) / 2

    def _apply_fake_italic(self, mask_image, shear=0.25):
        bbox = mask_image.getbbox()
        if bbox is None:
            return mask_image

        cropped = mask_image.crop(bbox)
        width, height = cropped.size
        x_shift = int(round(abs(shear) * height))
        transformed = cropped.transform(
            (width + x_shift, height),
            Image.AFFINE,
            (1, shear, -x_shift if shear > 0 else 0, 0, 1, 0),
            resample=Image.BICUBIC,
        )

        result = Image.new("L", mask_image.size, 0)
        paste_x = max(0, bbox[0] - x_shift // 2)
        paste_y = bbox[1]
        result.paste(transformed, (paste_x, paste_y))
        return result

    def _apply_effects(
        self,
        text_mask,
        width,
        height,
        fill_type,
        fill_color_1,
        fill_color_2,
        fill_direction,
        outline_width,
        outline_opacity,
        outline_type,
        outline_color_1,
        outline_color_2,
        outline_direction,
        effect_settings,
    ):
        result = Image.new("RGBA", (width, height), (0, 0, 0, 0))

        if effect_settings["shadow_enabled"] and effect_settings["shadow_opacity"] > 0:
            shadow_layer = self._make_shadow_layer(
                text_mask,
                effect_settings["shadow_color"],
                effect_settings["shadow_opacity"],
                effect_settings["shadow_offset_x"],
                effect_settings["shadow_offset_y"],
                effect_settings["shadow_blur"],
            )
            result = Image.alpha_composite(result, shadow_layer)

        if effect_settings["glow_enabled"] and effect_settings["glow_opacity"] > 0:
            glow_layer = self._make_glow_layer(
                text_mask,
                effect_settings["glow_color"],
                effect_settings["glow_opacity"],
                effect_settings["glow_blur"],
                effect_settings["glow_intensity"],
            )
            result = Image.alpha_composite(result, glow_layer)

        if fill_type != "none":
            fill_layer = self._make_fill_layer(
                text_mask, width, height, fill_type, fill_color_1, fill_color_1, fill_color_2, fill_direction
            )
            result = Image.alpha_composite(result, fill_layer)

        if outline_type != "none" and outline_width > 0 and outline_opacity > 0:
            outline_layer = self._make_outline_layer(
                text_mask,
                width,
                height,
                outline_width,
                outline_opacity,
                outline_type,
                outline_color_1,
                outline_color_1,
                outline_color_2,
                outline_direction,
            )
            result = Image.alpha_composite(result, outline_layer)

        if effect_settings["inner_shadow_enabled"] and effect_settings["inner_shadow_opacity"] > 0:
            inner_shadow_layer = self._make_inner_shadow_layer(
                text_mask,
                effect_settings["inner_shadow_color"],
                effect_settings["inner_shadow_opacity"],
                effect_settings["inner_shadow_offset_x"],
                effect_settings["inner_shadow_offset_y"],
                effect_settings["inner_shadow_blur"],
            )
            result = Image.alpha_composite(result, inner_shadow_layer)

        return result

    def _effect_settings(self, preset, strength):
        scale = max(0.0, float(strength))
        base = {
            "shadow_enabled": False,
            "shadow_color": "#000000",
            "shadow_opacity": 0.0,
            "shadow_offset_x": 0,
            "shadow_offset_y": 0,
            "shadow_blur": 0.0,
            "inner_shadow_enabled": False,
            "inner_shadow_color": "#000000",
            "inner_shadow_opacity": 0.0,
            "inner_shadow_offset_x": 0,
            "inner_shadow_offset_y": 0,
            "inner_shadow_blur": 0.0,
            "glow_enabled": False,
            "glow_color": "#00d4ff",
            "glow_opacity": 0.0,
            "glow_blur": 0.0,
            "glow_intensity": 1.0,
        }
        if preset == "soft_shadow":
            base.update({
                "shadow_enabled": True,
                "shadow_opacity": min(1.0, 0.35 * scale),
                "shadow_offset_x": int(round(4 * scale)),
                "shadow_offset_y": int(round(4 * scale)),
                "shadow_blur": max(0.0, 6.0 * scale),
            })
        elif preset == "heavy_shadow":
            base.update({
                "shadow_enabled": True,
                "shadow_opacity": min(1.0, 0.6 * scale),
                "shadow_offset_x": int(round(8 * scale)),
                "shadow_offset_y": int(round(8 * scale)),
                "shadow_blur": max(0.0, 10.0 * scale),
            })
        elif preset == "soft_glow":
            base.update({
                "glow_enabled": True,
                "glow_opacity": min(1.0, 0.35 * scale),
                "glow_blur": max(0.1, 10.0 * scale),
                "glow_intensity": max(0.1, 1.0 * scale),
            })
        elif preset == "neon_glow":
            base.update({
                "glow_enabled": True,
                "glow_opacity": min(1.0, 0.55 * scale),
                "glow_blur": max(0.1, 14.0 * scale),
                "glow_intensity": max(0.1, 1.6 * scale),
            })
        elif preset == "engraved":
            base.update({
                "inner_shadow_enabled": True,
                "inner_shadow_opacity": min(1.0, 0.55 * scale),
                "inner_shadow_offset_x": int(round(2 * scale)),
                "inner_shadow_offset_y": int(round(2 * scale)),
                "inner_shadow_blur": max(0.1, 4.0 * scale),
            })
        elif preset == "shadow_glow":
            base.update({
                "shadow_enabled": True,
                "shadow_opacity": min(1.0, 0.4 * scale),
                "shadow_offset_x": int(round(5 * scale)),
                "shadow_offset_y": int(round(5 * scale)),
                "shadow_blur": max(0.1, 7.0 * scale),
                "glow_enabled": True,
                "glow_opacity": min(1.0, 0.3 * scale),
                "glow_blur": max(0.1, 10.0 * scale),
                "glow_intensity": max(0.1, 1.2 * scale),
            })
        return base

    def _resolve_shadow_settings(
        self,
        effect_settings,
        shadow_mode,
        shadow_color,
        shadow_offset_x,
        shadow_offset_y,
        shadow_opacity,
        shadow_blur,
    ):
        settings = dict(effect_settings)
        settings["shadow_color"] = shadow_color
        if shadow_mode == "custom":
            settings.update({
                "shadow_enabled": shadow_opacity > 0,
                "shadow_color": shadow_color,
                "shadow_opacity": max(0.0, min(1.0, float(shadow_opacity))),
                "shadow_offset_x": int(shadow_offset_x),
                "shadow_offset_y": int(shadow_offset_y),
                "shadow_blur": max(0.0, float(shadow_blur)),
            })
        return settings

    def _make_shadow_layer(self, text_mask, color, opacity, offset_x, offset_y, blur):
        shadow_mask = text_mask
        if blur > 0:
            shadow_mask = shadow_mask.filter(ImageFilter.GaussianBlur(radius=blur))
        shadow_mask = ImageChops.offset(shadow_mask, int(offset_x), int(offset_y))
        shadow_mask = self._scale_mask(shadow_mask, opacity)
        layer = Image.new("RGBA", text_mask.size, self._hex_to_rgba(color))
        layer.putalpha(shadow_mask)
        return layer

    def _make_glow_layer(self, text_mask, color, opacity, blur, intensity):
        blurred = text_mask.filter(ImageFilter.GaussianBlur(radius=max(0.1, blur)))
        outside = ImageChops.subtract(blurred, text_mask)
        if intensity != 1.0:
            outside = outside.point(lambda value: max(0, min(255, int(value * intensity))))
        outside = self._scale_mask(outside, opacity)
        layer = Image.new("RGBA", text_mask.size, self._hex_to_rgba(color))
        layer.putalpha(outside)
        return layer

    def _make_fill_layer(self, text_mask, width, height, fill_type, fill_color, color1, color2, direction):
        if fill_type == "solid":
            layer = Image.new("RGBA", (width, height), self._hex_to_rgba(fill_color))
        else:
            layer = self._gradient_image(width, height, color1, color2, direction)
        layer.putalpha(text_mask)
        return layer

    def _make_outline_layer(
        self,
        text_mask,
        width,
        height,
        outline_width,
        outline_opacity,
        outline_type,
        outline_color,
        color1,
        color2,
        direction,
    ):
        dilated = text_mask.copy()
        for _ in range(int(outline_width)):
            dilated = dilated.filter(ImageFilter.MaxFilter(3))
        outline_only = ImageChops.subtract(dilated, text_mask)
        outline_only = self._scale_mask(outline_only, outline_opacity)

        if outline_type == "solid":
            layer = Image.new("RGBA", (width, height), self._hex_to_rgba(outline_color))
        else:
            layer = self._gradient_image(width, height, color1, color2, direction)
        layer.putalpha(outline_only)
        return layer

    def _make_inner_shadow_layer(self, text_mask, color, opacity, offset_x, offset_y, blur):
        blurred = text_mask.filter(ImageFilter.GaussianBlur(radius=max(0.1, blur)))
        shifted = ImageChops.offset(blurred, int(offset_x), int(offset_y))
        inverted = ImageChops.invert(shifted)
        inner_mask = ImageChops.multiply(text_mask, inverted)
        inner_mask = self._scale_mask(inner_mask, opacity)
        layer = Image.new("RGBA", text_mask.size, self._hex_to_rgba(color))
        layer.putalpha(inner_mask)
        return layer

    def _gradient_image(self, width, height, color1, color2, direction):
        rgb1 = self._hex_to_rgba(color1)
        rgb2 = self._hex_to_rgba(color2)
        xs = np.linspace(0.0, 1.0, num=max(1, width), dtype=np.float32)
        ys = np.linspace(0.0, 1.0, num=max(1, height), dtype=np.float32)
        x_grid, y_grid = np.meshgrid(xs, ys)

        if direction == "left_right":
            t = x_grid
        elif direction == "right_left":
            t = 1.0 - x_grid
        elif direction == "top_bottom":
            t = y_grid
        elif direction == "bottom_top":
            t = 1.0 - y_grid
        elif direction == "diagonal":
            t = (x_grid + y_grid) / 2.0
        elif direction == "diagonal_reverse":
            t = ((1.0 - x_grid) + y_grid) / 2.0
        else:
            t = y_grid

        red = (rgb1[0] * (1.0 - t) + rgb2[0] * t).astype(np.uint8)
        green = (rgb1[1] * (1.0 - t) + rgb2[1] * t).astype(np.uint8)
        blue = (rgb1[2] * (1.0 - t) + rgb2[2] * t).astype(np.uint8)
        alpha = np.full((height, width), 255, dtype=np.uint8)
        image = np.stack([red, green, blue, alpha], axis=-1)
        return Image.fromarray(image, "RGBA")

    def _scale_mask(self, mask, opacity):
        alpha_scale = max(0.0, min(1.0, float(opacity)))
        if alpha_scale >= 1.0:
            return mask
        return mask.point(lambda value: max(0, min(255, int(value * alpha_scale))))

    def _hex_to_rgba(self, hex_color, alpha=255):
        color = str(hex_color).strip().lstrip("#")
        if len(color) == 3:
            color = "".join(char * 2 for char in color)
        if len(color) != 6:
            color = "ffffff"
        return (
            int(color[0:2], 16),
            int(color[2:4], 16),
            int(color[4:6], 16),
            max(0, min(255, int(alpha))),
        )

    def _pil_to_tensor(self, image):
        array = np.array(image).astype(np.float32) / 255.0
        return torch.from_numpy(array)[None, ...]

    def _alpha_to_mask(self, image):
        alpha = np.array(image.getchannel("A")).astype(np.float32) / 255.0
        return torch.from_numpy(alpha)[None, ..., None]


__all__ = ["MeuxArtisticTextPreview"]
