"""
Image Annotator — Adds child-friendly educational pill badges to illustrations.

Decouples visual art from typography:
- AI generates a clean, charming storybook illustration (no fuzzy text).
- ImageAnnotator overlays crisp, candy-colored rounded badges with pointer
  lines, guaranteed correct spelling, and zero distortion.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# ── Kid-Friendly Badge Color Palette ─────────────────────────────────
# Soft, vibrant candy pastels that look great against both light and dark art.
BADGE_PALETTE = [
    {"bg": (255, 107, 129), "border": (230, 80, 100), "text": (255, 255, 255)},   # Coral Pink
    {"bg": (38, 166, 154),  "border": (20, 140, 130),  "text": (255, 255, 255)},   # Mint Teal
    {"bg": (255, 183, 77),  "border": (235, 160, 50),  "text": (50, 40, 20)},      # Honey Gold
    {"bg": (149, 117, 205), "border": (125, 95, 185),  "text": (255, 255, 255)},   # Soft Lavender
    {"bg": (79, 195, 247),  "border": (50, 170, 220),  "text": (20, 45, 65)},      # Sky Blue
    {"bg": (129, 199, 132), "border": (100, 175, 105), "text": (25, 50, 25)},      # Fresh Green
]

# Regex patterns to extract intended diagram labels from LLM descriptions
_LABEL_EXTRACT_PATTERNS = [
    # Quoted text: "Petal", 'Stem', ‘Leaf’, “Roots”
    r"""(?i)[`'‘’“"]([a-zA-Z0-9\s-]{1,25})['‘’”"]""",
    # "labels: flower, stem, leaf"
    r"""(?i)labels?:\s*([a-zA-Z0-9,\s-]+)""",
]


def extract_labels(description: str, max_labels: int = 5) -> list[str]:
    """
    Extract diagram labels from an image description.

    Finds quoted words or explicit label listings in the LLM's text.
    Returns a cleaned, deduplicated list of concise labels.
    """
    if not description:
        return []

    found: list[str] = []

    # 1. Check for quoted words first
    quoted = re.findall(r"""[`'‘’“"]([a-zA-Z0-9\s-]{1,25})['‘’”"]""", description)
    for q in quoted:
        cleaned = q.strip().strip(" ,.-")
        # Ignore common non-label quoted words
        if cleaned.lower() not in {"a", "an", "the", "image", "diagram", "illustration", "null"}:
            if cleaned and cleaned not in found:
                found.append(cleaned.capitalize())

    # 2. If no quotes, check for "labels: a, b, c"
    if not found:
        match = re.search(r"""(?i)labels?:\s*([a-zA-Z0-9,\s-]+)""", description)
        if match:
            raw_items = match.group(1).split(",")
            for item in raw_items:
                cleaned = item.strip().strip(" ,.-")
                if cleaned and cleaned not in found:
                    found.append(cleaned.capitalize())

    return found[:max_labels]


def _get_font(size: int = 16) -> ImageFont.ImageFont:
    """Load a clean modern TrueType font, falling back to default."""
    font_candidates = [
        "SegoeUI-Bold.ttf",
        "segoeuib.ttf",
        "arialbd.ttf",
        "Arial.ttf",
        "DejaVuSans-Bold.ttf",
        "DejaVuSans.ttf",
    ]
    for name in font_candidates:
        try:
            return ImageFont.truetype(name, size=size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def overlay_labels(image: Image.Image, labels: list[str]) -> Image.Image:
    """
    Overlay charming, rounded candy-pill badges onto an educational image.

    Badges are positioned along the left and right margins with subtle
    pointer lines and anchor dots pointing toward the central illustration.
    """
    if not labels or not image:
        return image

    img = image.copy().convert("RGBA")
    width, height = img.size

    # Dedicated transparent overlay for crisp anti-aliased drawing
    overlay = ImageDraw.Draw(img)

    font_size = max(14, int(height * 0.032))
    font = _get_font(font_size)

    n = len(labels)
    padding_x = 12
    padding_y = 6
    radius = 12

    # Distribute badges along left (even index) and right (odd index) margins
    # so the central illustration remains clear and visible.
    y_step = height / (n + 1)

    for i, label in enumerate(labels):
        color_scheme = BADGE_PALETTE[i % len(BADGE_PALETTE)]
        bg_color = color_scheme["bg"]
        border_color = color_scheme["border"]
        text_color = color_scheme["text"]

        # Calculate text bounding box
        bbox = font.getbbox(label)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        badge_w = text_w + padding_x * 2
        badge_h = text_h + padding_y * 2

        center_y = int((i + 1) * y_step)
        badge_top = center_y - badge_h // 2
        badge_bottom = badge_top + badge_h

        is_left = (i % 2 == 0)

        if is_left:
            badge_left = 18
            badge_right = badge_left + badge_w
            # Pointer line connects badge right edge toward the center
            pointer_start = (badge_right, center_y)
            pointer_end = (badge_right + 30, center_y)
            dot_center = (badge_right + 33, center_y)
        else:
            badge_right = width - 18
            badge_left = badge_right - badge_w
            # Pointer line connects badge left edge toward the center
            pointer_start = (badge_left, center_y)
            pointer_end = (badge_left - 30, center_y)
            dot_center = (badge_left - 33, center_y)

        # 1. Draw subtle drop shadow for depth
        shadow_box = [badge_left + 2, badge_top + 2, badge_right + 2, badge_bottom + 2]
        overlay.rounded_rectangle(shadow_box, radius=radius, fill=(0, 0, 0, 45))

        # 2. Draw rounded pill badge background
        badge_box = [badge_left, badge_top, badge_right, badge_bottom]
        overlay.rounded_rectangle(
            badge_box,
            radius=radius,
            fill=(*bg_color, 255),
            outline=(*border_color, 255),
            width=2,
        )

        # 3. Draw pointer line with rounded anchor dot
        overlay.line([pointer_start, pointer_end], fill=(*border_color, 230), width=2)
        dot_r = 4
        overlay.ellipse(
            [dot_center[0] - dot_r, dot_center[1] - dot_r,
             dot_center[0] + dot_r, dot_center[1] + dot_r],
            fill=(*bg_color, 255),
            outline=(*border_color, 255),
            width=2,
        )

        # 4. Draw label text inside the pill badge
        text_x = badge_left + padding_x - bbox[0]
        text_y = badge_top + padding_y - bbox[1]
        overlay.text((text_x, text_y), label, font=font, fill=(*text_color, 255))

    return img.convert("RGB")
