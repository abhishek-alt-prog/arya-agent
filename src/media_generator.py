"""
Media Generator — Handles local generation of Images (Stable Diffusion) and Audio (Edge TTS).

Assets are organized into structured subdirectories:
  /assets/{subject}/{unit_slug}/{topic_slug}/{type}_{short_id}.{ext}
"""
from __future__ import annotations

import os
import re
import asyncio
import logging
from typing import Optional
from uuid import uuid4

import numpy as np
from PIL import Image

# Setup logging
logger = logging.getLogger(__name__)

# Output directory for BFF static assets
DEFAULT_ASSETS_DIR = (
    "/Users/abhishek/Code/Arya/bff/src/main/resources/static/assets"
    if os.path.exists("/Users/abhishek/Code/Arya/bff/src/main/resources/static/assets")
    else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
)
ASSETS_DIR = os.getenv("ASSETS_DIR", DEFAULT_ASSETS_DIR)
BFF_ASSETS_URL_PREFIX = "/assets"

# Create the directory if it doesn't exist
try:
    os.makedirs(ASSETS_DIR, exist_ok=True)
except OSError:
    pass

# ── Validation thresholds ────────────────────────────────────────────
# An image is considered "blank" if its pixel standard deviation is below
# this threshold (i.e. nearly uniform color).
BLANK_IMAGE_STD_THRESHOLD = 5.0

# Maximum retries for image generation when a blank is detected.
MAX_IMAGE_RETRIES = 2


def slugify(text: str) -> str:
    """Convert text to a filesystem-safe slug (lowercase, hyphens, no special chars)."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)        # remove non-word chars
    text = re.sub(r"[\s_]+", "-", text)          # collapse whitespace / underscores
    text = re.sub(r"-{2,}", "-", text)           # collapse multiple hyphens
    return text.strip("-")


def _build_asset_dir(subject: str = "", unit_name: str = "", topic_name: str = "") -> str:
    """
    Build a structured subdirectory path under ASSETS_DIR.

    Examples:
        ("SCIENCE", "Plants", "Parts of a plant")
        → /assets/science/plants/parts-of-a-plant/

        If no context is given, falls back to the root ASSETS_DIR.
    """
    parts = [ASSETS_DIR]
    if subject:
        parts.append(slugify(subject))
    if unit_name:
        parts.append(slugify(unit_name))
    if topic_name:
        parts.append(slugify(topic_name))

    dir_path = os.path.join(*parts)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def _build_url_prefix(subject: str = "", unit_name: str = "", topic_name: str = "") -> str:
    """Build the URL prefix that mirrors the directory structure."""
    parts = [BFF_ASSETS_URL_PREFIX]
    if subject:
        parts.append(slugify(subject))
    if unit_name:
        parts.append(slugify(unit_name))
    if topic_name:
        parts.append(slugify(topic_name))
    return "/".join(parts)


def is_blank_image(image: "Image.Image") -> bool:
    """
    Detect if a PIL Image is blank (single colour or nearly uniform).

    Converts to a numpy array and checks if the spatial standard
    deviation across pixels is below a threshold. A real illustration
    will have far higher variance than an empty / solid-colour image.
    """
    arr = np.array(image, dtype=np.float32)
    if arr.ndim == 3:
        # Spatial standard deviation across height and width for each channel
        spatial_std = float(np.std(arr, axis=(0, 1)).max())
    else:
        spatial_std = float(arr.std())

    is_blank = bool(spatial_std < BLANK_IMAGE_STD_THRESHOLD)
    if is_blank:
        logger.warning(
            "Blank image detected (spatial_std=%.2f < threshold %.2f)",
            spatial_std, BLANK_IMAGE_STD_THRESHOLD,
        )
    return is_blank


# ── Prompt transformation ────────────────────────────────────────────
# Style prefix injected before every prompt to steer SDXL Turbo toward
# clean educational visuals without text rendering.
_STYLE_PREFIX = (
    "flat vector educational illustration for children, "
    "clean simple design, no text, no labels, no letters, no words, no writing, "
    "no numbers, no captions, no annotations, "
)

# Quality suffix appended after the subject matter.
_QUALITY_SUFFIX = (
    ", vibrant colors, white background, sharp lines, "
    "high quality digital art, simple shapes"
)

# Alternative style tokens used on retry to get a different result.
_RETRY_STYLE_PREFIX = (
    "colorful storybook illustration for young children, "
    "cute friendly style, no text, no labels, no letters, no words, "
    "no writing, no numbers, no captions, "
)

# Phrases in LLM descriptions that ask for text rendering — strip them.
_TEXT_INSTRUCTION_PATTERNS = [
    r"(?i)\blabel{1,2}ed?\b",
    r"(?i)\bwith\s+(the\s+)?(word|text|label|letter|number|caption|annotation|title)s?\b",
    r"(?i)\bsaying\b",
    r"(?i)\bthat\s+(says?|reads?)\b",
    r"(?i)\b(word|text|label|caption|annotation|title)s?\s+(showing|reading|saying)\b",
    r"(?i)\bwrite\b",
    r"(?i)\bwritten\b",
    r"""(?i)["'][^"']{1,30}["']""",  # quoted text like "petal" or 'petal'
]


def _transform_prompt_for_diffusion(
    raw_description: str,
    *,
    use_retry_style: bool = False,
) -> str:
    """
    Transform an LLM-generated image description into an SDXL-optimized prompt.

    This does three things:
    1. Strips instructions that ask the model to render text (labels, captions, etc.)
    2. Prepends style tokens for clean educational illustrations
    3. Appends quality tokens for sharp, vibrant output

    The retry variant uses a different visual style to get varied results on
    blank-image retries.
    """
    cleaned = raw_description.strip()

    # Remove text-rendering instructions
    for pattern in _TEXT_INSTRUCTION_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned)

    # Collapse whitespace left behind by removals
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,.")

    if not cleaned:
        cleaned = "educational diagram"

    prefix = _RETRY_STYLE_PREFIX if use_retry_style else _STYLE_PREFIX
    return f"{prefix}{cleaned}{_QUALITY_SUFFIX}"


class MediaGenerator:
    def __init__(self):
        self.image_pipeline = None

    def _load_image_pipeline(self):
        """Lazy load the SDXL Turbo pipeline to save memory."""
        if self.image_pipeline is None:
            from .config import SD_MODEL_ID
            logger.info("Loading Stable Diffusion pipeline (%s)...", SD_MODEL_ID)
            from diffusers import AutoPipelineForText2Image
            import torch

            # Use CPU or MPS (Apple Silicon) if available
            device = "mps" if torch.backends.mps.is_available() else "cpu"

            # IMPORTANT: Always use float32 on MPS — float16 causes blank /
            # black images on Apple Silicon due to numerical instability.
            self.image_pipeline = AutoPipelineForText2Image.from_pretrained(
                SD_MODEL_ID,
                torch_dtype=torch.float32,
                safety_checker=None,  # avoid unnecessary memory usage
            )
            self.image_pipeline = self.image_pipeline.to(device)
            # Reduce memory usage
            self.image_pipeline.enable_attention_slicing()
            logger.info(
                "Stable Diffusion pipeline loaded on %s (float32): %s",
                device, SD_MODEL_ID,
            )

    def generate_image(
        self,
        prompt: str,
        prefix: str = "img",
        subject: str = "",
        unit_name: str = "",
        topic_name: str = "",
    ) -> Optional[str]:
        """
        Generates an image from a text prompt and saves it to the structured
        static assets folder.

        The raw LLM description is transformed into an SD-optimized prompt
        that avoids text rendering and uses educational illustration style tokens.

        Returns the URL path to the generated image, or None if generation
        failed after retries (so the lesson doesn't link a broken image).
        """
        from .config import SD_INFERENCE_STEPS

        self._load_image_pipeline()
        logger.info("Generating image for prompt: %s", prompt)

        # Transform the LLM description into an SD-optimized prompt
        full_prompt = _transform_prompt_for_diffusion(prompt)
        logger.debug("Transformed SD prompt: %s", full_prompt)

        asset_dir = _build_asset_dir(subject, unit_name, topic_name)
        url_prefix = _build_url_prefix(subject, unit_name, topic_name)

        for attempt in range(1, MAX_IMAGE_RETRIES + 1):
            logger.info("Image generation attempt %d/%d", attempt, MAX_IMAGE_RETRIES)

            # SDXL Turbo: guidance_scale=0.0 (distilled without CFG),
            # no negative_prompt (incompatible with turbo distillation).
            image = self.image_pipeline(
                full_prompt,
                num_inference_steps=SD_INFERENCE_STEPS,
                guidance_scale=0.0,
            ).images[0]

            if not is_blank_image(image):
                # Good image — save and return
                filename = f"{prefix}_{uuid4().hex[:8]}.png"
                filepath = os.path.join(asset_dir, filename)
                image.save(filepath)
                logger.info("Saved image to %s", filepath)
                return f"{url_prefix}/{filename}"

            # Blank image — switch to retry style for next attempt
            logger.warning(
                "Attempt %d produced a blank image, retrying with alternate style",
                attempt,
            )
            full_prompt = _transform_prompt_for_diffusion(
                prompt, use_retry_style=True,
            )

        logger.error(
            "Image generation failed after %d attempts for prompt: %s",
            MAX_IMAGE_RETRIES, prompt,
        )
        return None

    def generate_audio(
        self,
        text: str,
        prefix: str = "audio",
        subject: str = "",
        unit_name: str = "",
        topic_name: str = "",
    ) -> str | None:
        """
        Generates speech from text using edge-tts and saves it to the
        structured static assets folder.

        Returns the URL path to the generated audio, or None on failure.
        """
        if not text:
            return None

        logger.info("Generating audio for text (length %d)...", len(text))
        import edge_tts

        asset_dir = _build_asset_dir(subject, unit_name, topic_name)
        url_prefix = _build_url_prefix(subject, unit_name, topic_name)

        filename = f"{prefix}_{uuid4().hex[:8]}.mp3"
        filepath = os.path.join(asset_dir, filename)

        # Voice selection: "en-US-AriaNeural" — fitting for Arya!
        voice = "en-US-AriaNeural"

        async def _generate():
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(filepath)

        # Run async in sync context
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                loop.run_until_complete(_generate())
            else:
                loop.run_until_complete(_generate())
        except RuntimeError:
            # If no event loop
            asyncio.run(_generate())
        except Exception as e:
            logger.error("Audio generation failed: %s", e)
            return None

        logger.info("Saved audio to %s", filepath)
        return f"{url_prefix}/{filename}"

    def unload_models(self):
        """Free up memory."""
        if self.image_pipeline is not None:
            del self.image_pipeline
            self.image_pipeline = None
            logger.info("Unloaded Stable Diffusion pipeline.")
