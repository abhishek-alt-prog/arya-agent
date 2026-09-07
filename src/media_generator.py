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
ASSETS_DIR = "/Users/abhishek/Code/Arya/bff/src/main/resources/static/assets"
BFF_ASSETS_URL_PREFIX = "/assets"

# Create the directory if it doesn't exist
os.makedirs(ASSETS_DIR, exist_ok=True)

# ── Validation thresholds ────────────────────────────────────────────
# An image is considered "blank" if its pixel standard deviation is below
# this threshold (i.e. nearly uniform color).
BLANK_IMAGE_STD_THRESHOLD = 5.0

# Maximum retries for image generation when a blank is detected.
MAX_IMAGE_RETRIES = 2

# Inference steps — higher = better quality, slower.  30 is a good
# balance for SD 1.5 on Apple Silicon.
DEFAULT_INFERENCE_STEPS = 30


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


class MediaGenerator:
    def __init__(self):
        self.image_pipeline = None

    def _load_image_pipeline(self):
        """Lazy load the Stable Diffusion pipeline to save memory."""
        if self.image_pipeline is None:
            logger.info("Loading Stable Diffusion pipeline...")
            from diffusers import AutoPipelineForText2Image
            import torch

            # Use CPU or MPS (Apple Silicon) if available
            device = "mps" if torch.backends.mps.is_available() else "cpu"

            # IMPORTANT: Always use float32 on MPS — float16 causes blank /
            # black images on Apple Silicon due to numerical instability.
            self.image_pipeline = AutoPipelineForText2Image.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float32,
                safety_checker=None,  # avoid unnecessary memory usage
            )
            self.image_pipeline = self.image_pipeline.to(device)
            # Reduce memory usage
            self.image_pipeline.enable_attention_slicing()
            logger.info("Stable Diffusion pipeline loaded on %s (float32).", device)

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

        Returns the URL path to the generated image, or None if generation
        failed after retries (so the lesson doesn't link a broken image).
        """
        self._load_image_pipeline()
        logger.info("Generating image for prompt: %s", prompt)

        # Use the model's exact generated description directly
        full_prompt = prompt.strip()
        negative_prompt = "blurry, distorted, deformed, low quality, dark"

        asset_dir = _build_asset_dir(subject, unit_name, topic_name)
        url_prefix = _build_url_prefix(subject, unit_name, topic_name)

        for attempt in range(1, MAX_IMAGE_RETRIES + 1):
            logger.info("Image generation attempt %d/%d", attempt, MAX_IMAGE_RETRIES)

            image = self.image_pipeline(
                full_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=DEFAULT_INFERENCE_STEPS,
            ).images[0]

            if not is_blank_image(image):
                # Good image — save and return
                filename = f"{prefix}_{uuid4().hex[:8]}.png"
                filepath = os.path.join(asset_dir, filename)
                image.save(filepath)
                logger.info("Saved image to %s", filepath)
                return f"{url_prefix}/{filename}"

            # Blank image — modify prompt for next attempt
            logger.warning(
                "Attempt %d produced a blank image, retrying with enhanced prompt",
                attempt,
            )
            full_prompt = (
                f"clear detailed educational textbook diagram for children, {prompt}, "
                f"highly detailed, sharp, clean vector illustration, informative visual aid"
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
