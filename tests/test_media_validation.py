"""
Tests for media generator — blank image detection, slugify, and structured paths.
"""
import os
import sys

import pytest
from PIL import Image

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.media_generator import (
    BLANK_IMAGE_STD_THRESHOLD,
    _build_asset_dir,
    _build_url_prefix,
    is_blank_image,
    slugify,
)


# ── slugify ──────────────────────────────────────────────────────────

class TestSlugify:
    def test_simple(self):
        assert slugify("Parts of a plant") == "parts-of-a-plant"

    def test_special_chars(self):
        assert slugify("What is soil made of?") == "what-is-soil-made-of"

    def test_dashes_and_underscores(self):
        assert slugify("Pushes and pulls") == "pushes-and-pulls"

    def test_multiple_spaces(self):
        assert slugify("  extra   spaces  ") == "extra-spaces"

    def test_mixed_case(self):
        assert slugify("SCIENCE") == "science"

    def test_unicode_and_punctuation(self):
        # em-dash, comma, etc.
        assert slugify("Nouns, verbs, and adjectives") == "nouns-verbs-and-adjectives"

    def test_already_slug(self):
        assert slugify("hello-world") == "hello-world"


# ── URL prefix building ─────────────────────────────────────────────

class TestBuildUrlPrefix:
    def test_full_context(self):
        url = _build_url_prefix("SCIENCE", "Plants", "Parts of a plant")
        assert url == "/assets/science/plants/parts-of-a-plant"

    def test_partial_context(self):
        url = _build_url_prefix("MATHS", "Fractions")
        assert url == "/assets/maths/fractions"

    def test_no_context(self):
        url = _build_url_prefix()
        assert url == "/assets"

    def test_subject_only(self):
        url = _build_url_prefix("ENGLISH")
        assert url == "/assets/english"


# ── Blank image detection ───────────────────────────────────────────

class TestIsBlankImage:
    def test_solid_white_is_blank(self):
        img = Image.new("RGB", (64, 64), color=(255, 255, 255))
        assert is_blank_image(img) is True

    def test_solid_black_is_blank(self):
        img = Image.new("RGB", (64, 64), color=(0, 0, 0))
        assert is_blank_image(img) is True

    def test_solid_color_is_blank(self):
        img = Image.new("RGB", (64, 64), color=(128, 64, 200))
        assert is_blank_image(img) is True

    def test_varied_image_is_not_blank(self):
        import numpy as np
        # Create a deliberately noisy image
        rng = np.random.default_rng(42)
        arr = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        assert is_blank_image(img) is False

    def test_gradient_is_not_blank(self):
        import numpy as np
        # Horizontal gradient — has high variance
        arr = np.zeros((64, 256, 3), dtype=np.uint8)
        for x in range(256):
            arr[:, x, :] = x
        img = Image.fromarray(arr)
        assert is_blank_image(img) is False


# ── Structured directory building ────────────────────────────────────

class TestBuildAssetDir:
    def test_creates_subdirectory(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.media_generator.ASSETS_DIR", str(tmp_path))

        result = _build_asset_dir("SCIENCE", "Plants", "Parts of a plant")
        expected = os.path.join(str(tmp_path), "science", "plants", "parts-of-a-plant")
        assert result == expected
        assert os.path.isdir(expected)

    def test_no_context_returns_root(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.media_generator.ASSETS_DIR", str(tmp_path))

        result = _build_asset_dir()
        assert result == str(tmp_path)
