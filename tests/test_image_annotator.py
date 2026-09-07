"""
Tests for image_annotator module — label extraction and candy-pill badge overlay.
"""
import pytest
from PIL import Image

from src.image_annotator import (
    BADGE_PALETTE,
    extract_labels,
    overlay_labels,
)


# ── Label Extraction ─────────────────────────────────────────────────

class TestExtractLabels:
    def test_extracts_double_quoted_labels(self):
        desc = 'A plant with labels "Flower", "Stem", "Leaf", "Roots"'
        labels = extract_labels(desc)
        assert labels == ["Flower", "Stem", "Leaf", "Roots"]

    def test_extracts_single_quoted_labels(self):
        desc = "A diagram showing 'Petal', 'Pollen', and 'Stem'"
        labels = extract_labels(desc)
        assert labels == ["Petal", "Pollen", "Stem"]

    def test_extracts_smart_quoted_labels(self):
        desc = "An illustration of a plant with ‘Leaf’ and “Root”"
        labels = extract_labels(desc)
        assert labels == ["Leaf", "Root"]

    def test_extracts_colon_separated_labels(self):
        desc = "Educational diagram of a flower. Labels: petal, stem, root"
        labels = extract_labels(desc)
        assert labels == ["Petal", "Stem", "Root"]

    def test_returns_empty_when_no_labels(self):
        desc = "A cute little owl reading a book under a tree"
        labels = extract_labels(desc)
        assert labels == []

    def test_caps_at_max_labels(self):
        desc = '"One", "Two", "Three", "Four", "Five", "Six", "Seven"'
        labels = extract_labels(desc, max_labels=5)
        assert len(labels) == 5

    def test_ignores_common_non_label_words(self):
        desc = 'This is "an" "image" with labels "Petal" and "Stem"'
        labels = extract_labels(desc)
        assert "An" not in labels
        assert "Image" not in labels
        assert "Petal" in labels
        assert "Stem" in labels

    def test_handles_empty_or_none(self):
        assert extract_labels("") == []
        assert extract_labels(None) == []


# ── Badge Overlay ────────────────────────────────────────────────────

class TestOverlayLabels:
    def test_overlay_preserves_dimensions(self):
        img = Image.new("RGB", (512, 512), color=(240, 240, 240))
        result = overlay_labels(img, ["Petal", "Stem", "Roots"])
        assert result.size == (512, 512)
        assert result.mode == "RGB"

    def test_overlay_modifies_pixels(self):
        # A plain white image should have different pixels after drawing badges
        img = Image.new("RGB", (512, 512), color=(255, 255, 255))
        result = overlay_labels(img, ["Flower", "Leaf"])
        assert img.tobytes() != result.tobytes()

    def test_overlay_empty_labels_returns_original(self):
        img = Image.new("RGB", (100, 100), color=(200, 200, 200))
        result = overlay_labels(img, [])
        assert img.tobytes() == result.tobytes()

    def test_overlay_cycles_palette_for_many_labels(self):
        img = Image.new("RGB", (600, 600), color=(250, 250, 250))
        labels = ["One", "Two", "Three", "Four", "Five", "Six"]
        result = overlay_labels(img, labels)
        assert result.size == (600, 600)
