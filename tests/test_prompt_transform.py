"""
Tests for the prompt transformation layer in media_generator.

Verifies that LLM-generated image descriptions are correctly transformed
into SDXL-optimized prompts that avoid text rendering.
"""

import pytest

from src.media_generator import _transform_prompt_for_diffusion


# ── Basic transformation ─────────────────────────────────────────────

class TestPromptTransformBasic:
    """Core prompt transformation behavior."""

    def test_adds_style_prefix(self):
        result = _transform_prompt_for_diffusion("a flower diagram")
        assert result.startswith("flat vector educational illustration")

    def test_adds_quality_suffix(self):
        result = _transform_prompt_for_diffusion("a flower diagram")
        assert "vibrant colors" in result
        assert "sharp lines" in result

    def test_includes_no_text_directives(self):
        result = _transform_prompt_for_diffusion("a simple plant")
        assert "no text" in result
        assert "no labels" in result
        assert "no letters" in result

    def test_preserves_core_subject(self):
        result = _transform_prompt_for_diffusion("a flower with pink petals and green stem")
        assert "flower" in result
        assert "pink petals" in result
        assert "green stem" in result

    def test_retry_style_uses_alternate_prefix(self):
        result = _transform_prompt_for_diffusion("a flower", use_retry_style=True)
        assert result.startswith("colorful storybook illustration")
        assert "flat vector" not in result

    def test_retry_style_still_blocks_text(self):
        result = _transform_prompt_for_diffusion("a flower", use_retry_style=True)
        assert "no text" in result
        assert "no labels" in result


# ── Text stripping ───────────────────────────────────────────────────

class TestTextStripping:
    """Ensures text-rendering instructions are stripped from prompts."""

    def test_strips_labelled(self):
        result = _transform_prompt_for_diffusion(
            "a labelled diagram of a flower"
        )
        assert "labelled" not in result.lower()
        assert "flower" in result

    def test_strips_labeled_american_spelling(self):
        result = _transform_prompt_for_diffusion(
            "a labeled diagram of a plant cell"
        )
        assert "labeled" not in result.lower()
        assert "plant cell" in result

    def test_strips_with_text(self):
        result = _transform_prompt_for_diffusion(
            "a diagram with text showing the parts"
        )
        assert "with text" not in result.lower()

    def test_strips_with_labels(self):
        result = _transform_prompt_for_diffusion(
            "a diagram with labels pointing to each part"
        )
        assert "with labels" not in result.lower()

    def test_strips_with_the_word(self):
        result = _transform_prompt_for_diffusion(
            "a flower with the word 'petal' pointing to each part"
        )
        assert "with the word" not in result.lower()
        assert "'petal'" not in result

    def test_strips_quoted_text(self):
        result = _transform_prompt_for_diffusion(
            'a diagram with "stem" and "root" labels'
        )
        assert '"stem"' not in result
        assert '"root"' not in result

    def test_strips_saying(self):
        result = _transform_prompt_for_diffusion(
            "a banner saying Welcome to Science"
        )
        assert "saying" not in result.lower()

    def test_strips_that_says(self):
        result = _transform_prompt_for_diffusion(
            "a sign that says 'hello'"
        )
        assert "that says" not in result.lower()

    def test_strips_written(self):
        result = _transform_prompt_for_diffusion(
            "a chalkboard with written equations"
        )
        assert "written" not in result.lower()

    def test_strips_write(self):
        result = _transform_prompt_for_diffusion(
            "write the formula on the board"
        )
        assert "write" not in result.lower()


# ── Edge cases ───────────────────────────────────────────────────────

class TestEdgeCases:
    """Edge cases and fallback behavior."""

    def test_empty_string_gets_fallback(self):
        result = _transform_prompt_for_diffusion("")
        assert "educational diagram" in result

    def test_whitespace_only_gets_fallback(self):
        result = _transform_prompt_for_diffusion("   ")
        assert "educational diagram" in result

    def test_all_text_stripped_gets_fallback(self):
        # If removing text instructions leaves nothing meaningful
        result = _transform_prompt_for_diffusion('"hello"')
        assert "educational diagram" in result

    def test_no_double_spaces_after_stripping(self):
        result = _transform_prompt_for_diffusion(
            "a labelled diagram with text showing parts"
        )
        assert "  " not in result

    def test_no_leading_trailing_commas_in_subject(self):
        result = _transform_prompt_for_diffusion(
            ", labelled, "
        )
        # After stripping, should not start/end with bare commas
        # (the style prefix/suffix add their own)
        assert ",," not in result
