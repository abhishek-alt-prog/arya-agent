"""
Tests for the prompt transformation layer in media_generator.

Verifies that LLM-generated image descriptions are transformed into
clean, kid-friendly storybook illustration prompts while stripping out
label text instructions so the AI generates clean art without pseudo-text.
"""

import pytest

from src.media_generator import _transform_prompt_for_diffusion


# ── Basic transformation ─────────────────────────────────────────────

class TestPromptTransformBasic:
    """Core prompt transformation behavior for charming storybook art."""

    def test_adds_storybook_style_prefix(self):
        result = _transform_prompt_for_diffusion("a friendly owl")
        assert result.startswith("whimsical children's book illustration")

    def test_adds_quality_suffix(self):
        result = _transform_prompt_for_diffusion("a friendly owl")
        assert "kid-friendly" in result
        assert "charming" in result

    def test_preserves_core_subject(self):
        result = _transform_prompt_for_diffusion("a flower with pink petals and green stem")
        assert "flower" in result
        assert "pink petals" in result
        assert "green stem" in result

    def test_retry_style_uses_alternate_prefix(self):
        result = _transform_prompt_for_diffusion("a flower", use_retry_style=True)
        assert result.startswith("colorful 3d claymation storybook illustration")


# ── Label Stripping from Diffusion Prompt ─────────────────────────────

class TestLabelStrippingForDiffusion:
    """Ensures label instructions are removed from the AI prompt so the AI doesn't draw pseudo-text."""

    def test_strips_quoted_labels(self):
        result = _transform_prompt_for_diffusion(
            'a diagram of a plant with labels "Flower", "Stem", "Roots"'
        )
        assert '"Flower"' not in result
        assert '"Stem"' not in result
        assert '"Roots"' not in result
        assert "plant" in result

    def test_strips_single_quoted_labels(self):
        result = _transform_prompt_for_diffusion(
            "a diagram of a plant with labels 'Petal', 'Leaf'"
        )
        assert "'Petal'" not in result
        assert "'Leaf'" not in result
        assert "plant" in result

    def test_strips_labels_colon_phrase(self):
        result = _transform_prompt_for_diffusion(
            "a flower. Labels: petal, stem, root"
        )
        assert "Labels:" not in result
        assert "flower" in result


# ── Edge cases ───────────────────────────────────────────────────────

class TestEdgeCases:
    """Edge cases and fallback behavior."""

    def test_empty_string_gets_fallback(self):
        result = _transform_prompt_for_diffusion("")
        assert "cute educational illustration for children" in result

    def test_whitespace_only_gets_fallback(self):
        result = _transform_prompt_for_diffusion("   ")
        assert "cute educational illustration for children" in result

    def test_all_labels_stripped_gets_fallback(self):
        result = _transform_prompt_for_diffusion('"Leaf"')
        assert "cute educational illustration for children" in result

    def test_no_double_spaces(self):
        result = _transform_prompt_for_diffusion(
            "a   diagram   with   extra   spaces"
        )
        assert "  " not in result

    def test_no_leading_trailing_commas_in_subject(self):
        result = _transform_prompt_for_diffusion(", flower, ")
        assert ",," not in result
