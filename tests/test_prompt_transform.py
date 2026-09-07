"""
Tests for the prompt transformation layer in media_generator.

Verifies that LLM-generated image descriptions are correctly transformed
into SD 3.5-optimized prompts that promote clear typography and normalize
quoted labels.
"""

import pytest

from src.media_generator import _transform_prompt_for_diffusion


# ── Basic transformation ─────────────────────────────────────────────

class TestPromptTransformBasic:
    """Core prompt transformation behavior for SD 3.5."""

    def test_adds_style_prefix(self):
        result = _transform_prompt_for_diffusion("a flower diagram")
        assert result.startswith("clear educational diagram illustration")

    def test_adds_quality_suffix(self):
        result = _transform_prompt_for_diffusion("a flower diagram")
        assert "vibrant colors" in result
        assert "sharp typography" in result

    def test_promotes_legible_text(self):
        result = _transform_prompt_for_diffusion("a simple plant")
        assert "legible text labels in English" in result

    def test_no_negative_words_in_positive_prompt(self):
        # Ensure the 'pink elephant' anti-text tokens are NOT in the positive prompt
        result = _transform_prompt_for_diffusion("a simple plant")
        assert "no text" not in result
        assert "no labels" not in result
        assert "no words" not in result

    def test_preserves_core_subject(self):
        result = _transform_prompt_for_diffusion("a flower with pink petals and green stem")
        assert "flower" in result
        assert "pink petals" in result
        assert "green stem" in result

    def test_retry_style_uses_alternate_prefix(self):
        result = _transform_prompt_for_diffusion("a flower", use_retry_style=True)
        assert result.startswith("colorful educational textbook diagram")

    def test_retry_style_includes_bold_labels(self):
        result = _transform_prompt_for_diffusion("a flower", use_retry_style=True)
        assert "clear bold English labels" in result


# ── Label Quote Normalization ────────────────────────────────────────

class TestLabelQuoteNormalization:
    """Ensures label words are formatted in double quotes for SD 3.5's T5 encoder."""

    def test_normalizes_single_quotes(self):
        result = _transform_prompt_for_diffusion(
            "a diagram of a plant with labels 'Flower', 'Stem', 'Roots'"
        )
        assert '"Flower"' in result
        assert '"Stem"' in result
        assert '"Roots"' in result

    def test_normalizes_smart_quotes(self):
        result = _transform_prompt_for_diffusion(
            "a diagram with ‘Petal’ and ‘Leaf’"
        )
        assert '"Petal"' in result
        assert '"Leaf"' in result

    def test_preserves_existing_double_quotes(self):
        result = _transform_prompt_for_diffusion(
            'a diagram with "Heart" and "Lungs"'
        )
        assert '"Heart"' in result
        assert '"Lungs"' in result

    def test_preserves_quoted_multiword_labels(self):
        result = _transform_prompt_for_diffusion(
            "a map with 'North America' and 'South America'"
        )
        assert '"North America"' in result
        assert '"South America"' in result


# ── Edge cases ───────────────────────────────────────────────────────

class TestEdgeCases:
    """Edge cases and fallback behavior."""

    def test_empty_string_gets_fallback(self):
        result = _transform_prompt_for_diffusion("")
        assert "educational diagram with clear labels" in result

    def test_whitespace_only_gets_fallback(self):
        result = _transform_prompt_for_diffusion("   ")
        assert "educational diagram with clear labels" in result

    def test_no_double_spaces(self):
        result = _transform_prompt_for_diffusion(
            "a   diagram   with   extra   spaces"
        )
        assert "  " not in result

    def test_no_leading_trailing_commas_in_subject(self):
        result = _transform_prompt_for_diffusion(", flower, ")
        assert ",," not in result
