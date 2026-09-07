"""
Tests for content evaluator — alignment evaluation and retry logic.
"""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.content_evaluator import (
    build_lesson_context_for_questions,
    evaluate_alignment,
)
from src.models import (
    AlignmentResult,
    ContentSection,
    LessonContent,
    Question,
    QuestionAlignment,
    QuestionType,
)


# ── Fixtures ─────────────────────────────────────────────────────────

def _make_lesson() -> LessonContent:
    """A lesson about roots, stems, and leaves — NOT about flowers."""
    return LessonContent(
        title="The Amazing Parts of a Plant!",
        introduction="Hello! Today we'll learn about roots, stems, and leaves.",
        sections=[
            ContentSection(
                heading="Roots",
                body="Roots grow underground. They suck up water and hold the plant in the soil.",
                imageDescription="A plant with visible roots underground",
                funFact="Some roots can grow 60 metres deep!",
            ),
            ContentSection(
                heading="Stems",
                body="The stem is like a straw. It carries water from the roots to the leaves.",
                imageDescription="A cross-section of a stem showing tubes",
                funFact="Some stems are as thick as a car!",
            ),
            ContentSection(
                heading="Leaves",
                body="Leaves catch sunlight and use it to make food for the plant. This is called photosynthesis.",
                imageDescription="A bright green leaf in the sunshine",
                funFact="Leaves change colour in autumn because they stop making green pigment.",
            ),
        ],
        summary="We learned about roots, stems, and leaves today. Great job!",
    )


def _make_aligned_questions() -> list[Question]:
    """Questions that ARE answerable from the lesson above."""
    return [
        Question(
            id="q1",
            type=QuestionType.MCQ,
            questionText="What do roots do?",
            hint="Think about what's underground!",
            options=["Suck up water", "Make flowers", "Catch sunlight", "Fly"],
            correctOptionIndex=0,
        ),
        Question(
            id="q2",
            type=QuestionType.FILL_BLANK,
            questionText="Fill in the blank!",
            hint="Think about the stem.",
            sentenceWithBlank="The stem carries ___ from the roots to the leaves.",
            correctAnswer="water",
            acceptableAnswers=["water", "Water"],
        ),
    ]


def _make_misaligned_questions() -> list[Question]:
    """Questions that are NOT answerable from the lesson above (asks about flowers)."""
    return [
        Question(
            id="q1",
            type=QuestionType.MCQ,
            questionText="What colour are most flower petals?",
            hint="Think about the garden!",
            options=["Red", "Blue", "Green", "Purple"],
            correctOptionIndex=0,
        ),
        Question(
            id="q2",
            type=QuestionType.MCQ,
            questionText="What do bees collect from flowers?",
            hint="It's sweet!",
            options=["Nectar", "Water", "Soil", "Leaves"],
            correctOptionIndex=0,
        ),
    ]


# ── Tests: evaluate_alignment ────────────────────────────────────────

class TestEvaluateAlignment:
    def test_aligned_content_passes(self):
        """When LLM says all questions are aligned, result.all_aligned is True."""
        mock_ollama = MagicMock()
        mock_ollama.generate_json.return_value = {
            "allAligned": True,
            "questions": [
                {"questionId": "q1", "aligned": True, "reason": "Covered in Section 1."},
                {"questionId": "q2", "aligned": True, "reason": "Covered in Section 2."},
            ],
            "summary": "All questions are properly aligned.",
        }

        lesson = _make_lesson()
        questions = _make_aligned_questions()

        result = evaluate_alignment(lesson, questions, mock_ollama)
        assert result.all_aligned is True
        assert len(result.misaligned_ids) == 0

    def test_misaligned_content_fails(self):
        """When LLM detects misaligned questions, result.all_aligned is False."""
        mock_ollama = MagicMock()
        mock_ollama.generate_json.return_value = {
            "allAligned": False,
            "questions": [
                {"questionId": "q1", "aligned": False, "reason": "Lesson does not cover flowers."},
                {"questionId": "q2", "aligned": False, "reason": "Lesson does not mention bees."},
            ],
            "summary": "Questions ask about flowers but lesson only covers roots/stems/leaves.",
        }

        lesson = _make_lesson()
        questions = _make_misaligned_questions()

        result = evaluate_alignment(lesson, questions, mock_ollama)
        assert result.all_aligned is False
        assert set(result.misaligned_ids) == {"q1", "q2"}

    def test_empty_questions_pass(self):
        """Edge case: no questions should still return aligned."""
        mock_ollama = MagicMock()
        lesson = _make_lesson()

        result = evaluate_alignment(lesson, [], mock_ollama)
        assert result.all_aligned is True
        mock_ollama.generate_json.assert_not_called()

    def test_llm_failure_assumes_aligned(self):
        """If the LLM call fails, we assume aligned to avoid blocking."""
        mock_ollama = MagicMock()
        mock_ollama.generate_json.side_effect = Exception("LLM down")

        lesson = _make_lesson()
        questions = _make_aligned_questions()

        result = evaluate_alignment(lesson, questions, mock_ollama)
        assert result.all_aligned is True  # graceful fallback


# ── Tests: build_lesson_context_for_questions ────────────────────────

class TestBuildLessonContext:
    def test_includes_all_sections(self):
        lesson = _make_lesson()
        context = build_lesson_context_for_questions(lesson)

        assert "Roots" in context
        assert "Stems" in context
        assert "Leaves" in context
        assert "photosynthesis" in context

    def test_includes_fun_facts(self):
        lesson = _make_lesson()
        context = build_lesson_context_for_questions(lesson)

        assert "60 metres" in context

    def test_includes_critical_instruction(self):
        lesson = _make_lesson()
        context = build_lesson_context_for_questions(lesson)

        assert "MUST be answerable" in context

    def test_includes_title_and_summary(self):
        lesson = _make_lesson()
        context = build_lesson_context_for_questions(lesson)

        assert lesson.title in context
        assert "roots, stems, and leaves" in context


# ── Tests: evaluate_model_output ─────────────────────────────────────

from src.content_evaluator import (
    apply_visual_improvements,
    evaluate_model_output,
)
from src.models import (
    ModelOutputEvaluation,
    Subject,
    VisualEvaluation,
)


class TestEvaluateModelOutput:
    def test_instructional_visuals_and_aligned_questions_pass(self):
        mock_ollama = MagicMock()
        mock_ollama.generate_json.return_value = {
            "passed": True,
            "lessonQualityScore": 1.0,
            "visualEvaluations": [
                {
                    "sectionIndex": 0,
                    "heading": "Roots",
                    "imageDescription": "A cross-section diagram showing roots absorbing water from soil particles",
                    "isInstructional": True,
                    "pedagogicalPurpose": "Shows root absorption process",
                    "critique": "Directly explains how roots work with clear diagrammatic visual.",
                    "improvedDescription": None,
                }
            ],
            "questionAlignments": [
                {
                    "questionId": "q1",
                    "aligned": True,
                    "reason": "Covered in Section 1.",
                }
            ],
            "summary": "High quality instructional visuals and aligned questions.",
        }

        lesson = _make_lesson()
        questions = _make_aligned_questions()

        result = evaluate_model_output(lesson, questions, Subject.SCIENCE, "Parts of a plant", mock_ollama)
        assert result.passed is True
        assert len(result.misaligned_question_ids) == 0
        assert len(result.unhelpful_visual_indices) == 0

    def test_decorative_visuals_flagged(self):
        mock_ollama = MagicMock()
        mock_ollama.generate_json.return_value = {
            "passed": False,
            "lessonQualityScore": 0.8,
            "visualEvaluations": [
                {
                    "sectionIndex": 0,
                    "heading": "Roots",
                    "imageDescription": "A cute owl smiling in a forest",
                    "isInstructional": False,
                    "pedagogicalPurpose": "None - decorative mascot",
                    "critique": "Merely decorative cartoon of an owl, does not explain how roots absorb water.",
                    "improvedDescription": "A detailed botanical diagram of root hairs drinking water droplets underground with arrows",
                }
            ],
            "questionAlignments": [
                {
                    "questionId": "q1",
                    "aligned": True,
                    "reason": "Covered in lesson.",
                }
            ],
            "summary": "Decorative visual detected.",
        }

        lesson = _make_lesson()
        questions = _make_aligned_questions()

        result = evaluate_model_output(lesson, questions, Subject.SCIENCE, "Parts of a plant", mock_ollama)
        assert result.passed is False
        assert 0 in result.unhelpful_visual_indices
        assert result.visual_evaluations[0].improved_description is not None

    def test_misaligned_questions_flagged(self):
        mock_ollama = MagicMock()
        mock_ollama.generate_json.return_value = {
            "passed": False,
            "lessonQualityScore": 0.9,
            "visualEvaluations": [
                {
                    "sectionIndex": 0,
                    "heading": "Roots",
                    "imageDescription": "A diagram of roots",
                    "isInstructional": True,
                    "pedagogicalPurpose": "Root diagram",
                    "critique": "Good diagram",
                    "improvedDescription": None,
                }
            ],
            "questionAlignments": [
                {
                    "questionId": "q1",
                    "aligned": False,
                    "reason": "Flower petals are not covered in the lesson.",
                }
            ],
            "summary": "Misaligned question detected.",
        }

        lesson = _make_lesson()
        questions = _make_misaligned_questions()

        result = evaluate_model_output(lesson, questions, Subject.SCIENCE, "Parts of a plant", mock_ollama)
        assert result.passed is False
        assert "q1" in result.misaligned_question_ids


class TestApplyVisualImprovements:
    def test_upgrades_decorative_visual_to_instructional(self):
        lesson = _make_lesson()
        lesson.sections[0].image_description = "A cute owl smiling"

        eval_result = ModelOutputEvaluation(
            passed=False,
            visualEvaluations=[
                VisualEvaluation(
                    sectionIndex=0,
                    heading="Roots",
                    imageDescription="A cute owl smiling",
                    isInstructional=False,
                    pedagogicalPurpose="None",
                    critique="Decorative",
                    improvedDescription="Diagram showing root absorption with blue water arrows",
                )
            ],
            questionAlignments=[],
        )

        improved_lesson = apply_visual_improvements(lesson, eval_result)
        assert (
            improved_lesson.sections[0].image_description
            == "Diagram showing root absorption with blue water arrows"
        )

    def test_removes_decorative_visual_when_no_improved_description(self):
        lesson = _make_lesson()
        lesson.sections[0].image_description = "Decorative border with stars"

        eval_result = ModelOutputEvaluation(
            passed=False,
            visualEvaluations=[
                VisualEvaluation(
                    sectionIndex=0,
                    heading="Roots",
                    imageDescription="Decorative border with stars",
                    isInstructional=False,
                    pedagogicalPurpose="None",
                    critique="Useless decorative border",
                    improvedDescription=None,
                )
            ],
            questionAlignments=[],
        )

        improved_lesson = apply_visual_improvements(lesson, eval_result)
        assert improved_lesson.sections[0].image_description is None

