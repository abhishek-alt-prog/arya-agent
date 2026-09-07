"""
Content Evaluator — validates model output: pedagogical visual utility & lesson-assessment alignment.

Ensures that:
1. Visual descriptions (images) provide genuine educational value (diagrams,
   comparisons, models, processes) rather than decorative filler/clip art.
2. Assessment questions are strictly answerable from the lesson content alone.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from .config import OLLAMA_EVAL_MODEL
from .models import (
    AlignmentResult,
    LessonContent,
    ModelOutputEvaluation,
    Question,
    QuestionAlignment,
    Subject,
    VisualEvaluation,
)

if TYPE_CHECKING:
    from .ollama_client import OllamaClient

logger = logging.getLogger(__name__)

# Maximum retries for the alignment/evaluation loop.
MAX_ALIGNMENT_RETRIES = 2

# ── System prompts ───────────────────────────────────────────────────

MODEL_EVAL_SYSTEM_PROMPT = """\
You are an expert primary-school curriculum evaluator and pedagogical visual designer.
Your job is to rigorously evaluate generated educational content for a 7-year-old child.

CRITERIA:
1. EDUCATIONAL VISUAL UTILITY (CRITICAL):
   Images in lessons MUST serve a direct teaching purpose. They must visually explain,
   diagram, or demonstrate the specific concept taught in the section (e.g. labeled
   parts of a plant, process flow with arrows, place-value block groupings, countable
   items, fraction pies, side-by-side comparisons).
   - Flag as UNHELPFUL / DECORATIVE (isInstructional=false) if the image is merely:
     * A cute mascot (like an owl) smiling, waving, or holding a book
     * Generic cartoon scenery without explanatory value
     * Decorative background or clip art that doesn't explain the concept
   - For any unhelpful image, provide an 'improvedDescription': a specific, concrete,
     informative diagram or visual model that actively teaches the concept.
     If the section does not benefit from a visual, set improvedDescription to null.

2. QUESTION-LESSON ALIGNMENT:
   Every quiz question MUST be answerable using ONLY the provided lesson text.
   If a question asks about facts, vocabulary, or concepts NOT explicitly taught
   in the lesson, it is MISALIGNED (aligned=false).

You will respond ONLY with valid JSON."""

ALIGNMENT_EVAL_SYSTEM_PROMPT = """\
You are a meticulous curriculum quality-assurance reviewer.
Your ONLY job is to check whether each quiz question can be answered
using ONLY the information in the provided lesson content.

A question is ALIGNED if a student who studied ONLY the lesson text
would have enough information to answer it correctly.

A question is MISALIGNED if it requires knowledge that is NOT present
in the lesson — even if the knowledge is generally true or common sense.

You will respond ONLY with valid JSON."""


def _summarise_lesson(content: LessonContent) -> str:
    """
    Flatten lesson content into a plain-text summary for the LLM evaluator.
    Includes title, introduction, all section bodies, fun facts, and the summary.
    """
    parts = [
        f"TITLE: {content.title}",
        f"INTRODUCTION: {content.introduction}",
    ]
    for i, section in enumerate(content.sections, 1):
        parts.append(f"\nSECTION {i} — {section.heading}:")
        parts.append(f"Body: {section.body}")
        if section.image_description:
            parts.append(f"Image Description: {section.image_description}")
        if section.fun_fact:
            parts.append(f"Fun fact: {section.fun_fact}")
    if content.summary:
        parts.append(f"\nSUMMARY: {content.summary}")
    return "\n".join(parts)


def _summarise_questions(questions: list[Question]) -> str:
    """Flatten questions into plain text for the evaluator."""
    lines = []
    for q in questions:
        line = f"[{q.id}] ({q.type.value}) {q.question_text}"
        if q.options:
            line += f"  Options: {q.options}"
        if q.sentence_with_blank:
            line += f"  Sentence: {q.sentence_with_blank}"
        if q.correct_answer:
            line += f"  Answer: {q.correct_answer}"
        if q.correct_option_index is not None and q.options:
            line += f"  Correct: {q.options[q.correct_option_index]}"
        lines.append(line)
    return "\n".join(lines)


def evaluate_model_output(
    lesson_content: LessonContent,
    questions: list[Question],
    subject: Subject,
    topic: str,
    ollama: "OllamaClient",
    eval_model: str | None = None,
) -> ModelOutputEvaluation:
    """
    Comprehensive evaluation of both educational visual utility and question alignment.

    Validates:
    - That images are instructional diagrams/visual aids (not decorative fluff).
    - That questions are strictly answerable from the lesson.
    """
    lesson_text = _summarise_lesson(lesson_content)
    question_text = _summarise_questions(questions)

    sections_info = []
    for i, s in enumerate(lesson_content.sections):
        sections_info.append(
            f"Section {i} ('{s.heading}'): imageDescription = {s.image_description or 'None'}"
        )
    sections_str = "\n".join(sections_info)

    prompt = f"""Evaluate this generated lesson and assessment for a 7-year-old on "{topic}" ({subject.value}).

=== LESSON CONTENT & VISUALS ===
{lesson_text}

=== SECTIONS TO EVALUATE FOR VISUAL UTILITY ===
{sections_str}

=== QUIZ QUESTIONS ===
{question_text}

Evaluate each section's visual utility and each quiz question's alignment.
Return a JSON object with this exact structure:
{{
  "passed": true/false,
  "lessonQualityScore": 1.0,
  "visualEvaluations": [
    {{
      "sectionIndex": 0,
      "heading": "Section Heading",
      "imageDescription": "Original description or null",
      "isInstructional": true/false,
      "pedagogicalPurpose": "What concept does this illustrate?",
      "critique": "Why is this helpful or why is it merely decorative?",
      "improvedDescription": "A concrete explanatory diagram prompt if decorative/unhelpful, or null if fine"
    }}
  ],
  "questionAlignments": [
    {{
      "questionId": "q1",
      "aligned": true/false,
      "reason": "Explanation of alignment"
    }}
  ],
  "summary": "Overall assessment summary"
}}

RULES:
- If an image description is generic or decorative (e.g. 'A cute cartoon owl smiling'), mark isInstructional: false, and in improvedDescription provide an educational visual aid (e.g. 'A clearly labeled diagram showing...').
- If all questions are aligned AND all visuals are instructional (or improved), set passed: true."""

    try:
        data = ollama.generate_json(
            prompt,
            system=MODEL_EVAL_SYSTEM_PROMPT,
            model=eval_model or OLLAMA_EVAL_MODEL,
        )
        result = ModelOutputEvaluation.model_validate(data)

        # Compute derived fields
        result.misaligned_question_ids = [
            q.question_id for q in result.question_alignments if not q.aligned
        ]
        result.unhelpful_visual_indices = [
            v.section_index for v in result.visual_evaluations if not v.is_instructional
        ]
        result.passed = (
            len(result.misaligned_question_ids) == 0
            and len(result.unhelpful_visual_indices) == 0
        )

        logger.info(
            "Model output evaluation: %s (questions %d/%d aligned, visuals %d/%d instructional)",
            "PASS" if result.passed else "NEEDS_REVISION",
            len(questions) - len(result.misaligned_question_ids),
            len(questions),
            len(result.visual_evaluations) - len(result.unhelpful_visual_indices),
            len(result.visual_evaluations),
        )
        return result

    except Exception as exc:
        logger.error("Model output evaluation failed: %s", exc)
        return ModelOutputEvaluation(
            passed=True,
            summary=f"Evaluation failed with exception: {exc}; accepting content.",
        )


def apply_visual_improvements(
    lesson_content: LessonContent,
    eval_result: ModelOutputEvaluation,
) -> LessonContent:
    """
    Upgrade lesson visuals based on the evaluation result:
    If a visual was deemed unhelpful/decorative and the evaluator provided
    an improved educational description, apply it.
    If it was unhelpful and no improvement was provided, remove it (None).
    """
    for v_eval in eval_result.visual_evaluations:
        idx = v_eval.section_index
        if 0 <= idx < len(lesson_content.sections):
            if not v_eval.is_instructional:
                if v_eval.improved_description:
                    logger.info(
                        "Upgrading section %d visual from decorative to instructional: %s",
                        idx, v_eval.improved_description[:80],
                    )
                    lesson_content.sections[idx].image_description = v_eval.improved_description
                else:
                    logger.info(
                        "Removing decorative visual from section %d (no visual needed)", idx
                    )
                    lesson_content.sections[idx].image_description = None
    return lesson_content


def evaluate_alignment(
    lesson_content: LessonContent,
    questions: list[Question],
    ollama: "OllamaClient",
    eval_model: str | None = None,
) -> AlignmentResult:
    """
    Check whether each question is answerable from the lesson content alone.
    Maintained for direct question-alignment checks.
    """
    if not questions:
        return AlignmentResult(
            allAligned=True, questions=[], misalignedIds=[], summary="No questions to evaluate."
        )

    lesson_text = _summarise_lesson(lesson_content)
    question_text = _summarise_questions(questions)

    prompt = f"""Below is a lesson followed by quiz questions.
For EACH question, decide if it is ALIGNED (answerable from the lesson alone) or MISALIGNED.

=== LESSON CONTENT ===
{lesson_text}

=== QUIZ QUESTIONS ===
{question_text}

Return a JSON object with this exact structure:
{{
  "allAligned": true/false,
  "questions": [
    {{
      "questionId": "q1",
      "aligned": true,
      "reason": "The lesson covers this in Section 2."
    }}
  ],
  "summary": "Overall assessment summary"
}}

Be strict: if a question asks about a concept not explicitly taught in the lesson, mark it MISALIGNED."""

    try:
        data = ollama.generate_json(
            prompt,
            system=ALIGNMENT_EVAL_SYSTEM_PROMPT,
            model=eval_model or OLLAMA_EVAL_MODEL,
        )
        result = AlignmentResult.model_validate(data)
        result.misaligned_ids = [
            q.question_id for q in result.questions if not q.aligned
        ]
        result.all_aligned = len(result.misaligned_ids) == 0
        return result

    except Exception as exc:
        logger.error("Alignment evaluation failed: %s", exc)
        return AlignmentResult(
            allAligned=True,
            questions=[],
            misalignedIds=[],
            summary=f"Evaluation failed ({exc}); assuming aligned.",
        )


def build_lesson_context_for_questions(lesson_content: LessonContent) -> str:
    """
    Build a context block from the lesson content that is injected into the
    question generation prompt. Ensures the LLM only creates questions
    about concepts that were actually taught.
    """
    parts = [
        "THE LESSON COVERS THE FOLLOWING CONTENT (questions MUST be based ONLY on this):",
        "",
        f"Title: {lesson_content.title}",
        f"Introduction: {lesson_content.introduction}",
        "",
    ]
    for i, section in enumerate(lesson_content.sections, 1):
        parts.append(f"Section {i} — {section.heading}:")
        parts.append(f"  {section.body}")
        if section.fun_fact:
            parts.append(f"  Fun fact: {section.fun_fact}")
        parts.append("")

    if lesson_content.summary:
        parts.append(f"Summary: {lesson_content.summary}")

    parts.append("")
    parts.append(
        "CRITICAL: Every question you generate MUST be answerable using "
        "ONLY the information above. Do NOT ask about concepts, facts, or "
        "vocabulary that are not explicitly covered in the lesson sections above."
    )
    return "\n".join(parts)
