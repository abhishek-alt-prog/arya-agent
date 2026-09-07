"""
Course Generator — uses the curriculum skeleton + Gemma 4 to generate
lesson content and assessment questions for each topic.

Content pipeline:
  1. Generate lesson content via LLM
  2. Generate assessment questions via LLM (with lesson content as context)
  3. Validate alignment between lesson and questions
  4. On misalignment → retry questions, then retry both (max 2 retries)
  5. Generate media (images + audio) into structured asset directories
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Optional

from .config import AGENT_VERSION
from .content_evaluator import (
    MAX_ALIGNMENT_RETRIES,
    apply_visual_improvements,
    build_lesson_context_for_questions,
    evaluate_alignment,
    evaluate_model_output,
)
from .curriculum import CURRICULUM, get_units_for_subject
from .models import (
    ContentSection,
    Course,
    Difficulty,
    Lesson,
    LessonContent,
    Question,
    QuestionType,
    Subject,
    Unit,
)
from .ollama_client import OllamaClient
from .media_generator import MediaGenerator

logger = logging.getLogger(__name__)

# ── System prompts ───────────────────────────────────────────────────

LESSON_SYSTEM_PROMPT = """\
You are an expert primary-school teacher and educational visual designer creating a lesson for a 7-year-old child.
Your writing style MUST be:
- Fun, warm, and encouraging
- Use simple words (Year 3 reading level)
- Include colourful analogies and real-world examples a child can relate to
- End with a cheerful summary

EDUCATIONAL VISUALS (CRITICAL RULES):
- Every image description MUST be a clear educational visual aid or explanatory diagram that directly helps the 7-year-old child understand the specific concept being taught in that section (e.g. anatomical parts with labels, process with arrows, countable groups of objects, place-value blocks, fractions breakdown).
- NEVER generate decorative drawings, generic cartoons, or pictures of a mascot smiling or waving. If a section does not need a visual explanation, set imageDescription to null.
- For diagrams that need text labels, ALWAYS enclose the exact words to render in double quotes (e.g. 'A diagram of a plant with labels "Flower", "Stem", "Leaf", "Roots" pointing to each part'). Keep labels short (1-2 words per label, max 3-5 labels per diagram) so they render clearly and legibly.
- Complement labels with clear colors, arrows, and spatial layout so the diagram is intuitive.

You will respond ONLY with valid JSON."""

ASSESSMENT_SYSTEM_PROMPT = """\
You are an expert primary-school teacher creating a short quiz for a 7-year-old.
Create questions that test understanding (not memorisation).
Keep language simple, warm, and encouraging.
Include a mix of question types.

CRITICAL RULE: Every question MUST be answerable using ONLY the lesson content
provided. Do NOT ask about concepts, vocabulary, or facts that are not
explicitly covered in the lesson.

You will respond ONLY with valid JSON."""


class CourseGenerator:
    """
    Generates courses, lessons, and assessments.

    On first run:  Generates the full course skeleton from the curriculum
                   and creates initial EASY lessons for the first unit of
                   each subject.

    On adaptation: Re-generates upcoming lessons based on progress data.
    """

    def __init__(
        self,
        store,
        ollama: OllamaClient,
    ):
        self.bff = store
        self.ollama = ollama
        self.media_gen = MediaGenerator()

    # ── Initial course generation ────────────────────────────────────

    def generate_initial_courses(self, child_id: str) -> list[Course]:
        """
        Generate course plans for all subjects and push to the BFF.
        Only generates if no course exists for a subject yet.
        """
        created_courses: list[Course] = []

        for subject in Subject:
            existing = self.bff.get_course(child_id, subject)
            if existing:
                logger.info("Course already exists for %s, skipping", subject.value)
                continue

            logger.info("Generating initial course for %s", subject.value)
            units = get_units_for_subject(subject)

            course = Course(
                childId=child_id,
                subject=subject,
                units=[
                    Unit(
                        name=u["name"],
                        topics=u["topics"],
                        difficultyRange=["EASY"],
                        sequenceOrder=idx,
                        completed=False,
                    )
                    for idx, u in enumerate(units)
                ],
                status="ACTIVE",
                generatedAt=datetime.now().isoformat(),
                generatedByAgentVersion=AGENT_VERSION,
            )

            saved = self.bff.upsert_course(course)
            course.id = saved.get("id")
            created_courses.append(course)
            logger.info("Created course %s (id=%s)", subject.value, course.id)

        return created_courses

    # ── Lesson generation ────────────────────────────────────────────

    def generate_lessons_for_unit(
        self,
        child_id: str,
        course_id: str,
        subject: Subject,
        unit_name: str,
        topics: list[str],
        difficulty: Difficulty = Difficulty.EASY,
        student_context: str = "",
    ) -> list[Lesson]:
        """
        Generate one lesson + assessment per topic in the unit.
        Uses Gemma 4 to create the content, then validates alignment.
        """
        lessons: list[Lesson] = []

        for seq, topic in enumerate(topics):
            logger.info("Generating lesson: %s / %s / %s", subject.value, unit_name, topic)

            # ── Generate & align lesson + questions ──────────────
            content, questions = self._generate_aligned_content(
                subject, unit_name, topic, difficulty, student_context,
            )

            # ── Generate media for the content ───────────────────
            # Media context for structured asset directories
            media_ctx = dict(
                subject=subject.value,
                unit_name=unit_name,
                topic_name=topic,
            )

            logger.info("Generating media for lesson: %s", topic)
            if content.introduction:
                content.audio_url = self.media_gen.generate_audio(
                    content.introduction, prefix="intro", **media_ctx,
                )

            for section in content.sections:
                if section.body:
                    section.audio_url = self.media_gen.generate_audio(
                        section.body, prefix="section", **media_ctx,
                    )
                if section.image_description:
                    section.image_url = self.media_gen.generate_image(
                        section.image_description, prefix="img", **media_ctx,
                    )

            if content.summary:
                content.summary_audio_url = self.media_gen.generate_audio(
                    content.summary, prefix="summary", **media_ctx,
                )

            for q in questions:
                if q.question_text:
                    q.audio_url = self.media_gen.generate_audio(
                        q.question_text, prefix="question", **media_ctx,
                    )

            lesson = Lesson(
                courseId=course_id,
                childId=child_id,
                subject=subject,
                unitName=unit_name,
                topicName=topic,
                sequenceOrder=seq,
                difficulty=difficulty,
                content=content,
                assessmentQuestions=questions,
                createdAt=datetime.now().isoformat(),
            )
            lessons.append(lesson)

        # Unload media models to free memory
        self.media_gen.unload_models()

        # Push to BFF in bulk
        if lessons:
            saved = self.bff.upsert_lessons(lessons)
            # Update IDs from response
            for i, s in enumerate(saved):
                lessons[i].id = s.get("id")
            logger.info("Saved %d lessons for unit '%s'", len(lessons), unit_name)

        return lessons

    def generate_first_unit_lessons(self, child_id: str) -> list[Lesson]:
        """
        Generate lessons for the first unit of each subject.
        Called during initial setup so the child has content right away.
        """
        all_lessons: list[Lesson] = []

        for subject in Subject:
            course_data = self.bff.get_course(child_id, subject)
            if not course_data:
                logger.warning("No course found for %s, skipping", subject.value)
                continue

            course_id = course_data.get("id", "")
            units = course_data.get("units", [])
            if not units:
                continue

            existing_lessons = self.bff.get_lessons_for_subject(child_id, subject)
            if existing_lessons:
                logger.info(
                    "Lessons already exist for %s (%d lessons), skipping",
                    subject.value, len(existing_lessons),
                )
                continue

            first_unit = units[0]
            lessons = self.generate_lessons_for_unit(
                child_id=child_id,
                course_id=course_id,
                subject=subject,
                unit_name=first_unit["name"],
                topics=first_unit["topics"],
                difficulty=Difficulty.EASY,
            )
            all_lessons.extend(lessons)

        return all_lessons

    # ── Aligned content generation ───────────────────────────────────

    def _generate_aligned_content(
        self,
        subject: Subject,
        unit_name: str,
        topic: str,
        difficulty: Difficulty,
        student_context: str = "",
    ) -> tuple[LessonContent, list[Question]]:
        """
        Generate lesson content and assessment questions with comprehensive
        model output evaluation (pedagogical visual utility and lesson alignment).

        Retries on misalignment or unhelpful visuals:
        Retry 1: Apply visual improvements + regenerate misaligned questions.
        Retry 2: Regenerate both lesson and questions from scratch.

        Returns the best (lesson, questions) pair available.
        """
        # ── Initial generation ───────────────────────────────────
        content = self._generate_lesson_content(
            subject, unit_name, topic, difficulty, student_context,
        )
        questions = self._generate_questions(
            subject, unit_name, topic, difficulty, student_context,
            lesson_content=content,
        )

        # ── Model output evaluation loop ─────────────────────────
        for retry in range(MAX_ALIGNMENT_RETRIES):
            eval_result = evaluate_model_output(
                content, questions, subject, topic, self.ollama,
            )

            # Apply educational improvements to visuals (replace decorative visuals with diagrams/models or prune)
            content = apply_visual_improvements(content, eval_result)

            if eval_result.passed:
                logger.info(
                    "Model output evaluation PASSED for '%s' (retry=%d)",
                    topic, retry,
                )
                return content, questions

            logger.warning(
                "Model output evaluation flagged issues for '%s': %d misaligned questions, %d unhelpful visuals (retry %d/%d)",
                topic,
                len(eval_result.misaligned_question_ids),
                len(eval_result.unhelpful_visual_indices),
                retry + 1,
                MAX_ALIGNMENT_RETRIES,
            )

            # Build misaligned question feedback
            feedback_notes = []
            for qa in eval_result.question_alignments:
                if not qa.aligned:
                    feedback_notes.append(f"- Question {qa.question_id}: {qa.reason}")
            feedback_str = "\n".join(feedback_notes)
            combined_context = student_context
            if feedback_str:
                combined_context = (
                    f"{student_context}\n\n"
                    f"PREVIOUS QUESTIONS FAILED ALIGNMENT:\n{feedback_str}\n"
                    f"CRITICAL: Every question must be directly answerable from the lesson text."
                )

            if retry == 0:
                # First retry: regenerate questions with evaluation feedback
                logger.info("Retrying: regenerating questions with evaluation feedback")
                questions = self._generate_questions(
                    subject, unit_name, topic, difficulty, combined_context,
                    lesson_content=content,
                )
            else:
                # Second retry: regenerate everything
                logger.info("Retrying: regenerating both lesson and questions")
                content = self._generate_lesson_content(
                    subject, unit_name, topic, difficulty, student_context,
                )
                questions = self._generate_questions(
                    subject, unit_name, topic, difficulty, student_context,
                    lesson_content=content,
                )

        # Final check after all retries
        final_result = evaluate_model_output(content, questions, subject, topic, self.ollama)
        content = apply_visual_improvements(content, final_result)
        if not final_result.passed:
            logger.warning(
                "Evaluation still flagged issues for '%s' after %d retries. "
                "Using best available content. Misaligned: %s, Unhelpful visuals: %s",
                topic, MAX_ALIGNMENT_RETRIES,
                final_result.misaligned_question_ids,
                final_result.unhelpful_visual_indices,
            )

        return content, questions

    # ── LLM content generation ───────────────────────────────────────

    def _generate_lesson_content(
        self,
        subject: Subject,
        unit_name: str,
        topic: str,
        difficulty: Difficulty,
        student_context: str = "",
    ) -> LessonContent:
        """Use Gemma 4 to generate lesson content, personalized with student context."""

        # Build the personalization block
        context_block = ""
        if student_context:
            context_block = f"""

{student_context}

IMPORTANT PERSONALIZATION INSTRUCTIONS:
- Spend MORE time explaining concepts the student struggled with or had misconceptions about.
- Use DIFFERENT analogies and examples than a typical lesson would — the student already saw the standard explanation and didn't fully grasp it.
- For concepts the student is confident with, keep explanations brief and move on.
- Do NOT explicitly say "you got this wrong last time". Instead, naturally weave in extra explanation, practice examples, and fun facts for the weak areas.
- If the student had a specific misconception (e.g. thought 1/4 > 1/3), directly address that misunderstanding with a clear, child-friendly counter-example."""

        prompt = f"""Create a lesson for a 7-year-old about: "{topic}"
Subject: {subject.value}
Unit: {unit_name}
Difficulty: {difficulty.value}
{context_block}

Return a JSON object with this exact structure:
{{
  "title": "A fun, child-friendly title",
  "introduction": "A warm greeting from the mascot owl Hoot (2-3 sentences)",
  "sections": [
    {{
      "heading": "Section heading",
      "body": "Explanation text (3-5 sentences, simple language)",
      "imageDescription": "An educational diagram or visual aid for this concept. If labels are needed, put the exact words in double quotes (e.g. a diagram with labels \"Petal\", \"Stem\", \"Roots\"). Keep labels concise (1-2 words each, max 4 labels). Set to null if no visual aid is needed.",
      "funFact": "An optional fun or surprising fact"
    }}
  ],
  "summary": "A cheerful recap of what was learned (2-3 sentences)"
}}

Create 7-10 sections to provide a deep, comprehensive lesson. Make the tone warm and playful."""

        try:
            data = self.ollama.generate_json(prompt, system=LESSON_SYSTEM_PROMPT)
            return LessonContent.model_validate(data)
        except Exception as exc:
            logger.error("LLM lesson generation failed for '%s': %s", topic, exc)
            return self._fallback_lesson_content(topic)

    def _generate_questions(
        self,
        subject: Subject,
        unit_name: str,
        topic: str,
        difficulty: Difficulty,
        student_context: str = "",
        lesson_content: LessonContent | None = None,
    ) -> list[Question]:
        """
        Use Gemma 4 to generate assessment questions.

        When lesson_content is provided, the questions are scoped strictly
        to the concepts covered in the lesson.  This is the primary mechanism
        for preventing misaligned questions.
        """

        # Build lesson context block (most important for alignment)
        lesson_block = ""
        if lesson_content:
            lesson_block = "\n\n" + build_lesson_context_for_questions(lesson_content)

        # Build targeting instructions from student context
        targeting_block = ""
        if student_context:
            targeting_block = f"""

{student_context}

TARGETING INSTRUCTIONS:
- Create at least 1-2 questions that specifically test the concepts the student previously struggled with or hesitated on.
- Phrase these questions differently from the previous assessment so the student is tested on understanding, not memory.
- Include 1-2 questions on concepts the student was confident with, to maintain their confidence.
- Do NOT make the questions easier just because the student struggled — test the same concept but approach it from a different angle."""

        prompt = f"""Create 4 quiz questions for a 7-year-old about: "{topic}"
Subject: {subject.value}
Difficulty: {difficulty.value}
{lesson_block}
{targeting_block}

Return a JSON array with this exact structure:
[
  {{
    "id": "q1",
    "type": "MCQ",
    "questionText": "The question in simple language",
    "hint": "A gentle hint",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correctOptionIndex": 0
  }},
  {{
    "id": "q2",
    "type": "FILL_BLANK",
    "questionText": "Fill in the blank",
    "hint": "A gentle hint",
    "sentenceWithBlank": "The sun is a ___",
    "correctAnswer": "star",
    "acceptableAnswers": ["star", "Star"]
  }}
]

Create exactly 4 questions:
- 2 MCQ questions (4 options each)
- 1 FILL_BLANK question
- 1 more MCQ question

For EASY: straightforward recall
For MEDIUM: requires understanding
For HARD: requires application"""

        try:
            data = self.ollama.generate_json(prompt, system=ASSESSMENT_SYSTEM_PROMPT)
            if isinstance(data, list):
                return [Question.model_validate(q) for q in data]
            return []
        except Exception as exc:
            logger.error("LLM question generation failed for '%s': %s", topic, exc)
            return self._fallback_questions(topic)

    # ── Fallbacks (when LLM is unavailable) ──────────────────────────

    @staticmethod
    def _fallback_lesson_content(topic: str) -> LessonContent:
        """Provide basic lesson content when the LLM is unavailable."""
        return LessonContent(
            title=f"Let's Learn About {topic}!",
            introduction=(
                f"Hello there! 🦉 I'm Hoot, your learning buddy! "
                f"Today we're going to explore something really exciting — {topic}! "
                f"Are you ready? Let's go!"
            ),
            sections=[
                ContentSection(
                    heading=f"What is {topic}?",
                    body=(
                        f"Today we're learning about {topic}. "
                        f"This is a really interesting part of what we're studying! "
                        f"Let's discover some amazing things together."
                    ),
                    imageDescription=f"A colourful illustration explaining {topic}",
                    funFact=f"Did you know? {topic} is one of the most fascinating things to learn about!",
                ),
            ],
            summary=(
                f"Great job today! 🌟 We learned all about {topic}. "
                f"You're doing brilliantly — keep it up!"
            ),
        )

    @staticmethod
    def _fallback_questions(topic: str) -> list[Question]:
        """Provide basic questions when the LLM is unavailable."""
        return [
            Question(
                id="q1",
                type=QuestionType.MCQ,
                questionText=f"What did we learn about today?",
                hint="Think about what Hoot taught you!",
                options=[topic, "Dinosaurs", "Space rockets", "Ice cream"],
                correctOptionIndex=0,
            ),
            Question(
                id="q2",
                type=QuestionType.MCQ,
                questionText=f"Which of these is related to {topic}?",
                hint="Remember what we covered in the lesson!",
                options=[
                    f"Something about {topic}",
                    "Playing football",
                    "Cooking dinner",
                    "Watching TV",
                ],
                correctOptionIndex=0,
            ),
            Question(
                id="q3",
                type=QuestionType.FILL_BLANK,
                questionText="Fill in the missing word",
                hint="You learned this in the lesson!",
                sentenceWithBlank=f"Today we learned about ___",
                correctAnswer=topic.split()[0].lower(),
                acceptableAnswers=[topic.lower(), topic.split()[0].lower()],
            ),
            Question(
                id="q4",
                type=QuestionType.MCQ,
                questionText="Did you enjoy learning today?",
                hint="There's no wrong answer here! 😊",
                options=["Yes, it was great!", "It was fun!", "I loved it!", "All of the above!"],
                correctOptionIndex=3,
            ),
        ]
