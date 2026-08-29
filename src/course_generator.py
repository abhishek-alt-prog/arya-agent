"""
Course Generator — uses the curriculum skeleton + Gemma 4 to generate
lesson content and assessment questions for each topic.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Optional

from .config import AGENT_VERSION
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
You are an expert primary-school teacher creating a lesson for a 7-year-old child.
Your writing style MUST be:
- Fun, warm, and encouraging
- Use simple words (Year 3 reading level)
- Include colourful analogies and real-world examples a child can relate to
- End with a cheerful summary

You will respond ONLY with valid JSON."""

ASSESSMENT_SYSTEM_PROMPT = """\
You are an expert primary-school teacher creating a short quiz for a 7-year-old.
Create questions that test understanding (not memorisation).
Keep language simple, warm, and encouraging.
Include a mix of question types.
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
        Uses Gemma 4 to create the content.
        """
        lessons: list[Lesson] = []

        for seq, topic in enumerate(topics):
            logger.info("Generating lesson: %s / %s / %s", subject.value, unit_name, topic)

            # Generate lesson content via LLM
            content = self._generate_lesson_content(subject, unit_name, topic, difficulty, student_context)

            # Generate media for the content
            logger.info("Generating media for lesson: %s", topic)
            if content.introduction:
                content.audio_url = self.media_gen.generate_audio(content.introduction, prefix="intro")
            
            for section in content.sections:
                if section.body:
                    section.audio_url = self.media_gen.generate_audio(section.body, prefix="section")
                if section.image_description:
                    section.image_url = self.media_gen.generate_image(section.image_description, prefix="img")

            if content.summary:
                content.summary_audio_url = self.media_gen.generate_audio(content.summary, prefix="summary")

            # Generate assessment questions via LLM
            questions = self._generate_questions(subject, unit_name, topic, difficulty, student_context)

            for q in questions:
                if q.question_text:
                    q.audio_url = self.media_gen.generate_audio(q.question_text, prefix="question")

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
      "imageDescription": "Description of an illustration that would help explain this",
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
    ) -> list[Question]:
        """Use Gemma 4 to generate assessment questions, targeted at weak areas."""

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
