"""
Pydantic models for the Arya Agent.
Mirror the BFF's data structures for type-safe communication.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Subject(str, Enum):
    SCIENCE = "SCIENCE"
    MATHS = "MATHS"
    ENGLISH = "ENGLISH"


class Difficulty(str, Enum):
    EASY = "EASY"
    MEDIUM = "MEDIUM"
    HARD = "HARD"


class LessonStatus(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"


class QuestionType(str, Enum):
    MCQ = "MCQ"
    DRAG_DROP = "DRAG_DROP"
    FILL_BLANK = "FILL_BLANK"


# ── Course ────────────────────────────────────────────────────────────

class Unit(BaseModel):
    name: str
    topics: list[str]
    difficulty_range: list[str] = Field(alias="difficultyRange", default_factory=lambda: ["EASY"])
    sequence_order: int = Field(alias="sequenceOrder", default=0)
    completed: bool = False

    model_config = {"populate_by_name": True}


class Course(BaseModel):
    id: Optional[str] = None
    child_id: str = Field(alias="childId")
    subject: Subject
    units: list[Unit] = []
    status: str = "ACTIVE"
    generated_at: Optional[str] = Field(alias="generatedAt", default=None)
    last_updated_at: Optional[str] = Field(alias="lastUpdatedAt", default=None)
    generated_by_agent_version: Optional[str] = Field(alias="generatedByAgentVersion", default=None)

    model_config = {"populate_by_name": True}


# ── Lesson Content ───────────────────────────────────────────────────

class ContentSection(BaseModel):
    heading: str
    body: str
    image_description: Optional[str] = Field(alias="imageDescription", default=None)
    image_url: Optional[str] = Field(alias="imageUrl", default=None)
    fun_fact: Optional[str] = Field(alias="funFact", default=None)

    model_config = {"populate_by_name": True}


class LessonContent(BaseModel):
    title: str
    introduction: str
    sections: list[ContentSection] = []
    summary: str = ""

    model_config = {"populate_by_name": True}


class Question(BaseModel):
    id: str
    type: QuestionType
    question_text: str = Field(alias="questionText")
    hint: Optional[str] = None

    # MCQ
    options: Optional[list[str]] = None
    correct_option_index: Optional[int] = Field(alias="correctOptionIndex", default=None)

    # FILL_BLANK
    sentence_with_blank: Optional[str] = Field(alias="sentenceWithBlank", default=None)
    correct_answer: Optional[str] = Field(alias="correctAnswer", default=None)
    acceptable_answers: Optional[list[str]] = Field(alias="acceptableAnswers", default=None)

    # DRAG_DROP
    draggable_items: Optional[list[str]] = Field(alias="draggableItems", default=None)
    drop_targets: Optional[list[str]] = Field(alias="dropTargets", default=None)
    correct_mapping: Optional[dict[str, str]] = Field(alias="correctMapping", default=None)

    model_config = {"populate_by_name": True}


class Lesson(BaseModel):
    id: Optional[str] = None
    course_id: Optional[str] = Field(alias="courseId", default=None)
    child_id: str = Field(alias="childId")
    subject: Subject
    unit_name: str = Field(alias="unitName")
    topic_name: str = Field(alias="topicName")
    sequence_order: int = Field(alias="sequenceOrder", default=0)
    difficulty: Difficulty = Difficulty.EASY
    content: Optional[LessonContent] = None
    assessment_questions: list[Question] = Field(alias="assessmentQuestions", default_factory=list)
    status: LessonStatus = LessonStatus.NOT_STARTED
    star_rating: Optional[int] = Field(alias="starRating", default=None)
    xp_earned: Optional[int] = Field(alias="xpEarned", default=None)
    created_at: Optional[str] = Field(alias="createdAt", default=None)

    model_config = {"populate_by_name": True}


# ── Progress & Assessment ────────────────────────────────────────────

class Progress(BaseModel):
    id: Optional[str] = None
    child_id: str = Field(alias="childId")
    subject: Subject
    unit_name: str = Field(alias="unitName")
    topic_name: str = Field(alias="topicName")
    mastery_score: float = Field(alias="masteryScore", default=0.0)
    lessons_completed: int = Field(alias="lessonsCompleted", default=0)
    lessons_total: int = Field(alias="lessonsTotal", default=0)
    current_difficulty: Optional[Difficulty] = Field(alias="currentDifficulty", default=None)

    model_config = {"populate_by_name": True}


class AnswerRecord(BaseModel):
    question_id: str = Field(alias="questionId")
    given_answer: str = Field(alias="givenAnswer")
    correct_answer: str = Field(alias="correctAnswer")
    correct: bool

    model_config = {"populate_by_name": True}


class AssessmentResult(BaseModel):
    id: Optional[str] = None
    lesson_id: str = Field(alias="lessonId")
    child_id: str = Field(alias="childId")
    subject: Subject
    unit_name: str = Field(alias="unitName")
    topic_name: str = Field(alias="topicName")
    answers: list[AnswerRecord] = []
    score: int = 0
    total_questions: int = Field(alias="totalQuestions", default=0)
    star_rating: int = Field(alias="starRating", default=0)
    xp_earned: int = Field(alias="xpEarned", default=0)
    submitted_at: Optional[str] = Field(alias="submittedAt", default=None)

    model_config = {"populate_by_name": True}
