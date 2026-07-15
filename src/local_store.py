"""
Local JSON-file storage — replaces BFF dependency for standalone operation.

Writes all generated data to JSON files under a `data/` directory.
These files are the same shape as what the BFF would store in MongoDB,
so they can be bulk-imported later.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import (
    AssessmentResult,
    Course,
    Lesson,
    Progress,
    Subject,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> list | dict:
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return []


def _save_json(path: Path, data) -> None:
    _ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.debug("Saved %s", path)


class LocalStore:
    """Drop-in replacement for BFFClient that stores data as JSON files."""

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = data_dir or DATA_DIR
        _ensure_dir(self.data_dir)
        self._counter = 0

    def _next_id(self) -> str:
        self._counter += 1
        return f"local-{self._counter:04d}"

    # ── Child / Profile ──────────────────────────────────────────────

    def create_child(self, name: str, age: int) -> dict:
        child = {
            "id": self._next_id(),
            "name": name,
            "age": age,
            "level": 1,
            "levelTitle": "Curious Beginner",
            "totalXp": 0,
            "currentStreak": 0,
            "longestStreak": 0,
            "avatarId": "owl",
            "createdAt": datetime.now().isoformat(),
            "updatedAt": datetime.now().isoformat(),
        }
        path = self.data_dir / "children.json"
        children = _load_json(path)
        if isinstance(children, list):
            children.append(child)
        else:
            children = [child]
        _save_json(path, children)
        return child

    def get_dashboard(self, child_id: str) -> dict:
        return {
            "childId": child_id,
            "level": 1,
            "levelTitle": "Curious Beginner",
            "totalXp": 0,
            "currentStreak": 0,
            "xpToNextLevel": 100,
            "subjects": [],
        }

    # ── Courses ──────────────────────────────────────────────────────

    def get_courses(self, child_id: str) -> list[dict]:
        path = self.data_dir / f"courses_{child_id}.json"
        data = _load_json(path)
        return data if isinstance(data, list) else []

    def get_course(self, child_id: str, subject: Subject) -> Optional[dict]:
        courses = self.get_courses(child_id)
        for c in courses:
            if c.get("subject") == subject.value:
                return c
        return None

    def upsert_course(self, course: Course) -> dict:
        child_id = course.child_id
        path = self.data_dir / f"courses_{child_id}.json"
        courses = self.get_courses(child_id)

        course_dict = course.model_dump(by_alias=True)
        if not course_dict.get("id"):
            course_dict["id"] = self._next_id()

        # Replace if exists, else append
        found = False
        for i, c in enumerate(courses):
            if c.get("subject") == course.subject.value:
                courses[i] = course_dict
                found = True
                break
        if not found:
            courses.append(course_dict)

        _save_json(path, courses)
        logger.info("Upserted course %s for child %s", course.subject.value, child_id)
        return course_dict

    # ── Lessons ──────────────────────────────────────────────────────

    def upsert_lessons(self, lessons: list[Lesson]) -> list[dict]:
        if not lessons:
            return []

        child_id = lessons[0].child_id
        path = self.data_dir / f"lessons_{child_id}.json"
        existing = _load_json(path)
        if not isinstance(existing, list):
            existing = []

        saved = []
        for lesson in lessons:
            lesson_dict = lesson.model_dump(by_alias=True)
            if not lesson_dict.get("id"):
                lesson_dict["id"] = self._next_id()
            existing.append(lesson_dict)
            saved.append(lesson_dict)

        _save_json(path, existing)
        logger.info("Upserted %d lessons", len(saved))
        return saved

    # ── Assessment results ───────────────────────────────────────────

    def get_assessment_results(self, child_id: str) -> list[AssessmentResult]:
        path = self.data_dir / f"assessments_{child_id}.json"
        data = _load_json(path)
        if isinstance(data, list):
            return [AssessmentResult.model_validate(r) for r in data]
        return []

    # ── Progress ─────────────────────────────────────────────────────

    def get_progress(self, child_id: str) -> list[Progress]:
        path = self.data_dir / f"progress_{child_id}.json"
        data = _load_json(path)
        if isinstance(data, list):
            return [Progress.model_validate(p) for p in data]
        return []

    # ── Health ───────────────────────────────────────────────────────

    def health_check(self) -> bool:
        return True
