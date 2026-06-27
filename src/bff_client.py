"""
BFF REST client — all communication with the Spring Boot BFF goes through here.
"""
from __future__ import annotations

import logging
from typing import Optional

import requests

from .config import BFF_BASE_URL
from .models import (
    AssessmentResult,
    Course,
    Lesson,
    Progress,
    Subject,
)

logger = logging.getLogger(__name__)


class BFFClient:
    """Thin wrapper around the BFF REST API."""

    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or BFF_BASE_URL).rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    # ── Child / Profile ──────────────────────────────────────────────

    def create_child(self, name: str, age: int) -> dict:
        """Create a new child profile."""
        resp = self.session.post(
            f"{self.base_url}/api/children",
            json={"name": name, "age": age},
        )
        resp.raise_for_status()
        return resp.json()

    def get_dashboard(self, child_id: str) -> dict:
        resp = self.session.get(f"{self.base_url}/api/children/{child_id}/dashboard")
        resp.raise_for_status()
        return resp.json()

    # ── Courses (agent endpoints) ────────────────────────────────────

    def get_courses(self, child_id: str) -> list[dict]:
        resp = self.session.get(f"{self.base_url}/api/agent/children/{child_id}/courses")
        resp.raise_for_status()
        return resp.json()

    def get_course(self, child_id: str, subject: Subject) -> Optional[dict]:
        resp = self.session.get(
            f"{self.base_url}/api/agent/children/{child_id}/courses/{subject.value}"
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    def upsert_course(self, course: Course) -> dict:
        resp = self.session.post(
            f"{self.base_url}/api/agent/courses",
            json=course.model_dump(by_alias=True),
        )
        resp.raise_for_status()
        logger.info("Upserted course %s for child %s", course.subject, course.child_id)
        return resp.json()

    # ── Lessons (agent endpoints) ────────────────────────────────────

    def upsert_lessons(self, lessons: list[Lesson]) -> list[dict]:
        payload = [l.model_dump(by_alias=True) for l in lessons]
        resp = self.session.post(
            f"{self.base_url}/api/agent/lessons",
            json=payload,
        )
        resp.raise_for_status()
        logger.info("Upserted %d lessons", len(lessons))
        return resp.json()

    # ── Assessment results (agent endpoints) ─────────────────────────

    def get_assessment_results(self, child_id: str) -> list[AssessmentResult]:
        resp = self.session.get(
            f"{self.base_url}/api/agent/children/{child_id}/assessment-results"
        )
        resp.raise_for_status()
        return [AssessmentResult.model_validate(r) for r in resp.json()]

    # ── Progress (agent endpoints) ───────────────────────────────────

    def get_progress(self, child_id: str) -> list[Progress]:
        resp = self.session.get(
            f"{self.base_url}/api/agent/children/{child_id}/progress"
        )
        resp.raise_for_status()
        return [Progress.model_validate(p) for p in resp.json()]

    # ── Health ───────────────────────────────────────────────────────

    def health_check(self) -> bool:
        try:
            resp = self.session.get(f"{self.base_url}/actuator/health", timeout=5)
            return resp.status_code == 200
        except requests.ConnectionError:
            return False
