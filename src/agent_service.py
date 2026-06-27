"""
Agent Service — adaptive tutoring logic.

Analyses assessment results and progress data to decide:
- Whether to advance to the next topic
- Whether to repeat/reinforce the current topic at a harder difficulty
- Whether to simplify and provide easier content
- What the next set of lessons should look like
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from .bff_client import BFFClient
from .course_generator import CourseGenerator
from .models import (
    AssessmentResult,
    Difficulty,
    Lesson,
    Progress,
    Subject,
)
from .ollama_client import OllamaClient

logger = logging.getLogger(__name__)

# ── Difficulty adaptation thresholds ─────────────────────────────────
# Based on average star rating over recent assessments for a topic.

ADVANCE_THRESHOLD = 4.0       # ≥ 4 stars avg → move to next topic
MAINTAIN_THRESHOLD = 2.5      # 2.5–3.9 stars → stay at same difficulty
SIMPLIFY_THRESHOLD = 2.5      # < 2.5 stars → simplify / provide reinforcement

DIFFICULTY_LADDER = [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD]


def _next_difficulty(current: Difficulty) -> Difficulty:
    """Move up one difficulty level, capped at HARD."""
    idx = DIFFICULTY_LADDER.index(current)
    return DIFFICULTY_LADDER[min(idx + 1, len(DIFFICULTY_LADDER) - 1)]


def _prev_difficulty(current: Difficulty) -> Difficulty:
    """Move down one difficulty level, capped at EASY."""
    idx = DIFFICULTY_LADDER.index(current)
    return DIFFICULTY_LADDER[max(idx - 1, 0)]


class AgentService:
    """
    Orchestrates the adaptive tutoring loop.

    Workflow:
    1. Fetch all progress and recent assessment results from the BFF.
    2. For each subject, analyse per-topic mastery.
    3. Decide what to generate next:
       - Advance → generate next topic's lessons
       - Maintain → generate same topic at same difficulty
       - Simplify → regenerate topic at lower difficulty
    4. Push new lessons to the BFF.
    """

    def __init__(self, bff: BFFClient, ollama: OllamaClient):
        self.bff = bff
        self.ollama = ollama
        self.generator = CourseGenerator(bff, ollama)

    # ── Initial setup ────────────────────────────────────────────────

    def initial_setup(self, child_id: str) -> dict:
        """
        First-run setup: create child profile (if needed), generate
        course plans, and create initial lessons.
        """
        logger.info("Running initial setup for child %s", child_id)

        # 1. Generate course plans for all subjects
        courses = self.generator.generate_initial_courses(child_id)
        logger.info("Created %d course plans", len(courses))

        # 2. Generate lessons for the first unit of each subject
        lessons = self.generator.generate_first_unit_lessons(child_id)
        logger.info("Created %d initial lessons", len(lessons))

        return {
            "courses_created": len(courses),
            "lessons_created": len(lessons),
            "subjects": [s.value for s in Subject],
            "timestamp": datetime.now().isoformat(),
        }

    # ── Nightly adaptation run ───────────────────────────────────────

    def run_adaptation(self, child_id: str) -> dict:
        """
        Analyse progress and generate/adjust upcoming lessons.
        This is designed to run nightly via cron.
        """
        logger.info("Starting adaptation run for child %s", child_id)

        progress_list = self.bff.get_progress(child_id)
        results = self.bff.get_assessment_results(child_id)

        if not progress_list:
            logger.info("No progress data yet — nothing to adapt")
            return {"status": "no_data", "actions": []}

        actions: list[dict] = []

        for subject in Subject:
            subject_progress = [
                p for p in progress_list if p.subject == subject
            ]
            subject_results = [
                r for r in results if r.subject == subject
            ]

            if not subject_progress:
                continue

            subject_actions = self._adapt_subject(
                child_id, subject, subject_progress, subject_results
            )
            actions.extend(subject_actions)

        logger.info("Adaptation complete: %d actions taken", len(actions))
        return {
            "status": "completed",
            "actions": actions,
            "timestamp": datetime.now().isoformat(),
        }

    def _adapt_subject(
        self,
        child_id: str,
        subject: Subject,
        progress_list: list[Progress],
        results: list[AssessmentResult],
    ) -> list[dict]:
        """Adapt lessons for a single subject based on progress."""
        actions: list[dict] = []

        # Get the course to know topic order
        course_data = self.bff.get_course(child_id, subject)
        if not course_data:
            return actions

        units = course_data.get("units", [])
        course_id = course_data.get("id", "")

        for unit in units:
            unit_name = unit["name"]
            topics = unit["topics"]

            for i, topic in enumerate(topics):
                # Find progress for this topic
                topic_progress = next(
                    (p for p in progress_list if p.topic_name == topic),
                    None,
                )

                if not topic_progress:
                    # No progress yet — skip (lessons not started)
                    continue

                if topic_progress.lessons_completed == 0:
                    continue

                mastery = topic_progress.mastery_score
                current_diff = topic_progress.current_difficulty or Difficulty.EASY

                decision = self._decide_action(mastery, current_diff, topic, i, len(topics))
                actions.append(decision)

                # Execute the decision
                if decision["action"] == "advance":
                    # Generate lessons for the next topic
                    if i + 1 < len(topics):
                        next_topic = topics[i + 1]
                        next_diff = decision.get("new_difficulty", Difficulty.EASY)
                        self.generator.generate_lessons_for_unit(
                            child_id=child_id,
                            course_id=course_id,
                            subject=subject,
                            unit_name=unit_name,
                            topics=[next_topic],
                            difficulty=next_diff,
                        )
                        logger.info(
                            "Advanced: generated %s lesson at %s",
                            next_topic, next_diff.value,
                        )

                elif decision["action"] == "reinforce":
                    # Regenerate same topic at same or higher difficulty
                    self.generator.generate_lessons_for_unit(
                        child_id=child_id,
                        course_id=course_id,
                        subject=subject,
                        unit_name=unit_name,
                        topics=[topic],
                        difficulty=decision.get("new_difficulty", current_diff),
                    )
                    logger.info(
                        "Reinforced: regenerated %s at %s",
                        topic, decision.get("new_difficulty", current_diff).value,
                    )

                elif decision["action"] == "simplify":
                    # Regenerate at lower difficulty
                    easier = _prev_difficulty(current_diff)
                    self.generator.generate_lessons_for_unit(
                        child_id=child_id,
                        course_id=course_id,
                        subject=subject,
                        unit_name=unit_name,
                        topics=[topic],
                        difficulty=easier,
                    )
                    logger.info(
                        "Simplified: regenerated %s at %s",
                        topic, easier.value,
                    )

        return actions

    def _decide_action(
        self,
        mastery: float,
        current_difficulty: Difficulty,
        topic: str,
        topic_index: int,
        total_topics: int,
    ) -> dict:
        """
        Decide what to do based on mastery score.

        Returns a dict describing the action to take.
        """
        if mastery >= ADVANCE_THRESHOLD:
            # Child is doing great!
            if current_difficulty == Difficulty.HARD:
                # Already at hardest — advance to next topic
                return {
                    "action": "advance",
                    "topic": topic,
                    "reason": f"Mastered at HARD (avg {mastery:.1f}★)",
                    "new_difficulty": Difficulty.EASY,
                }
            else:
                # Increase difficulty on same topic first
                new_diff = _next_difficulty(current_difficulty)
                return {
                    "action": "reinforce",
                    "topic": topic,
                    "reason": f"Strong performance (avg {mastery:.1f}★), increasing difficulty",
                    "new_difficulty": new_diff,
                }

        elif mastery >= MAINTAIN_THRESHOLD:
            # Decent but not mastered — keep practising
            return {
                "action": "reinforce",
                "topic": topic,
                "reason": f"Moderate performance (avg {mastery:.1f}★), practising more",
                "new_difficulty": current_difficulty,
            }

        else:
            # Struggling — simplify
            return {
                "action": "simplify",
                "topic": topic,
                "reason": f"Needs support (avg {mastery:.1f}★), simplifying",
                "new_difficulty": _prev_difficulty(current_difficulty),
            }

    # ── Status / Health ──────────────────────────────────────────────

    def get_status(self, child_id: str) -> dict:
        """Return a summary of the agent's view of the child's learning."""
        try:
            progress = self.bff.get_progress(child_id)
            results = self.bff.get_assessment_results(child_id)
            courses = self.bff.get_courses(child_id)

            return {
                "child_id": child_id,
                "courses": len(courses),
                "topics_with_progress": len(progress),
                "total_assessments": len(results),
                "bff_connected": True,
                "ollama_connected": self.ollama.health_check(),
                "ollama_model": self.ollama.model,
                "model_available": self.ollama.is_model_available(),
            }
        except Exception as exc:
            return {
                "child_id": child_id,
                "error": str(exc),
                "bff_connected": False,
            }
