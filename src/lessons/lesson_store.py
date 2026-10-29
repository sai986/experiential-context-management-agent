from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Lesson:
    lesson_id: str
    lesson_type: str
    operation: str
    trigger_tags: list[str]
    instruction: str
    supporting_decision_id: str
    original_score: float
    counterfactual_score: float
    status: str = "validated"


class LessonStore:
    def __init__(self) -> None:
        self.lessons: list[Lesson] = []

    def add(self, lesson: Lesson) -> None:
        duplicate = any(
            existing.operation == lesson.operation
            and set(existing.trigger_tags) == set(lesson.trigger_tags)
            and existing.instruction == lesson.instruction
            for existing in self.lessons
        )
        if not duplicate:
            self.lessons.append(lesson)

    def retrieve(self, tags: list[str], operation: str) -> list[Lesson]:
        tag_set = set(tags)
        return [
            lesson
            for lesson in self.lessons
            if lesson.operation == operation and tag_set.intersection(lesson.trigger_tags)
        ]

    def to_dict(self) -> list[dict]:
        return [asdict(lesson) for lesson in self.lessons]

