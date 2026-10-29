from __future__ import annotations

from src.context.ledger import ContextLedger
from src.lessons.lesson_store import LessonStore


def risky_compression(content: str) -> str:
    lowered = content.lower()
    if "offline" in lowered or "cloud" in lowered:
        return "The deployment has constraints."
    if "budget" in lowered or "5000" in lowered:
        return "The solution has budget constraints."
    if "windows" in lowered or "linux" in lowered:
        return "The environment has platform constraints."
    if "csv" in lowered or "json" in lowered:
        return "The output format has constraints."
    if "read-only" in lowered:
        return "The migration has database constraints."
    return "The user mentioned an implementation constraint."


class BaseManager:
    def manage_after_observation(self, ledger: ContextLedger, current_step: int) -> None:
        raise NotImplementedError


class FullContextManager(BaseManager):
    def manage_after_observation(self, ledger: ContextLedger, current_step: int) -> None:
        for item in ledger.active_items():
            ledger.keep_item(item.item_id, "Full-context baseline keeps every item.", current_step)


class FixedCompressionManager(BaseManager):
    def manage_after_observation(self, ledger: ContextLedger, current_step: int) -> None:
        for item in list(ledger.active_items()):
            if item.step <= current_step - 3 and item.immutable:
                ledger.compress_item(item.item_id, risky_compression(item.content), "Fixed compression of older items.", current_step)


class NonLearningManager(BaseManager):
    def manage_after_observation(self, ledger: ContextLedger, current_step: int) -> None:
        for item in list(ledger.active_items()):
            if "distractor" in item.tags and item.step <= current_step - 1:
                ledger.archive_item(item.item_id, "Archive stale distractor.", current_step)
            elif item.step <= current_step - 3 and item.immutable:
                ledger.compress_item(item.item_id, risky_compression(item.content), "Compress older information without lessons.", current_step)


class LessonAwareManager(BaseManager):
    def __init__(self, lesson_store: LessonStore) -> None:
        self.lesson_store = lesson_store

    def manage_after_observation(self, ledger: ContextLedger, current_step: int) -> None:
        for item in list(ledger.active_items()):
            lessons = self.lesson_store.retrieve(item.tags, operation="COMPRESS")
            if lessons and any(tag in item.tags for tag in ["critical", "hard_constraint", "negation", "delayed_dependency"]):
                ledger.keep_item(item.item_id, "Retrieved lesson says critical constraints should remain verbatim.", current_step)
            elif "distractor" in item.tags and item.step <= current_step - 1:
                ledger.archive_item(item.item_id, "Archive stale distractor.", current_step)
            elif item.step <= current_step - 3 and item.immutable:
                ledger.compress_item(item.item_id, risky_compression(item.content), "Compress older non-critical information.", current_step)

