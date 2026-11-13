from __future__ import annotations

from src.constants import OP_ARCHIVE, OP_COMPRESS, TAG_DISTRACTOR, TAG_TRIGGER
from src.context.ledger import ContextLedger
from src.lessons.lesson_store import LessonStore


def risky_compression(content: str) -> str:
    # Intentionally dumb static summary — the baseline that motivates learning.
    # Having keyword-specific branches would make it smarter than intended and
    # risk accidentally hiding required keywords behind vague phrases.
    return "The user mentioned an implementation constraint."


def intelligent_compression(content: str, model_client) -> str:
    """LLM-based compression that preserves critical details.

    Asks the model to summarise in one sentence while keeping negations,
    exact numbers, hard constraints, and deadlines verbatim.

    Falls back to risky_compression if no model client is available
    (i.e. USE_LLM=0 mode).
    """
    if model_client is None:
        return risky_compression(content)
    prompt = (
        "Summarise the following observation in one sentence. "
        "Preserve any negations, exact numeric values, hard constraints, "
        "and deadlines verbatim — do not paraphrase them.\n\n"
        f"Observation: {content}\n\nSummary:"
    )
    return model_client.complete(prompt)


class BaseManager:
    def manage_after_observation(self, ledger: ContextLedger, current_step: int) -> None:
        raise NotImplementedError


class FullContextManager(BaseManager):
    def manage_after_observation(self, ledger: ContextLedger, current_step: int) -> None:
        for item in ledger.active_items():
            if TAG_TRIGGER in item.tags:
                continue
            ledger.keep_item(item.item_id, "Full-context baseline keeps every item.", current_step)


class FixedCompressionManager(BaseManager):
    def manage_after_observation(self, ledger: ContextLedger, current_step: int) -> None:
        for item in list(ledger.active_items()):
            if TAG_TRIGGER in item.tags:
                continue
            if item.step <= current_step - 3 and item.immutable:
                ledger.compress_item(
                    item.item_id,
                    risky_compression(item.content),
                    "Fixed compression of older items.",
                    current_step,
                )


class NonLearningManager(BaseManager):
    def manage_after_observation(self, ledger: ContextLedger, current_step: int) -> None:
        for item in list(ledger.active_items()):
            if TAG_TRIGGER in item.tags:
                continue
            if TAG_DISTRACTOR in item.tags and item.step <= current_step - 1:
                ledger.archive_item(item.item_id, "Archive stale distractor.", current_step)
            elif item.step <= current_step - 3 and item.immutable:
                ledger.compress_item(
                    item.item_id,
                    risky_compression(item.content),
                    "Compress older information without lessons.",
                    current_step,
                )


class LessonAwareManager(BaseManager):
    def __init__(self, lesson_store: LessonStore, model_client=None) -> None:
        self.lesson_store = lesson_store
        self.model_client = model_client

    def manage_after_observation(self, ledger: ContextLedger, current_step: int) -> None:
        for item in list(ledger.active_items()):
            if TAG_TRIGGER in item.tags:
                continue

            # --- Archive step (most aggressive reduction) ---
            # Distractors are candidates for archiving.
            if TAG_DISTRACTOR in item.tags and item.step <= current_step - 1:
                archive_lessons = self.lesson_store.retrieve(item.tags, operation=OP_ARCHIVE)
                if archive_lessons:
                    # A lesson says archiving this tag pattern caused a failure before.
                    # Keep verbatim — it may contain information that looks like a
                    # distractor but is actually needed.
                    ledger.keep_item(
                        item.item_id,
                        "Lesson: archiving this item type causes task failures — keeping verbatim.",
                        current_step,
                    )
                else:
                    ledger.archive_item(item.item_id, "Archive stale distractor.", current_step)
                continue

            # --- Compress step ---
            if item.step <= current_step - 3 and item.immutable:
                compress_lessons = self.lesson_store.retrieve(item.tags, operation=OP_COMPRESS)
                if compress_lessons:
                    # A lesson says compressing this tag pattern caused a failure.
                    # Keep verbatim instead.
                    ledger.keep_item(
                        item.item_id,
                        "Lesson: compressing this item type causes task failures — keeping verbatim.",
                        current_step,
                    )
                else:
                    compressed = intelligent_compression(item.content, self.model_client)
                    ledger.compress_item(
                        item.item_id,
                        compressed,
                        "Compress older information using learned compression strategy.",
                        current_step,
                    )
