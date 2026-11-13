"""
Stores and retrieves lessons learned from counterfactual replay.

A lesson is a text rule like:
  "Never COMPRESS items tagged [critical, negation]."

Lessons go through three promotion stages based on how many times
the same pattern caused a failure:
  candidate    (1 confirmation)  - tentative, not yet used
  validated    (2+ confirmations) - used by the learning manager
  transferable (5+ confirmations) - strong enough to give to Agent B
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


# Maps status names to numeric ranks so we can compare them easily.
# Higher rank = more trustworthy lesson.
_STATUS_RANK: dict[str, int] = {
    "candidate":    0,
    "validated":    1,
    "transferable": 2,
}

# How many harmful-replay confirmations a lesson needs to be promoted.
# These are intentionally conservative — a false lesson would hurt performance.
_THRESHOLD_VALIDATED:    int = 2
_THRESHOLD_TRANSFERABLE: int = 5


@dataclass(frozen=True)
class Lesson:
    """
    A single causal rule learned from a harmful replay.

    frozen=True means the fields cannot be accidentally overwritten after creation.
    This is important because the same Lesson object may be shared between the
    main lesson store and the transfer store given to Agent B.

    Fields:
      lesson_id            - unique ID, e.g. "lesson_0001"
      lesson_type          - always "compression_safety" for now
      operation            - "COMPRESS" or "ARCHIVE" — what went wrong
      trigger_tags         - which tags on the item triggered this lesson
      instruction          - human-readable rule text
      supporting_decision_id - which ledger decision caused the failure
      original_score       - task score before counterfactual (always 0.0)
      counterfactual_score - task score after undoing the decision (always 1.0)
      source_agent         - which agent learned this lesson (for transfer tracking)
    """
    lesson_id:              str
    lesson_type:            str
    operation:              str
    trigger_tags:           tuple[str, ...]
    instruction:            str
    supporting_decision_id: str
    original_score:         float
    counterfactual_score:   float
    source_agent:           str = "agent_a"


class LessonStore:
    """
    Holds all lessons and manages their promotion lifecycle.

    Key design: lessons are deduplicated by (operation, trigger_tags, instruction).
    Every time the same pattern causes another failure, the support_count increments.
    When support_count crosses a threshold, the lesson is promoted.

    This separation — immutable Lesson object + mutable count in the store —
    means lesson data is never accidentally overwritten, only the confidence count.
    """

    def __init__(self) -> None:
        self.lessons: list[Lesson] = []
        # support_count and status live here, keyed by lesson_id.
        # They are NOT stored on the Lesson object because Lesson is frozen.
        self._support_count: dict[str, int] = {}
        self._status:        dict[str, str] = {}
        self._id_counter:    int = 0

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def add(self, lesson: Lesson) -> None:
        """
        Add a lesson or increment its confirmation count if it already exists.

        If the same (operation, trigger_tags, instruction) pattern has been seen
        before, we do NOT add a duplicate — we just count one more confirmation
        and check whether that pushes the lesson to the next promotion stage.
        """
        for existing in self.lessons:
            if (existing.operation    == lesson.operation
                    and set(existing.trigger_tags) == set(lesson.trigger_tags)
                    and existing.instruction       == lesson.instruction):
                # Same pattern seen again — increment and possibly promote.
                self._support_count[existing.lesson_id] += 1
                self._promote_if_ready(existing.lesson_id)
                return

        # Brand new pattern — assign an ID, start at candidate status.
        self._id_counter += 1
        lesson_id = f"lesson_{self._id_counter:04d}"

        # Dataclass is frozen so we must rebuild it with the assigned ID.
        # Convert trigger_tags to tuple here so callers can pass either list or tuple.
        new_lesson = Lesson(
            lesson_id=lesson_id,
            lesson_type=lesson.lesson_type,
            operation=lesson.operation,
            trigger_tags=tuple(lesson.trigger_tags),
            instruction=lesson.instruction,
            supporting_decision_id=lesson.supporting_decision_id,
            original_score=lesson.original_score,
            counterfactual_score=lesson.counterfactual_score,
            source_agent=lesson.source_agent,
        )
        self.lessons.append(new_lesson)
        self._support_count[lesson_id] = 1
        self._status[lesson_id]        = "candidate"

    def _promote_if_ready(self, lesson_id: str) -> None:
        """Check whether a lesson's support_count has crossed a promotion threshold."""
        count  = self._support_count[lesson_id]
        status = self._status[lesson_id]

        if status == "candidate" and count >= _THRESHOLD_VALIDATED:
            self._status[lesson_id] = "validated"
        elif status == "validated" and count >= _THRESHOLD_TRANSFERABLE:
            self._status[lesson_id] = "transferable"

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def get_status(self, lesson_id: str) -> str:
        """Return the current promotion status of a lesson."""
        return self._status.get(lesson_id, "candidate")

    def get_support_count(self, lesson_id: str) -> int:
        """Return how many times this lesson has been confirmed."""
        return self._support_count.get(lesson_id, 0)

    def retrieve(
        self,
        tags:       list[str],
        operation:  str,
        min_status: str = "validated",
    ) -> list[Lesson]:
        """
        Return all lessons relevant to an item with the given tags and operation.

        Uses set intersection: a lesson matches if ANY of its trigger_tags
        appears in the item's tags. This is intentionally exact — no fuzzy
        matching — so retrieval results are deterministic and reproducible.

        min_status filters out lessons that have not yet been confirmed enough:
          "candidate"    - return everything (rarely useful)
          "validated"    - return lessons confirmed 2+ times (default)
          "transferable" - return only lessons confirmed 5+ times
        """
        tag_set      = set(tags)
        min_rank     = _STATUS_RANK.get(min_status, 0)

        return [
            lesson
            for lesson in self.lessons
            if lesson.operation == operation
            and tag_set.intersection(lesson.trigger_tags)
            and _STATUS_RANK.get(self._status.get(lesson.lesson_id, "candidate"), 0) >= min_rank
        ]

    def get_transferable(self) -> list[Lesson]:
        """
        Return only lessons that have been confirmed enough times to share with Agent B.

        These are lessons so consistently validated that they are likely universal
        rules rather than quirks of Agent A's specific training tasks.
        """
        return [
            lesson for lesson in self.lessons
            if self._status.get(lesson.lesson_id) == "transferable"
        ]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> list[dict]:
        """
        Convert all lessons to a JSON-serialisable list of dicts.
        Includes support_count and status so the output JSON is fully informative.
        """
        result = []
        for lesson in self.lessons:
            row = asdict(lesson)
            row["support_count"] = self._support_count.get(lesson.lesson_id, 1)
            row["status"]        = self._status.get(lesson.lesson_id, "candidate")
            result.append(row)
        return result
