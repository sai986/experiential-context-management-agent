from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Callable

from src.constants import SUSPECT_TAGS
from src.context.ledger import ContextLedger, ManagementDecision
from src.environment.evaluator import evaluate_answer
from src.environment.task_generator import ControlledTask
from src.lessons.lesson_store import Lesson, LessonStore


@dataclass
class ReplayResult:
    target_decision_id: str
    intervention_type: str
    original_score: float
    counterfactual_score: float
    label: str
    original_answer: str
    counterfactual_answer: str


def _suspect_tag_count(decision: ManagementDecision, ledger: ContextLedger) -> int:
    """How many suspect tags appeared on the input items of this decision."""
    count = 0
    for item_id in decision.input_item_ids:
        item = ledger.items.get(item_id)
        if item:
            count += len(SUSPECT_TAGS.intersection(item.tags))
    return count


def find_suspect_compressions(ledger: ContextLedger) -> list[ManagementDecision]:
    """Find COMPRESS decisions that touched items with suspect tags.

    Sorted most-suspicious first so the most likely culprits are tested first
    if the caller limits replay budget.
    """
    suspects = []
    for decision in ledger.decisions:
        if decision.operation != "COMPRESS" or not decision.output_item_id:
            continue
        input_items = [ledger.items[iid] for iid in decision.input_item_ids]
        if any(SUSPECT_TAGS.intersection(item.tags) for item in input_items):
            suspects.append(decision)
    return sorted(suspects, key=lambda d: _suspect_tag_count(d, ledger), reverse=True)


def find_suspect_archives(ledger: ContextLedger) -> list[ManagementDecision]:
    """Find ARCHIVE decisions that touched items with suspect tags.

    Sorted most-suspicious first.
    """
    suspects = []
    for decision in ledger.decisions:
        if decision.operation != "ARCHIVE":
            continue
        input_items = [ledger.items[iid] for iid in decision.input_item_ids]
        if any(SUSPECT_TAGS.intersection(item.tags) for item in input_items):
            suspects.append(decision)
    return sorted(suspects, key=lambda d: _suspect_tag_count(d, ledger), reverse=True)


def replay_restore_original(
    task: ControlledTask,
    ledger: ContextLedger,
    decision: ManagementDecision,
    original_answer: str,
    agent_fn: Callable | None = None,
) -> ReplayResult:
    """Counterfactual for COMPRESS: restore the original item, re-run the agent.

    If the score improves, the compression caused the failure (label=HARMFUL).
    agent_fn defaults to simulated_task_agent if not provided.
    """
    if agent_fn is None:
        from src.agents.task_agent import simulated_task_agent
        agent_fn = simulated_task_agent

    original_eval = evaluate_answer(task, original_answer)
    replay_ledger = deepcopy(ledger)

    if decision.output_item_id:
        replay_ledger.restore_original(
            decision.output_item_id,
            "Counterfactual replay: restore original content to test compression impact.",
            step=999,
        )

    counterfactual_answer = agent_fn(task, replay_ledger.active_context())
    counterfactual_eval = evaluate_answer(task, counterfactual_answer)
    label = "HARMFUL" if counterfactual_eval.task_success > original_eval.task_success else "NEUTRAL"

    return ReplayResult(
        target_decision_id=decision.decision_id,
        intervention_type="RESTORE_ORIGINAL",
        original_score=float(original_eval.task_success),
        counterfactual_score=float(counterfactual_eval.task_success),
        label=label,
        original_answer=original_answer,
        counterfactual_answer=counterfactual_answer,
    )


def replay_archive_keep(
    task: ControlledTask,
    ledger: ContextLedger,
    decision: ManagementDecision,
    original_answer: str,
    agent_fn: Callable | None = None,
) -> ReplayResult:
    """Counterfactual for ARCHIVE: restore the archived item, re-run the agent.

    If the score improves, the archive caused the failure (label=HARMFUL).
    agent_fn defaults to simulated_task_agent if not provided.
    """
    if agent_fn is None:
        from src.agents.task_agent import simulated_task_agent
        agent_fn = simulated_task_agent

    original_eval = evaluate_answer(task, original_answer)
    replay_ledger = deepcopy(ledger)

    for item_id in decision.input_item_ids:
        replay_ledger.unarchive_item(
            item_id,
            "Counterfactual replay: undo archive to test if keeping was better.",
            step=999,
        )

    counterfactual_answer = agent_fn(task, replay_ledger.active_context())
    counterfactual_eval = evaluate_answer(task, counterfactual_answer)
    label = "HARMFUL" if counterfactual_eval.task_success > original_eval.task_success else "NEUTRAL"

    return ReplayResult(
        target_decision_id=decision.decision_id,
        intervention_type="UNARCHIVE_KEEP",
        original_score=float(original_eval.task_success),
        counterfactual_score=float(counterfactual_eval.task_success),
        label=label,
        original_answer=original_answer,
        counterfactual_answer=counterfactual_answer,
    )


def create_lessons_from_replays(
    replays: list[ReplayResult],
    ledger: ContextLedger,
    lesson_store: LessonStore,
) -> None:
    """Convert HARMFUL replay results into lessons and add them to the store.

    COMPRESS failures: filter tags to SUSPECT_TAGS — lesson is about which
    danger tags make compression unsafe.

    ARCHIVE failures: keep ALL tags on the archived item — stripping to
    SUSPECT_TAGS would remove tags like 'distractor' that are the actual
    cause, making the lesson wrong.
    """
    for replay in replays:
        if replay.label != "HARMFUL":
            continue

        decision = next(d for d in ledger.decisions if d.decision_id == replay.target_decision_id)

        all_tags: list[str] = []
        for item_id in decision.input_item_ids:
            all_tags.extend(ledger.items[item_id].tags)

        if decision.operation == "ARCHIVE":
            # Keep all tags — the trigger may be a non-SUSPECT tag like "distractor"
            trigger_tags = sorted(set(all_tags))
            instruction = "Never archive items with these tags — they are needed to complete the task."
            lesson_type = "archive_safety"
        else:
            # COMPRESS: filter to only known danger tags
            trigger_tags = sorted(tag for tag in set(all_tags) if tag in SUSPECT_TAGS)
            instruction = "Preserve critical constraints, negations, exact values, and delayed dependencies verbatim during compression."
            lesson_type = "compression_safety"

        if not trigger_tags:
            trigger_tags = ["critical"]

        lesson_store.add(
            Lesson(
                lesson_id="lesson_placeholder",  # LessonStore.add() assigns the real ID
                lesson_type=lesson_type,
                operation=decision.operation,
                trigger_tags=trigger_tags,
                instruction=instruction,
                supporting_decision_id=decision.decision_id,
                original_score=replay.original_score,
                counterfactual_score=replay.counterfactual_score,
            )
        )


def replay_results_to_dict(replays: list[ReplayResult]) -> list[dict]:
    return [asdict(replay) for replay in replays]
