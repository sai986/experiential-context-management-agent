from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass

from src.agents.task_agent import simulated_task_agent
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


def find_suspect_compressions(ledger: ContextLedger) -> list[ManagementDecision]:
    suspects = []
    for decision in ledger.decisions:
        if decision.operation != "COMPRESS" or not decision.output_item_id:
            continue
        input_items = [ledger.items[item_id] for item_id in decision.input_item_ids]
        if any("critical" in item.tags or "negation" in item.tags for item in input_items):
            suspects.append(decision)
    return suspects


def replay_restore_original(task: ControlledTask, ledger: ContextLedger, decision: ManagementDecision, original_answer: str) -> ReplayResult:
    original_eval = evaluate_answer(task, original_answer)
    replay_ledger = deepcopy(ledger)
    if decision.output_item_id:
        replay_ledger.restore_original(
            decision.output_item_id,
            "Counterfactual replay restored original content after compression.",
            step=999,
        )
    counterfactual_answer = simulated_task_agent(task, replay_ledger.active_context())
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


def create_lessons_from_replays(replays: list[ReplayResult], ledger: ContextLedger, lesson_store: LessonStore) -> None:
    for replay in replays:
        if replay.label != "HARMFUL":
            continue
        decision = next(d for d in ledger.decisions if d.decision_id == replay.target_decision_id)
        tags: list[str] = []
        for item_id in decision.input_item_ids:
            tags.extend(ledger.items[item_id].tags)
        trigger_tags = sorted(set(tag for tag in tags if tag in {"critical", "hard_constraint", "negation", "delayed_dependency"}))
        if not trigger_tags:
            trigger_tags = ["critical"]
        lesson_store.add(
            Lesson(
                lesson_id=f"lesson_{len(lesson_store.lessons) + 1:04d}",
                lesson_type="compression_safety",
                operation="COMPRESS",
                trigger_tags=trigger_tags,
                instruction="Preserve critical constraints, negations, exact values, and delayed dependencies verbatim during compression.",
                supporting_decision_id=decision.decision_id,
                original_score=replay.original_score,
                counterfactual_score=replay.counterfactual_score,
            )
        )


def replay_results_to_dict(replays: list[ReplayResult]) -> list[dict]:
    return [asdict(replay) for replay in replays]

