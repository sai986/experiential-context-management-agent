from __future__ import annotations

from dataclasses import dataclass

from src.environment.task_generator import ControlledTask


@dataclass(frozen=True)
class EvaluationResult:
    task_success: int
    constraint_satisfaction: float
    forbidden_output_violation: int
    missing_required_terms: list[str]
    forbidden_terms_found: list[str]


def evaluate_answer(task: ControlledTask, answer: str) -> EvaluationResult:
    answer_lower = answer.lower()
    forbidden_found = [term for term in task.forbidden_outputs if term.lower() in answer_lower]
    missing_required = [term for term in task.required_outputs if term.lower() not in answer_lower]

    forbidden_violation = int(bool(forbidden_found))
    required_score = 1.0
    if task.required_outputs:
        required_score = (len(task.required_outputs) - len(missing_required)) / len(task.required_outputs)

    constraint_satisfaction = 0.0 if forbidden_violation else required_score
    task_success = int(constraint_satisfaction == 1.0)
    return EvaluationResult(
        task_success=task_success,
        constraint_satisfaction=constraint_satisfaction,
        forbidden_output_violation=forbidden_violation,
        missing_required_terms=missing_required,
        forbidden_terms_found=forbidden_found,
    )

