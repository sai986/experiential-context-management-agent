from __future__ import annotations

from dataclasses import dataclass

from src.environment.task_generator import ControlledTask

# Words that negate a term when they appear within 6 words before it.
_NEGATION_WORDS: frozenset[str] = frozenset({
    "not", "no", "never", "without",
    "don't", "doesn't", "do not", "does not",
    "cannot", "can't",
})


@dataclass(frozen=True)
class EvaluationResult:
    task_success: int
    constraint_satisfaction: float
    forbidden_output_violation: int
    missing_required_terms: list[str]
    forbidden_terms_found: list[str]


def _term_satisfied(term: str, answer_lower: str, synonym_groups: list[list[str]]) -> bool:
    """Return True if term OR any of its synonyms appears in the answer.

    Checking synonyms here (rather than in a separate pass after required_outputs)
    prevents the required_outputs check from marking a term missing before the
    synonym check gets a chance to find a valid match.
    """
    term_lower = term.lower()
    if term_lower in answer_lower:
        return True
    for group in synonym_groups:
        group_lower = [s.lower() for s in group]
        if term_lower in group_lower:
            # This term belongs to a synonym group — any member counts.
            if any(s in answer_lower for s in group_lower):
                return True
    return False


def _is_negated(term: str, answer_lower: str) -> bool:
    """Return True if a negation word appears within 6 words before the term.

    Example: "do not use offline mode" — "offline" is negated.
    """
    idx = answer_lower.find(term.lower())
    if idx == -1:
        return False
    words_before = answer_lower[:idx].split()[-6:]
    return bool(set(words_before) & _NEGATION_WORDS)


def evaluate_answer(task: ControlledTask, answer: str) -> EvaluationResult:
    answer_lower = answer.lower()
    synonym_groups: list[list[str]] = getattr(task, "synonym_groups", [])

    forbidden_found = [term for term in task.forbidden_outputs if term.lower() in answer_lower]

    missing_required = [
        term for term in task.required_outputs
        if not _term_satisfied(term, answer_lower, synonym_groups)
        or _is_negated(term, answer_lower)
    ]

    forbidden_violation = int(bool(forbidden_found))
    required_score = 1.0
    if task.required_outputs:
        required_score = (len(task.required_outputs) - len(missing_required)) / len(task.required_outputs)

    constraint_satisfaction = 0.0 if forbidden_violation else required_score
    # Use >= 0.9999 instead of == 1.0 to avoid floating-point precision failures
    # (e.g. 2/2 computing as 0.9999999... on some platforms).
    task_success = int(constraint_satisfaction >= 0.9999)

    return EvaluationResult(
        task_success=task_success,
        constraint_satisfaction=constraint_satisfaction,
        forbidden_output_violation=forbidden_violation,
        missing_required_terms=missing_required,
        forbidden_terms_found=forbidden_found,
    )
