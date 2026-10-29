from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from src.agents.task_agent import simulated_task_agent
from src.attribution.replay import (
    create_lessons_from_replays,
    find_suspect_compressions,
    replay_results_to_dict,
    replay_restore_original,
)
from src.context.ledger import ContextLedger
from src.context.manager import FixedCompressionManager, FullContextManager, LessonAwareManager, NonLearningManager
from src.environment.evaluator import evaluate_answer
from src.environment.task_generator import ControlledTask, Observation, materialize_default_splits
from src.lessons.lesson_store import LessonStore


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "controlled"
RAW_RESULTS_DIR = ROOT / "results" / "raw"
PROCESSED_RESULTS_DIR = ROOT / "results" / "processed"


def load_tasks(path: Path) -> list[ControlledTask]:
    tasks: list[ControlledTask] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        observations = [Observation(**obs) for obs in row["observations"]]
        tasks.append(
            ControlledTask(
                task_id=row["task_id"],
                family=row["family"],
                observations=observations,
                question=row["question"],
                expected_constraints=row["expected_constraints"],
                forbidden_outputs=row["forbidden_outputs"],
                required_outputs=row["required_outputs"],
            )
        )
    return tasks


def run_episode(task: ControlledTask, manager, condition: str) -> dict:
    ledger = ContextLedger(episode_id=f"{condition}_{task.task_id}")
    for observation in task.observations:
        ledger.add_raw(observation.step, observation.content, observation.tags)
        manager.manage_after_observation(ledger, observation.step)

    answer = simulated_task_agent(task, ledger.active_context())
    evaluation = evaluate_answer(task, answer)
    return {
        "task_id": task.task_id,
        "condition": condition,
        "family": task.family,
        "answer": answer,
        "evaluation": asdict(evaluation),
        "active_context_tokens": ledger.token_estimate(),
        "ledger": ledger,
    }


def summarize(results: list[dict]) -> dict:
    by_condition: dict[str, list[dict]] = {}
    for result in results:
        by_condition.setdefault(result["condition"], []).append(result)

    summary = {}
    for condition, rows in by_condition.items():
        n = len(rows)
        summary[condition] = {
            "n": n,
            "success_rate": sum(row["evaluation"]["task_success"] for row in rows) / n,
            "avg_constraint_satisfaction": sum(row["evaluation"]["constraint_satisfaction"] for row in rows) / n,
            "avg_active_context_tokens": sum(row["active_context_tokens"] for row in rows) / n,
        }
    return summary


def serializable_result(result: dict) -> dict:
    out = {key: value for key, value in result.items() if key != "ledger"}
    out["ledger"] = result["ledger"].to_dict()
    return out


def main() -> None:
    RAW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    materialize_default_splits(DATA_DIR)
    train_tasks = load_tasks(DATA_DIR / "train.jsonl")
    test_tasks = load_tasks(DATA_DIR / "test.jsonl")

    lesson_store = LessonStore()

    # Learn lessons from non-learning manager failures on the train split.
    learning_results = []
    all_replays = []
    for task in train_tasks:
        result = run_episode(task, NonLearningManager(), "train_non_learning_manager")
        learning_results.append(result)
        if result["evaluation"]["task_success"] == 0:
            episode_replays = []
            suspects = find_suspect_compressions(result["ledger"])
            for suspect in suspects[:1]:
                replay = replay_restore_original(task, result["ledger"], suspect, result["answer"])
                episode_replays.append(replay)
                all_replays.append(replay)
            create_lessons_from_replays(episode_replays, result["ledger"], lesson_store)

    # Evaluate pilot conditions on the held-out test split.
    conditions = [
        ("full_context", FullContextManager()),
        ("fixed_compression", FixedCompressionManager()),
        ("non_learning_manager", NonLearningManager()),
        ("attributed_lesson_manager", LessonAwareManager(lesson_store)),
    ]
    test_results = []
    for condition, manager in conditions:
        for task in test_tasks:
            test_results.append(run_episode(task, manager, condition))

    raw_payload = {
        "learned_lessons": lesson_store.to_dict(),
        "training_replays": replay_results_to_dict(all_replays),
        "training_results": [serializable_result(result) for result in learning_results],
        "test_results": [serializable_result(result) for result in test_results],
    }
    (RAW_RESULTS_DIR / "pilot_trace.json").write_text(json.dumps(raw_payload, indent=2), encoding="utf-8")

    summary = {
        "summary": summarize(test_results),
        "learned_lessons": lesson_store.to_dict(),
        "num_training_replays": len(all_replays),
        "num_harmful_replays": sum(1 for replay in all_replays if replay.label == "HARMFUL"),
    }
    (PROCESSED_RESULTS_DIR / "pilot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
