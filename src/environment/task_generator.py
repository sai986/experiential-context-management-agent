from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Observation:
    step: int
    content: str
    tags: list[str]


@dataclass(frozen=True)
class ControlledTask:
    task_id: str
    family: str
    observations: list[Observation]
    question: str
    expected_constraints: list[str]
    forbidden_outputs: list[str]
    required_outputs: list[str]


def _hard_constraint_task(i: int) -> ControlledTask:
    variants = [
        (
            "The deployment must run offline and must not use any cloud API.",
            ["offline", "no cloud API"],
            ["cloud API", "hosted endpoint", "online service", "SaaS inference"],
            ["local", "offline"],
        ),
        (
            "The budget must not exceed 5000 INR and paid managed services are not allowed.",
            ["budget <= 5000 INR", "no paid managed services"],
            ["paid service", "managed service", "premium API"],
            ["under 5000", "free"],
        ),
        (
            "The final answer must contain exactly three bullet points and no paragraph summary.",
            ["exactly three bullet points", "no paragraph summary"],
            ["paragraph summary", "four bullet", "five bullet"],
            ["three bullet"],
        ),
    ]
    critical, expected, forbidden, required = variants[i % len(variants)]
    observations = [
        Observation(1, critical, ["critical", "hard_constraint", "negation"]),
        Observation(2, "The user prefers a simple implementation with familiar tools.", ["preference"]),
        Observation(3, "A previous prototype used a more complex architecture than needed.", ["distractor"]),
        Observation(4, "The final recommendation should be practical and concise.", ["preference"]),
        Observation(5, "There are several optional deployment approaches available.", ["distractor"]),
    ]
    return ControlledTask(
        task_id=f"hard_constraint_{i:03d}",
        family="hard_constraint",
        observations=observations,
        question="Recommend a solution that satisfies all user requirements.",
        expected_constraints=expected,
        forbidden_outputs=forbidden,
        required_outputs=required,
    )


def _delayed_dependency_task(i: int) -> ControlledTask:
    variants = [
        (
            "The user's machine is Windows-only; Linux commands are not acceptable.",
            ["Windows-only", "no Linux commands"],
            ["apt-get", "bash script", "systemctl", "Linux"],
            ["Windows", "PowerShell"],
        ),
        (
            "The database is read-only during migration; destructive writes are not allowed.",
            ["read-only database", "no destructive writes"],
            ["DELETE", "DROP TABLE", "overwrite"],
            ["read-only", "backup"],
        ),
        (
            "The final artifact must be a CSV file; JSON output is not accepted.",
            ["CSV output", "no JSON output"],
            ["JSON", ".json"],
            ["CSV", ".csv"],
        ),
    ]
    critical, expected, forbidden, required = variants[i % len(variants)]
    observations = [
        Observation(1, critical, ["critical", "delayed_dependency", "negation"]),
        Observation(2, "The task initially appears to be a generic setup recommendation.", ["distractor"]),
        Observation(3, "The user asks for the shortest reliable path.", ["preference"]),
        Observation(4, "Several unrelated prior attempts are mentioned but not required.", ["distractor"]),
        Observation(5, "Now provide the concrete final procedure.", ["trigger"]),
    ]
    return ControlledTask(
        task_id=f"delayed_dependency_{i:03d}",
        family="delayed_dependency",
        observations=observations,
        question="Provide the final procedure while respecting earlier constraints.",
        expected_constraints=expected,
        forbidden_outputs=forbidden,
        required_outputs=required,
    )


def generate_tasks(n_per_family: int = 10) -> list[ControlledTask]:
    tasks: list[ControlledTask] = []
    for i in range(n_per_family):
        tasks.append(_hard_constraint_task(i))
        tasks.append(_delayed_dependency_task(i))
    return tasks


def write_jsonl(path: Path, tasks: Iterable[ControlledTask]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for task in tasks:
            row = asdict(task)
            row["observations"] = [asdict(obs) for obs in task.observations]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def materialize_default_splits(base_dir: Path = Path("data/controlled")) -> None:
    tasks = generate_tasks(n_per_family=15)
    write_jsonl(base_dir / "train.jsonl", tasks[:20])
    write_jsonl(base_dir / "validation.jsonl", tasks[20:25])
    write_jsonl(base_dir / "test.jsonl", tasks[25:30])


if __name__ == "__main__":
    materialize_default_splits()

