from __future__ import annotations

from src.environment.task_generator import ControlledTask


def simulated_task_agent(task: ControlledTask, managed_context: str) -> str:
    """Deterministic frozen agent used for the mid-submission pilot.

    The behavior intentionally depends on whether exact constraints survive in
    the managed context. This makes the replay mechanism measurable without API
    keys or hosted-model nondeterminism.
    """
    context_lower = managed_context.lower()

    if task.family == "hard_constraint":
        if "offline" in context_lower and "cloud api" in context_lower:
            return "Use a local offline model with no cloud API."
        if "5000 inr" in context_lower and "paid managed services" in context_lower:
            return "Use free local tooling under 5000 INR."
        if "exactly three bullet points" in context_lower:
            return "Return exactly three bullet points."
        return "Use a hosted endpoint through a cloud API for the fastest deployment."

    if task.family == "delayed_dependency":
        if "windows-only" in context_lower or "windows only" in context_lower:
            return "Use a Windows PowerShell procedure."
        if "read-only" in context_lower:
            return "Use a read-only backup and validation procedure."
        if "csv file" in context_lower:
            return "Export the final artifact as a CSV .csv file."
        return "Use a Linux bash script with apt-get and JSON output."

    return "No recommendation available."

