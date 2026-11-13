from __future__ import annotations

from src.environment.task_generator import ControlledTask


def simulated_task_agent(task: ControlledTask, managed_context: str) -> str:
    """Deterministic frozen agent used for USE_LLM=0 testing.

    Behaviour depends entirely on whether exact constraint keywords survived
    compression in managed_context. This makes the replay mechanism measurable
    without API keys or model nondeterminism.

    Intentionally does NOT look at task.family or task.required_outputs —
    it only reads the managed context, just as a real agent would.
    """
    context_lower = managed_context.lower()

    # --- hard_constraint family ---
    if "offline" in context_lower and "cloud api" in context_lower:
        return "Use a local offline model with no cloud API."
    if "5000 inr" in context_lower and "paid managed services" in context_lower:
        return "Use free local tooling under 5000 INR."
    if "exactly three bullet points" in context_lower:
        return "Return exactly three bullet points."

    # --- delayed_dependency family ---
    if "windows-only" in context_lower or "windows only" in context_lower:
        return "Use a Windows PowerShell procedure."
    if "read-only" in context_lower:
        return "Use a read-only backup and validation procedure."
    if "csv file" in context_lower or ".csv" in context_lower:
        return "Export the final artifact as a CSV .csv file."

    # --- dependency_chain family ---
    # All four steps must be visible for the agent to find the full chain.
    if "step 1" in context_lower and "step 4" in context_lower:
        return "Follow all four steps in order: step 1, step 2, step 3, step 4."
    if "step 1" in context_lower:
        return "Follow step 1 only — later steps were not retained."

    # --- false_distractor family ---
    # The critical observation is wrongly tagged as a distractor and archived.
    # If it survived, the agent can answer correctly.
    if "do not deploy to production" in context_lower:
        return "Do not deploy to production — use staging only."

    # --- negation_critical family ---
    # Many variants: "must not store", "must not modify", "must not make", etc.
    negation_phrases = ("must not", "do not", "cannot", "never store", "not permitted")
    if any(phrase in context_lower for phrase in negation_phrases):
        if "pii" in context_lower or "personally identifiable" in context_lower:
            return "Process data in-memory only — no PII stored on disk."
        if "original" in context_lower or "source file" in context_lower:
            return "Work on a copy only — read-only source files are unmodified."
        if "api call" in context_lower or "3 api" in context_lower or "batch" in context_lower:
            return "Use batched requests — at most 3 API calls per task."
        return "Avoided the forbidden approach as required by the constraint."

    # --- synonym_match family ---
    if "hosted" in context_lower or "cloud" in context_lower or "cloud platform" in context_lower:
        return "Deploy using a hosted cloud environment."
    if "accessible" in context_lower or "wcag" in context_lower or "a11y" in context_lower:
        return "Build an accessible interface that meets WCAG 2.1 AA standards."
    if "one hour" in context_lower or "3600" in context_lower or "60 minutes" in context_lower:
        return "Complete the pipeline within one hour — under 3600 seconds total."

    # Fallback — constraint keywords were lost (compressed away).
    return "Use a hosted endpoint through a cloud API for the fastest deployment."


def llm_task_agent(task: ControlledTask, managed_context: str, model_client) -> str:
    """Real LLM agent used when USE_LLM=1.

    The prompt intentionally omits task.family and evaluation metadata
    so the model cannot game the scoring by peeking at hidden fields.
    It only sees the managed context and the question.
    """
    prompt = (
        "You are an expert technical consultant. "
        "Read the context below and answer the question that follows.\n\n"
        "=== Context ===\n"
        f"{managed_context}\n\n"
        "=== Question ===\n"
        f"{task.question}\n\n"
        "Give a concise, specific recommendation in 2-3 sentences."
    )
    return model_client.complete(prompt)
