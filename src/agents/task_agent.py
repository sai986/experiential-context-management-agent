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
    if "under 500 usd" in context_lower or "500 usd" in context_lower:
        return "Use free open-source tooling — zero cost, no paid SaaS required."
    if "single json file" in context_lower or "json file" in context_lower:
        return "Output a single JSON file — no CSV or XML."

    # --- delayed_dependency family ---
    if "windows server" in context_lower or "windows-only" in context_lower or "windows only" in context_lower:
        return "Use a Windows PowerShell procedure."
    if "read-only" in context_lower and "replica" in context_lower:
        return "Use a read-only backup and validation procedure."
    if "pdf report" in context_lower or "pdf" in context_lower:
        return "Deliver a PDF report as the final output."

    # --- dependency_chain family ---
    # All four steps must be visible for the agent to reproduce the full chain.
    # Variant 0: Back up / Drain / dry-run / health check
    if "back up" in context_lower and "drain" in context_lower:
        return "Back up the configuration, Drain connections, apply with dry-run flag, re-enable after health check passes."
    # Variant 1: Export / checksums / transformation pipeline / sign-off
    if "export" in context_lower and "checksums" in context_lower:
        return "Export records, validate checksums, run the transformation pipeline, promote after sign-off."
    # Variant 2: Freeze writes / Replicate / replica lag / DNS record
    if "freeze writes" in context_lower and "replicate" in context_lower:
        return "Freeze writes to the shard, Replicate to new cluster, verify replica lag, switch the DNS record."
    # Step 1 was compressed away — only later steps visible
    if "drain" in context_lower or "checksums" in context_lower or "replica lag" in context_lower:
        return "Incomplete procedure — the first step was lost during compression."

    # --- false_distractor family ---
    # The critical observation is wrongly tagged as a distractor and archived.
    # If it survived, the agent can answer correctly.
    if "staging only" in context_lower or "use staging" in context_lower:
        return "Use staging only — do not deploy to production."
    if "second engineer" in context_lower or "reviewed by a second" in context_lower:
        return "Require a second engineer code review before merge."
    if "200ms" in context_lower and "synchronous" in context_lower:
        return "Design for async processing — respond within 200ms, no synchronous external calls."

    # --- negation_critical family ---
    # Many variants: "must not store", "must not modify", "must not make", etc.
    negation_phrases = ("must not", "do not", "cannot", "never store", "not permitted")
    if any(phrase in context_lower for phrase in negation_phrases):
        if "pii" in context_lower or "personally identifiable" in context_lower:
            return "Process data in-memory only — no PII stored on disk."
        if "original" in context_lower or "source file" in context_lower:
            return "Work on a copy only — original source files are unmodified."
        if "api call" in context_lower or "3 api" in context_lower or "batch" in context_lower:
            return "Use batched requests — at most 3 API calls per task."
        return "Avoided the forbidden approach as required by the constraint."

    # --- synonym_match family ---
    if "hosted" in context_lower or "cloud platform" in context_lower:
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
