"""
Investigation Agent (Agent 1).

Reads a raw incident description and produces a list of structured observations,
each with content and tags. These observations are then fed into the context
ledger one by one so the context manager can decide what to keep, compress,
or archive.

Two implementations:
  simulated_investigation_agent — deterministic, no API calls (USE_LLM=0)
  llm_investigation_agent       — real LLM call with few-shot prompt (USE_LLM=1)
"""
from __future__ import annotations

import json
import re

from src.constants import (
    TAG_CRITICAL,
    TAG_DELAYED_DEPENDENCY,
    TAG_DISTRACTOR,
    TAG_HARD_CONSTRAINT,
    TAG_NEGATION,
    TAG_PREFERENCE,
    TAG_TRIGGER,
)
from src.environment.task_generator import Observation

# ---------------------------------------------------------------------------
# Few-shot examples for the LLM prompt
# ---------------------------------------------------------------------------

_FEW_SHOT_EXAMPLES = """
Example 1
Input: "The system must not store user passwords in plaintext. The team uses PostgreSQL. An older version cached tokens on disk."
Output:
[
  {"step": 1, "content": "The system must not store user passwords in plaintext.", "tags": ["critical", "hard_constraint", "negation"]},
  {"step": 2, "content": "The team uses PostgreSQL.", "tags": ["preference"]},
  {"step": 3, "content": "An older version cached tokens on disk.", "tags": ["distractor"]}
]

Example 2
Input: "Step 1: Drain load balancer. Step 2: Run migration. Step 3: Re-enable traffic. The team prefers minimal downtime. Previous migrations used a different tool."
Output:
[
  {"step": 1, "content": "Step 1: Drain load balancer.", "tags": ["critical", "delayed_dependency"]},
  {"step": 2, "content": "Step 2: Run migration.", "tags": ["critical", "delayed_dependency"]},
  {"step": 3, "content": "Step 3: Re-enable traffic.", "tags": ["critical", "delayed_dependency"]},
  {"step": 4, "content": "The team prefers minimal downtime.", "tags": ["preference"]},
  {"step": 5, "content": "Previous migrations used a different tool.", "tags": ["distractor"]}
]
""".strip()

# Tag descriptions shown to the model so it knows how to assign them.
_TAG_GUIDE = f"""
Tag guide:
  {TAG_CRITICAL}           — the most important constraint; must never be lost
  {TAG_HARD_CONSTRAINT}    — a binding rule that cannot be relaxed
  {TAG_NEGATION}           — content contains "not", "must not", "do not", "never"
  {TAG_DELAYED_DEPENDENCY} — constraint stated early, only needed at the end
  {TAG_DISTRACTOR}         — present in the incident but not needed to answer
  {TAG_PREFERENCE}         — a preference, not a hard rule
  {TAG_TRIGGER}            — the question itself (last item only)
""".strip()


# ---------------------------------------------------------------------------
# Simulated agent (USE_LLM=0)
# ---------------------------------------------------------------------------

def simulated_investigation_agent(incident_text: str) -> list[Observation]:
    """Deterministic fallback that splits an incident into keyword-tagged observations.

    Sentences containing constraint keywords get critical/hard_constraint/negation tags.
    Sentences with step markers get delayed_dependency tags.
    Everything else becomes a distractor or preference.
    The last item is always tagged as a trigger.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", incident_text.strip()) if s.strip()]
    if not sentences:
        return [Observation(step=1, content=incident_text, tags=[TAG_TRIGGER])]

    negation_words = {"not", "must not", "do not", "never", "no ", "without"}
    constraint_words = {"must", "required", "mandatory", "only", "exclusively"}
    step_words = {"step", "first", "then", "finally", "next", "after"}

    observations: list[Observation] = []
    for idx, sentence in enumerate(sentences, start=1):
        lower = sentence.lower()
        tags: list[str] = []

        has_negation   = any(w in lower for w in negation_words)
        has_constraint = any(w in lower for w in constraint_words)
        has_step       = any(w in lower for w in step_words)

        if has_negation or has_constraint:
            tags.append(TAG_CRITICAL)
            tags.append(TAG_HARD_CONSTRAINT)
            if has_negation:
                tags.append(TAG_NEGATION)
        elif has_step:
            tags.append(TAG_CRITICAL)
            tags.append(TAG_DELAYED_DEPENDENCY)
        elif any(w in lower for w in {"prefer", "want", "like", "hope"}):
            tags.append(TAG_PREFERENCE)
        else:
            tags.append(TAG_DISTRACTOR)

        observations.append(Observation(step=idx, content=sentence, tags=tags))

    # Last item is the question trigger.
    last = observations[-1]
    observations[-1] = Observation(
        step=last.step,
        content=last.content,
        tags=[TAG_TRIGGER],
    )
    return observations


# ---------------------------------------------------------------------------
# LLM agent (USE_LLM=1)
# ---------------------------------------------------------------------------

def _strip_fences(text: str) -> str:
    """Remove markdown code fences (```json ... ```) if present."""
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


def llm_investigation_agent(incident_text: str, model_client) -> list[Observation]:
    """LLM-based investigation agent.

    Sends the incident text to the model with a few-shot prompt and parses
    the JSON response into a list of Observations.

    Falls back to simulated_investigation_agent if JSON parsing fails,
    so the pipeline never crashes on a malformed model response.
    """
    prompt = (
        f"{_TAG_GUIDE}\n\n"
        f"Examples:\n{_FEW_SHOT_EXAMPLES}\n\n"
        "Now analyse this incident and return ONLY a JSON array in the same format "
        "(no explanation, no markdown, just the array):\n\n"
        f"Input: \"{incident_text}\""
    )

    raw = model_client.complete(prompt)
    cleaned = _strip_fences(raw)

    try:
        rows = json.loads(cleaned)
    except json.JSONDecodeError:
        # Model returned something unparseable — fall back to the simulated agent.
        return simulated_investigation_agent(incident_text)

    observations: list[Observation] = []
    for row in rows:
        try:
            observations.append(
                Observation(
                    step=int(row["step"]),
                    content=str(row["content"]),
                    tags=[str(t) for t in row.get("tags", [])],
                )
            )
        except (KeyError, TypeError, ValueError):
            continue  # skip malformed rows; don't crash the whole pipeline

    if not observations:
        return simulated_investigation_agent(incident_text)

    return observations
