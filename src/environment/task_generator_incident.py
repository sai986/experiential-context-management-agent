"""
Incident task generator for the final submission experiment.

Produces tasks across 6 families, each designed to stress a different
failure mode in context management:

  hard_constraint      — binding rule that must not be paraphrased
  delayed_dependency   — constraint stated early, relevant only at the end
  dependency_chain     — 4-step procedure where losing any step breaks the answer
  false_distractor     — critical item is mistakenly tagged as a distractor
  negation_critical    — constraint expressed as a negation ("do not use X")
  synonym_match        — required term has an acceptable synonym in the answer

Each task wraps a ControlledTask with incident-specific metadata in IncidentTask.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from src.constants import (
    FAMILY_DELAYED_DEPENDENCY,
    FAMILY_DEPENDENCY_CHAIN,
    FAMILY_FALSE_DISTRACTOR,
    FAMILY_HARD_CONSTRAINT,
    FAMILY_NEGATION_CRITICAL,
    FAMILY_SYNONYM_MATCH,
    TAG_CRITICAL,
    TAG_DELAYED_DEPENDENCY,
    TAG_DISTRACTOR,
    TAG_HARD_CONSTRAINT,
    TAG_NEGATION,
    TAG_PREFERENCE,
    TAG_TRIGGER,
)
from src.environment.task_generator import ControlledTask, Observation


@dataclass(frozen=True)
class IncidentTask:
    """ControlledTask plus incident metadata for the final experiment.

    failure_mode  — a short label describing what this family tests
    critical_step — which observation step holds the key constraint
                    (used by the analysis script to verify replay found the right item)
    task:         — the underlying ControlledTask
    """
    failure_mode:  str
    critical_step: int
    task:          ControlledTask


# ---------------------------------------------------------------------------
# Family generators
# ---------------------------------------------------------------------------

def _hard_constraint_incident(i: int) -> IncidentTask:
    variants = [
        (
            "The system must run fully offline — no cloud API calls are permitted.",
            ["cloud", "API", "online"],
            ["offline", "local"],
            [["cloud", "hosted"], ["API", "endpoint"]],
        ),
        (
            "The total cost must stay under 500 USD — paid SaaS tools are not allowed.",
            ["paid", "SaaS", "subscription"],
            ["under 500", "free"],
            [["cost", "price"], ["SaaS", "managed service"]],
        ),
        (
            "The output must be a single JSON file — CSV and XML formats are rejected.",
            ["CSV", "XML", ".csv", ".xml"],
            ["JSON", ".json"],
            [["JSON", "json format"]],
        ),
    ]
    critical, forbidden, required, synonyms = variants[i % len(variants)]
    observations = [
        Observation(1, critical, [TAG_CRITICAL, TAG_HARD_CONSTRAINT, TAG_NEGATION]),
        Observation(2, "The team has experience with several deployment tools.", [TAG_PREFERENCE]),
        Observation(3, "An earlier prototype used a different approach than the current plan.", [TAG_DISTRACTOR]),
        Observation(4, "The stakeholder wants a practical and maintainable solution.", [TAG_PREFERENCE]),
        Observation(5, "There are multiple valid technical paths to consider.", [TAG_DISTRACTOR]),
        Observation(6, "Provide your final architecture recommendation.", [TAG_TRIGGER]),
    ]
    task = ControlledTask(
        task_id=f"inc_hard_{i:03d}",
        family=FAMILY_HARD_CONSTRAINT,
        observations=observations,
        question="Recommend a solution architecture that meets all stated constraints.",
        expected_constraints=[critical],
        forbidden_outputs=forbidden,
        required_outputs=required,
        synonym_groups=synonyms,
    )
    return IncidentTask(failure_mode="constraint_loss", critical_step=1, task=task)


def _delayed_dependency_incident(i: int) -> IncidentTask:
    variants = [
        (
            "The target machine is Windows Server 2019 — shell scripts using bash are not supported.",
            ["bash", "apt-get", "Linux", "chmod"],
            ["PowerShell", "Windows"],
            [["PowerShell", "ps1"], ["Windows", "Win"]],
        ),
        (
            "The database is in read-only replica mode during the migration window.",
            ["INSERT", "UPDATE", "DELETE", "DROP"],
            ["read-only", "SELECT"],
            [["read-only", "readonly"]],
        ),
        (
            "The delivery format must be a PDF report — interactive notebooks are not acceptable.",
            ["notebook", ".ipynb", "Jupyter"],
            ["PDF", ".pdf"],
            [["PDF", "pdf report"]],
        ),
    ]
    critical, forbidden, required, synonyms = variants[i % len(variants)]
    observations = [
        Observation(1, critical, [TAG_CRITICAL, TAG_DELAYED_DEPENDENCY, TAG_NEGATION]),
        Observation(2, "The initial scoping phase appears to be a standard technical review.", [TAG_DISTRACTOR]),
        Observation(3, "The user wants the fastest reliable path to completion.", [TAG_PREFERENCE]),
        Observation(4, "Several past attempts used different approaches.", [TAG_DISTRACTOR]),
        Observation(5, "Now produce the concrete step-by-step procedure.", [TAG_TRIGGER]),
    ]
    task = ControlledTask(
        task_id=f"inc_delayed_{i:03d}",
        family=FAMILY_DELAYED_DEPENDENCY,
        observations=observations,
        question="Provide the final procedure while respecting all earlier constraints.",
        expected_constraints=[critical],
        forbidden_outputs=forbidden,
        required_outputs=required,
        synonym_groups=synonyms,
    )
    return IncidentTask(failure_mode="delayed_constraint_loss", critical_step=1, task=task)


def _dependency_chain_incident(i: int) -> IncidentTask:
    """4-step procedure where losing any single step breaks the full answer.

    Distractors are inserted between step 1 and steps 2-4 so that when the
    manager runs at step 6, only step 1 is old enough (3+ steps ago) to be
    compressed. Steps 2-4 are recent enough to survive compression.
    This isolates step 1 as the replay suspect.
    """
    variants = [
        (
            [
                "Step 1: Back up the current configuration to a versioned snapshot.",
                "Step 2: Drain all active connections from the load balancer.",
                "Step 3: Apply the schema migration with a dry-run flag first.",
                "Step 4: Re-enable traffic only after the health check passes.",
            ],
            ["Step 1", "Step 2", "Step 3", "Step 4"],
        ),
        (
            [
                "Step 1: Export all user records to a staging table.",
                "Step 2: Validate checksums against the original table.",
                "Step 3: Run the transformation pipeline on the staging table.",
                "Step 4: Promote the staging table to production after sign-off.",
            ],
            ["Step 1", "Step 2", "Step 3", "Step 4"],
        ),
        (
            [
                "Step 1: Freeze writes to the primary shard.",
                "Step 2: Replicate data to the new cluster.",
                "Step 3: Verify replica lag is under 100ms.",
                "Step 4: Switch the DNS record to the new cluster endpoint.",
            ],
            ["Step 1", "Step 2", "Step 3", "Step 4"],
        ),
    ]
    steps, required = variants[i % len(variants)]
    observations = [
        Observation(1, steps[0], [TAG_CRITICAL, TAG_DELAYED_DEPENDENCY]),
        # Distractors pad the gap so step 1 ages past the compression threshold
        # before the later steps arrive.
        Observation(2, "The team is familiar with the existing infrastructure.", [TAG_DISTRACTOR]),
        Observation(3, "Previous attempts used manual processes that caused downtime.", [TAG_DISTRACTOR]),
        Observation(4, steps[1], [TAG_CRITICAL, TAG_DELAYED_DEPENDENCY]),
        Observation(5, steps[2], [TAG_CRITICAL, TAG_DELAYED_DEPENDENCY]),
        Observation(6, steps[3], [TAG_CRITICAL, TAG_DELAYED_DEPENDENCY]),
        Observation(7, "Now execute the full procedure in the correct order.", [TAG_TRIGGER]),
    ]
    task = ControlledTask(
        task_id=f"inc_chain_{i:03d}",
        family=FAMILY_DEPENDENCY_CHAIN,
        observations=observations,
        question="Reproduce the complete procedure in the correct order.",
        expected_constraints=steps,
        forbidden_outputs=["Step 1 only", "partial"],
        required_outputs=required,
        synonym_groups=[],
    )
    return IncidentTask(failure_mode="chain_step_loss", critical_step=1, task=task)


def _false_distractor_incident(i: int) -> IncidentTask:
    """Critical item is mistakenly tagged as a distractor.

    NonLearningManager will archive it. Replay should detect the ARCHIVE
    as harmful and create a lesson: never archive items tagged [distractor]
    when they also contain critical constraint keywords.

    Note: in a real incident, the tagger made an error. This family tests
    whether the system can learn from that mistake.
    """
    variants = [
        (
            "CRITICAL CONSTRAINT: Do not deploy to the production environment — use staging only.",
            ["production", "prod"],
            ["staging"],
        ),
        (
            "CRITICAL CONSTRAINT: All changes must be reviewed by a second engineer before merge.",
            ["unreviewed", "solo merge", "skip review"],
            ["reviewed", "second engineer"],
        ),
        (
            "CRITICAL CONSTRAINT: The API must respond within 200ms — no synchronous external calls.",
            ["synchronous", "blocking call", "external API call"],
            ["200ms", "async"],
        ),
    ]
    critical, forbidden, required = variants[i % len(variants)]
    observations = [
        # Tagged as distractor — this is the intentional mislabelling.
        # The system must learn not to archive items with these keywords.
        Observation(1, critical, [TAG_DISTRACTOR, TAG_CRITICAL, TAG_NEGATION]),
        Observation(2, "The team has a standard deployment pipeline in place.", [TAG_PREFERENCE]),
        Observation(3, "Several unrelated configuration options were discussed earlier.", [TAG_DISTRACTOR]),
        Observation(4, "The stakeholder wants to proceed quickly.", [TAG_PREFERENCE]),
        Observation(5, "What is your recommendation for the deployment?", [TAG_TRIGGER]),
    ]
    task = ControlledTask(
        task_id=f"inc_distractor_{i:03d}",
        family=FAMILY_FALSE_DISTRACTOR,
        observations=observations,
        question="Provide a deployment recommendation that meets all stated constraints.",
        expected_constraints=[critical],
        forbidden_outputs=forbidden,
        required_outputs=required,
        synonym_groups=[],
    )
    return IncidentTask(failure_mode="false_archive", critical_step=1, task=task)


def _negation_critical_incident(i: int) -> IncidentTask:
    variants = [
        (
            "The solution must not store any personally identifiable information (PII) on disk.",
            ["store PII", "write PII", "save user data", "log personal"],
            ["no PII", "in-memory"],
            [["in-memory", "memory only"], ["no PII", "PII-free"]],
        ),
        (
            "The process must not modify the original source files — work on a copy only.",
            ["modify original", "overwrite source", "edit in place"],
            ["copy", "read-only source"],
            [["copy", "duplicate"], ["read-only", "unmodified"]],
        ),
        (
            "The agent must not make more than 3 API calls per task — batch requests where possible.",
            ["4 API calls", "5 API calls", "unbounded calls", "loop over API"],
            ["3 calls", "batched"],
            [["batched", "batch request"], ["3 calls", "at most 3"]],
        ),
    ]
    critical, forbidden, required, synonyms = variants[i % len(variants)]
    observations = [
        Observation(1, critical, [TAG_CRITICAL, TAG_HARD_CONSTRAINT, TAG_NEGATION]),
        Observation(2, "The team explored several implementation options.", [TAG_PREFERENCE]),
        Observation(3, "An older version of the system had no such restrictions.", [TAG_DISTRACTOR]),
        Observation(4, "The user wants a clean and auditable solution.", [TAG_PREFERENCE]),
        Observation(5, "Provide the recommended implementation approach.", [TAG_TRIGGER]),
    ]
    task = ControlledTask(
        task_id=f"inc_negation_{i:03d}",
        family=FAMILY_NEGATION_CRITICAL,
        observations=observations,
        question="Recommend an implementation approach that satisfies all constraints.",
        expected_constraints=[critical],
        forbidden_outputs=forbidden,
        required_outputs=required,
        synonym_groups=synonyms,
    )
    return IncidentTask(failure_mode="negation_loss", critical_step=1, task=task)


def _synonym_match_incident(i: int) -> IncidentTask:
    variants = [
        (
            "The infrastructure must be hosted on a cloud platform.",
            ["on-premise", "bare metal", "local server"],
            ["cloud"],
            [["cloud", "hosted", "cloud-hosted", "cloud platform"]],
        ),
        (
            "The interface must be accessible to users (meet WCAG 2.1 AA accessibility standards).",
            ["inaccessible", "no screen reader", "skip accessibility"],
            ["accessible"],
            [["accessible", "WCAG", "a11y", "accessibility"]],
        ),
        (
            "The pipeline must complete within one hour (3600 seconds).",
            ["multi-day", "overnight", "week-long"],
            ["one hour"],
            [["one hour", "60 minutes", "3600 seconds", "< 1 hour"]],
        ),
    ]
    critical, forbidden, required, synonyms = variants[i % len(variants)]
    observations = [
        Observation(1, critical, [TAG_CRITICAL, TAG_HARD_CONSTRAINT]),
        Observation(2, "The team is evaluating different architectural patterns.", [TAG_PREFERENCE]),
        Observation(3, "A competitor product uses a different approach entirely.", [TAG_DISTRACTOR]),
        Observation(4, "The stakeholder values maintainability and cost-efficiency.", [TAG_PREFERENCE]),
        Observation(5, "What architecture do you recommend?", [TAG_TRIGGER]),
    ]
    task = ControlledTask(
        task_id=f"inc_synonym_{i:03d}",
        family=FAMILY_SYNONYM_MATCH,
        observations=observations,
        question="Recommend an architecture that satisfies all constraints.",
        expected_constraints=[critical],
        forbidden_outputs=forbidden,
        required_outputs=required,
        synonym_groups=synonyms,
    )
    return IncidentTask(failure_mode="synonym_loss", critical_step=1, task=task)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_GENERATORS = [
    _hard_constraint_incident,
    _delayed_dependency_incident,
    _dependency_chain_incident,
    _false_distractor_incident,
    _negation_critical_incident,
    _synonym_match_incident,
]


def generate_incident_tasks(n_per_family: int = 10) -> list[IncidentTask]:
    # Interleave families so every split (train/val/test) contains all 6 families.
    # Without interleaving a sequential 70/15/15 split would put entire families
    # only in training, making val/test results meaningless.
    tasks: list[IncidentTask] = []
    for i in range(n_per_family):
        for gen in _GENERATORS:
            tasks.append(gen(i))
    return tasks


def write_incident_jsonl(path: Path, tasks: Iterable[IncidentTask]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for inc in tasks:
            row = {
                "failure_mode":  inc.failure_mode,
                "critical_step": inc.critical_step,
                "task":          asdict(inc.task),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def materialize_incident_splits(
    base_dir: Path = Path("data/incident"),
    n_per_family: int = 10,
) -> None:
    tasks = generate_incident_tasks(n_per_family=n_per_family)
    total = len(tasks)
    # 70 / 15 / 15 split
    train_end = int(total * 0.70)
    val_end   = int(total * 0.85)
    write_incident_jsonl(base_dir / "train.jsonl",      tasks[:train_end])
    write_incident_jsonl(base_dir / "validation.jsonl", tasks[train_end:val_end])
    write_incident_jsonl(base_dir / "test.jsonl",       tasks[val_end:])


if __name__ == "__main__":
    materialize_incident_splits()
