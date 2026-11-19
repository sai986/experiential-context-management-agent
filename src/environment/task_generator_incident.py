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
    TAG_INFO,
    TAG_NEGATION,
    TAG_PREFERENCE,
    TAG_TRIGGER,
)
from src.environment.task_generator import ControlledTask, Observation

# ---------------------------------------------------------------------------
# Production-stream noise observations
# ---------------------------------------------------------------------------
# These 8 entries simulate the INFO/DEBUG flood typical of a real Kafka or
# Splunk stream. They carry no constraint information — they are safe to
# compress or archive. Including them forces the context manager to make
# meaningful compression decisions, making the RQ2 word-count metric
# interpretable: LessonAwareManager will keep critical items (via lessons)
# and compress/archive these freely, while FullContextManager keeps all 14.
#
# Step numbers are intentionally left at 0 here and re-numbered inside
# _inject_noise() so they fit correctly between existing observations.
_NOISE_TEMPLATES: list[tuple[str, list[str]]] = [
    ("System log: service heartbeat normal, all pods healthy.",            [TAG_INFO]),
    ("Metric: CPU utilisation at 38%, memory at 51%.",                    [TAG_INFO]),
    ("Debug: cache hit ratio 0.91 — no eviction pressure.",               [TAG_INFO]),
    ("System log: scheduled backup completed successfully at 03:00 UTC.", [TAG_INFO]),
    ("Metric: average request latency 142ms over the last 5 minutes.",    [TAG_INFO]),
    ("Debug: garbage collection pause 18ms — within normal range.",       [TAG_INFO]),
    ("System log: TLS certificate renewal successful, expiry extended.",  [TAG_INFO]),
    ("Metric: disk I/O at 12% — well below threshold.",                   [TAG_INFO]),
]


def _inject_noise(observations: list[Observation], i: int) -> list[Observation]:
    """Insert noise observations before the trigger step, re-numbering all steps.

    Uses i (task index) to rotate through the noise pool so each task gets a
    slightly different noise sequence, preventing the manager from learning to
    skip the same fixed entries by position.
    """
    trigger_step = max(obs.step for obs in observations)
    # Insert noise at steps immediately before the trigger.
    n = len(_NOISE_TEMPLATES)
    noise_obs = [
        Observation(
            step=trigger_step + j,  # temporarily after trigger; re-numbered below
            content=_NOISE_TEMPLATES[(i + j) % n][0],
            tags=_NOISE_TEMPLATES[(i + j) % n][1],
        )
        for j in range(8)
    ]
    # Bump the trigger step past the noise, then re-number everything cleanly.
    combined = observations[:-1] + noise_obs + [observations[-1]]
    return [
        Observation(step=idx + 1, content=obs.content, tags=obs.tags)
        for idx, obs in enumerate(combined)
    ]


@dataclass(frozen=True)
class IncidentTask:
    """ControlledTask plus incident metadata for the final experiment.

    failure_mode      — a short label describing what this family tests
    critical_step     — which observation step holds the key constraint
                        (used by the analysis script to verify replay found the right item)
    task              — the underlying ControlledTask
    raw_incident_text — raw natural-language paragraph fed to the Investigation Agent
                        (Agent 1) in the 3-agent end-to-end pipeline. The LLM parses
                        this into structured observations with tags. In USE_LLM=0 mode
                        the pre-built task.observations are used instead.
    """
    failure_mode:      str
    critical_step:     int
    task:              ControlledTask
    raw_incident_text: str = ""


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
            (
                "We are deploying to an air-gapped data centre. The system must run fully "
                "offline — no cloud API calls are permitted under any circumstances. "
                "The team has experience with several deployment tools and prefers practical "
                "solutions. An earlier prototype used a different approach than the current "
                "plan but was decommissioned. There are multiple valid technical paths. "
                "Provide your final architecture recommendation."
            ),
        ),
        (
            "The total cost must stay under 500 USD — paid SaaS tools are not allowed.",
            ["paid", "SaaS", "subscription"],
            # "free" and "open-source" are what the LLM and simulator naturally say
            # for a budget constraint; "under 500" as a literal phrase is too fragile.
            ["free", "open-source"],
            [["free", "open source", "open-source"], ["cost", "price"]],
            (
                "Budget is extremely tight for this project. The total cost must stay "
                "under 500 USD and paid SaaS tools are not allowed. The team is comfortable "
                "with open-source tooling. A previous spike used a commercial service but "
                "that option is now off the table. The stakeholder wants a maintainable "
                "solution. Provide your final architecture recommendation."
            ),
        ),
        (
            "The output must be a single JSON file — CSV and XML formats are rejected.",
            ["CSV", "XML", ".csv", ".xml"],
            # Drop ".json" — "JSON" subsumes it and the LLM says "JSON file" not ".json"
            ["JSON"],
            [["JSON", "json format", ".json"]],
            (
                "The downstream consumer only accepts structured data in a specific format. "
                "The output must be a single JSON file — CSV and XML formats are rejected "
                "by the ingestion pipeline. The team has experience with multiple serialisation "
                "options. An older version of this pipeline produced CSV output but that "
                "integration has been deprecated. There are multiple valid technical paths. "
                "Provide your final architecture recommendation."
            ),
        ),
    ]
    critical, forbidden, required, synonyms, raw_text = variants[i % len(variants)]
    observations = [
        Observation(1, critical, [TAG_CRITICAL, TAG_HARD_CONSTRAINT, TAG_NEGATION]),
        Observation(2, "The team has experience with several deployment tools.", [TAG_PREFERENCE]),
        Observation(3, "An earlier prototype used a different approach than the current plan.", [TAG_DISTRACTOR]),
        Observation(4, "The stakeholder wants a practical and maintainable solution.", [TAG_PREFERENCE]),
        Observation(5, "There are multiple valid technical paths to consider.", [TAG_DISTRACTOR]),
        Observation(6, "Provide your final architecture recommendation.", [TAG_TRIGGER]),
    ]
    observations = _inject_noise(observations, i)
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
    return IncidentTask(failure_mode="constraint_loss", critical_step=1, task=task,
                        raw_incident_text=raw_text)


def _delayed_dependency_incident(i: int) -> IncidentTask:
    variants = [
        (
            "The target machine is Windows Server 2019 — shell scripts using bash are not supported.",
            # "bash" removed: LLM correctly mentions it only to dismiss it ("bash is unsupported"),
            # which _is_negated() misses because the negation is post-nominal ("unsupported").
            # apt-get/chmod are unambiguous positive violations.
            ["apt-get", "Linux", "chmod"],
            ["PowerShell", "Windows"],
            [["PowerShell", "ps1"], ["Windows", "Win", "Windows Server"]],
            (
                "The client's deployment target is a locked-down Windows Server 2019 VM. "
                "The target machine is Windows Server 2019 — shell scripts using bash are "
                "not supported on this environment. The initial scoping looked like a standard "
                "technical review. The user wants the fastest reliable path to completion. "
                "Several past attempts at automation used cross-platform tools but those are "
                "unavailable here. Now produce the concrete step-by-step deployment procedure."
            ),
        ),
        (
            "The database is in read-only replica mode during the migration window.",
            ["INSERT", "UPDATE", "DELETE", "DROP"],
            # Drop "SELECT" — too literal; LLM says "read-only queries" not "SELECT".
            # "read-only" is what both simulator and LLM naturally produce.
            ["read-only"],
            [["read-only", "readonly", "read only"]],
            (
                "A planned maintenance window is in effect for the primary database cluster. "
                "The database is in read-only replica mode during the migration window — "
                "write operations will be rejected. The initial scoping appeared to be a "
                "standard health check procedure. The user wants the fastest reliable path. "
                "Several past migration scripts used destructive writes but those cannot be "
                "applied now. Produce the concrete step-by-step migration procedure."
            ),
        ),
        (
            "The delivery format must be a PDF report — interactive notebooks are not acceptable.",
            ["notebook", ".ipynb", "Jupyter"],
            # Drop ".pdf" — LLM says "PDF report" not ".pdf"; "PDF" subsumes it.
            ["PDF"],
            [["PDF", "pdf report", ".pdf"]],
            (
                "The client's compliance team has strict requirements on deliverable format. "
                "The delivery format must be a PDF report — interactive notebooks are not "
                "acceptable because they cannot be signed or archived by the compliance system. "
                "The initial scoping looked like a standard reporting task. The user wants the "
                "fastest path to completion. Several analysts prefer Jupyter notebooks but "
                "that option is ruled out. Produce the concrete reporting procedure."
            ),
        ),
    ]
    critical, forbidden, required, synonyms, raw_text = variants[i % len(variants)]
    observations = [
        Observation(1, critical, [TAG_CRITICAL, TAG_DELAYED_DEPENDENCY, TAG_NEGATION]),
        Observation(2, "The initial scoping phase appears to be a standard technical review.", [TAG_DISTRACTOR]),
        Observation(3, "The user wants the fastest reliable path to completion.", [TAG_PREFERENCE]),
        Observation(4, "Several past attempts used different approaches.", [TAG_DISTRACTOR]),
        Observation(5, "Now produce the concrete step-by-step procedure.", [TAG_TRIGGER]),
    ]
    observations = _inject_noise(observations, i)
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
    return IncidentTask(failure_mode="delayed_constraint_loss", critical_step=1, task=task,
                        raw_incident_text=raw_text)


def _dependency_chain_incident(i: int) -> IncidentTask:
    """4-step procedure where losing any single step breaks the full answer.

    Distractors are inserted between step 1 and steps 2-4 so that when the
    manager runs at step 6, only step 1 is old enough (3+ steps ago) to be
    compressed. Steps 2-4 are recent enough to survive compression.
    This isolates step 1 as the replay suspect.
    """
    # required_outputs use distinctive content keywords from each step, not step labels.
    # Rationale: the LLM reproduces all step content correctly but uses "1." "2." etc,
    # never the literal string "Step 1". Content keywords are robust to this.
    variants = [
        (
            [
                "Step 1: Back up the current configuration to a versioned snapshot.",
                "Step 2: Drain all active connections from the load balancer.",
                "Step 3: Apply the schema migration with a dry-run flag first.",
                "Step 4: Re-enable traffic only after the health check passes.",
            ],
            ["Back up", "Drain", "dry-run", "health check"],
            (
                "The engineering team is performing a live database schema migration. "
                "The approved runbook defines four mandatory steps that must all be followed "
                "in order. Step 1: Back up the current configuration to a versioned snapshot. "
                "The team is familiar with the existing infrastructure and has done similar "
                "work before. Previous attempts used manual processes that caused downtime and "
                "were abandoned. Step 2: Drain all active connections from the load balancer. "
                "Step 3: Apply the schema migration with a dry-run flag first. "
                "Step 4: Re-enable traffic only after the health check passes. "
                "Now execute the full procedure in the correct order."
            ),
        ),
        (
            [
                "Step 1: Export all user records to a staging table.",
                "Step 2: Validate checksums against the original table.",
                "Step 3: Run the transformation pipeline on the staging table.",
                "Step 4: Promote the staging table to production after sign-off.",
            ],
            ["Export", "checksums", "transformation pipeline", "sign-off"],
            (
                "A data migration between legacy and new schemas is underway. "
                "The compliance team has mandated a four-step process with no shortcuts. "
                "Step 1: Export all user records to a staging table. "
                "The engineering team is comfortable with SQL and ETL tooling. "
                "A previous ad-hoc migration skipped validation and caused data loss. "
                "Step 2: Validate checksums against the original table. "
                "Step 3: Run the transformation pipeline on the staging table. "
                "Step 4: Promote the staging table to production after sign-off. "
                "Reproduce the complete procedure in the correct order."
            ),
        ),
        (
            [
                "Step 1: Freeze writes to the primary shard.",
                "Step 2: Replicate data to the new cluster.",
                "Step 3: Verify replica lag is under 100ms.",
                "Step 4: Switch the DNS record to the new cluster endpoint.",
            ],
            ["Freeze writes", "Replicate", "replica lag", "DNS record"],
            (
                "The team is migrating the primary database to a new cluster with zero downtime. "
                "The operations runbook requires four sequential steps. "
                "Step 1: Freeze writes to the primary shard. "
                "The team has successfully run similar migrations on staging environments. "
                "An earlier attempt that skipped the freeze step caused split-brain issues. "
                "Step 2: Replicate data to the new cluster. "
                "Step 3: Verify replica lag is under 100ms. "
                "Step 4: Switch the DNS record to the new cluster endpoint. "
                "Reproduce the complete procedure in the correct order."
            ),
        ),
    ]
    steps, required, raw_text = variants[i % len(variants)]
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
    observations = _inject_noise(observations, i)
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
    return IncidentTask(failure_mode="chain_step_loss", critical_step=1, task=task,
                        raw_incident_text=raw_text)


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
            (
                "An urgent hotfix is ready to ship. The release manager sent an early note "
                "that read like a routine reminder: do not deploy to the production environment "
                "— use staging only until sign-off is complete. The team has a standard "
                "deployment pipeline in place. Several unrelated configuration options were "
                "discussed in earlier threads. The stakeholder wants to proceed quickly. "
                "What is your recommendation for the deployment?"
            ),
        ),
        (
            "CRITICAL CONSTRAINT: All changes must be reviewed by a second engineer before merge.",
            ["unreviewed", "solo merge", "skip review"],
            # "reviewed" → "review": LLM says "code review" not past tense "reviewed".
            ["review", "second engineer"],
            (
                "The team is working under time pressure to merge a critical fix. "
                "An early process note stated: all changes must be reviewed by a second "
                "engineer before merge — this is a non-negotiable compliance requirement. "
                "The team has a standard CI/CD pipeline in place. Several unrelated "
                "config discussions happened in the same thread and have already been resolved. "
                "The stakeholder wants to proceed as quickly as possible. "
                "What is your recommendation for merging this change?"
            ),
        ),
        (
            "CRITICAL CONSTRAINT: The API must respond within 200ms — no synchronous external calls.",
            # "synchronous" removed: LLM correctly says "no synchronous calls" (negated dismissal)
            # but _is_negated() misses post-nominal negation. "blocking" is an unambiguous violation.
            ["blocking call", "external API call"],
            ["200ms", "async"],
            (
                "A new latency SLA has been agreed with the customer. "
                "The technical lead posted early in the thread: the API must respond within "
                "200ms — no synchronous external calls are permitted in the request path. "
                "The team has a standard service mesh in place. Various architecture options "
                "were debated in earlier messages but most were ruled out. "
                "The stakeholder wants a clean and auditable implementation. "
                "What is your recommendation for the API architecture?"
            ),
        ),
    ]
    critical, forbidden, required, raw_text = variants[i % len(variants)]
    observations = [
        # Tagged as distractor — this is the intentional mislabelling.
        # The system must learn not to archive items with these keywords.
        Observation(1, critical, [TAG_DISTRACTOR, TAG_CRITICAL, TAG_NEGATION]),
        Observation(2, "The team has a standard deployment pipeline in place.", [TAG_PREFERENCE]),
        Observation(3, "Several unrelated configuration options were discussed earlier.", [TAG_DISTRACTOR]),
        Observation(4, "The stakeholder wants to proceed quickly.", [TAG_PREFERENCE]),
        Observation(5, "What is your recommendation for the deployment?", [TAG_TRIGGER]),
    ]
    observations = _inject_noise(observations, i)
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
    return IncidentTask(failure_mode="false_archive", critical_step=1, task=task,
                        raw_incident_text=raw_text)


def _negation_critical_incident(i: int) -> IncidentTask:
    variants = [
        (
            "The solution must not store any personally identifiable information (PII) on disk.",
            ["store PII", "write PII", "save user data", "log personal"],
            # "no PII" dropped — LLM says "in-memory" and "never written to disk", never the
            # exact phrase "no PII". "in-memory" is robust across all correct phrasings.
            ["in-memory"],
            [["in-memory", "memory only", "in memory"]],
            (
                "A new data processing feature is being designed for a healthcare client. "
                "Legal has confirmed: the solution must not store any personally identifiable "
                "information (PII) on disk under any circumstances — all PII processing must "
                "remain in-memory and be discarded after the request completes. "
                "The team explored several implementation options in the design review. "
                "An older version of the system had no such restrictions and stored user logs "
                "freely, but that design is no longer compliant. "
                "The user wants a clean and auditable solution. "
                "Provide the recommended implementation approach."
            ),
        ),
        (
            "The process must not modify the original source files — work on a copy only.",
            ["modify original", "overwrite source", "edit in place"],
            # "read-only source" dropped — too literal; LLM says "work on a copy".
            # "copy" is the natural token in every correct answer.
            ["copy"],
            [["copy", "duplicate", "copied"]],
            (
                "The team is running a batch transformation over a large archive of source files. "
                "The data governance policy is clear: the process must not modify the original "
                "source files — work on a copy only to preserve the audit trail. "
                "The team explored several in-place transformation approaches in earlier spikes. "
                "An older pipeline edited files directly but that caused audit failures. "
                "The user wants a clean and reproducible solution. "
                "Provide the recommended implementation approach."
            ),
        ),
        (
            "The agent must not make more than 3 API calls per task — batch requests where possible.",
            ["4 API calls", "5 API calls", "unbounded calls", "loop over API"],
            # "3 calls" dropped — LLM says "batch requests" not "3 calls".
            # "batch" covers "batched", "batch requests", "batching" via substring.
            ["batch"],
            [["batch", "batched", "batch request"]],
            (
                "An automated agent is being designed to process customer support tickets. "
                "The API provider has a strict rate-limit policy: the agent must not make "
                "more than 3 API calls per task — batch requests where possible to stay "
                "within the allowed quota. "
                "The team explored several retrieval strategies including per-item calls. "
                "An earlier prototype made unbounded API calls and hit rate limits in production. "
                "The user wants a clean and cost-efficient solution. "
                "Provide the recommended implementation approach."
            ),
        ),
    ]
    critical, forbidden, required, synonyms, raw_text = variants[i % len(variants)]
    observations = [
        Observation(1, critical, [TAG_CRITICAL, TAG_HARD_CONSTRAINT, TAG_NEGATION]),
        Observation(2, "The team explored several implementation options.", [TAG_PREFERENCE]),
        Observation(3, "An older version of the system had no such restrictions.", [TAG_DISTRACTOR]),
        Observation(4, "The user wants a clean and auditable solution.", [TAG_PREFERENCE]),
        Observation(5, "Provide the recommended implementation approach.", [TAG_TRIGGER]),
    ]
    observations = _inject_noise(observations, i)
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
    return IncidentTask(failure_mode="negation_loss", critical_step=1, task=task,
                        raw_incident_text=raw_text)


def _synonym_match_incident(i: int) -> IncidentTask:
    variants = [
        (
            "The infrastructure must be hosted on a cloud platform.",
            ["on-premise", "bare metal", "local server"],
            ["cloud"],
            [["cloud", "hosted", "cloud-hosted", "cloud platform"]],
            (
                "The company is redesigning its core infrastructure for the next fiscal year. "
                "The CTO has mandated: the infrastructure must be hosted on a cloud platform "
                "to enable global scalability and reduce on-premise operational overhead. "
                "The team is evaluating different architectural patterns for the migration. "
                "A competitor product runs entirely on bare metal but that model does not "
                "fit the company's growth plans. "
                "The stakeholder values maintainability and cost-efficiency. "
                "What architecture do you recommend?"
            ),
        ),
        (
            "The interface must be accessible to users (meet WCAG 2.1 AA accessibility standards).",
            ["inaccessible", "no screen reader", "skip accessibility"],
            ["accessible"],
            [["accessible", "WCAG", "a11y", "accessibility"]],
            (
                "A government contract requires the new citizen-facing portal to comply with "
                "accessibility regulations. The procurement team confirmed: the interface must "
                "be accessible to users and meet WCAG 2.1 AA accessibility standards. "
                "The team is evaluating different front-end frameworks for the build. "
                "A competitor product skipped accessibility testing entirely but that approach "
                "is not acceptable for a government contract. "
                "The stakeholder values maintainability and long-term support. "
                "What architecture do you recommend?"
            ),
        ),
        (
            "The pipeline must complete within one hour (3600 seconds).",
            ["multi-day", "overnight", "week-long"],
            ["one hour"],
            # Added "3,600 seconds" (comma-formatted) — LLM sometimes writes the number
            # with a thousands separator; the evaluator lowercases but doesn't strip commas.
            [["one hour", "60 minutes", "3600 seconds", "3,600 seconds", "< 1 hour"]],
            (
                "The nightly data pipeline feeds a dashboard used in the morning stand-up. "
                "Operations has set a hard SLA: the pipeline must complete within one hour "
                "(3600 seconds) or the dashboard will show stale data. "
                "The team is evaluating different processing frameworks and parallelisation "
                "strategies. A previous version of the pipeline ran overnight but that "
                "cadence no longer meets the reporting requirements. "
                "The stakeholder values reliability and cost-efficiency. "
                "What architecture do you recommend?"
            ),
        ),
    ]
    critical, forbidden, required, synonyms, raw_text = variants[i % len(variants)]
    observations = [
        Observation(1, critical, [TAG_CRITICAL, TAG_HARD_CONSTRAINT]),
        Observation(2, "The team is evaluating different architectural patterns.", [TAG_PREFERENCE]),
        Observation(3, "A competitor product uses a different approach entirely.", [TAG_DISTRACTOR]),
        Observation(4, "The stakeholder values maintainability and cost-efficiency.", [TAG_PREFERENCE]),
        Observation(5, "What architecture do you recommend?", [TAG_TRIGGER]),
    ]
    observations = _inject_noise(observations, i)
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
    return IncidentTask(failure_mode="synonym_loss", critical_step=1, task=task,
                        raw_incident_text=raw_text)


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
                "failure_mode":      inc.failure_mode,
                "critical_step":     inc.critical_step,
                "raw_incident_text": inc.raw_incident_text,
                "task":              asdict(inc.task),
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
