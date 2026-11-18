#!/usr/bin/env python3
"""
Generate Android logcat incidents from LogHub Android_2k.log_structured.csv.

Outputs 180 incidents (30 per family, 6 families) and splits them into:
  data/android/incidents.jsonl    — master list (180 lines)
  data/android/train.jsonl        — 125 tasks  (rounds 0-20 partial)
  data/android/validation.jsonl   — 28 tasks   (rounds 20-24 partial)
  data/android/test.jsonl         — 27 tasks   (rounds 24-29)

Each incident wraps a real Android log window (5-7 real lines + 8 noise lines
interleaved) with a synthetic constraint sentence that maps to known
simulated_task_agent triggers so USE_LLM=0 results are meaningful.

Usage:
    cd <project_root>
    python3 data/android/generate_incidents.py

Output: data/android/{incidents,train,validation,test}.jsonl
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[2]
CSV_IN  = Path(__file__).resolve().parent / "android_structured.csv"
OUT     = Path(__file__).resolve().parent / "incidents.jsonl"
TRAIN_OUT = Path(__file__).resolve().parent / "train.jsonl"
VAL_OUT   = Path(__file__).resolve().parent / "validation.jsonl"
TEST_OUT  = Path(__file__).resolve().parent / "test.jsonl"

# ---------------------------------------------------------------------------
# Background noise lines injected per incident (8 total, interleaved with
# real log lines). Each ends with a sentence terminator so the investigation
# agent creates a separate observation for each.
# ---------------------------------------------------------------------------

_NOISE_POOL: list[str] = [
    "I/Choreographer: Skipped 3 frames — the application may be doing too much work.",
    "D/PowerManagerService: wakefulness=Awake userActivitySummary=0x1.",
    "I/ActivityManager: Displayed com.android.launcher3/.Launcher: +234ms.",
    "D/WifiStateMachine: CMD_START_SCAN received in state ConnectedState.",
    "V/Sensor: Hall sensor changed: type=5 accuracy=3.",
    "D/dalvikvm: GC_CONCURRENT freed 2048K, 55% free 8192K/16384K.",
    "I/SystemServer: Starting DropBoxManagerService for event logging.",
    "D/NetworkManagementService: listInterfaces() called by system.",
    "I/InputDispatcher: Delivering touch input to window.",
    "D/PackageManager: Running dexopt on updated app packages.",
    "I/AudioService: stream 3 muted.",
    "D/ConnectivityService: reporting network disconnection.",
]


# ---------------------------------------------------------------------------
# Family definitions — each maps to a simulated_task_agent trigger so that
# USE_LLM=0 runs produce meaningful (non-zero) baseline success rates.
#
# Format per pattern:
#   constraint_template : f-string with {component} placeholder
#   required_outputs    : list[str]  (must appear in simulated agent's answer)
#   forbidden_outputs   : list[str]  (must NOT appear un-negated in answer)
#   synonym_groups      : list[list[str]]  (at least 2 synonyms per required term)
#   question_template   : str (family-specific question)
#   failure_mode        : string label
# ---------------------------------------------------------------------------

FAMILIES: list[dict] = [
    # -------------------------------------------------------------------
    # Family 1: hard_constraint
    # simulated_task_agent triggers: "offline" + "cloud api"
    #                                 "500 usd" / "open-source"
    #                                 "json file" / "single json"
    # -------------------------------------------------------------------
    {
        "family": "hard_constraint",
        "failure_mode": "constraint_compression",
        "patterns": [
            {
                "constraint": (
                    "CONSTRAINT: {component} diagnostics must run fully offline "
                    "— no cloud API calls are permitted during incident triage."
                ),
                "required_outputs": ["offline", "local"],
                "forbidden_outputs": ["cloud API", "hosted endpoint", "SaaS inference"],
                "synonym_groups": [
                    ["cloud API", "hosted endpoint", "remote endpoint"],
                    ["offline", "local", "on-device"],
                ],
                "question": "What constraint must be preserved when resolving this {component} issue?",
            },
            {
                "constraint": (
                    "CONSTRAINT: {component} fix must use only open-source tools "
                    "— total cost must stay under 500 USD and paid services are not allowed."
                ),
                "required_outputs": ["free", "open-source"],
                "forbidden_outputs": ["paid service", "managed service", "premium API"],
                "synonym_groups": [
                    ["free", "open source", "open-source", "no-cost"],
                    ["cost", "price", "budget"],
                ],
                "question": "What constraint must be preserved when resolving this {component} issue?",
            },
            {
                "constraint": (
                    "CONSTRAINT: {component} diagnostic report must be a single JSON file "
                    "— CSV and XML formats are rejected by the ingestion pipeline."
                ),
                "required_outputs": ["JSON"],
                "forbidden_outputs": ["CSV", "XML", ".csv", ".xml"],
                "synonym_groups": [
                    ["JSON", "json format", ".json", "json output"],
                ],
                "question": "What constraint must be preserved when resolving this {component} issue?",
            },
        ],
    },

    # -------------------------------------------------------------------
    # Family 2: delayed_dependency
    # simulated_task_agent triggers: "windows server" / "windows-only"
    #                                 "read-only" + "replica"
    #                                 "pdf report" / "pdf"
    # -------------------------------------------------------------------
    {
        "family": "delayed_dependency",
        "failure_mode": "delayed_constraint_loss",
        "patterns": [
            {
                "constraint": (
                    "CONSTRAINT: {component} target system is Windows Server 2019 "
                    "— shell scripts using bash are not supported on this environment."
                ),
                "required_outputs": ["PowerShell", "Windows"],
                "forbidden_outputs": ["apt-get", "Linux", "chmod"],
                "synonym_groups": [
                    ["PowerShell", "ps1", "Windows scripting"],
                    ["Windows", "Win", "Windows Server"],
                ],
                "question": "What prerequisite from the log history is needed to fix this {component} issue?",
            },
            {
                "constraint": (
                    "CONSTRAINT: {component} database is in read-only replica mode "
                    "— write operations are not permitted during the migration window."
                ),
                "required_outputs": ["read-only"],
                "forbidden_outputs": ["INSERT", "UPDATE", "DELETE", "DROP"],
                "synonym_groups": [
                    ["read-only", "readonly", "read only", "no writes"],
                ],
                "question": "What prerequisite from the log history is needed to fix this {component} issue?",
            },
            {
                "constraint": (
                    "CONSTRAINT: {component} incident report must be delivered as a PDF report "
                    "— interactive notebooks are not acceptable for compliance reasons."
                ),
                "required_outputs": ["PDF"],
                "forbidden_outputs": ["notebook", ".ipynb", "Jupyter"],
                "synonym_groups": [
                    ["PDF", "pdf report", ".pdf", "portable document"],
                ],
                "question": "What prerequisite from the log history is needed to fix this {component} issue?",
            },
        ],
    },

    # -------------------------------------------------------------------
    # Family 3: dependency_chain
    # simulated_task_agent triggers for the 3 chain variants:
    #   "back up" + "drain"
    #   "export" + "checksums"
    #   "freeze writes" + "replicate"
    # -------------------------------------------------------------------
    {
        "family": "dependency_chain",
        "failure_mode": "chain_step_loss",
        "patterns": [
            {
                "constraint": (
                    "RECOVERY PROCEDURE: Step 1: Back up current {component} configuration to a snapshot. "
                    "Step 2: Drain the event queue from all active threads. "
                    "Step 3: Apply the patch with a dry-run flag first. "
                    "Step 4: Re-enable the service only after the health check passes."
                ),
                "required_outputs": ["Back up", "Drain", "dry-run", "health check"],
                "forbidden_outputs": ["skip step", "partial rollout", "Step 1 only"],
                "synonym_groups": [
                    ["Back up", "snapshot", "backup"],
                    ["Drain", "flush queue", "empty queue"],
                    ["dry-run", "dry run", "rehearsal"],
                    ["health check", "readiness probe", "status check"],
                ],
                "question": "List all required steps in order to resolve this {component} incident.",
            },
            {
                "constraint": (
                    "RECOVERY PROCEDURE: Step 1: Export all {component} state records to a staging table. "
                    "Step 2: Validate checksums against the original baseline. "
                    "Step 3: Run the transformation pipeline on the staging table. "
                    "Step 4: Promote the staging table to production after sign-off."
                ),
                "required_outputs": ["Export", "checksums", "transformation pipeline", "sign-off"],
                "forbidden_outputs": ["skip validation", "partial export", "Step 1 only"],
                "synonym_groups": [
                    ["Export", "dump", "extract records"],
                    ["checksums", "hash verification", "integrity check"],
                    ["transformation pipeline", "ETL pipeline", "transform step"],
                    ["sign-off", "approval", "signoff"],
                ],
                "question": "List all required steps in order to resolve this {component} incident.",
            },
            {
                "constraint": (
                    "RECOVERY PROCEDURE: Step 1: Freeze writes to the {component} primary shard. "
                    "Step 2: Replicate data to the recovery cluster. "
                    "Step 3: Verify replica lag is under 100ms before continuing. "
                    "Step 4: Switch the DNS record to the recovery cluster endpoint."
                ),
                "required_outputs": ["Freeze writes", "Replicate", "replica lag", "DNS record"],
                "forbidden_outputs": ["skip replication", "partial failover", "Step 1 only"],
                "synonym_groups": [
                    ["Freeze writes", "halt writes", "stop writes"],
                    ["Replicate", "sync data", "copy data"],
                    ["replica lag", "replication delay", "sync lag"],
                    ["DNS record", "DNS entry", "DNS update"],
                ],
                "question": "List all required steps in order to resolve this {component} incident.",
            },
        ],
    },

    # -------------------------------------------------------------------
    # Family 4: false_distractor
    # The constraint sentence INTENTIONALLY uses no constraint keywords so
    # simulated_investigation_agent tags it as [distractor].
    # NonLearningManager archives it; FullContextManager keeps it.
    # simulated_task_agent trigger: "second engineer" in context
    # -------------------------------------------------------------------
    {
        "family": "false_distractor",
        "failure_mode": "false_archive",
        "patterns": [
            {
                "constraint": (
                    "Team practice: a second engineer reviews all {component} changes before merging."
                ),
                "required_outputs": ["review", "second engineer"],
                "forbidden_outputs": ["unreviewed", "skip review", "solo merge"],
                "synonym_groups": [
                    ["review", "code review", "peer review"],
                    ["second engineer", "another developer", "peer"],
                ],
                "question": "What is the actual root cause, ignoring any {component} background entries?",
            },
            {
                "constraint": (
                    "Release protocol: a second engineer approves {component} code changes before deployment."
                ),
                "required_outputs": ["review", "second engineer"],
                "forbidden_outputs": ["unreviewed", "skip review", "solo merge"],
                "synonym_groups": [
                    ["review", "approval", "sign-off"],
                    ["second engineer", "another reviewer", "approver"],
                ],
                "question": "What is the actual root cause, ignoring any {component} background entries?",
            },
            {
                "constraint": (
                    "Code review policy: each {component} change involves a second engineer for approval."
                ),
                "required_outputs": ["review", "second engineer"],
                "forbidden_outputs": ["unreviewed", "skip review", "solo merge"],
                "synonym_groups": [
                    ["review", "code review", "approval"],
                    ["second engineer", "reviewer", "pair"],
                ],
                "question": "What is the actual root cause, ignoring any {component} background entries?",
            },
        ],
    },

    # -------------------------------------------------------------------
    # Family 5: negation_critical
    # simulated_task_agent triggers:
    #   negation ("must not") + "pii" / "personally identifiable" → in-memory
    #   negation + "original" / "source file"                     → copy
    #   negation + "api call" / "batch"                           → batch
    # -------------------------------------------------------------------
    {
        "family": "negation_critical",
        "failure_mode": "negation_loss",
        "patterns": [
            {
                "constraint": (
                    "CONSTRAINT: {component} debug output must not store any personally "
                    "identifiable information (PII) on disk — all PII processing must remain in-memory."
                ),
                "required_outputs": ["in-memory"],
                "forbidden_outputs": ["write to disk", "persist PII", "store user data"],
                "synonym_groups": [
                    ["in-memory", "memory only", "in memory", "RAM only"],
                ],
                "question": "What action is explicitly prohibited for this {component} error?",
            },
            {
                "constraint": (
                    "CONSTRAINT: {component} processing must not modify the original source files "
                    "— work on a copy only to preserve the audit trail."
                ),
                "required_outputs": ["copy"],
                "forbidden_outputs": ["modify original", "overwrite source", "edit in place"],
                "synonym_groups": [
                    ["copy", "duplicate", "copied file", "replica"],
                ],
                "question": "What action is explicitly prohibited for this {component} error?",
            },
            {
                "constraint": (
                    "CONSTRAINT: {component} agent must not make more than 3 API calls per task "
                    "— batch requests where possible to stay within the allowed quota."
                ),
                "required_outputs": ["batch"],
                "forbidden_outputs": ["unbounded calls", "loop over API", "per-item request"],
                "synonym_groups": [
                    ["batch", "batched", "batch request", "bulk request"],
                ],
                "question": "What action is explicitly prohibited for this {component} error?",
            },
        ],
    },

    # -------------------------------------------------------------------
    # Family 6: synonym_match
    # simulated_task_agent triggers:
    #   "hosted" / "cloud platform"  → "Deploy using a hosted cloud environment."
    #   "accessible" / "wcag"        → accessibility answer
    #   "one hour" / "3600"          → "Complete the pipeline within one hour..."
    # -------------------------------------------------------------------
    {
        "family": "synonym_match",
        "failure_mode": "synonym_loss",
        "patterns": [
            {
                "constraint": (
                    "CONSTRAINT: {component} thread stack traces must be reported to a hosted "
                    "cloud platform for centralized analysis and monitoring."
                ),
                "required_outputs": ["cloud"],
                "forbidden_outputs": ["on-premise", "bare metal", "local server"],
                "synonym_groups": [
                    ["cloud", "hosted", "cloud platform", "cloud-hosted", "cloud environment"],
                ],
                "question": "What system must be reported to for this {component} incident?",
            },
            {
                "constraint": (
                    "CONSTRAINT: {component} monitoring interface must be accessible to all users "
                    "and meet WCAG 2.1 AA accessibility standards."
                ),
                "required_outputs": ["accessible"],
                "forbidden_outputs": ["inaccessible", "no screen reader", "skip accessibility"],
                "synonym_groups": [
                    ["accessible", "WCAG", "a11y", "accessibility", "screen reader"],
                ],
                "question": "What system must be reported to for this {component} incident?",
            },
            {
                "constraint": (
                    "CONSTRAINT: {component} diagnostic pipeline must complete within one hour "
                    "(3600 seconds) for real-time incident response requirements."
                ),
                "required_outputs": ["one hour"],
                "forbidden_outputs": ["multi-day", "overnight", "week-long"],
                "synonym_groups": [
                    ["one hour", "60 minutes", "3600 seconds", "< 1 hour", "within 1h"],
                ],
                "question": "What system must be reported to for this {component} incident?",
            },
        ],
    },
]


# ---------------------------------------------------------------------------
# Window extraction helpers
# ---------------------------------------------------------------------------

def _fmt_log_line(row: dict) -> str:
    """Format a CSV row as a compact log line string ending with a period.

    The trailing period ensures the simulated_investigation_agent creates
    a separate Observation for each log line (it splits on sentence-enders).
    """
    content = row["Content"][:100].strip()
    # Strip trailing punctuation to avoid double-punctuation, then add period.
    content = content.rstrip(".!?,;:")
    return (
        f"{row['Date']} {row['Time']}  {row['Level']}  "
        f"{row['Component']}  {content}."
    )


def extract_windows(rows: list[dict], n: int = 180) -> list[dict]:
    """
    Select n evenly-spaced windows centred on W/E/F rows.

    Each window is a dict with:
      center_component : str  (Component of the centre W/E/F line)
      center_date      : str
      log_lines        : list[str]  (formatted lines from the window, with trailing .)
      event_template   : str  (for deduplication)
    """
    wef_indices = [i for i, r in enumerate(rows) if r["Level"] in ("E", "W", "F")]

    if len(wef_indices) < n:
        step = 1
    else:
        step = max(1, len(wef_indices) // n)

    seen_templates: set[str] = set()
    windows: list[dict] = []
    cand_indices = wef_indices[::step]

    for ci in cand_indices:
        if len(windows) >= n:
            break
        center_row = rows[ci]
        tmpl = center_row.get("EventTemplate", "")

        if tmpl in seen_templates:
            continue
        seen_templates.add(tmpl)

        # Take 6-8 real log lines around the center W/E/F row.
        window_rows = rows[max(0, ci - 3) : ci + 5]
        log_lines = [_fmt_log_line(r) for r in window_rows]
        windows.append(
            {
                "center_component": center_row["Component"] or "SystemServer",
                "center_date":      center_row["Date"],
                "log_lines":        log_lines,
                "event_template":   tmpl,
            }
        )

    # Pad with repeated windows if fewer than n were found after dedup.
    while len(windows) < n:
        windows.append(windows[len(windows) % max(1, len(windows))])

    return windows[:n]


# ---------------------------------------------------------------------------
# Incident builder
# ---------------------------------------------------------------------------

def _pick_noise(round_num: int, fam_idx: int, count: int = 8) -> list[str]:
    """Return `count` background noise lines, varied by round/family so
    different incidents don't repeat the exact same noise sequence."""
    offset = (round_num * 6 + fam_idx) % len(_NOISE_POOL)
    pool = _NOISE_POOL[offset:] + _NOISE_POOL[:offset]
    return pool[:count]


def _interleave(real_lines: list[str], noise_lines: list[str]) -> list[str]:
    """Interleave real log lines and noise lines so they alternate."""
    result: list[str] = []
    n_real  = len(real_lines)
    n_noise = len(noise_lines)
    for i in range(max(n_real, n_noise)):
        if i < n_real:
            result.append(real_lines[i])
        if i < n_noise:
            result.append(noise_lines[i])
    return result


def build_incident(
    incident_id:    str,
    family_def:     dict,
    pattern_idx:    int,
    window:         dict,
    round_num:      int,
    fam_idx:        int,
) -> dict:
    """Assemble one incident JSONL record with 13-15 observations."""
    pat       = family_def["patterns"][pattern_idx % len(family_def["patterns"])]
    component = window["center_component"]
    date_str  = window["center_date"]

    # --- Constraint sentence (1 sentence) ----------------------------------
    constraint = pat["constraint"].format(component=component)

    # --- Real log lines (5-7 from window) ----------------------------------
    real_lines = window["log_lines"][:6]   # up to 6 real lines

    # --- Background noise (8 lines, varied by position) --------------------
    noise_lines = _pick_noise(round_num, fam_idx, count=8)

    # --- Interleave real + noise --------------------------------------------
    body_lines = _interleave(real_lines, noise_lines)   # 14 lines interleaved

    # --- Background sentence -----------------------------------------------
    background = (
        f"BACKGROUND: Android system log recorded on {date_str} from {component}."
    )

    # --- Question (last sentence = trigger observation) --------------------
    question_sent = pat["question"].format(component=component)
    # Ensure question ends with "?"
    if not question_sent.endswith("?"):
        question_sent = question_sent.rstrip(".") + "?"

    # --- Compose raw_text --------------------------------------------------
    # Each sentence ends in ./?/! so the investigation agent splits them into
    # separate observations. Total: 1 (constraint) + ~14 (body) + 1 (bg) + 1 (q)
    # = ~17 sentences → 15-17 observations after split.
    parts = [constraint] + body_lines + [background, question_sent]
    raw_text = " ".join(parts)

    return {
        "incident_id":    incident_id,
        "family":         family_def["family"],
        "failure_mode":   family_def["failure_mode"],
        "source":         "android_logcat_loghub",
        "component":      component,
        "raw_text":       raw_text,
        "question":       question_sent,
        "required_outputs":   pat["required_outputs"],
        "forbidden_outputs":  pat["forbidden_outputs"],
        "synonym_groups":     pat["synonym_groups"],
        "expected_constraints": [constraint],
    }


# ---------------------------------------------------------------------------
# JSONL writer
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, incidents: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for inc in incidents:
            f.write(json.dumps(inc, ensure_ascii=False) + "\n")
    print(f"Wrote {len(incidents)} incidents to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not CSV_IN.exists():
        print(f"ERROR: {CSV_IN} not found.")
        print("Download it first:")
        print("  curl -L 'https://raw.githubusercontent.com/logpai/loghub/master/Android/Android_2k.log_structured.csv' \\")
        print(f"       -o {CSV_IN}")
        sys.exit(1)

    # --- Load CSV -----------------------------------------------------------
    with CSV_IN.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Loaded {len(rows)} rows from {CSV_IN.name}")

    # --- Level distribution -------------------------------------------------
    from collections import Counter
    level_dist = Counter(r["Level"] for r in rows)
    comp_dist  = Counter(r["Component"] for r in rows)
    wef        = [r for r in rows if r["Level"] in ("E", "W", "F")]
    print(f"\nLevel distribution:")
    for lv, cnt in sorted(level_dist.items(), key=lambda x: -x[1]):
        print(f"  {lv}: {cnt}")
    print(f"\nTop 10 components by frequency:")
    for comp, cnt in comp_dist.most_common(10):
        print(f"  {comp}: {cnt}")
    print(f"\nW/E/F lines (candidate window centres): {len(wef)}")

    # --- Extract 180 windows ------------------------------------------------
    N_TOTAL    = 180
    N_FAMILIES = len(FAMILIES)   # 6
    N_ROUNDS   = 30              # 30 instances per family

    windows = extract_windows(rows, n=N_TOTAL)
    print(f"\nExtracted {len(windows)} windows")

    # --- Generate incidents --------------------------------------------------
    # Interleave families: for each round r (0-29), emit all 6 families.
    # This ensures every split (train/val/test) contains all families.
    incidents: list[dict] = []

    for round_num in range(N_ROUNDS):
        for fam_idx, family_def in enumerate(FAMILIES):
            flat_idx    = round_num * N_FAMILIES + fam_idx
            family_name = family_def["family"]
            incident_id = f"android_{family_name}_{round_num + 1:03d}"
            window      = windows[flat_idx % len(windows)]
            pattern_idx = round_num % 3   # rotate through 3 patterns per family

            inc = build_incident(
                incident_id = incident_id,
                family_def  = family_def,
                pattern_idx = pattern_idx,
                window      = window,
                round_num   = round_num,
                fam_idx     = fam_idx,
            )
            incidents.append(inc)

    # --- Verify counts -------------------------------------------------------
    from collections import Counter as C
    fam_counts = C(inc["family"] for inc in incidents)
    print(f"\nFamily counts: {dict(fam_counts)}")
    assert len(incidents) == N_TOTAL, f"Expected {N_TOTAL} incidents, got {len(incidents)}"
    assert all(cnt == N_ROUNDS for cnt in fam_counts.values()), \
        f"Each family should have {N_ROUNDS}: {fam_counts}"
    print(f"\nVerification passed: {N_TOTAL} incidents, {N_ROUNDS} per family.")

    # --- Check observation count (approximate) --------------------------------
    # Sample a few to confirm 13-17 sentences in raw_text.
    import re
    sample = incidents[0]
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", sample["raw_text"].strip()) if s.strip()]
    print(f"\nSample observation count (incident 0): {len(sentences)} sentences")

    # --- Split into train / validation / test --------------------------------
    # 125 / 28 / 27  =  69.4% / 15.6% / 15.0%
    TRAIN_END = 125
    VAL_END   = 153   # 125 + 28

    train_inc = incidents[:TRAIN_END]
    val_inc   = incidents[TRAIN_END:VAL_END]
    test_inc  = incidents[VAL_END:]

    print(f"\nSplit: {len(train_inc)} train / {len(val_inc)} validation / {len(test_inc)} test")

    # --- Write all JSONL files -----------------------------------------------
    _write_jsonl(OUT,       incidents)
    _write_jsonl(TRAIN_OUT, train_inc)
    _write_jsonl(VAL_OUT,   val_inc)
    _write_jsonl(TEST_OUT,  test_inc)

    print("\nSample incident IDs:")
    for inc in incidents[:6]:
        print(f"  {inc['incident_id']}  family={inc['family']}  component={inc['component']}")
    print("  ...")
    for inc in incidents[-3:]:
        print(f"  {inc['incident_id']}  family={inc['family']}")


if __name__ == "__main__":
    main()
