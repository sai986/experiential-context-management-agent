#!/usr/bin/env python3
"""Generate enriched BGL server-log incidents JSONL from bgl_structured.csv.

Reads   data/bgl/bgl_structured.csv
Writes  data/bgl/incidents.jsonl    -- all 180 incidents (master list)
        data/bgl/train.jsonl        -- first 125 (Phase 1 training)
        data/bgl/validation.jsonl   -- next 28
        data/bgl/test.jsonl         -- last 27 (Phase 2+3 evaluation)

Incidents per family:  6 families × 30 = 180 total (interleaved round-robin)
Observations per incident: 1 constraint + 5-6 real BGL lines + 8 background
noise lines = 14-15 total lines in raw_text.

Window sampling:
  - Candidate lines: Label != '-' (anomaly) OR Level in {FATAL, SEVERE}.
  - Window around candidate i: rows [i-4 .. i+8] (13 rows).
  - Deduplicate by EventTemplate of the center row (keep first occurrence).
  - Wrap around when 180 windows needed but only 60 unique exist.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

HERE     = Path(__file__).resolve().parent
CSV_PATH = HERE / "bgl_structured.csv"
OUT_PATH = HERE / "incidents.jsonl"

FAMILIES = [
    "hard_constraint",
    "delayed_dependency",
    "dependency_chain",
    "false_distractor",
    "negation_critical",
    "synonym_match",
]

N_PER_FAMILY   = 30   # 30 × 6 = 180 total
WINDOW_BEFORE  = 4
WINDOW_AFTER   = 8    # center + 8 lines after
N_REAL_LINES   = 6    # real BGL lines per incident
N_NOISE_LINES  = 8    # background noise lines per incident

# Split sizes: 125 train / 28 validation / 27 test
_TRAIN_END      = 125
_VALID_END      = 153   # 125 + 28
# test: [153:]

# ---------------------------------------------------------------------------
# Background noise line pool — realistic BGL INFO/DEBUG lines
# ---------------------------------------------------------------------------

_NOISE_TEMPLATES = [
    "INFO kernel: job {job_id} checkpoint saved to /bgl/jobs/{job_id}/ckpt",
    "DEBUG mmcs: heartbeat {node} OK latency=2ms",
    "INFO ciod: I/O buffer {pct}% utilized on partition {rack}",
    "DEBUG rts: MPI rank sync OK barrier_count={cnt}",
    "INFO mailbox: network link {node} bandwidth nominal 3.4 GB/s",
    "DEBUG bgl_core: torus router tables consistent {node}",
    "INFO db: power measurement rack={rack} voltage=3.3V current=12A",
    "DEBUG service_node: NFS mount /bgl/home healthy inode_count=142891",
    "INFO mmcs: job allocation confirmed partition={rack}",
    "DEBUG kernel: software interrupt handled IRQ=15 latency=0.3ms",
    "INFO ciod: stderr/stdout redirection OK rank=0",
    "DEBUG rts: collective allreduce latency=142us max_rank={cnt}",
]


def _make_noise_line(template_idx: int, node: str, incident_seq: int) -> str:
    """Format one noise line, deriving pseudo-values from node string."""
    tmpl  = _NOISE_TEMPLATES[template_idx % len(_NOISE_TEMPLATES)]
    # Extract rack from node id (e.g., "R02-M1-N0-C:J12-U11" -> "R02")
    rack  = node.split("-")[0] if "-" in node else "R00"
    job_id = 10000 + incident_seq * 37 + template_idx * 7
    pct    = 40 + (incident_seq * 13 + template_idx * 5) % 50
    cnt    = 1 + (incident_seq * 3 + template_idx * 11) % 512
    return tmpl.format(node=node, rack=rack, job_id=job_id, pct=pct, cnt=cnt)


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Window helpers
# ---------------------------------------------------------------------------

def find_candidate_indices(rows: list[dict]) -> list[int]:
    idx_set: set[int] = set()
    for i, row in enumerate(rows):
        if row["Label"] != "-":
            idx_set.add(i)
        if row["Level"] in ("FATAL", "SEVERE"):
            idx_set.add(i)
    return sorted(idx_set)


def collect_unique_windows(
    rows: list[dict], candidates: list[int]
) -> list[tuple[int, list[dict]]]:
    """Return (center_idx, window) for up to 60 deduplicated candidates."""
    seen: set[str] = set()
    windows: list[tuple[int, list[dict]]] = []
    for idx in candidates:
        tmpl = rows[idx]["EventTemplate"].strip()
        if tmpl in seen:
            continue
        seen.add(tmpl)
        start  = max(0, idx - WINDOW_BEFORE)
        end    = min(len(rows), idx + WINDOW_AFTER + 1)
        window = rows[start:end]
        if len(window) < 3:
            continue
        windows.append((idx, window))
        if len(windows) >= 60:
            break
    return windows


def _pick_real_lines(window: list[dict], center_idx: int, n: int) -> list[dict]:
    """Pick n rows from window, biased toward the center and surrounding rows."""
    # Find center position within window slice.
    # window starts at max(0, center_idx - WINDOW_BEFORE), so local index ≈ min(center_idx, WINDOW_BEFORE).
    local_center = min(center_idx, WINDOW_BEFORE)
    # Collect lines: 2 before center, center itself, then lines after.
    before = window[max(0, local_center - 2): local_center]
    after  = window[local_center + 1: local_center + n]
    center = [window[local_center]] if local_center < len(window) else []
    combined = before + center + after
    # If fewer than n, pad from start of window.
    if len(combined) < n:
        for row in window:
            if row not in combined:
                combined.append(row)
            if len(combined) >= n:
                break
    return combined[:n]


def format_log_line(row: dict) -> str:
    return (
        f"[{row['Level']}] {row['Component']} ({row['Node']}) "
        f"{row['Time']}: {row['Content']}"
    )


# ---------------------------------------------------------------------------
# Per-family incident builders
# ---------------------------------------------------------------------------

def _build_hard_constraint(
    incident_id: str, node: str, component: str, error_level: str,
    real_lines: list[str], noise_lines: list[str]
) -> dict:
    constraint = (
        f"CONSTRAINT: Node {node} {error_level} errors MUST require manual "
        f"operator intervention. Automatic job migration is PROHIBITED for "
        f"RAS events on this node."
    )
    log_text = _interleave(constraint, real_lines, noise_lines)
    return {
        "incident_id":     incident_id,
        "family":          "hard_constraint",
        "failure_mode":    "constraint_compression",
        "source":          "bgl_loghub",
        "node":            node,
        "raw_text":        log_text,
        "question": (
            f"What constraint applies to handling node {node} {error_level} failures, "
            f"and what are the required steps for the operator?"
        ),
        "required_outputs":  ["manual intervention", "operator", "migration prohibited"],
        "forbidden_outputs": ["automatic migration", "auto-migrate", "auto migrate"],
        "synonym_groups": [
            ["manual intervention", "manual override", "manual handling", "human intervention"],
            ["operator", "on-call engineer", "ops team", "administrator"],
            ["migration prohibited", "no automatic migration", "do not migrate",
             "must not migrate", "auto-migration disabled", "migration is prohibited"],
        ],
    }


def _build_delayed_dependency(
    incident_id: str, node: str, component: str, error_level: str,
    real_lines: list[str], noise_lines: list[str]
) -> dict:
    constraint = (
        f"CONSTRAINT: Component {component} on node {node} requires kernel "
        f"version 2.6 minimum for safe RAS event processing. Earlier kernel "
        f"versions do not support the required interrupt handling."
    )
    log_text = _interleave(constraint, real_lines, noise_lines)
    return {
        "incident_id":     incident_id,
        "family":          "delayed_dependency",
        "failure_mode":    "delayed_constraint_loss",
        "source":          "bgl_loghub",
        "node":            node,
        "raw_text":        log_text,
        "question": (
            f"What hardware requirement from the early log entries must be satisfied "
            f"before safe recovery of {component} on {node}?"
        ),
        "required_outputs":  ["kernel 2.6", "kernel version", "version requirement"],
        "forbidden_outputs": ["kernel 2.4", "kernel 2.5"],
        "synonym_groups": [
            ["kernel 2.6", "kernel version 2.6", "Linux 2.6", "kernel 2.6.16"],
            ["kernel version", "kernel release", "OS version", "kernel level"],
            ["version requirement", "minimum version", "version prerequisite",
             "compatibility requirement", "version check"],
        ],
    }


def _build_dependency_chain(
    incident_id: str, node: str, component: str, error_level: str,
    real_lines: list[str], noise_lines: list[str]
) -> dict:
    constraint = (
        f"CONSTRAINT: Hardware error recovery for {component} on {node} requires "
        f"the following sequence: "
        f"Step 1 isolate the node, "
        f"Step 2 drain jobs, "
        f"Step 3 run diagnostics, "
        f"Step 4 re-integrate. "
        f"Steps must NOT be skipped or reordered."
    )
    log_text = _interleave(constraint, real_lines, noise_lines)
    return {
        "incident_id":     incident_id,
        "family":          "dependency_chain",
        "failure_mode":    "chain_step_loss",
        "source":          "bgl_loghub",
        "node":            node,
        "raw_text":        log_text,
        "question": (
            f"List the complete recovery sequence for the {error_level} error "
            f"on node {node}. Steps must not be skipped."
        ),
        "required_outputs":  ["isolate", "drain jobs", "run diagnostics", "re-integrate"],
        "forbidden_outputs": [],
        "synonym_groups": [
            ["isolate", "quarantine", "fence off", "mark offline", "isolate node"],
            ["drain jobs", "evacuate jobs", "checkpoint and stop",
             "graceful shutdown", "job evacuation"],
            ["run diagnostics", "execute diagnostics", "diagnostic check",
             "hardware diagnostics", "diagnostics"],
            ["re-integrate", "bring back online", "re-enable node",
             "reintegrate", "restore to service"],
        ],
    }


def _build_false_distractor(
    incident_id: str, node: str, component: str, error_level: str,
    real_lines: list[str], noise_lines: list[str]
) -> dict:
    constraint = (
        f"CONSTRAINT: Torus network link failures on {node} MUST NOT be "
        f"classified as node memory errors. The error type determines the "
        f"recovery path and escalation tier."
    )
    log_text = _interleave(constraint, real_lines, noise_lines)
    return {
        "incident_id":     incident_id,
        "family":          "false_distractor",
        "failure_mode":    "false_archive",
        "source":          "bgl_loghub",
        "node":            node,
        "raw_text":        log_text,
        "question": (
            f"What is the true failure type at {node}? "
            f"Categorize and handle the failure correctly."
        ),
        "required_outputs":  ["network failure", "torus link", "interconnect fault"],
        "forbidden_outputs": ["memory error", "ECC error", "DRAM"],
        "synonym_groups": [
            ["network failure", "network link failure", "interconnect failure",
             "link failure", "network fault"],
            ["torus link", "torus network", "BGL torus",
             "torus interconnect", "torus"],
            ["interconnect fault", "network fault", "link fault",
             "communication failure", "network"],
        ],
    }


def _build_negation_critical(
    incident_id: str, node: str, component: str, error_level: str,
    real_lines: list[str], noise_lines: list[str]
) -> dict:
    constraint = (
        f"CONSTRAINT: {error_level} RAS events on {node} MUST NOT be downgraded "
        f"to WARNING severity in the event log. Event severity determines "
        f"automated escalation policy."
    )
    log_text = _interleave(constraint, real_lines, noise_lines)
    return {
        "incident_id":     incident_id,
        "family":          "negation_critical",
        "failure_mode":    "negation_loss",
        "source":          "bgl_loghub",
        "node":            node,
        "raw_text":        log_text,
        "question": (
            f"What MUST NOT be done when handling this {error_level} RAS event "
            f"at {node}, and what is the required escalation response?"
        ),
        "required_outputs":  ["FATAL severity", "escalate", "do not suppress"],
        "forbidden_outputs": ["downgrade to WARNING", "suppress event", "ignore"],
        "synonym_groups": [
            ["FATAL severity", "FATAL level", "fatal classification",
             "fatal rating", "severity FATAL", "FATAL"],
            ["escalate", "escalation required", "notify tier-2",
             "raise alert", "escalate to ops", "escalation"],
            ["do not suppress", "must not suppress", "cannot be suppressed",
             "no suppression", "suppression is prohibited", "do not downgrade"],
        ],
    }


def _build_synonym_match(
    incident_id: str, node: str, component: str, error_level: str,
    real_lines: list[str], noise_lines: list[str]
) -> dict:
    constraint = (
        f"CONSTRAINT: Compute node failures on {node} MUST report the hardware "
        f"location code (FRU identifier) for maintenance tracking. The rack "
        f"position and hardware address must be included in the incident record."
    )
    log_text = _interleave(constraint, real_lines, noise_lines)
    return {
        "incident_id":     incident_id,
        "family":          "synonym_match",
        "failure_mode":    "synonym_loss",
        "source":          "bgl_loghub",
        "node":            node,
        "raw_text":        log_text,
        "question": (
            f"What must be recorded about this hardware failure on node {node} "
            f"for maintenance purposes?"
        ),
        "required_outputs":  ["location code", "FRU identifier", "hardware address"],
        "forbidden_outputs": [],
        "synonym_groups": [
            ["location code", "FRU", "FRU location", "hardware location code",
             "FRU code", "location"],
            ["FRU identifier", "field replaceable unit", "FRU ID",
             "component identifier", "FRU number"],
            ["hardware address", "node identifier", "rack position",
             "physical location", "hardware location"],
        ],
    }


_BUILDERS = {
    "hard_constraint":    _build_hard_constraint,
    "delayed_dependency": _build_delayed_dependency,
    "dependency_chain":   _build_dependency_chain,
    "false_distractor":   _build_false_distractor,
    "negation_critical":  _build_negation_critical,
    "synonym_match":      _build_synonym_match,
}


def _interleave(constraint: str, real_lines: list[str], noise_lines: list[str]) -> str:
    """Build raw_text: constraint first, then real + noise lines interleaved."""
    combined: list[str] = [constraint]
    real_iter  = iter(real_lines)
    noise_iter = iter(noise_lines)
    real_done  = False
    noise_done = False
    # Alternate: 1 real, 2 noise, 1 real, 2 noise ... until exhausted
    while not (real_done and noise_done):
        if not real_done:
            line = next(real_iter, None)
            if line is None:
                real_done = True
            else:
                combined.append(line)
        for _ in range(2):
            if not noise_done:
                n = next(noise_iter, None)
                if n is None:
                    noise_done = True
                    break
                combined.append(n)
    return "\n".join(combined)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found.")
        sys.exit(1)

    print(f"Loading {CSV_PATH} ...", flush=True)
    rows = load_csv(CSV_PATH)
    print(f"Total rows: {len(rows)}", flush=True)

    candidates = find_candidate_indices(rows)
    print(f"Anomaly/FATAL/SEVERE candidate lines: {len(candidates)}", flush=True)

    windows = collect_unique_windows(rows, candidates)
    print(f"Unique deduplicated windows: {len(windows)}", flush=True)

    if not windows:
        print("ERROR: no windows found.", flush=True)
        sys.exit(1)

    # Build 180 incidents interleaved by family:
    # For round r (0..29), family_idx f (0..5):
    #   flat position  = r * 6 + f
    #   window index   = flat_pos % len(windows)  (wraps for round >= 10)
    incidents: list[dict] = []
    family_seq: dict[str, int] = {f: 0 for f in FAMILIES}
    global_seq = 0

    for r in range(N_PER_FAMILY):
        for f_idx, family in enumerate(FAMILIES):
            win_idx    = (r * len(FAMILIES) + f_idx) % len(windows)
            center_idx, window = windows[win_idx]
            center_row = rows[center_idx]

            node      = center_row["Node"]
            component = center_row["Component"]
            level     = center_row["Level"] if center_row["Level"] in ("FATAL", "SEVERE") else "ERROR"
            seq       = family_seq[family]
            inc_id    = f"bgl_{family}_{seq + 1:03d}"

            # Pick real log lines
            picked     = _pick_real_lines(window, center_idx, N_REAL_LINES)
            real_lines = [format_log_line(w) for w in picked]

            # Generate noise lines
            noise_lines = [
                _make_noise_line(i, node, global_seq)
                for i in range(N_NOISE_LINES)
            ]

            builder = _BUILDERS[family]
            inc     = builder(inc_id, node, component, level, real_lines, noise_lines)
            incidents.append(inc)
            family_seq[family] += 1
            global_seq += 1

    # Write master list
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        for inc in incidents:
            fh.write(json.dumps(inc, ensure_ascii=False) + "\n")
    print(f"\nWritten {len(incidents)} incidents to {OUT_PATH}", flush=True)

    # Write train / validation / test splits
    _write_split(incidents[:_TRAIN_END],     HERE / "train.jsonl",      "train")
    _write_split(incidents[_TRAIN_END:_VALID_END], HERE / "validation.jsonl", "validation")
    _write_split(incidents[_VALID_END:],     HERE / "test.jsonl",       "test")

    # Distribution summary
    print("\nFamily distribution:", flush=True)
    dist = Counter(inc["family"] for inc in incidents)
    for fam in FAMILIES:
        print(f"  {fam}: {dist[fam]}", flush=True)

    # Verify split counts
    train_inc = incidents[:_TRAIN_END]
    val_inc   = incidents[_TRAIN_END:_VALID_END]
    test_inc  = incidents[_VALID_END:]
    print(f"\nSplit sizes: train={len(train_inc)}, validation={len(val_inc)}, test={len(test_inc)}", flush=True)
    print("Per-family in test:", flush=True)
    test_dist = Counter(inc["family"] for inc in test_inc)
    for fam in FAMILIES:
        print(f"  {fam}: {test_dist[fam]}", flush=True)


def _write_split(incidents: list[dict], path: Path, name: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for inc in incidents:
            fh.write(json.dumps(inc, ensure_ascii=False) + "\n")
    print(f"Written {len(incidents)} to {path.name} ({name})", flush=True)


if __name__ == "__main__":
    main()
