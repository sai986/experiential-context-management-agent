# Experiential Context-Management Agent

> **Can a context-management agent use counterfactual replay to attribute
> downstream task outcomes to its earlier retention, compression, and archival
> decisions, convert validated effects into reusable lessons, and transfer those
> lessons across frozen task agents without parameter updates?**

A research system that learns from its own context-management mistakes.
Instead of using a fixed compression policy, the agent runs counterfactual
replays after each failure, attributes losses to specific context decisions,
and builds a lesson store that improves future behaviour — including transfer
to a fresh agent that never ran the original tasks.

This repository is the **final submission** for the group project
(deadline: November 15, 2026).

---

## Research Questions

| RQ | Question |
|----|----------|
| RQ1 | Does lesson learning improve task success over a non-learning baseline? |
| RQ2 | Does the lesson-aware manager preserve performance while reducing active context words? |
| RQ3 | Do lessons transfer to a fresh agent (Agent B) that never saw the training tasks? |

---

## Repository Structure

```
src/
  agents/
    model_client.py          # Thin Anthropic API wrapper (USE_LLM=0/1)
    task_agent.py            # Simulated + LLM task agents
    investigation_agent.py   # Tags observations with context-safety labels
  attribution/
    replay.py                # Counterfactual replay engine + lesson creation
  context/
    ledger.py                # Immutable context ledger (KEEP/COMPRESS/ARCHIVE)
    manager.py               # Four manager variants (full/fixed/non-learning/lesson-aware)
  environment/
    evaluator.py             # Substring + synonym + negation-aware scoring
    task_generator.py        # Base controlled task dataclasses
    task_generator_incident.py  # 6-family incident task generator
  experiments/
    run_pilot.py             # Original mid-submission pilot (simulated only)
    run_incident_pilot.py    # Final experiment (simulated + LLM, all 3 RQs)
  lessons/
    lesson_store.py          # Lesson lifecycle: candidate → validated → transferable
  constants.py               # Shared tag and operation constants
```

---

## Quick Start

### Simulated mode (free, no API key needed)

Tests the full pipeline without any LLM calls. Recommended for verifying setup.

```bash
pip install -r requirements.txt
python -m src.experiments.run_incident_pilot
```

### LLM mode (real API calls)

**Step 1** — Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY and optionally ANTHROPIC_BASE_URL
```

**Step 2** — Run:

```bash
python -m src.experiments.run_incident_pilot
```

The `.env` file sets `USE_LLM=1` automatically. The experiment logs live
progress to the terminal and writes results incrementally so a crash can be
resumed by re-running the same command.

---

## Output Files

| File | Contents |
|------|----------|
| `results/incident/raw/episode_results.jsonl` | One line per episode as it completes |
| `results/incident/raw/all_replays.jsonl` | One line per counterfactual replay |
| `results/incident/processed/summary.json` | Condition × family success rates + lessons |

A human-readable **Analysis Report** is printed to the terminal at the end of
every run, covering RQ1 / RQ2 / RQ3 findings, per-family breakdown, and lesson
details.

---

## Task Families

The incident task set covers 6 failure modes that stress different aspects of
context management:

| Family | What it tests |
|--------|--------------|
| `hard_constraint` | Binding rule that must not be paraphrased or compressed |
| `delayed_dependency` | Constraint stated early, only relevant at the final step |
| `dependency_chain` | 4-step procedure where losing any step breaks the answer |
| `false_distractor` | Critical item is mislabelled as a distractor and archived |
| `negation_critical` | Constraint expressed as a negation ("do not use X") |
| `synonym_match` | Required term has an acceptable synonym in the answer |

---

## Experiment Conditions

| Condition | Description |
|-----------|-------------|
| `full_context` | No compression — all observations kept verbatim (ceiling) |
| `fixed_compression` | Aggressive single-pass compression, no learning |
| `non_learning_manager` | Archives distractors and compresses old items, no lessons |
| `lesson_aware_agent_a` | Agent A with lessons learned from training failures |
| `lesson_aware_agent_b` | Fresh Agent B seeded with Agent A's transferable lessons |

---

## Key Design Decisions

- **Counterfactual replay**: deepcopy ledger → undo one decision → re-run agent → compare scores. Proves causation, not correlation.
- **Immutable ledger**: raw observations are never deleted; `active=False` means archived, not removed.
- **Lesson lifecycle**: `candidate` (1 confirmation) → `validated` (≥2) → `transferable` (≥5). Only `transferable` lessons cross the agent boundary.
- **Tag-based retrieval**: set intersection over `trigger_tags`, fully deterministic — no vectors, preserves experimental control.
- **`temperature=0.0`**: all LLM calls use temperature 0 for reproducibility.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_LLM` | `0` | Set to `1` to enable real API calls |
| `ANTHROPIC_API_KEY` | — | API key (Anthropic or LiteLLM virtual key) |
| `ANTHROPIC_BASE_URL` | Anthropic default | Custom endpoint for a proxy or internal gateway |

See `.env.example` for a template.

---

## Mid-Submission Pilot

The original mid-submission pilot (`run_pilot.py`) is preserved for reference.
It uses a simpler controlled task set (no incident families) and runs in
simulated mode only:

```bash
python -m src.experiments.run_pilot
```

Output goes to `results/raw/` and `results/processed/pilot_summary.json`.

---

## Project Documents

| Document | Description |
|----------|-------------|
| `Experiential_Context_Management_Project_Blueprint.md` | Full research specification and design rationale |
| `TEAM_IMPLEMENTATION_PLAN.md` | Stage-by-stage implementation plan |
| `ARCHITECTURE_AND_PLAN.md` | Architecture decisions, limitations, and report-ready write-ups |
| `IMPLEMENTATION_LEARNINGS.md` | 59 concepts learned during implementation (stages 1–14) |
| `docs/GROUP_PROJECT_MID_SUBMISSION_REPORT.md` | Mid-submission report |

---

## Dependencies

```
anthropic>=0.40.0,<1.0.0
python-dotenv>=1.0.0
```

Python 3.10+ required (`X | Y` union types in type hints).
