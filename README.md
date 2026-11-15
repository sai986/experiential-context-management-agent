# Experiential Context Management
## Learning Transferable Lessons from Counterfactual Agent Outcomes

**Team ALiBi** | M26CL3.410: Language Models and Agents | IIIT Hyderabad, Monsoon 2026

| Name | Roll Number | Email |
|------|-------------|-------|
| Ajit Kumar Yadav | 2026901008 | ajit.yadav@research.iiit.ac.in |
| Padakanti Sai Prasad | 2026901002 | saiprasad.padakanti@research.iiit.ac.in |
| Anil Setti | 2026901010 | anil.setti@research.iiit.ac.in |
| Hitesh Kumar | 2026901011 | hitesh.k@research.iiit.ac.in |

> **Research question:** Can a context-management agent use counterfactual replay to attribute
> downstream task outcomes to its earlier retention, compression, and archival decisions,
> convert validated effects into reusable lessons, and transfer those lessons across frozen
> task agents without parameter updates?

---

## Key Results

Evaluated on 180 tasks across 6 incident failure-mode families using a real frozen LLM:

| Condition | Success rate | Avg context words |
|-----------|-------------|-------------------|
| Full context (ceiling) | 73% | 131 |
| Non-learning manager (baseline) | 4% | 27 |
| Lesson-aware Agent A | 62% | 53 |
| Lesson-aware Agent B (transfer) | 60% | 35 |

- **RQ1:** Lesson learning raises success from 4% to 62% (+58 pp over non-learning baseline).
- **RQ2:** Agent A uses 60% fewer context words than full context while staying within 11 pp of the full-context ceiling.
- **RQ3:** 4 transferred lessons give a fresh Agent B 60% success with no additional training or parameter updates.

---

## What is Novel

A common first reaction is: "the context manager is just if/else rules - where is the intelligence?" This points at the wrong component. The intelligence is not in how the manager executes a decision at runtime. It is in **how the rules that govern those decisions were discovered**.

Every rule in the lesson store was generated automatically through counterfactual causal attribution. No human wrote "never compress observations tagged `[critical, negation]`." The system discovered this rule by running controlled experiments on its own past failures:

```
1. Manager compresses observation X (tagged [critical, negation])
2. Task agent fails
3. Attribution engine deepcopies the ledger, undoes exactly that one compression,
   re-runs the task agent
4. Score improves from 0 to 1 -> causal proof that the compression caused the failure
5. Rule created: "COMPRESS on [critical, negation] = harmful"
6. Rule confirmed independently 5+ more times across different episodes
7. Rule promoted to transferable and given to Agent B, which never saw those episodes
8. Agent B succeeds on new tasks it has never seen, guided only by the transferred rule
```

**The contribution is a pipeline that converts downstream task failures into causal, reusable, transferable context-management rules, with no human annotation and no model retraining.** The runtime decision procedure is intentionally simple and deterministic - this is a deliberate design choice, not a limitation. A deterministic runtime means that any change in task success rate is 100% attributable to the content of the lesson store, isolating the lesson acquisition pipeline as the sole experimental variable.

---

## Motivation

Modern production systems emit continuous, unbounded observation streams - Kafka topics, Splunk log pipelines, alerting queues - that an LLM agent cannot hold in full context. Each log entry carries a natural importance signal in its log level: CRITICAL/ERROR entries must not be lost; INFO entries are often background noise; DEBUG entries are almost always distractors. A naive context window that fills up and drops the oldest entries will eventually discard a CRITICAL observation essential to answering the final incident question.

The key insight is that **log levels are an imperfect signal**. A single DEBUG log saying "fallback disabled" may be the most important entry in a 10,000-line trace. A flood of ERROR entries during routine load testing may be pure noise. What is needed is not a static priority filter but a **context manager that learns from its own filtering mistakes**: when the agent fails because a critical observation was compressed away, that failure should generate a reusable rule that prevents the same mistake on future incidents.

---

## Theoretical Framing

This project is best understood through the lens of **Case-Based Reasoning (CBR)** (Kolodner, 1993). Each lesson in the lesson store is a case: a past failure situation described by the observation's tag pattern and the operation applied, paired with the corrective action of keeping the observation verbatim. Unlike classical CBR where cases are added by human experts, our case base grows through **causal attribution** - a lesson is only created when a counterfactual test proves that the operation caused the failure, not merely correlated with it.

The cross-agent transfer component (RQ3) aligns with **continual and lifelong learning** (Thrun, 1996; Ring, 1994). Agent A accumulates lessons over a training distribution. The most strongly confirmed lessons (transferable threshold: 5+ independent confirmations) are handed to Agent B. Agent B benefits from Agent A's experience without inheriting task-specific overfitting, because the lessons are tag-pattern rules rather than model weight updates. This is **transfer via distilled symbolic rules**, not weight sharing or fine-tuning.

**Compared to related work:** MemGPT (Packer et al., 2023) uses fixed heuristics with no learning. StreamingLLM (Xiao et al., 2024) handles streaming context via attention sinks but applies no learning. Reflexion (Shinn et al., 2023) learns from failures but its lessons target the task agent's behaviour, not the context manager's retention policy. AdaCoM (Lu et al., arXiv:2605.30785, 2026) learns context manager policy through RL weight updates, requiring GPU training; this project achieves the same goal through structured text rules with no parameter updates. ACON (Kang et al., ICML 2026) refines compression guidelines from failure analysis but covers only the compression operation and no cross-agent transfer.

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
    model_client.py              # Thin API wrapper (USE_LLM=0/1 toggle)
    task_agent.py                # Simulated + LLM task agents
    investigation_agent.py       # Tags observations with context-safety labels (future work)
  attribution/
    replay.py                    # Counterfactual replay engine + lesson creation
  context/
    ledger.py                    # Immutable context ledger (KEEP/COMPRESS/ARCHIVE)
    manager.py                   # 4 manager variants (full/fixed/non-learning/lesson-aware)
  environment/
    evaluator.py                 # Substring + synonym + negation-aware scoring
    task_generator.py            # Base controlled task dataclasses
    task_generator_incident.py   # 6-family incident task generator
  experiments/
    run_pilot.py                 # Original mid-submission pilot (simulated only)
    run_incident_pilot.py        # Final experiment (simulated + LLM, all 3 RQs)
  lessons/
    lesson_store.py              # Lesson lifecycle: candidate -> validated -> transferable
  constants.py                   # Shared tag and operation constants
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
# edit .env: set ANTHROPIC_API_KEY and optionally ANTHROPIC_BASE_URL
```

**Step 2** — Run:

```bash
python -m src.experiments.run_incident_pilot
```

Set `USE_LLM=1` in `.env` to enable LLM mode. The experiment logs live progress to the terminal and writes results incrementally so a crash can be resumed by re-running the same command.

---

## Output Files

| File | Contents |
|------|----------|
| `results/incident/raw/episode_results.jsonl` | One line per episode as it completes |
| `results/incident/raw/all_replays.jsonl` | One line per counterfactual replay |
| `results/incident/raw/lesson_store_a.json` | Serialised lesson store after Phase 1 (re-running skips Phase 1) |
| `results/incident/processed/summary.json` | Condition x family success rates + transferable lessons |

A human-readable **Analysis Report** is printed to the terminal at the end of every run, covering RQ1/RQ2/RQ3 findings, per-family breakdown, and lesson details.

---

## Task Families

| Family | Observations | Failure mode tested |
|--------|-------------|---------------------|
| `hard_constraint` | 14 (6 + 8 noise) | Binding rule stated early must survive until the trigger |
| `delayed_dependency` | 13 (5 + 8 noise) | Constraint stated at step 1 is only needed at the final step |
| `dependency_chain` | 15 (7 + 8 noise) | 4-step procedure where losing any step breaks the full answer |
| `false_distractor` | 13 (5 + 8 noise) | Critical item is mislabelled as a distractor and wrongly archived |
| `negation_critical` | 13 (5 + 8 noise) | Constraint expressed as a negation ("do not use X") must survive compression |
| `synonym_match` | 13 (5 + 8 noise) | Required term has an acceptable synonym; the evaluator checks both |

8 background `info`-tagged noise observations are injected per task to simulate a realistic Kafka/Splunk stream. The 70/30 train/test split gives 125 training tasks (Phase 1) and 55 test tasks (Phase 2 + 3).

---

## Experiment Conditions

| Condition | Description |
|-----------|-------------|
| `full_context` | No compression - all observations kept verbatim (ceiling) |
| `non_learning_manager` | Archives distractors and compresses old items; no lessons (baseline) |
| `lesson_aware_agent_a` | Agent A with lessons learned from Phase 1 training failures |
| `lesson_aware_agent_b` | Fresh Agent B seeded with Agent A's 4 transferable lessons |

---

## Key Design Decisions

- **Counterfactual replay:** deepcopy ledger -> undo 1 decision -> re-run frozen agent -> compare scores. Proves causation, not correlation.
- **Immutable ledger:** raw observations are never deleted; `active=False` means archived, not removed. Replay can always undo any decision.
- **Lesson lifecycle:** `candidate` (1 confirmation) -> `validated` (>=2) -> `transferable` (>=5). Only `transferable` lessons cross the agent boundary.
- **Tag-based retrieval:** set intersection over `trigger_tags`, fully deterministic - no vectors, no embeddings, no extra API calls. Any score change is 100% attributable to the manager's decision.
- **`temperature=0.0`:** all LLM calls are deterministic within a single model snapshot.
- **Frozen task agent:** the task agent's weights are never updated. Any improvement in success rate is caused solely by the context manager.
- **Lesson store persistence:** the store is serialised to `results/incident/raw/lesson_store_a.json` after Phase 1. Re-running loads the saved store and skips Phase 1. Delete the file to force a full retrain.

---

## Known Limitations

**Tag-pattern lessons without content awareness:** lessons encode only the operation and tag pattern, not observation content. An LLM-generated lesson encoding *what* made an item critical would generalise across tag combinations and handle content-driven criticality that tags cannot express. This is the primary motivation for content-aware lesson generation as future work. Items tagged only as `info` are also never replay candidates, so a critical observation mislabelled as `info` would be archived indefinitely with no lesson ever learned.

**Independent counterfactual tests:** the attribution engine tests 1 suspect decision at a time. It does not detect joint causation where 2 decisions together cause a failure but neither does alone.

**Lesson granularity and overlapping tags:** lessons are created at the exact tag combination observed. Overlapping tag sets (e.g. `[critical, hard_constraint]` and `[critical, hard_constraint, negation]`) produce separate lessons, fragmenting support counts and slowing promotion to `transferable`.

**Stationarity assumption:** the lesson store's `support_count` only increases. If the underlying log stream changes, stale lessons may remain active. Drift-aware lesson deprecation is left as future work.

**Synthetic benchmark:** the benchmark is fully controlled. Each family has exactly 1 critical observation and a deterministic correct answer. Validation on real production incident streams is left as future work.

---

## Future Scope

The following extensions are architecturally prepared in the codebase but not activated in this experiment:

- **Investigation Agent:** `src/agents/investigation_agent.py` implements an LLM-based parser that converts raw incident text into structured tagged observations. Activating it completes the full 3-agent pipeline for real streaming input.
- **Embedding-based lesson retrieval:** replacing tag set-intersection with dense vector similarity would allow lessons to generalise across unseen tag combinations and scale to high-cardinality production log fields.
- **Bidirectional lesson evidence:** extending the lesson format to record confirmed-safe operations would allow the manager to distinguish "no evidence of harm" from "confirmed safe."
- **Drift-aware lesson deprecation:** reducing `support_count` when a previously harmful operation no longer causes failures would allow the store to adapt to concept drift.

---

## Reproducibility

**Model.** All LLM calls use `claude-sonnet-4-6` with `temperature=0.0` (defined in `src/constants.py`). Results are deterministic within a single model snapshot. Any change in model version may produce different answers from the task agent and therefore different success rates.

**Dataset generation.** The task generator (`src/environment/task_generator_incident.py`) is fully deterministic — it uses no random seed because all task content is generated from fixed templates and counters. Running the generator twice produces identical tasks in identical order.

**Committed results.** The exact results reported in the paper are committed to this repository and can be inspected without any API calls:
- `results/incident/processed/summary.json` — all 4 condition × 6 family success rates and the 4 transferable lessons
- `results/incident/raw/lesson_store_a.json` — Agent A's full lesson store after Phase 1

To verify the reported numbers, open `summary.json` directly. No LLM calls are needed.

**Skipping Phase 1 (saves ~125 training API calls).** If `results/incident/raw/lesson_store_a.json` is present when the experiment starts, Phase 1 is skipped entirely and the saved store is loaded instead. The committed `lesson_store_a.json` is the exact store used to produce the reported results. To reproduce Phase 1 from scratch, delete the file before running:

```bash
rm results/incident/raw/lesson_store_a.json
python -m src.experiments.run_incident_pilot
```

**Full reproduction (Phase 1 + 2 + 3) requires approximately 180 LLM calls** for Phase 2/3 evaluation, plus 143 counterfactual replay calls during Phase 1 if the store is deleted. At `temperature=0.0` with the same model snapshot, results should be identical to those in `summary.json`.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_LLM` | `0` | Set to `1` to enable real LLM API calls |
| `ANTHROPIC_API_KEY` | - | API key (Anthropic direct or LiteLLM virtual key) |
| `ANTHROPIC_BASE_URL` | Anthropic default | Custom endpoint for a proxy or internal gateway |

See `.env.example` for a template.

---

## Deliverables

| File | Description |
|------|-------------|
| `FINAL_REPORT.pdf` | 6-page two-column LaTeX report (all 9 sections + references) |
| `FINAL_REPORT.md` | Markdown source of the final report |
| `FINAL_REPORT.tex` | LaTeX source (recompile with `pdflatex FINAL_REPORT.tex`) |

---

## Mid-Submission Pilot

The original mid-submission pilot (`run_pilot.py`) is preserved for reference.
It uses a simpler controlled task set (no incident families) and runs in simulated mode only:

```bash
python -m src.experiments.run_pilot
```

Output goes to `results/raw/` and `results/processed/pilot_summary.json`.

---

## Dependencies

```
anthropic>=0.40.0,<1.0.0
python-dotenv>=1.0.0
```

Python 3.10+ required (`X | Y` union types in type hints).
