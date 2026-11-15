# Experiential Context-Management Agent

> **Can a context-management agent use counterfactual replay to attribute
> downstream task outcomes to its earlier retention, compression, and archival
> decisions, convert validated effects into reusable lessons, and transfer those
> lessons across frozen task agents without parameter updates?**

A research system that learns from its own context-management mistakes.
Instead of using a fixed compression policy, the agent runs counterfactual
replays after each failure, attributes losses to specific context decisions,
and builds a lesson store that improves future behaviour -- including transfer
to a fresh agent that never ran the original tasks.

This repository is the **final submission** for the group project
(deadline: November 15, 2026).

---

## What is Novel — Answering the Right Question

A common first reaction to this project is: "the context manager is just if/else
rules -- where is the intelligence?" This question points at the wrong component.
The intelligence is not in how the manager executes a decision at runtime. It is
in **how the rules that govern those decisions were discovered**.

Every rule in the lesson store was generated automatically through a process
the manager itself cannot perform: counterfactual causal attribution. No human
wrote "never compress observations tagged [critical, negation]." The system
discovered this rule by running controlled experiments on its own past failures:

```
1. Manager compresses observation X (tagged [critical, negation])
2. Task agent fails
3. Attribution engine makes a deepcopy of the ledger, undoes exactly
   that one compression, re-runs the task agent
4. Score improves from 0 to 1 → causal proof that the compression caused the failure
5. Rule created: "COMPRESS on [critical, negation] = harmful"
6. Rule confirmed independently 5+ more times across different episodes
7. Rule promoted to transferable and given to Agent B, which never saw those episodes
8. Agent B succeeds on new tasks it has never seen, guided only by the transferred rule
```

This is the contribution: **a pipeline that converts downstream task failures
into causal, reusable, transferable context-management rules, with no human
annotation and no model retraining.** The runtime decision procedure is
intentionally simple and deterministic -- this is a deliberate design choice,
not a limitation. A deterministic runtime means that any change in task success
rate is 100% attributable to the content of the lesson store, isolating the
lesson acquisition pipeline as the sole experimental variable.

The analogy is a decision tree: a decision tree "is just rules" at inference
time, but the interesting science is in how those rules were learned from data.
The lesson store is a decision tree grown from causal replay experiments rather
than from labelled examples.

---

## Real-World Motivation

Modern production systems emit continuous, unbounded observation streams --
Kafka topics, Splunk log pipelines, alerting queues -- that an LLM agent cannot
hold in full context. Each log entry already carries a natural importance signal
in its log level: CRITICAL/ERROR entries correspond to observations that must
not be lost; INFO entries are often background noise; DEBUG entries are almost
always distractors. A naive context window that fills up and drops the oldest
entries will eventually discard a CRITICAL observation that is essential to
answering the final incident question.

The key insight motivating this project is that **log levels are an imperfect
signal**. A single DEBUG log saying "fallback disabled" may be the most
important entry in a 10,000-line trace. A flood of ERROR entries during routine
load testing may be pure noise. What is needed is not a static priority filter
but a **context manager that learns from its own filtering mistakes**: when the
agent fails because a critical observation was compressed away, that failure
should generate a reusable rule -- "never compress observations tagged
[critical, negation] in this stream" -- that prevents the same mistake on
future incidents.

This experiment validates that core learning loop on a controlled benchmark.
The streaming deployment context (Kafka/Splunk ingestion, real-time observation
feeds, production incident triage) is the intended production target, with the
controlled synthetic benchmark serving as a reproducible proxy for the
failure modes that arise in real incident streams.

---

## Theoretical Framing

This project is best understood through the lens of **Case-Based Reasoning
(CBR)** (Kolodner, 1993) applied to context management decisions. Each lesson
in the lesson store is a case: a past failure situation (described by the
observation's tag pattern and the operation that was applied) paired with the
corrective action (keep verbatim instead). When the context manager encounters
a new observation, it retrieves the most relevant prior cases by tag-set
intersection and adapts the default policy accordingly. Unlike classical CBR,
the case base here grows through **causal attribution** -- a lesson is only
created when a counterfactual test proves that the operation caused the failure,
not merely correlated with it.

The cross-agent transfer component (RQ3) aligns with **continual and lifelong
learning** (Thrun, 1996; Ring, 1994). Agent A accumulates lessons over a
training distribution. The most strongly confirmed lessons -- those that survive
the transferable threshold (>=5 independent confirmations) -- are handed to
Agent B, a fresh agent that never ran the training tasks. Agent B benefits from
Agent A's experience without inheriting any task-specific overfitting, because
the lessons are expressed as tag-pattern rules rather than as updates to model
weights. This is structurally analogous to **transfer via distilled rules**
rather than via weight sharing or fine-tuning.

Compared to closely related work: MemGPT / Letta (Packer et al., 2023)
implements explicit in/out-of-context memory management but uses fixed
heuristics with no learning from past decisions. StreamingLLM (Xiao et al.,
2023) handles streaming context via attention sinks but applies no learning
at all. Reflexion (Shinn et al., 2023) learns from failures, but its lessons
target the **task agent's behaviour**, not the context manager's retention
policy. AdaCoM learns context manager policy through RL weight updates,
requiring GPU training infrastructure; this project achieves the same goal
through structured text rules with no parameter updates, making it compatible
with any frozen hosted LLM.

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
| `non_learning_manager` | Archives distractors and compresses old items, no lessons (baseline) |
| `lesson_aware_agent_a` | Agent A with lessons learned from training failures |
| `lesson_aware_agent_b` | Fresh Agent B seeded with Agent A's transferable lessons |

---

## Key Design Decisions

- **Counterfactual replay**: deepcopy ledger -> undo one decision -> re-run agent -> compare scores. Proves causation, not correlation -- the score change is caused by the manager decision, not by any other variable.
- **Immutable ledger**: raw observations are never deleted; `active=False` means archived, not removed. Replay can always undo any decision because the original is still there.
- **Lesson lifecycle**: `candidate` (1 confirmation) -> `validated` (>=2) -> `transferable` (>=5). Only `transferable` lessons cross the agent boundary. Conservative thresholds reduce false lessons that would hurt performance.
- **Tag-based retrieval**: set intersection over `trigger_tags`, fully deterministic -- no vectors, no embeddings, no extra API calls. Any score change is 100% attributable to the manager's decision, not to the retrieval mechanism.
- **`temperature=0.0`**: all LLM calls use temperature 0 for reproducibility. Results are deterministic within a single model snapshot.
- **Frozen task agent**: the task agent's weights are never updated during the experiment. If the score improves, the context manager is the only possible explanation -- the task agent did not change.
- **Lesson store persistence**: the lesson store is serialised to JSON after Phase 1 (`results/incident/raw/lesson_store_a.json`). Re-running the experiment loads the saved store and skips Phase 1 entirely, saving all training API calls. Delete the file to force a full re-train.

---

## Known Limitations

**Stationarity assumption**: the lesson store's `support_count` only increases -- it never decreases. If the underlying log stream changes (a service is retired, an error pattern shifts), old lessons may remain active even though they no longer apply. Detecting and deprecating stale lessons (concept drift handling) is left as future work.

**Tag-scalability**: in the controlled benchmark, observations carry a small fixed tag vocabulary (6 types). In a real Splunk or Kafka deployment, observations have dozens of fields (`service_name`, `pod_id`, `error_code`, `region`, `latency_percentile`). Pure set intersection over such a space leads to combinatorial explosion of possible tag patterns. The natural upgrade path is embedding-based retrieval over observation feature vectors, which generalises across unseen tag combinations.

**Fixed transfer threshold**: the `transferable` threshold (>=5 confirmations) is a single global constant. Different production streams have different error costs -- a financial trading stream where failures are expensive should require more confirmations before transferring a lesson than a development log stream. Per-stream threshold configuration is a straightforward extension.

**Negative-only lessons**: lessons encode only harmful operations -- what the manager must not do. The absence of a lesson means "no evidence of harm," not "confirmed safe." In the controlled benchmark these are equivalent by task design. In production with noisier tagging they can diverge.

**Independent counterfactual tests**: the attribution engine tests one suspect decision at a time. It does not detect joint causation -- cases where two decisions together cause failure but neither does alone. Task families are designed to have a single primary culprit, making this safe for the benchmark.

**Tag-only suspect detection and content-identical items**: the replay engine only tests items carrying `SUSPECT_TAGS` (`critical`, `hard_constraint`, `negation`, `delayed_dependency`) as candidates for counterfactual undo. Items tagged `[info]` are never tested, even if mislabelled by the investigation agent. In a real Kafka or Splunk stream where tagging is imperfect, a critical observation could be permanently archived with no lesson ever learned -- the failure would recur indefinitely. The same problem arises when two items share identical tag sets but differ in content: any lesson created applies to both, eliminating selective treatment. The fix requires two upgrades: (1) content-aware suspect detection that tests all archived items regardless of tags, and (2) LLM-generated content-pattern lessons that encode *what* made an item critical rather than only *which tags* it carried. This is the primary motivation for embedding-based lesson retrieval listed in Future Scope.

**Lesson granularity and overlapping trigger tags**: lessons are created at the granularity of the exact tag combination observed on the failing item. When multiple failure episodes involve items with overlapping but distinct tag sets (e.g. `[critical, hard_constraint]` and `[critical, hard_constraint, negation]`), the store retains separate lessons rather than merging them. This is a deliberate fidelity choice: each lesson reflects exactly the evidence from the episode that generated it, and merging via subsumption would attribute causal weight to tags not present in those episodes. At retrieval, all matching lessons fire (the manager errs toward KEEP), so the runtime decision is always correct. The cost is that support counts are fragmented across overlapping lessons, which slows promotion toward the `transferable` threshold and may cause RQ3 transfer results to be conservative.

---

## Future Scope

The following extensions are architecturally prepared in the codebase but not activated in this experiment:

- **Investigation Agent (Agent 1)**: `src/agents/investigation_agent.py` implements an LLM-based parser that converts raw incident text into structured tagged observations. It is not invoked in the current experiment -- observations come from the controlled task generator directly. Activating it completes the full three-agent pipeline for real streaming input.
- **Embedding-based lesson retrieval**: replacing tag set-intersection with dense vector similarity over observation features would allow lessons to generalise across unseen tag combinations and scale to high-cardinality production log fields.
- **Bidirectional lesson evidence**: extending the lesson format to record both harmful and confirmed-safe operations would allow the manager to distinguish "no lesson found because this is safe" from "no lesson found yet."
- **Drift-aware lesson deprecation**: a lesson deprecation mechanism that reduces `support_count` when a previously harmful operation no longer causes failures would allow the store to adapt to concept drift in the underlying stream.

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
