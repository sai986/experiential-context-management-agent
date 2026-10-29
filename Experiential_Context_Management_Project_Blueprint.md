# Experiential Context-Management Agent

## Project Blueprint for Mid-Submission and Final Evaluation

**Working research question**

> **Can an experiential context manager adapt to different task agents without retraining by learning reusable context-management lessons from interaction?**

**Sharpened technical formulation**

> **Can a context-management agent use counterfactual replay to attribute downstream task outcomes to its earlier retention, compression, archival, retrieval, and forgetting decisions, convert validated effects into reusable lessons, and transfer those lessons across frozen task agents without parameter updates?**

**Project status:** implementation-ready research specification  
**Mid-submission deadline:** October 30, 2026  
**Final-submission deadline:** November 15, 2026  

---

## 1. Executive Summary

Long-horizon LLM agents accumulate user constraints, intermediate reasoning, tool outputs, failed attempts, partial findings, and repeated observations. Retaining everything increases token cost and can make the active context noisy. Compressing or removing information reduces cost, but risks deleting exact values, negations, dependencies, evidence, or unresolved constraints needed later.

Existing work attacks different parts of this problem:

- **RAG/DPR** retrieve relevant external information.
- **MemGPT/Letta** organize memory into active and external tiers.
- **Reflexion** converts task feedback into reusable verbal lessons.
- **AdaCoM** trains an external model to manage the context of a frozen task agent.
- **CausalFlow/Causal Agent Replay** use interventions and counterfactual execution to identify which trajectory decisions contributed to failure.

Our project combines these ideas around a narrower target: **the context manager itself should learn from experience**. It should not merely reflect on whether a task failed. It should test whether one of its context decisions contributed to the outcome. Only supported lessons should enter persistent experience memory.

The core loop is:

```text
Raw trajectory
    ↓
Experiential Context Manager
    ↓
KEEP / COMPRESS / ARCHIVE / RETRIEVE / RESTORE
    ↓
Managed context
    ↓
Frozen task-solving agent
    ↓
Task outcome
    ↓
Counterfactual replay of suspected context decisions
    ↓
Causal attribution with confidence
    ↓
Validated context-management lesson
    ↓
Persistent lesson memory
    ↓
Reuse on later tasks and different task agents
```

The main claim is **not** that using an agent for context management is new. The intended contribution is:

1. make context-management decisions explicitly traceable;
2. attribute outcomes to those decisions through intervention rather than unverified reflection;
3. convert supported effects into reusable context-management lessons;
4. test whether lessons transfer across frozen task agents without retraining.

---

## 2. Motivation and Problem Statement

### 2.1 Why active context needs management

A long-running agent may accumulate:

- original task instructions;
- hard constraints and preferences;
- intermediate plans;
- tool outputs;
- repetitive observations;
- failed hypotheses;
- tentative conclusions;
- retrieval results;
- contradictions and corrections;
- partially completed subtasks.

The complete trajectory is useful as a record, but it is not always the best reasoning input. A useful context manager must reduce clutter without losing information that remains causally important for the task.

### 2.2 The weakness of one-shot context management

A conventional manager may decide:

```text
Observation 7 → COMPRESS
Observation 11 → ARCHIVE
Memory 4 → RETRIEVE
```

If the final task fails, the system usually knows only that the answer was wrong. It does not reliably know whether:

- essential evidence was compressed incorrectly;
- a constraint was archived too early;
- the wrong memory was retrieved;
- retrieval never surfaced the relevant item;
- the task agent ignored correctly supplied evidence;
- the task agent lacked knowledge or reasoning ability;
- a tool returned incomplete information.

Without attribution, reflection can generate false lessons and pollute memory.

### 2.3 Research gap targeted by this project

The targeted gap is not generic context compression, memory hierarchy, retrieval, self-reflection, or external context management. The targeted gap is:

> **Experience-driven improvement of context-management decisions using credible downstream attribution, with reusable lessons evaluated for transfer across frozen task agents.**

This is a meta-level learning problem. The learned object is not primarily the solution to the task. The learned object is a rule about **how to construct useful context for a task-solving agent**.

---

## 3. Relationship to Prior Work

### 3.1 RAG and DPR

RAG retrieves passages and gives them to a generator. DPR provides dense semantic retrieval using separate query and passage encoders. These establish the retrieval layer, but they do not by themselves learn lessons from the consequences of context-retention or compression decisions.

### 3.2 MemGPT and Letta

MemGPT introduces an operating-system-inspired memory hierarchy in which the limited context window behaves like active memory and external storage holds information that can be recalled. Letta operationalizes and extends this style of stateful agent architecture.

Our work can use a similar separation between active and archived information. The difference is that our primary question is not where information lives. It is whether the manager can learn, from attributed outcomes, **better policies for moving and transforming information**.

### 3.3 Reflexion

Reflexion transforms task feedback into natural-language lessons that influence later attempts without updating model weights. This provides our experiential-learning inspiration.

However, ordinary Reflexion asks questions such as:

> What should the task-solving agent do differently next time?

Our manager asks:

> Which context-management decision affected the downstream agent, what evidence supports that attribution, and what reusable context-management rule follows?

### 3.4 AdaCoM

AdaCoM is the closest architectural baseline. It uses an external LLM to manage the context of a frozen task agent and trains the manager through reinforcement learning. Its public description reports that different agents prefer different fidelity/compression regimes and that transfer is strongest among agents with similar capability.

Our proposed distinction is:

| Dimension | AdaCoM-style approach | Proposed approach |
|---|---|---|
| Task agent | Frozen | Frozen |
| Context manager | External model | External agentic manager |
| Improvement mechanism | Parameter learning through RL | Persistent, retrieved experiential lessons; no required online weight update |
| Feedback | Downstream task reward | Downstream outcome plus counterfactual attribution |
| Reflection | Not the central learning representation | Explicit lesson generation |
| Cross-agent question | Transfer trained manager | Transfer validated context-management lessons |
| Main unit of learning | Manager policy parameters | Evidence-backed verbal/structured lessons |

### 3.5 CausalFlow and Causal Agent Replay

Counterfactual agent-attribution systems intervene on an execution step and rerun downstream behavior to estimate whether changing that step changes the outcome. Our proposal applies this principle specifically to **context-management operations**.

This distinction matters because an LLM judge inspecting a trace can produce a plausible explanation without showing that the suspected decision changed the result. Our system therefore treats reflection as a hypothesis generator and replay as the validation mechanism.

### 3.6 Novelty boundary

The project should not claim that the following are individually novel:

- external context managers;
- context compression or summarization;
- archival and retrieval;
- reflection;
- agents with memory;
- counterfactual replay;
- transfer across model families.

The proposed contribution is the combination and evaluation of:

1. context-management actions as explicit intervention points;
2. downstream attribution focused on these actions;
3. validated, reusable context-management lessons;
4. cross-task and cross-agent transfer without retraining the task agent.

---

## 4. Research Questions and Hypotheses

### RQ1: Experiential improvement

**Can persistent context-management lessons improve held-out task performance relative to a non-learning context manager?**

**H1:** A manager retrieving validated lessons will improve task success or constraint satisfaction under an equal context budget.

### RQ2: Attribution quality

**Does counterfactual attribution produce better lessons than naive LLM-only reflection?**

**H2:** Counterfactual attribution will reduce false blame and identify damaging context-management decisions more accurately on controlled tasks with known ground truth.

### RQ3: Context efficiency

**Can experiential management preserve task quality while reducing active-context tokens?**

**H3:** The proposed manager will improve the task-quality/context-cost trade-off relative to full context, fixed windows, and fixed summarization.

### RQ4: Cross-agent transfer

**Can lessons learned while managing Agent A improve an unseen frozen Agent B?**

**H4:** At least a subset of lessons, such as preserving exact numeric constraints or negations, will transfer across task agents.

### RQ5: Negative transfer

**When do lessons learned for one agent harm another agent?**

**H5:** Agent-specific lessons will be less transferable than content-structural lessons; lesson gating by task, content type, and agent profile will reduce negative transfer.

---

## 5. Scope

### 5.1 In scope

- One external context-management agent.
- Two frozen task agents for the final cross-agent experiment.
- Explicit KEEP, COMPRESS, ARCHIVE, RETRIEVE, and RESTORE operations.
- Persistent structured lesson memory.
- Counterfactual replay of suspected context decisions.
- Controlled synthetic benchmark with ground-truth critical evidence.
- Optional extension to one established long-horizon benchmark if time and compute permit.
- Baselines, ablations, token accounting, and failure analysis.

### 5.2 Out of scope for the first version

- Foundation-model pretraining.
- Fine-tuning every task agent.
- Full AdaCoM RL reproduction requiring large multi-GPU infrastructure.
- Arbitrary open-world browser automation as the initial benchmark.
- Multi-agent negotiation unrelated to context management.
- Claims of lifelong learning over months or years.
- Universal causal identification under uncontrolled real-world conditions.

### 5.3 Why the controlled benchmark comes first

A controlled benchmark allows the team to know:

- which evidence is essential;
- which information is a distractor;
- which facts may be safely compressed;
- which information must remain verbatim;
- which archived item must later be retrieved;
- which manager decision caused a designed failure.

This makes attribution measurable. An open-ended web benchmark should be an extension, not the first validation environment.

---

## 6. System Architecture

### 6.1 Components

#### A. Environment / task generator

Produces long-horizon tasks with staged observations, constraints, distractors, contradictions, and delayed evidence.

#### B. Context ledger

Stores every raw context item and every transformation. Raw information is immutable; managed views are derived artifacts.

#### C. Context-management agent

Chooses and applies KEEP, COMPRESS, ARCHIVE, RETRIEVE, and RESTORE operations.

#### D. Active context builder

Constructs the exact prompt supplied to the task agent.

#### E. Frozen task agent

Solves the task using the managed context. The task agent is not fine-tuned or updated.

#### F. Evaluator

Scores final correctness, constraint satisfaction, evidence use, and context cost.

#### G. Attribution engine

Selects suspected context decisions, performs interventions, replays downstream execution, and estimates outcome differences.

#### H. Reflection engine

Converts attributed effects into candidate lessons.

#### I. Lesson validator and store

Rejects unsupported lessons, merges duplicates, tracks evidence, and promotes repeatedly supported lessons.

#### J. Lesson retriever

Retrieves the most relevant validated lessons before a new context-management decision.

### 6.2 End-to-end flow

```text
1. Task begins.
2. Environment emits observations over multiple steps.
3. Context ledger stores each raw item.
4. Manager retrieves relevant prior lessons.
5. Manager chooses context operations.
6. Active-context builder creates the task-agent view.
7. Frozen task agent acts or answers.
8. Evaluator scores the outcome.
9. If outcome is poor or uncertain, attribution engine selects suspect decisions.
10. Counterfactual variants are replayed.
11. Outcome differences are estimated.
12. Reflection engine generates a lesson only for attribution-supported effects.
13. Lesson store records evidence and confidence.
14. Future episodes retrieve applicable lessons.
```

---

## 7. Context-Management Action Space

### 7.1 KEEP

Retain an item in active context without changing its content.

Use when:

- the item is an unresolved hard constraint;
- exact wording or numeric values matter;
- the item is current evidence for a live hypothesis;
- the item is required by the final output format.

### 7.2 COMPRESS

Replace one or more items with a shorter representation while retaining references to the originals.

Compression must preserve, where applicable:

- names and entities;
- values and units;
- negations;
- logical dependencies;
- uncertainty markers;
- source provenance;
- unresolved contradictions.

### 7.3 ARCHIVE

Remove an item from active context while keeping it retrievable.

Use when:

- the information is not currently relevant;
- the item is duplicated elsewhere;
- the associated hypothesis is inactive but may return;
- the item is too detailed for the current reasoning stage.

### 7.4 RETRIEVE

Return an archived raw or compressed item to active context.

A retrieval decision should record:

- query;
- retrieved IDs;
- similarity or ranking metadata where available;
- reason for retrieval;
- whether the retrieved item was used by the task agent.

### 7.5 RESTORE

Replace a compressed representation with the original content. RESTORE is especially important for counterfactual replay.

---

## 8. Data Model and Trace Schema

### 8.1 Context item

```json
{
  "item_id": "ctx_0042",
  "episode_id": "ep_0012",
  "step": 7,
  "type": "tool_observation",
  "content": "The deployment must operate offline and use no cloud API.",
  "source": "task_environment",
  "created_at": "...",
  "immutable": true,
  "tags": ["hard_constraint", "negation", "deployment"],
  "criticality_ground_truth": "critical"
}
```

### 8.2 Management decision

```json
{
  "decision_id": "dec_0081",
  "episode_id": "ep_0012",
  "step": 8,
  "operation": "COMPRESS",
  "input_item_ids": ["ctx_0042", "ctx_0043"],
  "output_item_id": "ctx_view_0031",
  "output_content": "The user requires a constrained deployment.",
  "rationale": "Reduce repetitive deployment details.",
  "active_tokens_before": 7450,
  "active_tokens_after": 6910,
  "retrieved_lesson_ids": ["lesson_0011"],
  "manager_model": "manager-model-id",
  "prompt_version": "cm_v1"
}
```

### 8.3 Task outcome

```json
{
  "episode_id": "ep_0012",
  "task_agent": "agent_A",
  "final_answer": "Use a hosted cloud inference endpoint.",
  "task_score": 0.0,
  "constraint_score": 0.5,
  "violated_constraints": ["must operate offline"],
  "active_context_tokens": 6910,
  "total_input_tokens": 22100,
  "tool_calls": 8
}
```

### 8.4 Counterfactual intervention

```json
{
  "intervention_id": "int_0092",
  "episode_id": "ep_0012",
  "target_decision_id": "dec_0081",
  "intervention_type": "RESTORE_ORIGINAL",
  "original_operation": "COMPRESS",
  "counterfactual_context_change": "Restore ctx_0042 and ctx_0043",
  "rollout_seed": 17,
  "counterfactual_task_score": 1.0,
  "counterfactual_constraint_score": 1.0
}
```

### 8.5 Lesson

```json
{
  "lesson_id": "lesson_0023",
  "status": "candidate",
  "lesson_type": "compression_safety",
  "trigger": {
    "operation": "COMPRESS",
    "content_types": ["hard_constraint", "negation"],
    "task_phase": "planning"
  },
  "instruction": "Preserve hard constraints, negations, quantities, and units verbatim during compression.",
  "supporting_decisions": ["dec_0081"],
  "original_mean_score": 0.0,
  "counterfactual_mean_score": 1.0,
  "estimated_effect": 1.0,
  "rollouts": 5,
  "confidence": 0.88,
  "source_agent": "agent_A",
  "scope": "candidate_cross_agent",
  "created_from": "counterfactual_replay"
}
```

---

## 9. Causal Attribution Design

### 9.1 Why reflection alone is insufficient

An LLM can look at a failed trace and produce a plausible blame statement. Plausibility does not establish that changing the suspected context decision would change the outcome.

Therefore:

- reflection proposes suspects;
- intervention tests suspects;
- only supported effects become lessons.

### 9.2 Outcome variable

Let the outcome score be:

```text
Y = w1 × task_correctness
  + w2 × constraint_satisfaction
  + w3 × evidence_support
  - w4 × context_cost
```

For the first experiment, use a simpler decomposition rather than hiding everything inside one scalar:

- task correctness;
- constraint satisfaction;
- active-context tokens;
- total input tokens.

The team can report a combined utility separately.

### 9.3 Decision-level effect

For a management decision `d_i`:

```text
Effect(d_i) = E[Y | intervene with alternative decision]
            - E[Y | retain original decision]
```

Examples:

```text
COMPRESS → RESTORE original
ARCHIVE  → KEEP active
RETRIEVE → OMIT retrieved item
KEEP     → ARCHIVE or remove
OMIT     → inject missing item
```

### 9.4 Replay protocol

1. Reconstruct the exact state immediately before the target context decision.
2. Replace only the selected manager operation.
3. Preserve all recorded upstream observations.
4. Rerun downstream task-agent behavior.
5. Repeat because hosted models can be stochastic.
6. Compare the distribution of outcomes, not only one run.

### 9.5 Screening and confirmation

To control cost:

- first screen the last 3–5 relevant context decisions before failure;
- perform a small number of replays per decision;
- allocate additional replays only to decisions showing meaningful outcome change;
- label unstable effects as uncertain rather than forcing a conclusion.

The exact replay count is an experimental parameter, not a claim from prior work. A practical starting point is three screening rollouts and up to five confirmation rollouts for promising decisions.

### 9.6 Attribution labels

```text
HARMFUL:
Reversing the decision reliably improves the outcome.

HELPFUL:
Reversing the decision reliably worsens the outcome.

NEUTRAL:
Intervention produces negligible change.

UNCERTAIN:
Results are inconsistent or confidence is insufficient.
```

Only HARMFUL and HELPFUL decisions create candidate lessons. UNCERTAIN decisions remain recorded but do not update lesson memory.

### 9.7 Multi-decision interactions

Two context decisions may jointly cause failure. For example:

- compressing a constraint may be harmless if a later tool output repeats it;
- archiving the later output makes the earlier compression damaging.

The MVP should focus on single-decision interventions. Pairwise interventions can be an extension for the final submission if single-decision attribution is insufficient.

### 9.8 Attribution ground truth

Controlled tasks should contain planted failure mechanisms so attribution accuracy can be measured. Example:

```text
Critical fact: "must not use external network access"
Planted damaging action: compression removes the negation
Expected causal decision: the compression operation
```

This supports metrics such as top-1 localization accuracy and false-attribution rate.

---

## 10. Experiential Lesson Lifecycle

### 10.1 Candidate generation

The reflection engine receives:

- original decision;
- original context;
- downstream outcome;
- counterfactual intervention;
- counterfactual outcome;
- effect estimate;
- uncertainty information.

It produces a general rule, not an episode-specific complaint.

Bad lesson:

> Do not compress observation 7 in episode 12.

Better lesson:

> Preserve exact negations and mandatory constraints when compressing requirement-bearing messages.

### 10.2 Validation

A candidate lesson should satisfy:

- causal support above a chosen threshold;
- no contradiction with higher-confidence lessons;
- enough specificity to be actionable;
- enough generality to apply beyond the source task;
- no dependence on hidden ground-truth labels at inference time.

### 10.3 Promotion states

```text
candidate
    ↓ repeated support on another episode
validated
    ↓ success on held-out task
transferable
    ↓ success with another task agent
cross_agent_validated
```

### 10.4 Deduplication

Semantically similar lessons should be merged while retaining all supporting evidence. Example:

- preserve exact numbers during compression;
- do not paraphrase numeric limits;
- retain quantities and units verbatim.

These may form one canonical lesson with multiple evidence records.

### 10.5 Contradictions and scope

Lessons can conflict:

- preserve more raw context for Agent A;
- compress aggressively for Agent B.

Do not force a universal rule. Store scope:

```json
{
  "scope": {
    "agent_family": "agent_A_family",
    "capability_band": "smaller_model",
    "task_type": "multi_constraint_qa",
    "context_state": "high_redundancy"
  }
}
```

### 10.6 Lesson retrieval

Retrieve lessons using a representation of:

- candidate operation;
- content type;
- current task type;
- task phase;
- active-context size;
- source/target agent profile;
- known failure pattern.

Only a small top-k set should be injected. Otherwise lesson memory becomes another source of context overload.

---

## 11. Benchmark Design

### 11.1 Controlled task families

#### Family A: hard-constraint preservation

The task includes exact requirements such as:

- budget limits;
- latency thresholds;
- memory limits;
- offline-only operation;
- forbidden tools;
- required output fields.

Failure occurs when compression removes exact values or negations.

#### Family B: delayed dependency

An early observation appears unimportant but becomes necessary after a later observation. The manager must archive and retrieve rather than permanently discard.

#### Family C: contradiction resolution

A later observation corrects an earlier one. The manager must preserve provenance and avoid merging contradictory facts into an incorrect summary.

#### Family D: distractor overload

Many semantically related but irrelevant observations surround a small number of decisive facts.

#### Family E: repeated evidence

Several observations repeat the same fact. Compression should reduce cost without changing task outcome.

#### Family F: exact procedural sequence

The task requires ordered steps. Compression that removes ordering should cause failure.

#### Family G: agent-specific context preference

The same task is run with two task agents. One may perform better with concise structured context; another may benefit from higher-fidelity raw observations.

### 11.2 Task template

Each generated task should include:

```json
{
  "task_id": "constraint_001",
  "query": "Select the valid deployment configuration.",
  "observations": [...],
  "critical_item_ids": [...],
  "safe_to_compress_item_ids": [...],
  "distractor_item_ids": [...],
  "required_retrieval_item_ids": [...],
  "gold_answer": "...",
  "gold_constraints": [...],
  "planted_failure_points": [...]
}
```

### 11.3 Data split

Use disjoint splits:

- **train/experience split:** lessons may be created;
- **validation split:** lesson thresholds and retrieval settings are selected;
- **test split:** learning is frozen and final results are reported;
- **cross-agent test split:** Agent-B evaluation using Agent-A lessons.

Do not generate and evaluate on identical templates with only entity names changed. Vary the reasoning structure and failure type.

### 11.4 Optional established benchmark

If the controlled benchmark succeeds, add one bounded established benchmark. Candidate choices include the BrowseComp-Plus or MCP-Bench workflows represented in the AdaCoM repository, or the previously considered bounded OpenRCA-style setting. This should remain optional until the core mechanism is validated.

---

## 12. Baselines

### B0: Full context

All available context is retained. This measures task quality and token cost without context management.

### B1: Sliding window

Retain the most recent N tokens/messages. This tests a simple recency policy.

### B2: Fixed summarization

Periodically summarize older context using one fixed prompt. No learning and no retrieval of lessons.

### B3: Relevance-based retrieval

Archive items and retrieve by semantic similarity. No experiential lesson memory.

### B4: Non-learning context-management agent

Use KEEP/COMPRESS/ARCHIVE/RETRIEVE actions, but reset between episodes and do not store lessons.

### B5: Manager with naive reflection

After failure, an LLM generates a lesson directly from the trace. No counterfactual replay validates the blame.

### B6: Manager with attributed reflection

Generate lessons only after intervention-supported attribution. This is the proposed within-agent system.

### B7: Cross-agent transfer

Use lessons learned while managing Agent A to manage Agent B, with Agent B frozen and no Agent-B-specific lessons initially.

---

## 13. Required Ablations

| Variant | Manager actions | Reflection | Counterfactual replay | Persistent lessons | Cross-agent transfer |
|---|---:|---:|---:|---:|---:|
| Full context | No | No | No | No | No |
| Fixed summarization | Fixed | No | No | No | No |
| Non-learning manager | Yes | No | No | No | No |
| Naive experiential manager | Yes | Yes | No | Yes | Optional |
| Attributed manager without lesson retrieval | Yes | Yes | Yes | Stored only | No |
| Attributed experiential manager | Yes | Yes | Yes | Yes | No |
| Full proposed system | Yes | Yes | Yes | Yes | Yes |

Additional ablations if capacity permits:

- remove lesson confidence filtering;
- retrieve all lessons instead of top-k;
- remove agent-profile conditioning;
- use only harmful-decision lessons;
- use only helpful-decision lessons;
- disable RESTORE;
- use one replay instead of repeated replay;
- generate lessons from successes only, failures only, or both.

---

## 14. Metrics

### 14.1 Task quality

- task success rate;
- exact-match or structured-answer accuracy;
- constraint satisfaction rate;
- evidence-supported answer rate;
- unsupported-answer rate;
- abstention quality where applicable.

### 14.2 Context efficiency

- average active-context tokens;
- peak active-context tokens;
- total input tokens per task;
- compression ratio;
- number of archived items;
- number of retrievals;
- retrieved-item utilization;
- tokens per successful task.

### 14.3 Attribution quality

- top-1 causal-decision localization accuracy;
- top-k localization recall;
- false-attribution rate;
- intervention outcome-flip rate;
- effect-size confidence or variance;
- proportion labeled uncertain;
- replay fidelity diagnostics.

### 14.4 Learning quality

- improvement across experience episodes;
- held-out gain from validated lessons;
- lesson reuse frequency;
- percentage of candidate lessons promoted;
- lesson contradiction rate;
- lesson-memory token overhead;
- harmful lesson rate.

### 14.5 Cross-agent transfer

```text
Transfer Gain(A → B)
= Score(B with Agent-A lessons)
- Score(B without lessons)
```

Also report:

```text
Transfer Efficiency
= Transfer Gain / additional lesson tokens
```

and:

```text
Negative Transfer Rate
= fraction of tasks where transferred lessons reduce score
```

### 14.6 Cost and latency

- manager calls;
- task-agent calls;
- replay calls;
- end-to-end latency;
- total token usage;
- optional API cost accounting if hosted models are used.

---

## 15. Experimental Protocol

### 15.1 Phase 1: deterministic mechanism validation

Goal: confirm that the ledger, context operations, replay, and lesson store work.

Use rule-based or deterministic task scoring. Plant known failure points. Verify that restoring a deleted/over-compressed critical fact changes the outcome as expected.

### 15.2 Phase 2: naive versus attributed reflection

For each failed task:

1. ask naive reflector to identify the damaging decision;
2. run counterfactual replay for selected suspects;
3. compare both against planted ground truth;
4. create lesson memories separately;
5. evaluate both lesson stores on held-out tasks.

This is the most important experiment for defending causal attribution.

### 15.3 Phase 3: within-agent experiential learning

Run Agent A across experience tasks. Add only validated lessons. Evaluate on an unseen Agent-A test split with:

- no lessons;
- naive lessons;
- attributed lessons.

### 15.4 Phase 4: cross-agent transfer

Run frozen Agent B on the same held-out distribution:

```text
B0: no context manager
B1: manager without lessons
B2: manager with Agent-A lessons
B3: manager with Agent-B-specific lessons
B4: manager with combined A+B lessons and scope gating
```

Interpretation:

- B2 > B1 demonstrates cross-agent transfer;
- B3 > B2 indicates meaningful agent-specific adaptation;
- B2 ≈ B3 suggests broadly reusable lessons;
- B2 < B1 reveals negative transfer;
- B4 tests whether gating can combine universal and agent-specific experience.

### 15.5 Randomness and repeated runs

Where model output is stochastic:

- fix prompt versions;
- record model identifiers and decoding parameters;
- use repeated trials;
- report means and dispersion;
- preserve all raw outputs;
- do not present a single favorable replay as causal proof.

---

## 16. Mid-Submission Plan for October 30

The mid-submission requires progress, datasets, baselines, ablations, experimental results, challenges, mitigations, final timeline, and updated GitHub. The minimum defensible package is below.

### 16.1 Freeze the project statement

Use the exact proposed title and research question from this document. Avoid further scope expansion before the checkpoint.

### 16.2 Deliver a small pilot rather than the entire system

Minimum pilot:

- 20 controlled tasks;
- one task-agent model;
- one manager model or deterministic manager scaffold;
- four variants: full context, non-learning manager, naive reflection, attributed reflection;
- one planted causal context error per failure-oriented task family;
- actual measured results.

If model/API access blocks the full pilot, implement deterministic replay and a small model sample, clearly separating completed results from planned evaluation.

### 16.3 Mid-submission results table

Populate with actual values only:

| Method | Task success | Constraint satisfaction | Avg active tokens | Attribution accuracy | Total model calls |
|---|---:|---:|---:|---:|---:|
| Full context | TBD | TBD | TBD | N/A | TBD |
| Non-learning manager | TBD | TBD | TBD | N/A | TBD |
| Naive reflection | TBD | TBD | TBD | TBD | TBD |
| Attributed reflection | TBD | TBD | TBD | TBD | TBD |

### 16.4 Evidence to include

- architecture diagram;
- sample raw trajectory;
- sample manager decisions;
- failed output;
- counterfactual intervention;
- changed outcome;
- generated lesson;
- later task where lesson is retrieved;
- GitHub commit showing reproducible code and configuration.

### 16.5 Honest checkpoint framing

The mid-submission should state:

- causal attribution is validated first on controlled tasks;
- cross-agent transfer is the final experiment;
- full RL reproduction is outside the initial scope;
- current results are pilot results, not final generalization claims.

---

## 17. Final-Submission Plan for November 15

### Stage 1: stabilize the controlled benchmark

- freeze task templates and splits;
- verify ground-truth critical evidence;
- verify planted failure interventions;
- add automated scoring.

### Stage 2: complete attribution engine

- decision screening;
- replay support;
- confidence calculations;
- uncertain-effect handling;
- trace visualization.

### Stage 3: complete lesson memory

- candidate generation;
- validation thresholds;
- deduplication;
- promotion states;
- retrieval and scoping.

### Stage 4: within-agent study

- run all baselines;
- run required ablations;
- report quality, cost, and attribution metrics;
- analyze failure modes.

### Stage 5: cross-agent study

- freeze Agent A lessons;
- run Agent B transfer;
- measure gain and negative transfer;
- analyze universal versus agent-specific lessons.

### Stage 6: optional external benchmark

Only after the core claims are supported on controlled tasks, run a bounded established benchmark.

### Stage 7: final report and evaluation preparation

- write method and limitations precisely;
- include both positive and negative findings;
- prepare architecture, attribution, and transfer diagrams;
- rehearse answers to novelty and causality questions.

---

## 18. GitHub Repository Structure

```text
experiential-context-manager/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .env.example
├── configs/
│   ├── manager.yaml
│   ├── agent_a.yaml
│   ├── agent_b.yaml
│   └── experiments/
│       ├── full_context.yaml
│       ├── naive_reflection.yaml
│       ├── attributed_reflection.yaml
│       └── cross_agent_transfer.yaml
├── data/
│   ├── controlled/
│   │   ├── train.jsonl
│   │   ├── validation.jsonl
│   │   └── test.jsonl
│   └── schemas/
├── src/
│   ├── environment/
│   │   ├── task_generator.py
│   │   └── evaluator.py
│   ├── context/
│   │   ├── ledger.py
│   │   ├── actions.py
│   │   ├── manager.py
│   │   ├── archive.py
│   │   └── active_context.py
│   ├── agents/
│   │   ├── task_agent.py
│   │   └── model_client.py
│   ├── attribution/
│   │   ├── suspect_selector.py
│   │   ├── interventions.py
│   │   ├── replay.py
│   │   └── effect_estimator.py
│   ├── lessons/
│   │   ├── reflector.py
│   │   ├── validator.py
│   │   ├── lesson_store.py
│   │   └── retriever.py
│   ├── experiments/
│   │   ├── run_baselines.py
│   │   ├── run_ablation.py
│   │   └── run_transfer.py
│   └── utils/
│       ├── tokens.py
│       ├── logging.py
│       └── seeds.py
├── tests/
│   ├── test_ledger.py
│   ├── test_actions.py
│   ├── test_replay.py
│   ├── test_attribution.py
│   └── test_lesson_store.py
├── scripts/
│   ├── generate_tasks.py
│   ├── run_pilot.py
│   └── aggregate_results.py
├── results/
│   ├── raw/
│   ├── processed/
│   └── figures/
├── notebooks/
│   └── pilot_analysis.ipynb
└── docs/
    ├── PROJECT_BLUEPRINT.md
    ├── MID_SUBMISSION.md
    └── FINAL_REPORT_OUTLINE.md
```

### Repository rules

- Every experiment must use a committed configuration file.
- Raw traces must remain immutable.
- Generated summaries and lessons must reference source IDs.
- Results must record model names, prompt versions, seeds, and run IDs.
- Secrets must never be committed.
- Reported metrics must be reproducible from saved raw outputs.

---

## 19. Core Algorithms

### 19.1 Episode execution

```text
function RUN_EPISODE(task, manager, task_agent, lesson_store):
    ledger = initialize_ledger(task)

    while not task.finished:
        state = ledger.current_state()
        candidate_lessons = lesson_store.retrieve(state)

        decisions = manager.manage(
            raw_items=ledger.raw_items,
            current_view=ledger.active_view,
            lessons=candidate_lessons
        )

        ledger.apply(decisions)
        managed_context = ledger.build_active_context()

        action = task_agent.act(managed_context)
        observation = task.step(action)
        ledger.append_raw(action, observation)

    outcome = evaluate(task, ledger, task_agent)
    return ledger, outcome
```

### 19.2 Attribution

```text
function ATTRIBUTE(ledger, outcome, task_agent):
    suspects = select_suspect_decisions(ledger, outcome)
    effects = []

    for decision in suspects:
        alternatives = generate_interventions(decision, ledger)

        for alternative in alternatives:
            cf_scores = replay_multiple_times(
                checkpoint=ledger.before(decision),
                intervention=alternative,
                task_agent=task_agent
            )

            effect = compare(cf_scores, outcome.scores)
            effects.append(effect)

    return classify_effects(effects)
```

### 19.3 Lesson creation

```text
function CREATE_LESSONS(attributed_effects):
    lessons = []

    for effect in attributed_effects:
        if effect.label in {HARMFUL, HELPFUL} and effect.confident:
            lesson = reflect_into_general_rule(effect)
            if validate_structure(lesson):
                lessons.append(lesson)

    return lessons
```

### 19.4 Cross-agent transfer

```text
lessons_A = learn_lessons(agent_A, experience_split)
freeze(lessons_A)

score_B_no_lessons = evaluate(agent_B, test_split, lessons=[])
score_B_transfer = evaluate(agent_B, test_split, lessons=lessons_A)

transfer_gain = score_B_transfer - score_B_no_lessons
```

---

## 20. Failure Modes and Mitigations

### 20.1 Incorrect blame assignment

**Risk:** the system blames context management when the task agent reasoned incorrectly.

**Mitigation:** require counterfactual outcome change before creating a lesson; allow UNCERTAIN attribution.

### 20.2 Replay nondeterminism

**Risk:** outcome changes due to model randomness rather than intervention.

**Mitigation:** repeated rollouts, recorded upstream state, matched parameters, confidence reporting, and deterministic controlled tasks for the initial study.

### 20.3 Overgeneralized lessons

**Risk:** a lesson from one task harms unrelated tasks.

**Mitigation:** attach scope metadata, maintain candidate/validated states, test on held-out tasks, and measure negative transfer.

### 20.4 Lesson-memory overload

**Risk:** the lesson store itself inflates context.

**Mitigation:** top-k retrieval, deduplication, compact canonical lessons, and token-overhead reporting.

### 20.5 Compression hallucination

**Risk:** the manager invents or alters facts while summarizing.

**Mitigation:** retain immutable originals, compare critical fields, allow RESTORE, and test exact-value preservation.

### 20.6 Excessive replay cost

**Risk:** replay multiplies model calls.

**Mitigation:** screen only a small suspect set, early-stop weak effects, cache deterministic tool outputs, and separate research-time learning cost from deployment-time inference cost.

### 20.7 Negative cross-agent transfer

**Risk:** a lesson suited to Agent A harms Agent B.

**Mitigation:** agent-profile conditioning, confidence thresholds, universal versus agent-specific labels, and fallback to a no-lesson manager.

### 20.8 No measurable improvement

**Risk:** persistent lessons add overhead without improving success.

**Mitigation:** report the negative result and identify whether failures arise from attribution, lesson abstraction, retrieval, or task-agent insensitivity.

### 20.9 Scope explosion

**Risk:** attempting RL, several benchmarks, many models, and full causal interaction analysis simultaneously.

**Mitigation:** freeze the MVP to controlled tasks, one manager, one task agent, single-decision replay, and the four main configurations. Add Agent B only after the loop works.

---

## 21. Expected Results and Interpretation

The project should not pre-commit to positive results. The following are possible outcomes.

### Outcome A: attributed lessons improve within-agent performance

Interpretation: context-management experience is useful, and counterfactual validation produces actionable lessons.

### Outcome B: naive reflection and attributed reflection perform similarly

Interpretation: controlled tasks may be too simple, or attribution may not add value beyond reflection under the chosen setting.

### Outcome C: attribution is accurate but lessons do not help

Interpretation: lesson abstraction or retrieval is the bottleneck.

### Outcome D: Agent-A lessons help Agent B

Interpretation: at least some context-management principles transfer across agents.

### Outcome E: Agent-A lessons harm Agent B

Interpretation: context preferences are agent-specific; gating and agent profiling are necessary. This is still a meaningful result.

### Outcome F: full context remains best

Interpretation: the benchmark context may be too short, or management errors outweigh long-context degradation. Report the cost/quality trade-off rather than forcing a superiority claim.

---

## 22. Contribution Claims the Team Can Defend

If supported by experiments, the final report may claim:

1. a context-management trace representation supporting explicit intervention;
2. a counterfactual attribution procedure focused on context-management actions;
3. a persistent store of evidence-backed context-management lessons;
4. a controlled benchmark for context-retention/compression/retrieval failures;
5. an ablation comparing naive reflection with attributed reflection;
6. an empirical analysis of cross-agent lesson transfer and negative transfer.

Avoid unsupported claims such as:

- universal causal identification;
- human-like lifelong learning;
- guaranteed improvement for all agents;
- replacement of RAG, MemGPT, or AdaCoM;
- production readiness after a small benchmark;
- causality based only on an LLM explanation without intervention.

---

## 23. Recommended Project Title Options

### Preferred

**Experiential Context Management: Learning Transferable Lessons from Counterfactual Agent Outcomes**

### More technical

**Causal Experiential Context Management for Cross-Agent Transfer**

### More descriptive

**Learning What to Keep: Counterfactual Attribution and Reusable Context-Management Lessons for Frozen LLM Agents**

### Conservative academic wording

**Evaluating Counterfactually Attributed Context-Management Experience Across Frozen LLM Agents**

---

## 24. Mid-Submission Report Outline

```text
1. Title and team
2. Problem statement
3. Research question
4. Motivation
5. Related work
6. Gap and proposed contribution
7. System architecture
8. Dataset / controlled benchmark
9. Baselines
10. Ablation plan
11. Completed implementation
12. Pilot experimental setup
13. Pilot results
14. Qualitative example
15. Challenges faced
16. Mitigation plan
17. Final submission timeline
18. GitHub repository and reproducibility status
19. Limitations and ethical considerations
20. References
```

### Required qualitative case study

Show one full example:

```text
Original critical fact
→ context-manager decision
→ managed context
→ task-agent failure
→ suspected decision
→ counterfactual intervention
→ corrected outcome
→ generated lesson
→ reuse on another task
```

This makes the contribution understandable in a few minutes.

---

## 25. Team Work Breakdown

### Workstream A: benchmark and evaluator

- task templates;
- splits;
- ground-truth critical evidence;
- deterministic scoring;
- planted failures.

### Workstream B: context ledger and manager

- data schemas;
- context actions;
- active context construction;
- archival and retrieval.

### Workstream C: attribution and replay

- checkpoints;
- interventions;
- repeated replay;
- effect estimates;
- attribution metrics.

### Workstream D: lessons and transfer

- reflection prompt;
- lesson schema;
- validation and deduplication;
- retrieval;
- Agent-A to Agent-B experiment.

### Workstream E: experiments and report

- configurations;
- token accounting;
- results aggregation;
- plots and tables;
- mid/final report;
- GitHub reproducibility checks.

Each workstream should own tests and documentation, not only code.

---

## 26. Immediate Checklist

### Before mid-submission

- [ ] Freeze title and research question.
- [ ] Create GitHub repository structure.
- [ ] Commit this blueprint as `docs/PROJECT_BLUEPRINT.md`.
- [ ] Implement task schema and at least two task families.
- [ ] Implement context ledger.
- [ ] Implement KEEP, COMPRESS, ARCHIVE, RETRIEVE, RESTORE.
- [ ] Implement full-context baseline.
- [ ] Implement non-learning manager baseline.
- [ ] Implement naive reflection baseline.
- [ ] Implement one counterfactual replay intervention.
- [ ] Save all raw traces.
- [ ] Run small pilot.
- [ ] Populate results with actual numbers.
- [ ] Document challenges and mitigations.
- [ ] Link updated GitHub in the submission.

### Before final submission

- [ ] Freeze controlled dataset splits.
- [ ] Complete attribution metrics.
- [ ] Complete lesson lifecycle.
- [ ] Run within-agent experiments.
- [ ] Run required ablations.
- [ ] Add Agent B.
- [ ] Run cross-agent transfer.
- [ ] Measure negative transfer.
- [ ] Report token and replay costs.
- [ ] Complete qualitative error analysis.
- [ ] Freeze code and environment.
- [ ] Produce final report and evaluation slides.

---

## 27. Questions the Evaluators Are Likely to Ask

### “Is this just AdaCoM plus Reflexion?”

Proposed answer:

> AdaCoM demonstrates learned external context management for frozen agents, while Reflexion stores verbal lessons from task feedback. Our evaluation focuses on a different unit of learning: evidence-backed lessons about the context manager's own operations. We intervene on those operations, replay downstream execution, and store a lesson only when changing the context decision changes the outcome. We then test whether these lessons transfer to a different frozen task agent.

### “Why is the context manager an agent?”

Proposed answer:

> The manager observes state, retrieves prior management lessons, chooses among context operations, records decisions, evaluates downstream consequences, and updates persistent experience. The agentic loop enables sequential decisions and experience reuse; the novelty claim does not rely on the label “agent.”

### “How do you know the manager caused the failure?”

Proposed answer:

> We do not infer causality from explanation alone. We restore or replace a suspected management decision and replay the downstream task. We compare outcome distributions and report uncertain cases rather than forcing attribution.

### “Why not train the manager with RL?”

Proposed answer:

> RL is a strong baseline but may require substantial training infrastructure. This project studies whether a manager can improve through explicit, inspectable experiential lessons and whether those lessons transfer without updating the task agent. The two approaches are complementary.

### “What transfers across agents?”

Proposed answer:

> Structured lessons about context operations, such as preserving exact constraints, retaining contradiction provenance, or retrieving archived evidence when a hypothesis reactivates. We measure transfer empirically and separately report negative transfer.

### “What makes the evaluation causal?”

Proposed answer:

> We intervene on a recorded context-management decision and rerun downstream behavior. Controlled tasks provide known failure points, and repeated replay accounts for stochasticity. We avoid claiming universal causality outside the evaluated intervention setting.

### “What if full context is best?”

Proposed answer:

> The research question concerns the task-quality/context-cost trade-off. A result showing that context management saves tokens but harms accuracy, or only helps beyond a certain trajectory length, is still informative.

---

## 28. Limitations

- Counterfactual replay can be expensive.
- Hosted-model nondeterminism makes exact replay difficult.
- Controlled benchmarks may simplify real-world context-management failures.
- Single-decision interventions may miss interactions.
- Verbal lessons may be sensitive to the reflector model and prompt.
- Transferred lessons may encode source-agent preferences.
- Causal claims are limited to the intervention design and measured outcomes.
- A small number of task agents cannot establish universal transfer.
- Token efficiency does not automatically imply latency or monetary efficiency.

These limitations should be visible in the report, not hidden.

---

## 29. Success Criteria

The project is successful if it produces a reproducible answer to the research question, including a negative answer.

The strongest positive result would satisfy all of the following:

1. attributed reflection localizes damaging context decisions more accurately than naive reflection;
2. validated lessons improve held-out tasks for Agent A;
3. Agent-A lessons improve Agent B without retraining Agent B;
4. gains remain meaningful after accounting for lesson and replay overhead;
5. negative-transfer cases are identified and analyzed.

A credible partial success would satisfy the first two and present cross-agent transfer as an inconclusive or negative result with evidence.

---

## 30. References and Implementation Starting Points

1. Lewis et al. **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.** 2020.
2. Karpukhin et al. **Dense Passage Retrieval for Open-Domain Question Answering.** 2020.
3. Packer et al. **MemGPT: Towards LLMs as Operating Systems.**
4. Shinn et al. **Reflexion: Language Agents with Verbal Reinforcement Learning.**
5. Wang et al. **Voyager: An Open-Ended Embodied Agent with Large Language Models.**
6. Yi et al. **Learning Agent-Compatible Context Management for Long-Horizon Tasks (AdaCoM).** [arXiv abstract](https://arxiv.org/abs/2605.30785)
7. AdaCoM authors. **AdaCoM code repository.** [GitHub repository](https://github.com/luyi256/AdaCoM)
8. Bonagiri et al. **CausalFlow: Causal Attribution and Counterfactual Repair for LLM Agent Failures.** [arXiv abstract](https://arxiv.org/abs/2605.25338)
9. Shah. **Causal Agent Replay: Counterfactual Attribution for LLM-Agent Failures.** [Technical write-up](https://jaineet17.github.io/causal-agent-replay/writeup.html)
10. Shah. **Causal Agent Replay code repository.** [GitHub repository](https://github.com/jaineet17/causal-agent-replay)

---

## 31. Final One-Paragraph Proposal

We propose an **Experiential Context-Management Agent** for long-horizon LLM tasks. The manager controls which trajectory information is kept, compressed, archived, retrieved, or restored before being presented to a frozen task-solving agent. After an outcome, the system does not immediately trust an LLM-generated explanation of failure. Instead, it performs counterfactual replay by reversing or replacing suspected context-management decisions and measuring whether downstream outcomes change. Supported effects are converted into structured, reusable context-management lessons with evidence, confidence, and scope metadata. The project evaluates whether these lessons improve held-out tasks, whether causal attribution outperforms naive reflection, and whether lessons learned while managing one frozen task agent transfer to another without parameter retraining. The contribution is therefore not a new memory tier or summarizer, but an experimental framework for **causally attributed, experience-driven, cross-agent context management**.
