# Experiential Context-Management Agent

## Step-by-Step Team Implementation Plan

This document converts the project blueprint into an implementation plan the team can follow.

The goal is to build a system where a context manager decides what information to keep, compress, archive, retrieve, or restore before giving context to a frozen task-solving LLM. After the task succeeds or fails, the system performs counterfactual replay to check whether a context-management decision affected the outcome. Supported effects become reusable lessons for future tasks.

---

## 1. Final System in One Paragraph

For each long-horizon task, we generate a sequence of observations containing important facts, distractors, repeated information, contradictions, and delayed constraints. A context ledger stores all raw observations. A context manager builds a smaller active context using actions such as KEEP, COMPRESS, ARCHIVE, RETRIEVE, and RESTORE. A frozen task agent solves the task from that managed context. If the answer is wrong or violates constraints, the attribution engine changes one suspected context decision and replays the downstream run. If restoring or changing the decision improves the result, the system creates a structured lesson and stores it. Future runs retrieve relevant lessons before making context decisions.

---

## 2. Minimum Version for Mid-Submission

The mid-submission should not attempt the full system. The minimum defensible version is:

1. Two controlled task families.
2. One frozen task agent.
3. One context manager.
4. Context ledger with immutable raw items.
5. Four actions: KEEP, COMPRESS, ARCHIVE, RESTORE.
6. One counterfactual replay intervention: restore original content after a harmful compression.
7. Three baselines:
   - full context;
   - fixed compression;
   - context manager without lessons.
8. One attributed-learning version:
   - manager uses stored lessons from previous failures.
9. Small pilot table with actual numbers.
10. One qualitative case study showing:
    - original critical fact;
    - bad context decision;
    - failed task output;
    - counterfactual restore;
    - improved task output;
    - learned lesson.

Cross-agent transfer should be described as final-submission work unless the basic loop is already working.

---

## 3. Recommended Repository Structure

Create this structure:

```text
data/
  controlled/
    train.jsonl
    validation.jsonl
    test.jsonl
  schemas/
    task.schema.json
    trace.schema.json

src/
  environment/
    task_generator.py
    evaluator.py

  context/
    ledger.py
    actions.py
    manager.py
    active_context.py

  agents/
    task_agent.py
    model_client.py

  attribution/
    suspect_selector.py
    interventions.py
    replay.py
    effect_estimator.py

  lessons/
    lesson_store.py
    retriever.py
    reflector.py

  experiments/
    run_pilot.py
    run_baselines.py
    run_learning.py
    aggregate_results.py

results/
  raw/
  processed/
  figures/

docs/
  PROJECT_BLUEPRINT.md
  MID_SUBMISSION.md
  FINAL_REPORT_OUTLINE.md
```

For the mid-submission, only the core files need to work:

```text
src/environment/task_generator.py
src/environment/evaluator.py
src/context/ledger.py
src/context/actions.py
src/context/manager.py
src/agents/task_agent.py
src/agents/model_client.py
src/attribution/replay.py
src/lessons/lesson_store.py
src/experiments/run_pilot.py
src/experiments/aggregate_results.py
```

---

## 4. Phase 1: Define Controlled Tasks

Owner: benchmark/evaluation team

### 4.1 Implement Two Task Families

Start with these two families.

#### Family A: Hard-Constraint Preservation

Purpose: test whether the manager preserves exact requirements.

Example critical facts:

```text
The solution must run offline.
The budget must not exceed 5000 INR.
Do not use cloud APIs.
The output must include exactly three bullet points.
```

Failure mode:

```text
"Do not use cloud APIs" gets compressed to "deployment has constraints."
```

Then the task agent may recommend a cloud-based solution.

#### Family B: Delayed Dependency

Purpose: test whether earlier archived facts are restored when they become relevant later.

Example:

```text
Step 1: The client uses Windows only.
Step 2-8: Many distractors appear.
Step 9: Recommend an installation method.
```

Failure mode:

```text
The manager archives "Windows only" and never restores it.
```

Then the task agent may recommend Linux commands.

### 4.2 Task JSONL Format

Each task should be one JSON object per line:

```json
{
  "task_id": "hard_constraint_001",
  "family": "hard_constraint",
  "observations": [
    {
      "step": 1,
      "content": "The system must run offline and must not use any cloud API.",
      "tags": ["critical", "hard_constraint", "negation", "deployment"]
    },
    {
      "step": 2,
      "content": "The user prefers a simple setup.",
      "tags": ["preference"]
    }
  ],
  "question": "Recommend a deployment approach.",
  "expected_constraints": [
    "offline",
    "no cloud API"
  ],
  "forbidden_outputs": [
    "cloud API",
    "hosted endpoint",
    "SaaS inference"
  ]
}
```

### 4.3 Dataset Size for Pilot

For mid-submission:

```text
train: 10 tasks
validation: 5 tasks
test: 5 tasks
```

For final:

```text
train: 50-100 tasks
validation: 20 tasks
test: 30-50 tasks
```

---

## 5. Phase 2: Build the Context Ledger

Owner: context team

The ledger is the most important engineering component. It must preserve raw facts and record every context-management decision.

### 5.1 Context Item Schema

Implement a `ContextItem` object:

```json
{
  "item_id": "ctx_0001",
  "episode_id": "ep_0001",
  "step": 1,
  "content": "The system must run offline.",
  "source": "task_environment",
  "tags": ["critical", "hard_constraint"],
  "immutable": true
}
```

### 5.2 Decision Schema

Implement a `ManagementDecision` object:

```json
{
  "decision_id": "dec_0001",
  "episode_id": "ep_0001",
  "step": 3,
  "operation": "COMPRESS",
  "input_item_ids": ["ctx_0001"],
  "output_item_id": "ctx_view_0001",
  "output_content": "The deployment has constraints.",
  "rationale": "Reduce context length.",
  "active_tokens_before": 1200,
  "active_tokens_after": 900
}
```

### 5.3 Ledger Responsibilities

The ledger must support:

1. Add raw observation.
2. Apply management decision.
3. Build active context.
4. Return state before a decision.
5. Restore original item from compressed view.
6. Export full trace as JSON.

Do not delete raw context items.

---

## 6. Phase 3: Implement Context Actions

Owner: context team

Implement the actions in `src/context/actions.py`.

### 6.1 KEEP

The item remains active unchanged.

Use for:

```text
hard constraints
exact numbers
negations
output format rules
currently relevant evidence
```

### 6.2 COMPRESS

The item is replaced by a shorter version, but the original raw item remains in the ledger.

For mid-submission, compression can be simple:

```text
Rule-based compression or LLM compression.
```

Important: record the original item ID and compressed output ID.

### 6.3 ARCHIVE

The item is removed from active context but stays retrievable.

For mid-submission, archive can simply mark:

```json
{
  "active": false,
  "archived": true
}
```

### 6.4 RESTORE

Replace compressed content with the original raw content.

This is essential for counterfactual replay.

---

## 7. Phase 4: Implement the First Context Manager

Owner: manager team

The first manager should be simple and inspectable.

### 7.1 Non-Learning Manager

Initial rules:

```text
KEEP items tagged critical, hard_constraint, negation, exact_value.
COMPRESS repeated or long preference/distractor items.
ARCHIVE old distractors.
```

For pilot, also intentionally allow some risky compression so there are failures to analyze.

Example:

```text
Compress old hard constraints after several steps.
```

This creates controlled failures for replay.

### 7.2 Lesson-Aware Manager

Before deciding, retrieve lessons from the lesson store.

Example lesson:

```text
Preserve hard constraints, negations, exact numbers, units, and output format requirements verbatim during compression.
```

If this lesson is retrieved, the manager should KEEP the item or compress it safely.

---

## 8. Phase 5: Implement the Frozen Task Agent

Owner: agent team

Use one frozen task agent for mid-submission.

Recommended:

```text
Task agent A: gpt-5-mini
```

The task agent prompt should be stable and versioned.

Example prompt:

```text
You are solving a controlled task.
Use only the provided managed context.
Respect all explicit constraints.
Return a concise final answer.

Managed context:
{active_context}

Question:
{question}
```

Do not change the task agent prompt between experiment conditions unless the change is explicitly part of the experiment.

---

## 9. Phase 6: Implement Evaluation

Owner: benchmark/evaluation team

Use deterministic scoring first. Avoid LLM-as-judge unless necessary.

### 9.1 Metrics

Record:

```text
task_success: 0 or 1
constraint_satisfaction: 0 to 1
forbidden_output_violation: 0 or 1
active_context_tokens
total_input_tokens
```

### 9.2 Simple Scoring

For hard constraints:

```text
If output contains forbidden term, mark failure.
If required keyword is missing, mark constraint violation.
```

Example:

```text
Forbidden terms:
cloud API
hosted endpoint
online service
```

If any appear in the answer when the task says offline-only, score failure.

---

## 10. Phase 7: Implement Counterfactual Replay

Owner: attribution team

For mid-submission, implement only one replay type:

```text
RESTORE_ORIGINAL_AFTER_COMPRESSION
```

### 10.1 Replay Procedure

1. Take a failed episode.
2. Find a suspected COMPRESS decision involving a critical item.
3. Rebuild active context, but replace compressed content with original raw content.
4. Rerun the same frozen task agent on the changed context.
5. Evaluate the new output.
6. Compare original score and counterfactual score.

### 10.2 Attribution Label

Use simple labels:

```text
HARMFUL:
Restoring the original improves the score.

NEUTRAL:
Restoring the original does not change the score.

UNCERTAIN:
Results are inconsistent across repeated runs.
```

For the pilot, one replay per failed task is acceptable. For final experiments, use 3-5 repeated runs.

---

## 11. Phase 8: Create and Store Lessons

Owner: lessons team

When replay shows that a context decision was harmful, create a lesson.

### 11.1 Lesson Schema

```json
{
  "lesson_id": "lesson_0001",
  "lesson_type": "compression_safety",
  "trigger": {
    "operation": "COMPRESS",
    "tags": ["hard_constraint", "negation"]
  },
  "instruction": "Preserve hard constraints and negations verbatim during compression.",
  "supporting_decisions": ["dec_0001"],
  "original_score": 0,
  "counterfactual_score": 1,
  "status": "validated",
  "source_agent": "gpt-5-mini"
}
```

### 11.2 Lesson Retrieval

For mid-submission, retrieval can be rule-based:

```text
If current item tags match lesson trigger tags, retrieve the lesson.
```

Vector retrieval can be added later, but it is not necessary for the first version.

---

## 12. Phase 9: Run Baselines

Owner: experiments team

Run all conditions on the same test tasks.

### 12.1 Conditions for Mid-Submission

```text
B0_full_context:
Give all observations to the task agent.

B1_fixed_compression:
Compress old observations with no lessons.

B2_non_learning_manager:
Use manager rules, but no learned lessons.

B3_attributed_lesson_manager:
Use lessons created from counterfactual replay.
```

### 12.2 Results Table

Produce a table like:

```text
Condition                  Success   Constraint   Avg active tokens
Full context               0.90      0.95         4200
Fixed compression          0.60      0.65         1800
Non-learning manager       0.70      0.75         2100
Attributed lesson manager  0.85      0.90         2300
```

The numbers above are placeholders. The submitted report must use actual results.

---

## 13. Phase 10: Add Final-Submission Features

Only add these after the pilot loop works.

### 13.1 More Task Families

Add:

```text
contradiction resolution
distractor overload
repeated evidence
exact procedural sequence
agent-specific context preference
```

### 13.2 More Actions

Add RETRIEVE:

```text
Retrieve archived items when current task state needs them.
```

### 13.3 More Attribution

Add:

```text
ARCHIVE -> KEEP intervention
RETRIEVE -> OMIT intervention
KEEP -> REMOVE intervention
```

### 13.4 Cross-Agent Transfer

Use:

```text
Agent A: gpt-5-mini
Agent B: gpt-5-nano
```

Procedure:

1. Learn lessons using Agent A.
2. Freeze the lesson store.
3. Evaluate Agent B without lessons.
4. Evaluate Agent B with Agent-A lessons.
5. Measure transfer gain or negative transfer.

---

## 14. Team Work Breakdown

### Workstream A: Benchmark and Evaluation

Responsibilities:

```text
task_generator.py
evaluator.py
data splits
deterministic scoring
result tables
```

Deliverables:

```text
controlled train/validation/test JSONL files
scoring script
pilot metrics
```

### Workstream B: Context Ledger and Actions

Responsibilities:

```text
ledger.py
actions.py
active_context.py
trace export
token counting
```

Deliverables:

```text
working ledger
action application
saved traces
```

### Workstream C: Manager and Agents

Responsibilities:

```text
manager.py
task_agent.py
model_client.py
prompt versions
```

Deliverables:

```text
non-learning manager
lesson-aware manager
frozen task-agent wrapper
```

### Workstream D: Attribution and Lessons

Responsibilities:

```text
replay.py
interventions.py
effect_estimator.py
lesson_store.py
retriever.py
reflector.py
```

Deliverables:

```text
restore-original replay
effect labels
lesson creation
lesson retrieval
```

### Workstream E: Experiments and Report

Responsibilities:

```text
run_pilot.py
run_baselines.py
aggregate_results.py
MID_SUBMISSION.md
figures/tables
```

Deliverables:

```text
pilot results
qualitative example
mid-submission report
final timeline
```

---

## 15. Suggested Timeline

### Day 1: Skeleton and Dataset

```text
Create repository structure.
Define task JSONL schema.
Generate 20 pilot tasks.
Implement deterministic evaluator.
```

### Day 2: Ledger and Actions

```text
Implement context ledger.
Implement KEEP, COMPRESS, ARCHIVE, RESTORE.
Export full trace JSON.
```

### Day 3: Manager and Task Agent

```text
Implement non-learning manager.
Implement model client.
Implement frozen task agent prompt.
Run full-context and manager conditions.
```

### Day 4: Replay and Lessons

```text
Implement restore-original replay.
Create lessons from harmful compression.
Implement rule-based lesson retrieval.
Run lesson-aware manager.
```

### Day 5: Results and Report

```text
Aggregate metrics.
Write qualitative case study.
Prepare mid-submission report.
Document limitations and final plan.
```

For final submission, use the remaining time to expand task families, repeat runs, add Agent B, and analyze negative transfer.

---

## 16. What to Show in Mid-Submission

The mid-submission should contain:

1. Research question.
2. Short related-work positioning.
3. Architecture diagram.
4. Controlled benchmark design.
5. Context ledger and trace schema.
6. Baselines.
7. Pilot results table.
8. One qualitative replay example.
9. Current limitations.
10. Final-submission timeline.

The key qualitative example should look like this:

```text
Raw fact:
The system must run offline and must not use cloud APIs.

Manager decision:
Compressed to "The deployment has constraints."

Task-agent output:
Use a hosted cloud inference endpoint.

Evaluation:
Failed because it violates offline/no-cloud constraint.

Counterfactual intervention:
Restore the original raw fact in active context.

Replay output:
Use a local model running on the user's machine.

Lesson:
Preserve hard constraints and negations verbatim during compression.
```

---

## 17. Risks and Mitigations

### Risk: Scope Explosion

Mitigation:

```text
Do not implement all task families before the pilot works.
Do not start cross-agent transfer before the single-agent loop works.
Do not add vector retrieval before rule-based retrieval works.
```

### Risk: No Improvement Over Full Context

Mitigation:

```text
Report cost-quality tradeoff.
Full context can be best on short tasks.
The project is about whether management helps under context pressure.
```

### Risk: LLM Nondeterminism

Mitigation:

```text
Use deterministic scoring.
Use repeated runs for final experiments.
Report uncertain attribution instead of forcing claims.
```

### Risk: Lessons Are Just Hand-Written Rules

Mitigation:

```text
Include a hand-written heuristic baseline.
Show that replay validates which lesson matters in which failure.
```

### Risk: Evaluators Say It Is Just AdaCoM Plus Reflexion

Mitigation:

```text
Clearly state the distinction:
AdaCoM learns context-management policy parameters.
Reflexion learns task-agent behavioral lessons.
This project learns evidence-backed lessons about context-management operations.
```

---

## 18. Success Criteria

Minimum success:

```text
The system records context decisions, performs one counterfactual replay,
creates one validated lesson, and shows that the lesson prevents a similar
future context-management failure.
```

Strong final success:

```text
Attributed lessons improve held-out task performance or constraint satisfaction
relative to a non-learning manager under similar token budgets.
```

Best-case final success:

```text
Lessons learned with Agent A also improve Agent B, showing some cross-agent
transfer without retraining either task agent.
```

