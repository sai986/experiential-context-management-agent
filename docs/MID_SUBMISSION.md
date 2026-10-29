# Mid-Submission Update

## Project

Experiential Context Management: Learning Transferable Lessons from Counterfactual Agent Outcomes

## Current Progress

We implemented a minimal controlled pilot for the proposed context-management agent. The current codebase includes:

- controlled synthetic task generation;
- deterministic evaluator;
- immutable context ledger;
- context-management decisions for KEEP, COMPRESS, ARCHIVE, and RESTORE;
- full-context, fixed-compression, and non-learning-manager baselines;
- counterfactual replay by restoring original content after harmful compression;
- lesson creation from attributed failures;
- lesson-aware manager evaluation.

## Dataset

The pilot uses controlled synthetic tasks with two task families:

1. Hard-constraint preservation.
2. Delayed dependency preservation.

The generated splits are:

```text
data/controlled/train.jsonl
data/controlled/validation.jsonl
data/controlled/test.jsonl
```

## Baselines

Current baselines:

```text
full_context
fixed_compression
non_learning_manager
```

Main proposed method:

```text
attributed_lesson_manager
```

## Attribution Mechanism

For failed episodes, the system identifies suspicious COMPRESS decisions over critical or negated information. It then performs a counterfactual replay by restoring the original raw item and rerunning the frozen deterministic task agent. If the restored run succeeds where the original failed, the compression decision is labeled HARMFUL and a lesson is stored.

## Current Limitation

The current pilot uses a deterministic simulated task agent rather than a hosted LLM. This was intentional for the mid-submission so the core attribution and lesson loop can be tested without API-key or nondeterminism issues.

## Final Submission Plan

Next steps:

1. Replace or supplement the simulated task agent with `gpt-5-mini`.
2. Add repeated replay runs for stochastic models.
3. Add RETRIEVE and archive-to-keep interventions.
4. Expand controlled task families.
5. Add cross-agent transfer with Agent A and Agent B.
6. Report token cost, replay cost, attribution accuracy, and negative transfer.

