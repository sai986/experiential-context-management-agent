# Experiential Context-Management Agent

This repository contains a minimal runnable pilot for an experiential context-management agent.

The pilot demonstrates:

- controlled long-horizon task generation;
- immutable context ledger and management decisions;
- full-context, fixed-compression, and non-learning manager baselines;
- counterfactual replay by restoring a compressed critical fact;
- lesson creation from attributed failures;
- evaluation of a lesson-aware manager.

## Quick Start

Run the pilot:

```bash
python -m src.experiments.run_pilot
```

The script writes:

```text
data/controlled/train.jsonl
data/controlled/validation.jsonl
data/controlled/test.jsonl
results/raw/*.json
results/processed/pilot_summary.json
```

## Current Scope

This is the mid-submission MVP. It uses deterministic task-agent simulation and deterministic scoring so the core attribution loop can be tested without API keys.

The final version can replace the simulated task agent with a frozen LLM such as `gpt-5-mini`, add repeated stochastic replay, add retrieval, and run cross-agent transfer.

