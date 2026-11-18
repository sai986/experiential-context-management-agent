# Controlled replay and transfer validation results

Offline deterministic fact-retention study. These are not LLM or LogHub task success rates.

| Condition | Successes | Success rate | Mean context words |
|---|---:|---:|---:|
| full_context | 144/144 | 100.0% | 175.7 |
| empty_store | 24/144 | 16.7% | 11.8 |
| learned_a | 96/144 | 66.7% | 15.2 |
| legacy_transfer_b | 24/144 | 16.7% | 11.8 |
| trusted_transfer_b | 96/144 | 66.7% | 15.2 |
| without_archive_lessons | 72/144 | 50.0% | 13.2 |
| without_compression_lessons | 48/144 | 33.3% | 13.8 |

The first three loss families test learned protection; benign_recent is a safe control.
info_mistag and joint_loss are negative controls exposing unchanged policy/replay limitations.
All managed conditions share the executor, input cases, grader and compressor. Evaluation stores are frozen.
Each seed uses new keys, values and nuisance observations. Shared templates do not establish unseen-rule generalization.

Inspect each seed summary for family results, null checks, lesson counts and import states.
The raw gzip JSONL includes complete ledgers; replays include target decisions and both answers.
