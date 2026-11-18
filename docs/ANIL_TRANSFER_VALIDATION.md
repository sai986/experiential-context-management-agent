# Anil's contribution: held-out replay and transfer validation

This adds a **controlled experiment, measured ablations, an evidence-preserving transfer fix and an artifact verifier**. It builds on the team's dataset work. It does not replace the LogHub pilots or relabel their saved model results.

The `heldout-transfer-validation` branch starts from updated `FinalSubmission` (`747bd87`) and merges the dataset branch (`162b943`) so both the updated report and teammate code are available. The original working checkout remains on its original branch with its untracked design documents.

## Problem and change

Agent B previously received each source rule through `add()`. That function means “record one new confirmation,” so it reset imported rules to candidate/support 1. Default retrieval requires validated status. A serialized rule had arrived, but could not affect B's decisions.

`LessonStore.import_transferable(source)` now imports trusted qualifying source evidence. It preserves status/support, assigns destination IDs, leaves source records unchanged and makes repeated imports idempotent. Existing overlapping evidence uses the **maximum count**, rather than adding counts that may describe the same episodes. An inconsistent transferable source below the threshold is rejected.

All four existing pilot runners use this import interface for future runs. Historical dataset summaries are unchanged: their Agent B numbers have not been rerun and cannot be presented as validation of this correction.

Both non-learning and lesson-aware managers now accept an optional compressor callable. Their existing defaults are retained. The new experiment supplies exactly the same deterministic seven-word truncation function to every managed condition, eliminating compressor quality as a variable in its lesson comparison.

## The concrete experiment

The executor reads key/value facts from visible context and returns JSON for the keys named in the question. It accesses neither grading targets nor family/tag metadata. Missing values become null. Evaluation compares parsed JSON against the exact target dictionary. The existing replay engine uses these same generated values as its required outputs; the runner checks replay labels against the exact JSON evaluation.

This isolates information preservation and replay/store integration. It uses **no hosted model, model weights or natural-language reasoning**. Tags and task families are engineered. The deliberately lossy compressor is a stress condition, not a realistic summarization-quality claim.

Default protocol:

- Three seeds: 7, 19 and 41.
- Per seed: 48 training, 24 validation and 48 test tasks.
- New opaque keys, values and noise per split; every cross-split input/ID comparison must pass.
- Train A with the non-learning policy and identical compressor; replay up to two suspected compressions and two suspected archives per failed episode.
- Promote rules using the existing 2/5 thresholds. Save complete source and imported stores.
- Run validation with the frozen learned store; it is a separate check, with no parameter selection or lesson updates.
- Compare seven conditions on identical test cases; assert that evaluation did not modify A's store.
- Record null interventions on failed training cases: unchanged context must produce the unchanged answer.

| Family | What it tests |
|---|---|
| `critical_compression` | An early tagged setting disappears in compression; a learned compression protection can keep it |
| `delayed_compression` | An early value is requested only after intervening noise |
| `false_archive` | A needed fact tagged critical/distractor is incorrectly archived |
| `benign_recent` | Recent needed information stays available without learned protection; an old unrelated fact may be shortened |
| `info_mistag` | Needed information tagged only as noise is always archived and excluded from replay candidates |
| `joint_loss` | Two lost values are required; restoring either one alone does not change binary success, so no harmful lesson is created |

Held-out inputs/values are distinct. **Templates and tag rules are shared**: this is not an unseen-rule, semantic, cross-domain or cross-model generalization study. The three seeds create different cases and noise; they do not provide evidence about LLM variability.

## Conditions and measured result

The default run has **144 test tasks per condition**, totaling 1,008 test episodes. Including training and validation, the raw artifacts contain 1,224 episodes.

| Condition | Test successes | Purpose |
|---|---:|---|
| Full context | 144/144 | Reference: every required fact is available |
| Same manager, empty store | 24/144 | Compressor-matched learning baseline |
| Learned Agent A | 96/144 | Test learned protection on new inputs/values |
| Legacy transfer to B | 24/144 | Reproduce inactive candidate imports |
| Evidence-preserving transfer to B | 96/144 | Verify imported rules affect decisions and recover A's behavior |
| A without archive lessons | 72/144 | Measured archive-rule ablation |
| A without compression lessons | 48/144 | Measured compression-rule ablation |

See [generated results](../results/transfer-validation/RESULTS.md), [offline HTML](../results/transfer-validation/RESULTS.html) and [aggregate JSON](../results/transfer-validation/summary.json) for the actual run, including context word counts.

The checked-in [verification record](../results/transfer-validation/verification.json) confirms 1,224 episodes, 120 replay interventions and 120 null checks. A second independent default run reproduced all 41 generated artifacts byte for byte in the same source/Python environment; see the [comparison and hashes](../results/transfer-validation/reproducibility.json). Eleven automated tests pass, including rejection of fabricated answers, fabricated replay outcomes and missing null controls.

The first three loss families and benign control succeed with learned/trusted rules. The info-mistag and joint-loss cases still fail. These negative controls explain the 66.7% result and identify concrete future improvements: broaden suspect discovery and represent partial/joint intervention evidence. They are not fixed by this contribution.

Each seed learns three transferable rules from eight distinct harmful training episodes per pattern. Across seeds: 120 replays, including 72 harmful and 48 neutral; 120 null checks. The verifier checks support against distinct training episode IDs, preventing repeated records from masquerading as independent evidence **in this study**. The general lesson store still lacks a universal evidence-ID deduplication mechanism.

Context counts are whitespace words. This experiment does not measure model tokens, API cost, realistic compressor quality, statistical reliability on production data or global novelty. Its concrete contribution is a verified protocol and repair that make the project's intended transfer behavior reviewable.

## Reproduce and verify

From the branch root, Python 3.10+ and the standard library are sufficient:

```bash
python -m unittest discover -s tests -v
python -m src.experiments.run_transfer_validation
python -m src.experiments.verify_transfer_validation results/transfer-validation
```

The runner writes only `results/transfer-validation/` by default, preserving teammate data and prior summaries. To compare a fresh run without overwriting these new study artifacts:

```bash
python -m src.experiments.run_transfer_validation --output /tmp/anil-transfer-reproduction
python -m src.experiments.verify_transfer_validation /tmp/anil-transfer-reproduction
```

The manifest records implementation revision/hashes, Python version, seeds, and dataset hashes. Each seed includes generated splits, duplicate audit, full source/imported/ablated stores, summary, complete episode ledgers in gzip JSONL, and replay decisions with original/counterfactual answers. The verifier recomputes counts and context words from raw records, reconstructs visible facts and replay interventions, validates exact grading, checks source/data hashes and transfer evidence, and rejects inconsistent summaries. It does not require new model calls.

The separate [teammate dataset audit](../results/transfer-validation/teammate-dataset-audit.json) checks exact inputs and IDs. Android has overlapping inputs across splits. BGL/OpenStack passing this exact check does **not** establish independent log windows or unseen templates; those require a provenance-aware audit.

## What to say during the review

> My contribution validates the external learning mechanism with controlled held-out inputs. I fixed imported rules becoming inactive, kept the compressor identical across comparisons, added measured archive/compression ablations and saved enough evidence to reconstruct the numbers. Transfer now changes B's decisions and matches A in this deterministic study. The negative controls show that mislabelled noise and joint losses remain unsolved; separate LLM validation is still needed.
