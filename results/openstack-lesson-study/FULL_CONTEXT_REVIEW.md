# Review of failed full-context pilot answers

All six full-context failures were inspected after the unchanged phrase-grader evaluation. This triage explains why a failed score is not sufficient to identify context loss. It supplies no replacement semantic score, and no pilot scores were changed.

The exact answers, tasks and grader fields are in `hosted/episodes.jsonl`. The source constraints are preserved in the final full-context ledger in every case below.

| Case | Grader finding | Answer-level observation |
|---|---|---|
| `openstack_false_distractor_026` | Forbidden `scheduler` | The answer correctly names Glance and says the cause is “not Nova scheduler failures.” The em-dash-attached “problem—not” is not recognized as a standalone negation token. |
| `openstack_hard_constraint_027` | Forbidden `live migration`; missing `stop first` | The answer says live migration is prohibited and to “stop the instance first.” Phrase matching misses these forms. Its advice also shifts the RAM check to the target host; that detail requires domain review. |
| `openstack_hard_constraint_028` | Forbidden `live migration`; missing `stop first` | The answer prescribes cold migration “instead of live migration” and “stop the VM first,” respecting the expressed constraint but failing literal forms. |
| `openstack_negation_critical_026` | Forbidden `auto-renew` / `automatic token`; missing `user action` | The answer explicitly prohibits automatic renewal and requires the user to manually re-authenticate. Post-nominal prohibition and the paraphrase of user action are missed. |
| `openstack_negation_critical_027` | Same token-renewal terms | Again prohibits automatic refresh and requires manual re-authentication; grading fails on the literal forbidden/required phrases. |
| `inc_hard_000` | Forbidden `API` | The answer recommends local bundled deployment and embedded inference “rather than external APIs.” The substring match and unrecognized “rather than” construction trigger the failure. |

The eight separate semantic fixtures in `evaluator-audit.json` independently demonstrate post-nominal negation, unrelated-negation windows, negated synonyms and dismissed alternatives. Four disagree with the grader. They are targeted diagnostic cases, not a representative grader-accuracy benchmark.

For presentation, say: **Our scored pilot supports a controlled policy comparison, and we found grading limitations that must be addressed before interpreting it as semantic task accuracy.** Future grader revisions should use an independently specified test set and report both old and revised scores; they should not be tuned to make the lesson policy win.
