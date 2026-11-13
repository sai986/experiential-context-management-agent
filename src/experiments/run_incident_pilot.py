"""
Full incident pilot experiment.

Runs the experiential context-management pipeline on the 6-family incident
task set and measures whether learned lessons improve performance on held-out
tasks (RQ1 / RQ2) and transfer to Agent B (RQ3).

Usage:
    # Test mode — no API calls, free to run:
    python -m src.experiments.run_incident_pilot

    # Real LLM mode:
    USE_LLM=1 ANTHROPIC_API_KEY=sk-... python -m src.experiments.run_incident_pilot

Output files (all under results/incident/):
    raw/episode_results.jsonl   — one line per episode, written as each completes
    raw/all_replays.jsonl       — one line per replay result
    processed/summary.json      — condition × family success rates + lessons

Crash recovery:
    Re-running resumes from the last completed episode — already-written lines
    in episode_results.jsonl are skipped so no duplicate API calls are made.
    Delete episode_results.jsonl to force a full re-run from scratch.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from src.agents.task_agent import llm_task_agent, simulated_task_agent
from src.attribution.replay import (
    create_lessons_from_replays,
    find_suspect_archives,
    find_suspect_compressions,
    replay_archive_keep,
    replay_restore_original,
)
from src.context.ledger import ContextLedger
from src.context.manager import (
    FixedCompressionManager,
    FullContextManager,
    LessonAwareManager,
    NonLearningManager,
)
from src.environment.evaluator import evaluate_answer
from src.environment.task_generator import ControlledTask
from src.environment.task_generator_incident import (
    generate_incident_tasks,
    materialize_incident_splits,
)
from src.lessons.lesson_store import LessonStore

ROOT              = Path(__file__).resolve().parents[2]
DATA_DIR          = ROOT / "data" / "incident"
RAW_DIR           = ROOT / "results" / "incident" / "raw"
PROCESSED_DIR     = ROOT / "results" / "incident" / "processed"

# How many suspects to replay per failed episode.
# 1 keeps the pilot fast; increase for a thorough training run.
MAX_SUSPECTS_PER_EPISODE = 2


# ---------------------------------------------------------------------------
# Progress logging
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    """Print a timestamped progress line, flushed immediately."""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _log_episode_detail(result: dict) -> None:
    """Log the LLM answer and evaluation detail for root-cause analysis.

    Only prints on failure so successful episodes stay concise.
    On success, prints a one-line answer preview so we can confirm
    the right terms were matched.
    """
    ev = result["evaluation"]
    answer = result["answer"]
    # Truncate long answers for readability — full text is in the JSONL.
    answer_preview = answer[:120].replace("\n", " ") + ("..." if len(answer) > 120 else "")

    if ev["task_success"] == 1:
        _log(f"    -> answer: {answer_preview}")
        return

    # Failure: print everything needed to root-cause the miss.
    _log(f"    -> answer: {answer_preview}")
    missing = ev.get("missing_required_terms", [])
    forbidden = ev.get("forbidden_terms_found", [])
    if missing:
        _log(f"    -> MISSING required terms: {missing}")
    if forbidden:
        _log(f"    -> FORBIDDEN terms found:  {forbidden}")


# ---------------------------------------------------------------------------
# Checkpoint helpers — skip episodes already written to episode_results.jsonl
# ---------------------------------------------------------------------------

def _load_completed(episode_out: Path) -> set[tuple[str, str]]:
    """Return (task_id, condition) pairs already in episode_results.jsonl."""
    done: set[tuple[str, str]] = set()
    if not episode_out.exists():
        return done
    with episode_out.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                done.add((row["task_id"], row["condition"]))
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def _use_llm() -> bool:
    return os.environ.get("USE_LLM", "0") == "1"


def _build_agent_fn(model_client) -> Callable:
    if _use_llm():
        return lambda task, ctx: llm_task_agent(task, ctx, model_client)
    return simulated_task_agent


def _build_model_client():
    if not _use_llm():
        return None
    from src.agents.model_client import ModelClient
    return ModelClient()


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    task: ControlledTask,
    manager,
    condition: str,
    agent_fn: Callable,
) -> dict:
    ledger = ContextLedger(episode_id=f"{condition}_{task.task_id}")
    for observation in task.observations:
        ledger.add_raw(observation.step, observation.content, observation.tags)
        manager.manage_after_observation(ledger, observation.step)

    answer = agent_fn(task, ledger.active_context())
    evaluation = evaluate_answer(task, answer)
    return {
        "task_id":               task.task_id,
        "condition":             condition,
        "family":                task.family,
        "answer":                answer,
        "evaluation":            asdict(evaluation),
        "active_context_words":  ledger.word_estimate(),
        "ledger":                ledger,
    }


# ---------------------------------------------------------------------------
# Replay helpers
# ---------------------------------------------------------------------------

def run_replay_for_episode(
    task: ControlledTask,
    result: dict,
    agent_fn: Callable,
    lesson_store: LessonStore,
    replay_out: Path,
) -> list:
    """Run COMPRESS and ARCHIVE counterfactuals for a failed episode.

    Writes each replay result to replay_out as it completes (crash-safe).
    Returns all replay results for this episode.
    """
    episode_replays = []
    answer = result["answer"]
    ledger = result["ledger"]

    compress_suspects = find_suspect_compressions(ledger)[:MAX_SUSPECTS_PER_EPISODE]
    for suspect in compress_suspects:
        replay = replay_restore_original(task, ledger, suspect, answer, agent_fn)
        episode_replays.append(replay)
        with replay_out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(replay)) + "\n")

    archive_suspects = find_suspect_archives(ledger)[:MAX_SUSPECTS_PER_EPISODE]
    for suspect in archive_suspects:
        replay = replay_archive_keep(task, ledger, suspect, answer, agent_fn)
        episode_replays.append(replay)
        with replay_out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(replay)) + "\n")

    create_lessons_from_replays(episode_replays, ledger, lesson_store)
    return episode_replays


# ---------------------------------------------------------------------------
# Summarisation
# ---------------------------------------------------------------------------

def summarize(results: list[dict]) -> dict:
    by_condition: dict[str, list[dict]] = {}
    for r in results:
        by_condition.setdefault(r["condition"], []).append(r)

    summary: dict[str, dict] = {}
    for condition, rows in by_condition.items():
        n = len(rows)
        by_family: dict[str, dict] = {}
        families = sorted({r["family"] for r in rows})
        for family in families:
            frows = [r for r in rows if r["family"] == family]
            fn = len(frows)
            by_family[family] = {
                "n":            fn,
                "success_rate": sum(r["evaluation"]["task_success"] for r in frows) / fn,
            }
        summary[condition] = {
            "n":                           n,
            "success_rate":                sum(r["evaluation"]["task_success"] for r in rows) / n,
            "avg_constraint_satisfaction": sum(r["evaluation"]["constraint_satisfaction"] for r in rows) / n,
            "avg_active_context_words":    sum(r["active_context_words"] for r in rows) / n,
            "by_family":                   by_family,
        }
    return summary


def serializable_result(result: dict) -> dict:
    out = {k: v for k, v in result.items() if k != "ledger"}
    out["ledger"] = result["ledger"].to_dict()
    return out


def print_analysis(summary: dict) -> None:
    """Print a human-readable analysis report from the summary dict."""
    s = summary["summary"]

    def sr(condition: str) -> float:
        return s.get(condition, {}).get("success_rate", float("nan"))

    def words(condition: str) -> float:
        return s.get(condition, {}).get("avg_active_context_words", float("nan"))

    sep = "=" * 60
    print(sep)
    print("  EXPERIMENT ANALYSIS REPORT")
    print(sep)

    # --- RQ1: Does learning improve over the non-learning baseline? ---
    print("\n--- RQ1: Does lesson learning improve context management? ---")
    baseline     = sr("non_learning_manager_test")
    learned      = sr("lesson_aware_agent_a_test")
    full_ctx     = sr("full_context_test")
    improvement  = learned - baseline
    gap_to_full  = full_ctx - learned
    print(f"  Non-learning baseline (test):   {baseline:.3f}")
    print(f"  Lesson-aware Agent A (test):    {learned:.3f}  ({improvement:+.3f} vs baseline)")
    print(f"  Full-context ceiling (test):    {full_ctx:.3f}  (gap to ceiling: {gap_to_full:.3f})")
    if improvement > 0:
        print("  FINDING: Lesson learning improves over the non-learning baseline.")
    else:
        print("  NOTE: No improvement detected on test split — see discussion below.")

    # --- RQ2: Context efficiency ---
    print("\n--- RQ2: Context efficiency ---")
    words_full    = words("full_context_test")
    words_learned = words("lesson_aware_agent_a_test")
    words_base    = words("non_learning_manager_test")
    if words_full > 0:
        reduction = (words_full - words_learned) / words_full * 100
        print(f"  Full-context avg words:         {words_full:.1f}")
        print(f"  Lesson-aware Agent A avg words: {words_learned:.1f}  ({reduction:.1f}% reduction vs full-context)")
        print(f"  Non-learning avg words:         {words_base:.1f}")

    # --- RQ3: Cross-agent transfer ---
    print("\n--- RQ3: Cross-agent transfer ---")
    agent_a  = sr("lesson_aware_agent_a_test")
    agent_b  = sr("lesson_aware_agent_b_test")
    n_trans  = summary["num_transferable"]
    n_total  = len(summary["learned_lessons_a"])
    print(f"  Lessons learned by Agent A:     {n_total}")
    print(f"  Lessons transferred to Agent B: {n_trans}")
    print(f"  Agent A test success rate:      {agent_a:.3f}")
    print(f"  Agent B test success rate:      {agent_b:.3f}")
    non_learning = sr("non_learning_manager_test")
    if agent_a > 0 and agent_b >= agent_a * 0.9:
        print("  FINDING: Agent B matches Agent A — transfer is effective.")
    elif agent_b > non_learning:
        print("  FINDING: Agent B improves over non-learning baseline — partial transfer.")
    else:
        print("  NOTE: Agent B did not improve over the non-learning baseline.")
        if n_trans == 0:
            print(f"  Reason: 0/{n_total} lessons reached transferable threshold (need ≥5 confirmations).")
        else:
            print(f"  Reason: only {n_trans}/{n_total} lessons transferred — volume may be too low.")

    # --- Per-family breakdown (test split) ---
    print("\n--- Per-family success rates (test split) ---")
    families = sorted({
        fam
        for cond_data in s.values()
        for fam in cond_data.get("by_family", {})
    })
    conditions_test = [
        ("full_context_test",         "full_ctx"),
        ("non_learning_manager_test",  "no_learn"),
        ("lesson_aware_agent_a_test",  "agent_a "),
        ("lesson_aware_agent_b_test",  "agent_b "),
    ]
    header = f"  {'family':<24}" + "".join(f"  {label:<9}" for _, label in conditions_test)
    print(header)
    print("-" * len(header))
    for fam in families:
        row = f"  {fam:<24}"
        for cond, _ in conditions_test:
            fsr = s.get(cond, {}).get("by_family", {}).get(fam, {}).get("success_rate", float("nan"))
            row += f"  {fsr:<9.3f}"
        print(row)

    # --- Replay / lesson summary ---
    print("\n--- Training summary ---")
    print(f"  Total training replays run:     {summary['num_training_replays']}")
    print(f"  Harmful replays (caused losses): {summary['num_harmful_replays']}")
    print(f"  Lessons learned:                {n_total}")
    print(f"  Transferable lessons:           {n_trans}")
    for lesson in summary["learned_lessons_a"]:
        tags   = ", ".join(lesson["trigger_tags"])
        status = lesson["status"]
        count  = lesson["support_count"]
        print(f"    [{status:>12}] COMPRESS on [{tags}]  (confirmed {count}x)")

    print(f"\n  Results saved to: results/incident/processed/summary.json")
    print(sep)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main() -> None:
    # Load ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL, USE_LLM from .env if present.
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # python-dotenv not installed — user must set env vars manually

    mode = "LLM" if _use_llm() else "SIMULATED"
    _log(f"Starting experiment — mode={mode}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    materialize_incident_splits(DATA_DIR, n_per_family=10)
    incident_tasks = generate_incident_tasks(n_per_family=10)
    n_total = len(incident_tasks)
    split_a = int(n_total * 0.70)
    split_b = int(n_total * 0.85)

    train_tasks = [it.task for it in incident_tasks[:split_a]]
    val_tasks   = [it.task for it in incident_tasks[split_a:split_b]]
    test_tasks  = [it.task for it in incident_tasks[split_b:]]
    _log(f"Tasks: {len(train_tasks)} train / {len(val_tasks)} val / {len(test_tasks)} test")

    model_client = _build_model_client()
    agent_fn     = _build_agent_fn(model_client)

    episode_out = RAW_DIR / "episode_results.jsonl"
    replay_out  = RAW_DIR / "all_replays.jsonl"

    # Phase 2/3 checkpoint: skip eval episodes already written.
    # Phase 1 (training) always re-runs so lesson_store_a is fully rebuilt —
    # skipping training episodes would leave the lesson store incomplete.
    eval_completed = _load_completed(episode_out)
    eval_completed = {(tid, cond) for tid, cond in eval_completed if not cond.startswith("train_")}
    if eval_completed:
        _log(f"Resuming — {len(eval_completed)} eval episodes already done, skipping them.")

    # Always clear and re-write Phase 1 output so it stays consistent with
    # the freshly-rebuilt lesson store.
    episode_out.write_text("", encoding="utf-8")
    replay_out.write_text("", encoding="utf-8")

    # ------------------------------------------------------------------
    # Phase 1 — Train Agent A: learn lessons from NonLearningManager failures
    # ------------------------------------------------------------------
    _log(f"Phase 1 — training on {len(train_tasks)} tasks ...")
    lesson_store_a = LessonStore()
    all_replays: list = []
    for i, task in enumerate(train_tasks, 1):
        condition = "train_agent_a"
        _log(f"  [{i}/{len(train_tasks)}] {task.task_id} family={task.family}")
        result = run_episode(task, NonLearningManager(), condition, agent_fn)
        success = result["evaluation"]["task_success"]
        _log(f"    -> success={success}  words={result['active_context_words']}")
        _log_episode_detail(result)
        with episode_out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(serializable_result(result)) + "\n")
        if success == 0:
            _log(f"    -> running replays ...")
            replays = run_replay_for_episode(task, result, agent_fn, lesson_store_a, replay_out)
            all_replays.extend(replays)
            n_lessons = len(lesson_store_a.to_dict())
            _log(f"    -> {len(replays)} replays done  lessons={n_lessons}")

    _log(f"Phase 1 done — {len(all_replays)} replays, {len(lesson_store_a.to_dict())} lessons learned")

    # ------------------------------------------------------------------
    # Phase 2 — Evaluate Agent A on validation + test tasks (4 conditions)
    # ------------------------------------------------------------------
    conditions_a = [
        ("full_context",         FullContextManager()),
        ("fixed_compression",    FixedCompressionManager()),
        ("non_learning_manager", NonLearningManager()),
        ("lesson_aware_agent_a", LessonAwareManager(lesson_store_a, model_client)),
    ]
    test_results_a: list[dict] = []
    for eval_tasks, split_label in [(val_tasks, "val"), (test_tasks, "test")]:
        n_eval = len(eval_tasks)
        _log(f"Phase 2 — {split_label} split ({n_eval} tasks × {len(conditions_a)} conditions)")
        for condition, manager in conditions_a:
            for i, task in enumerate(eval_tasks, 1):
                cond_label = f"{condition}_{split_label}"
                if (task.task_id, cond_label) in eval_completed:
                    continue
                _log(f"  [{i}/{n_eval}] {cond_label}  {task.task_id}")
                result = run_episode(task, manager, cond_label, agent_fn)
                _log(f"    -> success={result['evaluation']['task_success']}  words={result['active_context_words']}")
                _log_episode_detail(result)
                test_results_a.append(result)
                with episode_out.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(serializable_result(result)) + "\n")

    _log("Phase 2 done")

    # ------------------------------------------------------------------
    # Phase 3 — Cross-agent transfer: Agent B receives Agent A's lessons
    # ------------------------------------------------------------------
    transferable = lesson_store_a.get_transferable()
    lesson_store_b = LessonStore()
    for lesson in transferable:
        lesson_store_b.add(lesson)
    _log(f"Phase 3 — {len(transferable)} transferable lessons given to Agent B")

    test_results_b: list[dict] = []
    for eval_tasks, split_label in [(val_tasks, "val"), (test_tasks, "test")]:
        n_eval = len(eval_tasks)
        _log(f"Phase 3 — {split_label} split ({n_eval} tasks)")
        for i, task in enumerate(eval_tasks, 1):
            cond_label = f"lesson_aware_agent_b_{split_label}"
            if (task.task_id, cond_label) in eval_completed:
                continue
            _log(f"  [{i}/{n_eval}] {cond_label}  {task.task_id}")
            result = run_episode(
                task,
                LessonAwareManager(lesson_store_b, model_client),
                cond_label,
                agent_fn,
            )
            _log(f"    -> success={result['evaluation']['task_success']}  words={result['active_context_words']}")
            _log_episode_detail(result)
            test_results_b.append(result)
            with episode_out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(serializable_result(result)) + "\n")

    _log("Phase 3 done")

    # ------------------------------------------------------------------
    # Write final summary
    # ------------------------------------------------------------------
    # On a resume run, test_results_a/b only contain episodes run THIS invocation.
    # Reload the full JSONL so the summary always covers all conditions.
    all_test_results: list[dict] = []
    with episode_out.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if not row.get("condition", "").startswith("train_"):
                    all_test_results.append(row)
            except json.JSONDecodeError:
                pass
    summary = {
        "summary":               summarize(all_test_results),
        "learned_lessons_a":     lesson_store_a.to_dict(),
        "transferred_lessons_b": lesson_store_b.to_dict(),
        "num_training_replays":  len(all_replays),
        "num_harmful_replays":   sum(1 for r in all_replays if r.label == "HARMFUL"),
        "num_transferable":      len(transferable),
    }
    (PROCESSED_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _log("Summary written to results/incident/processed/summary.json")
    print(json.dumps(summary, indent=2))
    print_analysis(summary)


if __name__ == "__main__":
    main()
