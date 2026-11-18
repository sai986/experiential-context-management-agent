"""
Android logcat pilot experiment.

Runs the same context-management pipeline as run_incident_pilot.py but on
incidents derived from real Android logcat data (LogHub / logpai) instead
of synthetically generated tasks.

Each incident in data/android/incidents.jsonl contains:
  - raw_text: a real Android log window wrapped with a scenario constraint
              that creates one of the 6 failure families
  - required_outputs / forbidden_outputs: evaluator ground-truth
  - family / failure_mode: family label for per-family breakdown

The Investigation Agent converts raw_text -> structured Observations with tags.
  USE_LLM=0  ->  simulated_investigation_agent (heuristic, free, no API calls)
  USE_LLM=1  ->  llm_investigation_agent (Claude via Qualcomm proxy, requires key)

After that the pipeline is identical to run_incident_pilot.py.

Usage:
    # Simulated mode — free, no API calls:
    USE_LLM=0 python -m src.experiments.run_android_pilot

    # Real LLM mode (Qualcomm QGenie proxy — pass base URL WITHOUT /v1):
    PYTHONPATH=/local/mnt/workspace/khitesh/MS/pylibs \\
    USE_LLM=1 \\
    ANTHROPIC_API_KEY=$(cat /tmp/apikey.txt) \\
    ANTHROPIC_BASE_URL=https://qgenie-api.qualcomm.com \\
        python -m src.experiments.run_android_pilot

Output:
    USE_LLM=0  ->  results/android/simulated/{raw,processed}/
    USE_LLM=1  ->  results/android/real/{raw,processed}/
"""
from __future__ import annotations

import sys
import os

# Allow running directly from project root without installing the package.
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from src.agents.investigation_agent import (
    llm_investigation_agent,
    simulated_investigation_agent,
)
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
    FullContextManager,
    LessonAwareManager,
    NonLearningManager,
)
from src.environment.evaluator import evaluate_answer
from src.environment.task_generator import ControlledTask, Observation
from src.environment.task_generator_incident import IncidentTask
from src.lessons.lesson_store import LessonStore

ROOT       = Path(__file__).resolve().parents[2]
DATA_DIR   = ROOT / "data" / "android"
TRAIN_FILE = DATA_DIR / "train.jsonl"
TEST_FILE  = DATA_DIR / "test.jsonl"

MAX_SUSPECTS_PER_EPISODE = 2


# ---------------------------------------------------------------------------
# Mode helpers
# ---------------------------------------------------------------------------

def _use_llm() -> bool:
    return os.environ.get("USE_LLM", "0") == "1"


def _build_model_client():
    if not _use_llm():
        return None
    from src.agents.model_client import ModelClient
    return ModelClient()


def _build_agent_fn(model_client) -> Callable:
    if _use_llm():
        return lambda task, ctx: llm_task_agent(task, ctx, model_client)
    return simulated_task_agent


def _results_dir() -> Path:
    mode = "real" if _use_llm() else "simulated"
    return ROOT / "results" / "android" / mode


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Load incidents and convert via Investigation Agent
# ---------------------------------------------------------------------------

def load_android_incidents(data_file: Path, model_client) -> list[IncidentTask]:
    """Read incidents.jsonl; run investigation agent on each raw_text."""
    incidents: list[IncidentTask] = []
    use_llm = _use_llm()

    raw_lines = data_file.read_text(encoding="utf-8").splitlines()
    _log(f"Loading {len(raw_lines)} incidents from {data_file.name} ...")

    for idx, line in enumerate(raw_lines, 1):
        line = line.strip()
        if not line:
            continue
        inc = json.loads(line)

        # -- Investigation Agent: raw_text -> Observations -------------------
        raw_text = inc["raw_text"]
        if use_llm and model_client is not None:
            observations = llm_investigation_agent(raw_text, model_client)
        else:
            observations = simulated_investigation_agent(raw_text)

        _log(
            f"  [{idx:02d}] {inc['incident_id']}  family={inc['family']}"
            f"  obs={len(observations)}"
            f"  tags={set(t for o in observations for t in o.tags)}"
        )

        # -- Wrap in ControlledTask ------------------------------------------
        task = ControlledTask(
            task_id=inc["incident_id"],
            family=inc["family"],
            observations=observations,
            question=inc["question"],
            expected_constraints=inc.get("expected_constraints", []),
            forbidden_outputs=inc["forbidden_outputs"],
            required_outputs=inc["required_outputs"],
            synonym_groups=inc.get("synonym_groups", []),
        )
        incident_task = IncidentTask(
            failure_mode=inc.get("failure_mode", "unknown"),
            critical_step=1,
            task=task,
        )
        incidents.append(incident_task)

    return incidents


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
    for obs in task.observations:
        ledger.add_raw(obs.step, obs.content, obs.tags)
        manager.manage_after_observation(ledger, obs.step)
    answer     = agent_fn(task, ledger.active_context())
    evaluation = evaluate_answer(task, answer)
    return {
        "task_id":              task.task_id,
        "condition":            condition,
        "family":               task.family,
        "question":             task.question,
        "answer":               answer,
        "evaluation":           asdict(evaluation),
        "active_context_words": ledger.word_estimate(),
        "ledger":               ledger,
    }


def _log_episode_detail(result: dict) -> None:
    ev      = result["evaluation"]
    answer  = result["answer"]
    preview = answer[:120].replace("\n", " ") + ("..." if len(answer) > 120 else "")
    _log(f"    -> answer: {preview}")
    if ev["task_success"] == 0:
        missing  = ev.get("missing_required_terms", [])
        found_fb = ev.get("forbidden_terms_found", [])
        if missing:
            _log(f"    -> MISSING: {missing}")
        if found_fb:
            _log(f"    -> FORBIDDEN found: {found_fb}")


def serializable_result(result: dict) -> dict:
    out = {k: v for k, v in result.items() if k != "ledger"}
    out["ledger"] = result["ledger"].to_dict()
    return out


def _load_completed(path: Path) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as f:
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
    episode_replays = []
    answer  = result["answer"]
    ledger  = result["ledger"]

    for suspect in find_suspect_compressions(ledger)[:MAX_SUSPECTS_PER_EPISODE]:
        replay = replay_restore_original(task, ledger, suspect, answer, agent_fn)
        episode_replays.append(replay)
        with replay_out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(replay)) + "\n")

    for suspect in find_suspect_archives(ledger)[:MAX_SUSPECTS_PER_EPISODE]:
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
        n        = len(rows)
        families = sorted({r["family"] for r in rows})
        by_fam: dict[str, dict] = {}
        for fam in families:
            fr  = [r for r in rows if r["family"] == fam]
            fn  = len(fr)
            by_fam[fam] = {
                "n":            fn,
                "success_rate": sum(r["evaluation"]["task_success"] for r in fr) / fn,
            }
        summary[condition] = {
            "n":                           n,
            "success_rate":                sum(r["evaluation"]["task_success"] for r in rows) / n,
            "avg_constraint_satisfaction": sum(r["evaluation"]["constraint_satisfaction"] for r in rows) / n,
            "avg_active_context_words":    sum(r["active_context_words"] for r in rows) / n,
            "by_family":                   by_fam,
        }
    return summary


def print_analysis(summary: dict) -> None:
    s   = summary["summary"]

    def sr(cond: str) -> float:
        return s.get(cond, {}).get("success_rate", float("nan"))

    def wc(cond: str) -> float:
        return s.get(cond, {}).get("avg_active_context_words", float("nan"))

    sep = "=" * 65
    mode_label = "LLM (USE_LLM=1)" if _use_llm() else "SIMULATED (USE_LLM=0)"
    print(sep)
    print(f"  ANDROID LOGCAT PILOT — ANALYSIS REPORT")
    print(f"  Mode: {mode_label}")
    print(f"  Dataset: Android LogHub (logpai/loghub) — 30 incidents, 5 per family")
    print(sep)

    conds = [
        ("full_context_test",        "full_context (ceiling)"),
        ("non_learning_manager_test","non_learning_manager"),
        ("lesson_aware_agent_a_test","lesson_aware_agent_a"),
        ("lesson_aware_agent_b_test","lesson_aware_agent_b (transfer)"),
    ]
    col_w = 32
    print(f"\n  {'Condition':<{col_w}}  {'Success':>8}  {'Avg words':>10}  {'n':>4}")
    print("  " + "-" * (col_w + 28))
    for cond, label in conds:
        success = sr(cond)
        avg_w   = wc(cond)
        n       = s.get(cond, {}).get("n", 0)
        success_str = f"{success:.0%}" if success == success else "  --"
        words_str   = f"{avg_w:.0f}"  if avg_w   == avg_w   else "--"
        print(f"  {label:<{col_w}}  {success_str:>8}  {words_str:>10}  {n:>4}")

    # --- Per-family breakdown -----------------------------------------------
    families = sorted({
        fam
        for cond_data in s.values()
        for fam in cond_data.get("by_family", {})
    })
    col_labels = [
        ("full_context_test",        "ceiling"),
        ("non_learning_manager_test","no-learn"),
        ("lesson_aware_agent_a_test","agent-A"),
        ("lesson_aware_agent_b_test","agent-B"),
    ]
    fam_w  = 24
    cell_w = 9
    print(f"\n  Per-family success rate (test split)")
    header = f"  {'Family':<{fam_w}}" + "".join(f"  {lbl:>{cell_w}}" for _, lbl in col_labels)
    print(header)
    print("  " + "-" * (fam_w + (cell_w + 2) * len(col_labels)))
    for fam in families:
        row = f"  {fam:<{fam_w}}"
        for cond, _ in col_labels:
            fsr = s.get(cond, {}).get("by_family", {}).get(fam, {}).get("success_rate", float("nan"))
            cell = f"{fsr:.0%}" if fsr == fsr else "--"
            row += f"  {cell:>{cell_w}}"
        print(row)

    # --- Findings -----------------------------------------------------------
    baseline    = sr("non_learning_manager_test")
    learned_a   = sr("lesson_aware_agent_a_test")
    learned_b   = sr("lesson_aware_agent_b_test")
    full_ctx    = sr("full_context_test")

    print(f"\n  Findings")
    print("  " + "-" * 40)

    improvement = learned_a - baseline if (learned_a == learned_a and baseline == baseline) else float("nan")
    rq1 = "IMPROVED" if improvement > 0 else "no improvement"
    sign = f"{improvement:+.0%}" if improvement == improvement else "??"
    print(f"  RQ1 (learning)   : Agent A {sign} vs non-learning  [{rq1}]")

    words_full = wc("full_context_test")
    words_a    = wc("lesson_aware_agent_a_test")
    if words_full and words_full > 0 and words_a == words_a:
        reduction = (words_full - words_a) / words_full * 100
        print(f"  RQ2 (efficiency) : {reduction:.0f}% fewer context words vs full-context")
    gap = full_ctx - learned_a if (full_ctx == full_ctx and learned_a == learned_a) else float("nan")
    if gap == gap:
        print(f"                     Gap to full-context ceiling: {gap:+.0%}")

    n_total = len(summary["learned_lessons_a"])
    n_trans  = summary["num_transferable"]
    if learned_a == learned_a and learned_b == learned_b and baseline == baseline:
        if learned_a > 0 and learned_b >= learned_a * 0.9:
            rq3 = "EFFECTIVE transfer"
        elif learned_b > baseline:
            rq3 = "PARTIAL transfer"
        else:
            rq3 = "no transfer benefit"
        print(f"  RQ3 (transfer)   : Agent B {learned_b:.0%}  [{rq3}]"
              f"  ({n_trans}/{n_total} lessons transferred)")

    # --- Lessons ---------------------------------------------------------------
    print(f"\n  Lessons learned by Agent A: {n_total}  |  Transferred to B: {n_trans}")
    print(f"  Training replays: {summary['num_training_replays']}  |"
          f"  Harmful: {summary['num_harmful_replays']}")
    if summary["learned_lessons_a"]:
        print()
        for les in summary["learned_lessons_a"]:
            tags  = ", ".join(les["trigger_tags"])
            count = les["support_count"]
            bar   = "#" * min(count, 5) + "." * (5 - min(count, 5))
            print(f"    [{bar}] {les['status']:<12}  {les['operation']:<8}  [{tags}]  {count}x")

    results_dir = _results_dir()
    print(f"\n  Results -> {results_dir / 'processed' / 'summary.json'}")
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Load .env if available.
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    mode = "LLM" if _use_llm() else "SIMULATED"
    _log(f"Android logcat pilot — mode={mode}")

    if not TRAIN_FILE.exists() or not TEST_FILE.exists():
        print(f"ERROR: Split files not found in {DATA_DIR}.")
        print("Generate them first:")
        print("  python3 data/android/generate_incidents.py")
        return

    results_dir = _results_dir()
    RAW_DIR       = results_dir / "raw"
    PROCESSED_DIR = results_dir / "processed"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    _log(f"Output directory: {results_dir}")

    model_client = _build_model_client()
    agent_fn     = _build_agent_fn(model_client)

    # -- Load incidents via Investigation Agent ------------------------------
    # Phase 1 uses train split; Phases 2+3 use test split.
    train_incident_tasks = load_android_incidents(TRAIN_FILE, model_client)
    test_incident_tasks  = load_android_incidents(TEST_FILE,  model_client)
    train_tasks = [it.task for it in train_incident_tasks]
    test_tasks  = [it.task for it in test_incident_tasks]
    _log(f"Tasks: {len(train_tasks)} train / {len(test_tasks)} test")

    episode_out = RAW_DIR / "episode_results.jsonl"
    replay_out  = RAW_DIR / "all_replays.jsonl"
    episode_out.write_text("", encoding="utf-8")
    replay_out.write_text("", encoding="utf-8")

    eval_completed = _load_completed(episode_out)
    eval_completed = {(tid, cond) for tid, cond in eval_completed
                      if not cond.startswith("train_")}

    # -- Phase 1: Train Agent A ----------------------------------------------
    _log(f"Phase 1 — training on {len(train_tasks)} tasks ...")
    lesson_store_a = LessonStore()
    all_replays: list = []
    for i, task in enumerate(train_tasks, 1):
        condition = "train_agent_a"
        _log(f"  [{i}/{len(train_tasks)}] {task.task_id} family={task.family}")
        result  = run_episode(task, NonLearningManager(), condition, agent_fn)
        success = result["evaluation"]["task_success"]
        _log(f"    -> success={success}  words={result['active_context_words']}")
        _log_episode_detail(result)
        with episode_out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(serializable_result(result)) + "\n")
        if success == 0:
            _log(f"    -> running replays ...")
            replays = run_replay_for_episode(task, result, agent_fn, lesson_store_a, replay_out)
            all_replays.extend(replays)
            _log(f"    -> {len(replays)} replays  lessons={len(lesson_store_a.to_dict())}")

    _log(f"Phase 1 done — {len(all_replays)} replays  {len(lesson_store_a.to_dict())} lessons")

    # -- Phase 2: Evaluate Agent A (3 conditions on test split) --------------
    conditions_a = [
        ("full_context",         FullContextManager()),
        ("non_learning_manager", NonLearningManager()),
        ("lesson_aware_agent_a", LessonAwareManager(lesson_store_a, model_client)),
    ]
    test_results_a: list[dict] = []
    n_eval = len(test_tasks)
    _log(f"Phase 2 — test split ({n_eval} tasks × {len(conditions_a)} conditions)")
    for condition, manager in conditions_a:
        for i, task in enumerate(test_tasks, 1):
            cond_label = f"{condition}_test"
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

    # -- Phase 3: Cross-agent transfer ---------------------------------------
    transferable   = lesson_store_a.get_transferable()
    lesson_store_b = LessonStore()
    for les in transferable:
        lesson_store_b.add(les)
    _log(f"Phase 3 — {len(transferable)} transferable lessons given to Agent B")

    test_results_b: list[dict] = []
    n_eval = len(test_tasks)
    _log(f"Phase 3 — test split ({n_eval} tasks)")
    for i, task in enumerate(test_tasks, 1):
        cond_label = "lesson_aware_agent_b_test"
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

    # -- Write summary -------------------------------------------------------
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
        "mode":                  mode,
        "summary":               summarize(all_test_results),
        "dataset":               "Android LogHub (logpai/loghub) — 30 incidents, 5 per family",
        "learned_lessons_a":     lesson_store_a.to_dict(),
        "transferred_lessons_b": lesson_store_b.to_dict(),
        "num_training_replays":  len(all_replays),
        "num_harmful_replays":   sum(1 for r in all_replays if r.label == "HARMFUL"),
        "num_transferable":      len(transferable),
    }
    (PROCESSED_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _log(f"Summary written to {PROCESSED_DIR / 'summary.json'}")
    print("\n" + "=" * 65)
    print("SUMMARY.JSON CONTENTS:")
    print("=" * 65)
    print(json.dumps(summary, indent=2))
    print_analysis(summary)


if __name__ == "__main__":
    main()
