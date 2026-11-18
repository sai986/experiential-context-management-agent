"""
OpenStack multi-service log pipeline experiment.

Runs the experiential context-management pipeline on 180 incidents derived from
real OpenStack log data (loghub OpenStack_2k) + synthetic multi-service lines
for Neutron, Keystone, Cinder, and Glance.

Each incident in data/openstack/{train,test}.jsonl contains:
  - raw_text:           real Nova log window (5-7 lines) + 8 background noise lines
                        from multiple services, prefixed with a scenario constraint
  - required_outputs / forbidden_outputs / synonym_groups: evaluator ground-truth
  - family / failure_mode: family label for per-family breakdown
  - services: OpenStack service(s) represented in the window

The Investigation Agent converts raw_text -> structured Observations with tags.
  USE_LLM=0  ->  simulated_investigation_agent (heuristic, free, no API calls)
  USE_LLM=1  ->  llm_investigation_agent (Claude, requires API key)

The pipeline is identical to run_incident_pilot.py after that point.

Usage
-----
    # Simulated mode — free, no API calls:
    python -m src.experiments.run_openstack_pilot

    # LLM mode:
    USE_LLM=1 ANTHROPIC_API_KEY=sk-... python -m src.experiments.run_openstack_pilot

Output (all under results/openstack/{simulated|real}/)
-------------------------------------------------------
    raw/episode_results.jsonl    — one line per episode
    raw/all_replays.jsonl        — one line per replay
    processed/summary.json       — condition × family success rates + lessons
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path regardless of invocation method.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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

# ---------------------------------------------------------------------------
# Paths — separate result dirs for simulated (USE_LLM=0) vs real (USE_LLM=1)
# ---------------------------------------------------------------------------
ROOT          = Path(__file__).resolve().parents[2]
DATA_DIR      = ROOT / "data" / "openstack"
TRAIN_FILE    = DATA_DIR / "train.jsonl"       # 125 incidents (Phase 1)
TEST_FILE     = DATA_DIR / "test.jsonl"        #  27 incidents (Phase 2+3)
# Keep backwards-compatible alias for existence check
DATA_FILE     = DATA_DIR / "incidents.jsonl"
_mode_dir     = "real" if os.environ.get("USE_LLM", "0") == "1" else "simulated"
RAW_DIR       = ROOT / "results" / "openstack" / _mode_dir / "raw"
PROCESSED_DIR = ROOT / "results" / "openstack" / _mode_dir / "processed"

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


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Load incidents via Investigation Agent
# ---------------------------------------------------------------------------

def load_openstack_incidents(data_file: Path, model_client) -> list[IncidentTask]:
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

        # -- Investigation Agent: raw_text -> Observations ---------------------
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

        # -- Wrap in ControlledTask --------------------------------------------
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
    status  = "PASS" if ev["task_success"] else "FAIL"
    cs      = ev.get("constraint_satisfaction", 0.0)
    _log(f"    [{status}]  cs={cs:.2f}  answer: {preview}")
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
    answer = result["answer"]
    ledger = result["ledger"]

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
            fr = [r for r in rows if r["family"] == fam]
            fn = len(fr)
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


def print_analysis(summary_doc: dict) -> None:
    s       = summary_doc["summary"]
    n_total = len(summary_doc["learned_lessons_a"])
    n_trans = summary_doc["num_transferable"]

    def sr(cond: str) -> float:
        return s.get(cond, {}).get("success_rate", float("nan"))

    def wc(cond: str) -> float:
        return s.get(cond, {}).get("avg_active_context_words", float("nan"))

    sep = "=" * 68
    print()
    print(sep)
    print("  OPENSTACK MULTI-SERVICE LOG PILOT — RESULTS")
    print("  Dataset: OpenStack LogHub (loghub/OpenStack_2k) + synthetic multi-service lines")
    print(sep)

    conds = [
        ("full_context_test",         "Full context (ceiling)        "),
        ("non_learning_manager_test",  "Non-learning manager          "),
        ("lesson_aware_agent_a_test",  "Lesson-aware Agent A          "),
        ("lesson_aware_agent_b_test",  "Lesson-aware Agent B (transfer)"),
    ]
    print(f"\n  {'Condition':<36}  {'Success':>8}  {'Avg words':>10}  {'n':>4}")
    print("  " + "-" * 64)
    for cond, label in conds:
        success = sr(cond)
        avg_w   = wc(cond)
        n       = s.get(cond, {}).get("n", 0)
        success_str = f"{success:.0%}" if success == success else "  --"
        words_str   = f"{avg_w:.0f}"  if avg_w   == avg_w   else "--"
        print(f"  {label:<36}  {success_str:>8}  {words_str:>10}  {n:>4}")

    # Per-family breakdown
    families = sorted({
        fam
        for cond_data in s.values()
        for fam in cond_data.get("by_family", {})
    })
    col_labels = [
        ("full_context_test",         "ceiling "),
        ("non_learning_manager_test",  "no-learn"),
        ("lesson_aware_agent_a_test",  "agent-A "),
        ("lesson_aware_agent_b_test",  "agent-B "),
    ]
    fam_w  = 24
    cell_w = 10
    print(f"\n  Per-family success rate (test split)")
    header = f"  {'Family':<{fam_w}}" + "".join(f"  {lb:>{cell_w}}" for _, lb in col_labels)
    print(header)
    print("  " + "-" * (fam_w + (cell_w + 2) * len(col_labels)))
    for fam in families:
        row = f"  {fam:<{fam_w}}"
        for cond, _ in col_labels:
            fsr = s.get(cond, {}).get("by_family", {}).get(fam, {}).get("success_rate", float("nan"))
            cell = f"{fsr:.0%}" if fsr == fsr else "--"
            row += f"  {cell:>{cell_w}}"
        print(row)

    # RQ findings
    baseline    = sr("non_learning_manager_test")
    learned_a   = sr("lesson_aware_agent_a_test")
    learned_b   = sr("lesson_aware_agent_b_test")
    full_ctx    = sr("full_context_test")
    improvement = learned_a - baseline
    words_full  = wc("full_context_test")
    words_a     = wc("lesson_aware_agent_a_test")

    print(f"\n  Findings")
    print("  " + "-" * 44)
    rq1 = "IMPROVED" if improvement > 0 else "no improvement"
    sign = f"{improvement:+.0%}"
    print(f"  RQ1 (learning)   : Agent A {sign} vs non-learning  [{rq1}]")
    if words_full > 0:
        reduction = (words_full - words_a) / words_full * 100
        print(f"  RQ2 (efficiency) : {reduction:.0f}% fewer context words vs full-context")
    gap = full_ctx - learned_a
    print(f"                     Gap to full-context ceiling: {gap:+.0%}")
    if learned_a > 0 and learned_b >= learned_a * 0.9:
        rq3 = "EFFECTIVE transfer"
    elif learned_b > baseline:
        rq3 = "PARTIAL transfer"
    else:
        rq3 = "no transfer benefit"
    print(f"  RQ3 (transfer)   : Agent B {learned_b:.0%}  [{rq3}]"
          f"  ({n_trans}/{n_total} lessons transferred)")

    print(f"\n  Lessons learned by Agent A: {n_total}  |  Transferred to B: {n_trans}")
    print(f"  Training replays: {summary_doc['num_training_replays']}  |"
          f"  Harmful: {summary_doc['num_harmful_replays']}")
    if summary_doc["learned_lessons_a"]:
        print()
        for les in summary_doc["learned_lessons_a"]:
            tags  = ", ".join(les["trigger_tags"])
            count = les["support_count"]
            bar   = "#" * min(count, 5) + "." * (5 - min(count, 5))
            print(f"    [{bar}] {les['status']:<12}  {les['operation']:<8}  [{tags}]  {count}x")

    mode_lbl = "real" if _use_llm() else "simulated"
    print(f"\n  Full results -> results/openstack/{mode_lbl}/processed/summary.json")
    print(sep)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    mode = "LLM" if _use_llm() else "SIMULATED"
    _log(f"OpenStack multi-service pilot — mode={mode}")

    for path, label in [(TRAIN_FILE, "train.jsonl"), (TEST_FILE, "test.jsonl")]:
        if not path.exists():
            print(f"ERROR: {path} not found.")
            print("Run the incident generator first:")
            print("  python3 data/openstack/generate_incidents.py")
            return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    model_client = _build_model_client()
    agent_fn     = _build_agent_fn(model_client)

    # -- Load train and test splits via Investigation Agent -------------------
    _log("Loading train split ...")
    train_its   = load_openstack_incidents(TRAIN_FILE, model_client)
    _log("Loading test split ...")
    test_its    = load_openstack_incidents(TEST_FILE, model_client)
    train_tasks = [it.task for it in train_its]
    test_tasks  = [it.task for it in test_its]
    _log(f"Tasks: {len(train_tasks)} train / {len(test_tasks)} test")

    episode_out = RAW_DIR / "episode_results.jsonl"
    replay_out  = RAW_DIR / "all_replays.jsonl"
    # Clear output files for a fresh run
    episode_out.write_text("", encoding="utf-8")
    replay_out.write_text("", encoding="utf-8")

    eval_completed = _load_completed(episode_out)
    eval_completed = {(tid, cond) for tid, cond in eval_completed
                      if not cond.startswith("train_")}

    # -- Phase 1 — Train Agent A: learn lessons from NonLearningManager failures
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
            replays = run_replay_for_episode(
                task, result, agent_fn, lesson_store_a, replay_out
            )
            all_replays.extend(replays)
            _log(f"    -> {len(replays)} replays  lessons={len(lesson_store_a.to_dict())}")

    _log(f"Phase 1 done — {len(all_replays)} replays  {len(lesson_store_a.to_dict())} lessons")

    # -- Phase 2 — Evaluate Agent A on test tasks (3 conditions) --------------
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

    # -- Phase 3 — Cross-agent transfer: Agent B receives Agent A's lessons ---
    transferable   = lesson_store_a.get_transferable()
    lesson_store_b = LessonStore()
    for les in transferable:
        lesson_store_b.add(les)
    _log(f"Phase 3 — {len(transferable)} transferable lessons given to Agent B")

    test_results_b: list[dict] = []
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

    # -- Write summary ---------------------------------------------------------
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

    summary_doc = {
        "summary":               summarize(all_test_results),
        "dataset":               (
            f"OpenStack LogHub (loghub/OpenStack_2k) + synthetic multi-service lines — "
            f"180 incidents (30 per family × 6 families), 13-15 obs each; "
            f"train={len(train_tasks)} test={len(test_tasks)}"
        ),
        "learned_lessons_a":     lesson_store_a.to_dict(),
        "transferred_lessons_b": lesson_store_b.to_dict(),
        "num_training_replays":  len(all_replays),
        "num_harmful_replays":   sum(1 for r in all_replays if r.label == "HARMFUL"),
        "num_transferable":      len(transferable),
    }
    (PROCESSED_DIR / "summary.json").write_text(
        json.dumps(summary_doc, indent=2), encoding="utf-8"
    )
    _log(f"Summary written to results/openstack/{_mode_dir}/processed/summary.json")
    print(json.dumps(summary_doc, indent=2))
    print_analysis(summary_doc)


if __name__ == "__main__":
    main()
