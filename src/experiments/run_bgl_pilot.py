"""
BGL server-log pilot experiment.

Runs the experiential context-management pipeline on 30 incidents derived
from the BGL (Blue Gene/L) server log dataset (logpai/loghub) instead of
synthetically generated tasks.

Each incident in data/bgl/incidents.jsonl contains:
  - raw_text: a real BGL log window wrapped with a scenario constraint that
              creates one of the 6 failure families
  - required_outputs / forbidden_outputs: evaluator ground-truth
  - family / failure_mode: family label for per-family breakdown

The Investigation Agent converts raw_text -> structured Observations with tags.
  USE_LLM=0  ->  simulated_investigation_agent (heuristic, free, no API calls)
  USE_LLM=1  ->  llm_investigation_agent (Claude, requires API key)

After that the pipeline is identical to run_incident_pilot.py.

Usage:
    # Simulated mode (free, no API calls):
    USE_LLM=0 python3 -m src.experiments.run_bgl_pilot

    # LLM mode:
    USE_LLM=1 ANTHROPIC_API_KEY=sk-... python3 -m src.experiments.run_bgl_pilot

Output (separate directories per mode):
    results/bgl/simulated/raw/episode_results.jsonl     (USE_LLM=0)
    results/bgl/simulated/raw/all_replays.jsonl
    results/bgl/simulated/raw/lesson_store_a.json
    results/bgl/simulated/processed/summary.json
    results/bgl/real/raw/episode_results.jsonl          (USE_LLM=1)
    results/bgl/real/raw/all_replays.jsonl
    results/bgl/real/raw/lesson_store_a.json
    results/bgl/real/processed/summary.json
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
# sys.path setup: allow running as `python3 -m src.experiments.run_bgl_pilot`
# from the project root or directly as a script.
# ---------------------------------------------------------------------------
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

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
from src.constants import TAG_TRIGGER
from src.environment.evaluator import evaluate_answer
from src.environment.task_generator import ControlledTask, Observation
from src.environment.task_generator_incident import IncidentTask
from src.lessons.lesson_store import LessonStore

ROOT       = Path(__file__).resolve().parents[2]
DATA_DIR   = ROOT / "data" / "bgl"
TRAIN_FILE = DATA_DIR / "train.jsonl"
TEST_FILE  = DATA_DIR / "test.jsonl"
# Keep for backward compat (used in docstring / fallback check)
DATA_FILE  = DATA_DIR / "incidents.jsonl"

# Output directories: separate subtree per mode so simulated and LLM runs
# never overwrite each other and both summaries can be compared side by side.
_MODE_SUBDIR  = "real" if os.environ.get("USE_LLM", "0") == "1" else "simulated"
RAW_DIR       = ROOT / "results" / "bgl" / _MODE_SUBDIR / "raw"
PROCESSED_DIR = ROOT / "results" / "bgl" / _MODE_SUBDIR / "processed"
LOG_DIR       = ROOT / "results" / "bgl" / _MODE_SUBDIR / "logs"

MAX_SUSPECTS_PER_EPISODE = 2
LOG_TO_CONSOLE = False


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


def _log_episode_detail(result: dict) -> None:
    ev      = result["evaluation"]
    status  = "PASS" if ev["task_success"] else "FAIL"
    cs      = ev.get("constraint_satisfaction", 0.0)
    missing = ev.get("missing_required_terms", [])
    extra   = f"  missing={missing}" if missing else ""
    _log(f"       [{status}]  cs={cs:.2f}{extra}")


# ---------------------------------------------------------------------------
# Context logger
# ---------------------------------------------------------------------------

class _ContextLogger:
    """Writes a human-readable episode record to a log file."""

    _W = 72

    _PHASE_LABELS: dict[str, str] = {
        "train":         "Phase 1 — TRAIN  (Agent A learns lessons from failures)",
        "eval_test":     "Phase 2 — EVAL TEST  (measure conditions on test split)",
        "transfer_test": "Phase 3 — TRANSFER TEST  (Agent B uses Agent A's lessons)",
    }

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file   = path.open("w", encoding="utf-8")
        self._episode_num  = 0
        self._current_phase: str = ""

    def _write_header(self, mode: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        w  = self._W
        block = "\n".join([
            "=" * w,
            f"  BGL PILOT — EXPERIENTIAL CONTEXT-MANAGEMENT AGENT  |  mode={mode}  |  {ts}",
            f"  Dataset: BGL (Blue Gene/L) server logs — logpai/loghub",
            "=" * w, "",
        ])
        self._file.write(block)
        self._file.flush()
        if LOG_TO_CONSOLE:
            print(block, flush=True)

    def log(self, result: dict, phase: str) -> None:
        if phase != self._current_phase:
            self._current_phase = phase
            label  = self._PHASE_LABELS.get(phase, phase)
            banner = "\n".join(["", "~" * self._W, f"  >>> {label}", "~" * self._W, ""])
            self._file.write(banner)
            self._file.flush()
            if LOG_TO_CONSOLE:
                print(banner, flush=True)

        self._episode_num += 1
        ev      = result["evaluation"]
        success = ev["task_success"]
        cs      = ev.get("constraint_satisfaction", "?")
        words   = result.get("active_context_words", "?")
        ts      = time.strftime("%H:%M:%S")
        verdict = "PASS" if success else "FAIL"
        div     = "=" * self._W
        sdiv    = "-" * self._W

        lines = [
            div,
            f"  #{self._episode_num:04d}  {verdict}  cs={cs:.3f}  words={words}  [{ts}]",
            f"  phase={phase}  task={result['task_id']}  "
            f"family={result['family']}  cond={result['condition']}",
            sdiv,
            f"  Q: {result.get('question', '(not recorded)')}",
            f"  A: {result['answer'][:200].replace(chr(10), ' | ')}",
            sdiv,
        ]
        missing  = ev.get("missing_required_terms", [])
        forb     = ev.get("forbidden_terms_found", [])
        if missing:
            lines.append(f"  MISSING: {missing}")
        if forb:
            lines.append(f"  FORBIDDEN: {forb}")
        if not missing and not forb:
            lines.append("  All constraints satisfied.")
        lines += ["", ""]
        block = "\n".join(lines)
        self._file.write(block)
        self._file.flush()
        if LOG_TO_CONSOLE:
            print(block, flush=True)

    def log_lesson_store(self, lesson_store, event_desc: str) -> None:
        sdiv    = "-" * self._W
        lessons = lesson_store.to_dict()
        lines   = ["", f"  LESSONS  {event_desc}  (total={len(lessons)})", sdiv]
        if not lessons:
            lines.append("  (none)")
        else:
            for les in lessons:
                tags  = ", ".join(les["trigger_tags"])
                count = les["support_count"]
                filled = min(count, 5)
                bar    = "#" * filled + "." * (5 - filled)
                lines.append(
                    f"  {les['lesson_id']}  {les['operation']:<8}  "
                    f"{les['status']:<12}  [{bar}] {count}x  [{tags}]"
                )
        lines += ["", ""]
        block = "\n".join(lines)
        self._file.write(block)
        self._file.flush()
        if LOG_TO_CONSOLE:
            print(block, flush=True)

    def close(self) -> None:
        self._file.close()


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_completed(episode_out: Path) -> set[tuple[str, str]]:
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


# ---------------------------------------------------------------------------
# Load BGL incidents via Investigation Agent
# ---------------------------------------------------------------------------

def _load_investigation_cache(cache_path: Path) -> dict:
    """Load investigation cache: maps task_id -> list of observation dicts."""
    if cache_path.exists() and cache_path.stat().st_size > 2:
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_investigation_cache(cache_path: Path, cache: dict) -> None:
    """Persist investigation cache to disk atomically."""
    tmp = cache_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, separators=(",", ":")), encoding="utf-8")
    tmp.replace(cache_path)


def load_bgl_incidents(
    data_file: Path,
    model_client,
    cache: dict | None = None,
) -> list[IncidentTask]:
    """Read a BGL JSONL split file; run investigation agent on each raw_text.

    cache — optional dict mapping task_id -> list of observation dicts.  When
    provided, cached results are used instead of calling the LLM and newly
    computed observations are added to the dict so the caller can persist it.
    """
    incidents: list[IncidentTask] = []
    use_llm = _use_llm()

    raw_lines = data_file.read_text(encoding="utf-8").splitlines()
    n_cached  = sum(1 for l in raw_lines if l.strip() and json.loads(l.strip()).get("incident_id", "") in (cache or {}))
    _log(f"Loading {len(raw_lines)} incidents from {data_file.name} "
         f"({n_cached} from cache, {len(raw_lines)-n_cached} via agent) ...")

    for idx, line in enumerate(raw_lines, 1):
        line = line.strip()
        if not line:
            continue
        inc     = json.loads(line)
        task_id = inc["incident_id"]

        # ── Investigation Agent: raw_text -> Observations ──────────────────
        if cache is not None and task_id in cache:
            obs_dicts = cache[task_id]
            observations = [
                Observation(
                    step=od["step"],
                    content=od["content"],
                    tags=set(od["tags"]),
                )
                for od in obs_dicts
            ]
            _log(f"  [{idx:03d}] {task_id}  family={inc['family']}"
                 f"  obs={len(observations)}  [cache]")
        else:
            raw_text = inc["raw_text"]
            if use_llm and model_client is not None:
                observations = llm_investigation_agent(raw_text, model_client)
            else:
                observations = simulated_investigation_agent(raw_text)
            _log(
                f"  [{idx:03d}] {task_id}  family={inc['family']}"
                f"  obs={len(observations)}"
                f"  tags={set(t for o in observations for t in o.tags)}"
            )
            if cache is not None:
                cache[task_id] = [
                    {"step": o.step, "content": o.content, "tags": list(o.tags)}
                    for o in observations
                ]

        # ── Wrap in ControlledTask ─────────────────────────────────────────
        task = ControlledTask(
            task_id=task_id,
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
    model_client=None,
    raw_incident_text: str = "",
) -> dict:
    """Run one episode through the full 3-agent pipeline.

    USE_LLM=0 (simulated):
      task.observations already built by load_bgl_incidents(); used directly.

    USE_LLM=1 (LLM):
      llm_investigation_agent re-parses raw_incident_text into observations.
    """
    if model_client is not None and raw_incident_text:
        # Full 3-agent pipeline: LLM re-parses.
        observations = llm_investigation_agent(raw_incident_text, model_client)
        obs_source   = "investigation_agent"
        # Re-number steps sequentially.
        observations = [
            Observation(step=i + 1, content=obs.content, tags=obs.tags)
            for i, obs in enumerate(observations)
        ]
        # Replace any LLM-generated trigger with the verbatim question.
        non_trigger  = [o for o in observations if TAG_TRIGGER not in o.tags]
        trigger_step = len(non_trigger) + 1
        observations = non_trigger + [
            Observation(step=trigger_step, content=task.question, tags=[TAG_TRIGGER])
        ]
    else:
        observations = task.observations
        obs_source   = "task_environment"

    ledger = ContextLedger(episode_id=f"{condition}_{task.task_id}")
    for obs in observations:
        ledger.add_raw(obs.step, obs.content, obs.tags, source=obs_source)
        manager.manage_after_observation(ledger, obs.step)

    managed_context = ledger.active_context()
    answer          = agent_fn(task, managed_context)
    evaluation      = evaluate_answer(task, answer)
    return {
        "task_id":              task.task_id,
        "condition":            condition,
        "family":               task.family,
        "question":             task.question,
        "obs_source":           obs_source,
        "active_context":       managed_context,
        "answer":               answer,
        "evaluation":           asdict(evaluation),
        "active_context_words": ledger.word_estimate(),
        "ledger":               ledger,
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
        by_fam   = {}
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


def serializable_result(result: dict) -> dict:
    exclude = {"ledger", "active_context"}
    out = {k: v for k, v in result.items() if k not in exclude}
    out["ledger"] = result["ledger"].to_dict()
    return out


def print_analysis(summary: dict) -> None:
    s       = summary["summary"]
    n_total = len(summary["learned_lessons_a"])
    n_trans = summary["num_transferable"]

    def sr(cond: str) -> float:
        return s.get(cond, {}).get("success_rate", float("nan"))

    def wc(cond: str) -> float:
        return s.get(cond, {}).get("avg_active_context_words", float("nan"))

    W   = 68
    div = "=" * W

    print()
    print(div)
    print("  BGL PILOT — EXPERIMENT RESULTS")
    print("  Dataset: BGL (Blue Gene/L) server logs — logpai/loghub")
    print(div)

    conds = [
        ("full_context_test",         "Full context (ceiling)"),
        ("non_learning_manager_test",  "Non-learning manager"),
        ("lesson_aware_agent_a_test",  "Lesson-aware Agent A"),
        ("lesson_aware_agent_b_test",  "Lesson-aware Agent B (transfer)"),
    ]
    col_w = 32
    print(f"\n  {'Condition':<{col_w}}  {'Success':>8}  {'Avg words':>10}  {'n':>4}")
    print("  " + "-" * (col_w + 28))
    for cond, label in conds:
        success     = sr(cond)
        avg_w       = wc(cond)
        n           = s.get(cond, {}).get("n", 0)
        success_str = f"{success:.0%}" if success == success else "  --"
        words_str   = f"{avg_w:.0f}"  if avg_w   == avg_w   else "--"
        print(f"  {label:<{col_w}}  {success_str:>8}  {words_str:>10}  {n:>4}")

    # Per-family breakdown
    families = sorted({
        fam
        for cond_data in s.values()
        for fam in cond_data.get("by_family", {})
    })
    col_labels = [
        ("full_context_test",         "ceiling"),
        ("non_learning_manager_test",  "no-learn"),
        ("lesson_aware_agent_a_test",  "agent-A"),
        ("lesson_aware_agent_b_test",  "agent-B"),
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
            fsr  = s.get(cond, {}).get("by_family", {}).get(fam, {}).get("success_rate", float("nan"))
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
    print("  " + "-" * 40)

    rq1 = "IMPROVED" if improvement > 0 else "no improvement"
    sign = f"{improvement:+.0%}"
    print(f"  RQ1 (learning)   : Agent A {sign} vs non-learning  [{rq1}]")

    if words_full and words_full > 0:
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
    print(f"  Training replays: {summary['num_training_replays']}  |"
          f"  Harmful: {summary['num_harmful_replays']}")
    if summary["learned_lessons_a"]:
        print()
        for les in summary["learned_lessons_a"]:
            tags  = ", ".join(les["trigger_tags"])
            count = les["support_count"]
            bar   = "#" * min(count, 5) + "." * (5 - min(count, 5))
            print(f"    [{bar}] {les['status']:<12}  {les['operation']:<8}  [{tags}]  {count}x")

    print(f"\n  Full results -> results/bgl/{_MODE_SUBDIR}/processed/summary.json")
    print(div)
    print()


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    mode = "LLM" if _use_llm() else "SIMULATED"
    _log(f"BGL pilot — mode={mode}")

    # Check that split files exist; fall back to generating from incidents.jsonl.
    if not TRAIN_FILE.exists() or not TEST_FILE.exists():
        if DATA_FILE.exists():
            print(f"WARNING: split files not found — falling back to incidents.jsonl "
                  f"with 70/30 split. Run data/bgl/generate_incidents.py for proper splits.")
        else:
            print(f"ERROR: Neither {TRAIN_FILE} nor {DATA_FILE} found.")
            print("Generate them first:")
            print("  python3 data/bgl/generate_incidents.py")
            return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    model_client = _build_model_client()
    agent_fn     = _build_agent_fn(model_client)

    # ── Early Phase-1 skip check (needed before loading decisions) ──────────
    _early_lesson_store_path = RAW_DIR / "lesson_store_a.json"
    _phase1_done = (
        _early_lesson_store_path.exists()
        and _early_lesson_store_path.stat().st_size > 2
    )

    # ── Investigation cache (avoids re-calling the LLM on resume) ───────────
    _inv_cache_path = RAW_DIR / "investigation_cache.json"
    _inv_cache      = _load_investigation_cache(_inv_cache_path)
    if _inv_cache:
        _log(f"Investigation cache: {len(_inv_cache)} entries loaded.")

    # ── Load incidents via Investigation Agent ──────────────────────────────
    if TRAIN_FILE.exists() and TEST_FILE.exists():
        # Use pre-made splits (data/bgl/train.jsonl, data/bgl/test.jsonl).
        if _phase1_done:
            # Phase 1 already completed — skip loading train incidents entirely.
            _log(f"Phase 1 already done — skipping train incident loading.")
            train_its = []
        else:
            _log(f"Loading train incidents from {TRAIN_FILE.name} ...")
            train_its = load_bgl_incidents(TRAIN_FILE, model_client, _inv_cache)
            _save_investigation_cache(_inv_cache_path, _inv_cache)
        _log(f"Loading test incidents from {TEST_FILE.name} ...")
        test_its = load_bgl_incidents(TEST_FILE, model_client, _inv_cache)
        _save_investigation_cache(_inv_cache_path, _inv_cache)
    else:
        # Fallback: load all from incidents.jsonl and split 70/30.
        all_incident_tasks = load_bgl_incidents(DATA_FILE, model_client, _inv_cache)
        _save_investigation_cache(_inv_cache_path, _inv_cache)
        split_a   = int(len(all_incident_tasks) * 0.70)
        train_its = all_incident_tasks[:split_a]
        test_its  = all_incident_tasks[split_a:]

    train_tasks = [it.task for it in train_its]
    test_tasks  = [it.task for it in test_its]
    _log(f"Tasks: {len(train_tasks)} train / {len(test_tasks)} test")

    ctx_log = _ContextLogger(LOG_DIR / "context_log.txt")
    ctx_log._write_header(mode=mode)
    _log(f"Context log: {LOG_DIR / 'context_log.txt'}")

    episode_out       = RAW_DIR / "episode_results.jsonl"
    replay_out        = RAW_DIR / "all_replays.jsonl"
    lesson_store_path = RAW_DIR / "lesson_store_a.json"

    eval_completed = _load_completed(episode_out)
    eval_completed = {
        (tid, cond) for tid, cond in eval_completed
        if not cond.startswith("train_")
    }
    if eval_completed:
        _log(f"Resuming — {len(eval_completed)} eval episodes already done.")

    # ------------------------------------------------------------------
    # Phase 1 — Train Agent A
    # ------------------------------------------------------------------
    # Guard against an empty lesson_store_a.json left by an interrupted run
    # (lesson_store_path.exists() but content is "").
    _lesson_store_valid = (
        lesson_store_path.exists()
        and lesson_store_path.stat().st_size > 2
    )
    if _lesson_store_valid:
        lesson_store_a = LessonStore.load(lesson_store_path)
        n_lessons = len(lesson_store_a.to_dict())
        _log(f"Phase 1 — skipped (loaded {n_lessons} lessons from {lesson_store_path})")
        all_replays: list = []
    else:
        episode_out.write_text("", encoding="utf-8")
        replay_out.write_text("", encoding="utf-8")
        _log(f"Phase 1 — training on {len(train_tasks)} tasks ...")
        lesson_store_a = LessonStore()
        all_replays: list = []

        for i, (task, inc) in enumerate(zip(train_tasks, train_its), 1):
            condition = "train_agent_a"
            _log(f"  [{i}/{len(train_tasks)}] {task.task_id}  family={task.family}")
            result  = run_episode(
                task, NonLearningManager(), condition, agent_fn,
                model_client=model_client,
            )
            success = result["evaluation"]["task_success"]
            _log(f"    -> success={success}  words={result['active_context_words']}"
                 f"  obs_source={result['obs_source']}")
            _log_episode_detail(result)
            ctx_log.log(result, phase="train")
            with episode_out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(serializable_result(result)) + "\n")
            if success == 0:
                _log(f"    -> running replays ...")
                replays = run_replay_for_episode(
                    task, result, agent_fn, lesson_store_a, replay_out
                )
                all_replays.extend(replays)
                n_lessons = len(lesson_store_a.to_dict())
                _log(f"    -> {len(replays)} replays done  lessons={n_lessons}")
                ctx_log.log_lesson_store(
                    lesson_store_a,
                    f"after task {i}/{len(train_tasks)} ({task.task_id})",
                )

        lesson_store_a.save(lesson_store_path)
        _log(f"Phase 1 done — {len(all_replays)} replays, "
             f"{len(lesson_store_a.to_dict())} lessons learned")
        _log(f"Lesson store saved -> {lesson_store_path}")
        ctx_log.log_lesson_store(lesson_store_a, "END OF PHASE 1")

    # ------------------------------------------------------------------
    # Phase 2 — Evaluate Agent A on test tasks (3 conditions)
    # ------------------------------------------------------------------
    conditions_a = [
        ("full_context",         FullContextManager()),
        ("non_learning_manager", NonLearningManager()),
        ("lesson_aware_agent_a", LessonAwareManager(lesson_store_a, model_client)),
    ]
    test_results_a: list[dict] = []
    n_eval = len(test_tasks)
    _log(f"Phase 2 — test split ({n_eval} tasks × {len(conditions_a)} conditions)")
    for condition, manager in conditions_a:
        for i, (task, inc) in enumerate(zip(test_tasks, test_its), 1):
            cond_label = f"{condition}_test"
            if (task.task_id, cond_label) in eval_completed:
                continue
            _log(f"  [{i}/{n_eval}] {cond_label}  {task.task_id}")
            result = run_episode(
                task, manager, cond_label, agent_fn,
                model_client=model_client,
            )
            _log(f"    -> success={result['evaluation']['task_success']}"
                 f"  words={result['active_context_words']}")
            _log_episode_detail(result)
            ctx_log.log(result, phase="eval_test")
            test_results_a.append(result)
            with episode_out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(serializable_result(result)) + "\n")

    _log("Phase 2 done")

    # ------------------------------------------------------------------
    # Phase 3 — Cross-agent transfer: Agent B receives Agent A's lessons
    # ------------------------------------------------------------------
    transferable   = lesson_store_a.get_transferable()
    lesson_store_b = LessonStore()
    lesson_store_b.import_transferable(lesson_store_a)
    _log(f"Phase 3 — {len(transferable)} transferable lessons given to Agent B")
    ctx_log.log_lesson_store(
        lesson_store_b,
        f"PHASE 3 START — {len(transferable)} transferable lessons"
        + (" (none reached threshold >=5)" if len(transferable) == 0 else ""),
    )

    test_results_b: list[dict] = []
    n_eval = len(test_tasks)
    _log(f"Phase 3 — test split ({n_eval} tasks)")
    for i, (task, inc) in enumerate(zip(test_tasks, test_its), 1):
        cond_label = "lesson_aware_agent_b_test"
        if (task.task_id, cond_label) in eval_completed:
            continue
        _log(f"  [{i}/{n_eval}] {cond_label}  {task.task_id}")
        result = run_episode(
            task,
            LessonAwareManager(lesson_store_b, model_client),
            cond_label, agent_fn,
            model_client=model_client,
        )
        _log(f"    -> success={result['evaluation']['task_success']}"
             f"  words={result['active_context_words']}")
        _log_episode_detail(result)
        ctx_log.log(result, phase="transfer_test")
        test_results_b.append(result)
        with episode_out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(serializable_result(result)) + "\n")

    _log("Phase 3 done")

    # ------------------------------------------------------------------
    # Write final summary
    # ------------------------------------------------------------------
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
        "dataset":               "BGL (Blue Gene/L) server logs — logpai/loghub — 180 incidents, 30 per family (train=125, test=27)",
        "learned_lessons_a":     lesson_store_a.to_dict(),
        "transferred_lessons_b": lesson_store_b.to_dict(),
        "num_training_replays":  len(all_replays),
        "num_harmful_replays":   sum(1 for r in all_replays if r.label == "HARMFUL"),
        "num_transferable":      len(transferable),
    }
    (PROCESSED_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    ctx_log.close()
    _log(f"Summary written to results/bgl/{_MODE_SUBDIR}/processed/summary.json")
    print(json.dumps(summary, indent=2))
    print_analysis(summary)


if __name__ == "__main__":
    main()
