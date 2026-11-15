"""
Full incident pilot experiment.

Runs the experiential context-management pipeline on the 6-family incident
task set and measures whether learned lessons improve performance on held-out
tasks (RQ1 / RQ2) and transfer to Agent B (RQ3).

Usage:
    # Test mode --no API calls, free to run:
    python -m src.experiments.run_incident_pilot

    # Real LLM mode:
    USE_LLM=1 ANTHROPIC_API_KEY=sk-... python -m src.experiments.run_incident_pilot

Output files (all under results/incident/):
    raw/episode_results.jsonl   --one line per episode, written as each completes
    raw/all_replays.jsonl       --one line per replay result
    processed/summary.json      --condition x family success rates + lessons

Crash recovery:
    Re-running resumes from the last completed episode --already-written lines
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

from src.agents.investigation_agent import llm_investigation_agent, simulated_investigation_agent
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
from src.environment.task_generator_incident import (
    generate_incident_tasks,
    materialize_incident_splits,
)
from src.lessons.lesson_store import LessonStore

ROOT              = Path(__file__).resolve().parents[2]
DATA_DIR          = ROOT / "data" / "incident"
RAW_DIR           = ROOT / "results" / "incident" / "raw"
PROCESSED_DIR     = ROOT / "results" / "incident" / "processed"
LOG_DIR           = ROOT / "results" / "incident" / "logs"

# How many suspects to replay per failed episode.
# 1 keeps the pilot fast; increase for a thorough training run.
MAX_SUSPECTS_PER_EPISODE = 2

# Set to True to also echo the full episode log to the console.
# False keeps the console clean (just progress lines); the log file always
# gets the full detail regardless of this setting.
LOG_TO_CONSOLE = False


# ---------------------------------------------------------------------------
# Progress logging
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    """Print a timestamped progress line, flushed immediately."""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _log_episode_detail(result: dict) -> None:
    """Print a compact one-liner verdict to the console progress stream."""
    ev      = result["evaluation"]
    status  = "PASS" if ev["task_success"] else "FAIL"
    cs      = ev.get("constraint_satisfaction", 0.0)
    missing = ev.get("missing_required_terms", [])
    extra   = f"  missing={missing}" if missing else ""
    _log(f"       [{status}]  cs={cs:.2f}{extra}")


# ---------------------------------------------------------------------------
# Context logger --writes full context + question to a log file per run
# ---------------------------------------------------------------------------

class _ContextLogger:
    """Writes a full human-readable episode record to a log file.

    Each episode block contains:
      - metadata (phase, task id, family, condition, timestamp)
      - ALL raw observations with their tags and final status (kept/compressed/archived)
      - every manager decision with before->after word counts
      - the managed context the agent actually saw
      - the question asked
      - the full agent answer
      - evaluation verdict with constraint satisfaction score
      - required terms (hit / missed), forbidden terms (found or not)

    This is enough to understand exactly why each episode succeeded or failed
    without opening any other file.

    Set LOG_TO_CONSOLE = True at the top of this file to also print to stdout.
    """

    _W = 72  # block width

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("w", encoding="utf-8")
        self._episode_num = 0
        self._current_phase: str = ""

    # Map internal phase codes to human-readable labels shown in every block.
    _PHASE_LABELS: dict[str, str] = {
        "train":         "Phase 1 --TRAIN  (Agent A learns lessons from failures)",
        "eval_test":     "Phase 2 --EVAL TEST  (measure 4 conditions on held-out test split)",
        "transfer_test": "Phase 3 --TRANSFER TEST  (Agent B uses Agent A's lessons, test split)",
    }

    def _write_header(self, mode: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        w  = self._W
        lines = [
            "=" * w,
            f"  EXPERIENTIAL CONTEXT-MANAGEMENT AGENT  |  mode={mode}  |  {ts}",
            f"  Detailed episode log. Each block: observations -> decisions -> answer -> verdict.",
            "=" * w,
            "",
        ]
        block = "\n".join(lines)
        self._file.write(block)
        self._file.flush()
        if LOG_TO_CONSOLE:
            print(block, flush=True)

    def log(self, result: dict, phase: str) -> None:
        # Emit a phase banner the first time we enter each new phase.
        if phase != self._current_phase:
            self._current_phase = phase
            label = self._PHASE_LABELS.get(phase, phase)
            banner_lines = [
                "",
                "~" * self._W,
                f"  >>> {label}",
                "~" * self._W,
                "",
            ]
            banner = "\n".join(banner_lines)
            self._file.write(banner)
            self._file.flush()
            if LOG_TO_CONSOLE:
                print(banner, flush=True)

        self._episode_num += 1
        ev        = result["evaluation"]
        task_id   = result["task_id"]
        condition = result["condition"]
        family    = result["family"]
        context   = result.get("active_context", "")
        question  = result.get("question", "")
        answer    = result["answer"]
        success   = ev["task_success"]
        cs        = ev.get("constraint_satisfaction", "?")
        missing   = ev.get("missing_required_terms", [])
        forbidden_found = ev.get("forbidden_terms_found", [])
        words     = result.get("active_context_words", "?")
        ledger    = result.get("ledger")
        ts        = time.strftime("%H:%M:%S")

        obs_source  = result.get("obs_source", "task_environment")
        phase_label = self._PHASE_LABELS.get(phase, phase)
        div   = "=" * self._W
        sdiv  = "-" * self._W
        verdict = "PASS" if success else "FAIL"
        lines = [
            div,
            f"  #{self._episode_num:04d}  {verdict}  cs={cs:.3f}  words={words}  [{ts}]",
            f"  phase={phase_label}",
            f"  task={task_id}  family={family}  condition={condition}  src={obs_source}",
            sdiv,
        ]

        # --- Observations with fate ---
        lines.append("  OBSERVATIONS")
        if ledger is not None:
            raw_items = sorted(
                [item for item in ledger.items.values()
                 if item.source in ("task_environment", "investigation_agent")],
                key=lambda it: it.step,
            )
            for item in raw_items:
                if item.active:
                    fate = "KEPT    "
                elif item.archived:
                    fate = "ARCHIVED"
                else:
                    fate = "REPLACED"
                tags_str = ", ".join(item.tags) if item.tags else "-"
                lines.append(f"  {fate}  [{tags_str}]  {item.content}")
        else:
            lines.append("  (ledger not available)")

        # --- Decisions ---
        lines += [sdiv, "  DECISIONS"]
        if ledger is not None and ledger.decisions:
            for dec in ledger.decisions:
                words_b = dec.active_tokens_before
                words_a = dec.active_tokens_after
                lines.append(f"  {dec.operation:<8}  step={dec.step}  words:{words_b}->{words_a}  {dec.rationale}")
                if dec.operation == "COMPRESS" and dec.output_content:
                    lines.append(f"           -> {dec.output_content}")
        else:
            lines.append("  (no decisions)")

        # --- Question and answer ---
        lines += [sdiv, f"  Q: {question}" if question else "  Q: (not recorded)"]
        lines.append("  A: " + answer.replace("\n", " | "))

        # --- Verdict ---
        lines += [sdiv]
        if missing:
            lines.append(f"  MISSING : {missing}")
        if forbidden_found:
            lines.append(f"  FORBIDDEN: {forbidden_found}")
        if not missing and not forbidden_found:
            lines.append("  All constraints satisfied.")
        lines += ["", ""]

        block = "\n".join(lines)
        self._file.write(block)
        self._file.flush()

        if LOG_TO_CONSOLE:
            print(block, flush=True)

    def log_replays(self, replays: list, task_id: str, family: str) -> None:
        if not replays:
            return
        sdiv = "-" * self._W
        lines = ["", f"  REPLAYS  task={task_id}  family={family}", sdiv]
        for i, r in enumerate(replays, 1):
            verdict = "HARMFUL" if r.label == "HARMFUL" else "neutral"
            lines.append(
                f"  [{i}] {verdict:<7}  {r.intervention_type}  {r.target_decision_id}"
                f"  score:{r.original_score:.2f}->{r.counterfactual_score:.2f}"
            )
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
                tags_str = ", ".join(les["trigger_tags"])
                count    = les["support_count"]
                filled   = min(count, 5)
                bar      = "#" * filled + "." * (5 - filled)
                lines.append(
                    f"  {les['lesson_id']}  {les['operation']:<8}  "
                    f"{les['status']:<12}  [{bar}] {count}x  [{tags_str}]"
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
# Checkpoint helpers --skip episodes already written to episode_results.jsonl
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
    model_client=None,
    raw_incident_text: str = "",
) -> dict:
    """Run one episode through the full 3-agent pipeline.

    Agent 1 (Investigation Agent):
      - USE_LLM=1 + raw_incident_text provided -> llm_investigation_agent parses
        the raw text into structured observations with tags (real 3-agent pipeline).
      - Otherwise -> use pre-built task.observations (controlled baseline / USE_LLM=0).

    Agent 2 (Context Manager):
      Receives observations one at a time and decides KEEP / COMPRESS / ARCHIVE.

    Agent 3 (Task Agent):
      Receives the managed context string and answers the question.
    """
    # --- Agent 1: generate observations ---
    if model_client is not None and raw_incident_text:
        # Full 3-agent pipeline: LLM parses raw incident text into tagged observations.
        observations = llm_investigation_agent(raw_incident_text, model_client)
        obs_source = "investigation_agent"

        # Gap 75 fix: re-number steps 1..N after LLM parsing.
        # The LLM sometimes returns non-sequential step numbers (e.g. 1, 1, 3, 5).
        # Staleness thresholds in manager.py compare item.step <= current_step - 3,
        # so a gap in step numbers would cause items to age out prematurely.
        observations = [
            Observation(step=i + 1, content=obs.content, tags=obs.tags)
            for i, obs in enumerate(observations)
        ]

        # Gap 76 fix: replace any trigger-tagged observation with task.question.
        # The LLM may generate a trigger sentence that paraphrases the question
        # rather than matching it exactly, causing substring evaluation to fail.
        # Discard any LLM-generated trigger and append the verbatim question.
        non_trigger = [o for o in observations if TAG_TRIGGER not in o.tags]
        trigger_step = len(non_trigger) + 1
        observations = non_trigger + [Observation(step=trigger_step, content=task.question, tags=[TAG_TRIGGER])]
    else:
        # Controlled baseline: use pre-built observations from the task generator.
        observations = task.observations
        obs_source = "task_environment"

    # --- Agent 2: build managed context via ledger + manager ---
    ledger = ContextLedger(episode_id=f"{condition}_{task.task_id}")
    for observation in observations:
        # Pass source so the ledger can record which agent generated each item.
        ledger.add_raw(observation.step, observation.content, observation.tags,
                       source=obs_source)
        manager.manage_after_observation(ledger, observation.step)

    # --- Agent 3: answer from managed context ---
    managed_context = ledger.active_context()
    answer = agent_fn(task, managed_context)
    evaluation = evaluate_answer(task, answer)
    return {
        "task_id":               task.task_id,
        "condition":             condition,
        "family":                task.family,
        "question":              task.question,
        "obs_source":            obs_source,
        "active_context":        managed_context,
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
    # Exclude the live ledger object and active_context string --both are
    # large, redundant (ledger.to_dict() captures the same data), and would
    # bloat the JSONL file unnecessarily.
    # obs_source is kept: tells readers whether Agent 1 (LLM) or pre-built
    # task observations generated the context for this episode.
    exclude = {"ledger", "active_context"}
    out = {k: v for k, v in result.items() if k not in exclude}
    out["ledger"] = result["ledger"].to_dict()
    return out


def print_analysis(summary: dict) -> None:
    s        = summary["summary"]
    n_total  = len(summary["learned_lessons_a"])
    n_trans  = summary["num_transferable"]

    def sr(cond: str) -> float:
        return s.get(cond, {}).get("success_rate", float("nan"))

    def wc(cond: str) -> float:
        return s.get(cond, {}).get("avg_active_context_words", float("nan"))

    W   = 68
    div = "=" * W

    # -------------------------------------------------------------------------
    print()
    print(div)
    print("  EXPERIMENT RESULTS")
    print(div)

    # ---- Overall condition table -------------------------------------------
    conds = [
        ("full_context_test",        "Full context (ceiling)"),
        ("non_learning_manager_test","Non-learning manager"),
        ("lesson_aware_agent_a_test","Lesson-aware Agent A"),
        ("lesson_aware_agent_b_test","Lesson-aware Agent B (transfer)"),
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

    # ---- Per-family breakdown -----------------------------------------------
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
    fam_w  = 22
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

    # ---- RQ findings --------------------------------------------------------
    baseline    = sr("non_learning_manager_test")
    learned_a   = sr("lesson_aware_agent_a_test")
    learned_b   = sr("lesson_aware_agent_b_test")
    full_ctx    = sr("full_context_test")
    improvement = learned_a - baseline
    words_full  = wc("full_context_test")
    words_a     = wc("lesson_aware_agent_a_test")

    print(f"\n  Findings")
    print("  " + "-" * 40)

    # RQ1
    rq1 = "IMPROVED" if improvement > 0 else "no improvement"
    sign = f"{improvement:+.0%}"
    print(f"  RQ1 (learning)   : Agent A {sign} vs non-learning  [{rq1}]")

    # RQ2
    if words_full > 0:
        reduction = (words_full - words_a) / words_full * 100
        print(f"  RQ2 (efficiency) : {reduction:.0f}% fewer context words vs full-context")
    gap = full_ctx - learned_a
    print(f"                     Gap to full-context ceiling: {gap:+.0%}")

    # RQ3
    if learned_a > 0 and learned_b >= learned_a * 0.9:
        rq3 = "EFFECTIVE transfer"
    elif learned_b > baseline:
        rq3 = "PARTIAL transfer"
    else:
        rq3 = "no transfer benefit"
    print(f"  RQ3 (transfer)   : Agent B {learned_b:.0%}  [{rq3}]"
          f"  ({n_trans}/{n_total} lessons transferred)")

    # ---- Lesson summary -----------------------------------------------------
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

    print(f"\n  Full results -> results/incident/processed/summary.json")
    print(div)
    print()


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main() -> None:
    # Load ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL, USE_LLM from .env if present.
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # python-dotenv not installed --user must set env vars manually

    mode = "LLM" if _use_llm() else "SIMULATED"
    _log(f"Starting experiment --mode={mode}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    materialize_incident_splits(DATA_DIR, n_per_family=30)
    incident_tasks = generate_incident_tasks(n_per_family=30)
    n_total = len(incident_tasks)
    split_a = int(n_total * 0.70)

    # 70/30 train/test split -- no validation split needed for a controlled benchmark.
    train_incidents = incident_tasks[:split_a]
    test_incidents  = incident_tasks[split_a:]
    train_tasks = [it.task for it in train_incidents]
    test_tasks  = [it.task for it in test_incidents]
    _log(f"Tasks: {len(train_tasks)} train / {len(test_tasks)} test")

    model_client = _build_model_client()
    agent_fn     = _build_agent_fn(model_client)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ctx_log = _ContextLogger(LOG_DIR / "context_log.txt")
    ctx_log._write_header(mode="LLM" if _use_llm() else "SIMULATED")
    _log(f"Context log: {LOG_DIR / 'context_log.txt'}  (console={'on' if LOG_TO_CONSOLE else 'off'})")

    episode_out    = RAW_DIR / "episode_results.jsonl"
    replay_out     = RAW_DIR / "all_replays.jsonl"
    lesson_store_path = RAW_DIR / "lesson_store_a.json"

    # Phase 2/3 checkpoint: skip eval episodes already written.
    eval_completed = _load_completed(episode_out)
    eval_completed = {(tid, cond) for tid, cond in eval_completed if not cond.startswith("train_")}
    if eval_completed:
        _log(f"Resuming --{len(eval_completed)} eval episodes already done, skipping them.")

    # ------------------------------------------------------------------
    # Phase 1 --Train Agent A: learn lessons from NonLearningManager failures
    #
    # If a saved lesson_store_a.json exists from a previous run, load it and
    # skip Phase 1 entirely --no API calls, instant resume.
    # Delete lesson_store_a.json to force a full re-train from scratch.
    # ------------------------------------------------------------------
    if lesson_store_path.exists():
        lesson_store_a = LessonStore.load(lesson_store_path)
        n_lessons = len(lesson_store_a.to_dict())
        _log(f"Phase 1 --skipped (loaded {n_lessons} lessons from {lesson_store_path})")
        all_replays: list = []
    else:
        # Clear episode and replay files so Phase 1 output is consistent with
        # the lesson store being built from scratch this run.
        episode_out.write_text("", encoding="utf-8")
        replay_out.write_text("", encoding="utf-8")
        _log(f"Phase 1 --training on {len(train_tasks)} tasks ...")
        lesson_store_a = LessonStore()
        all_replays: list = []
        for i, (task, inc) in enumerate(zip(train_tasks, train_incidents), 1):
            condition = "train_agent_a"
            _log(f"  [{i}/{len(train_tasks)}] {task.task_id} family={task.family}")
            result = run_episode(task, NonLearningManager(), condition, agent_fn,
                                 model_client=model_client)
            success = result["evaluation"]["task_success"]
            _log(f"    -> success={success}  words={result['active_context_words']}  obs_source={result['obs_source']}")
            _log_episode_detail(result)
            ctx_log.log(result, phase="train")
            with episode_out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(serializable_result(result)) + "\n")
            if success == 0:
                _log(f"    -> running replays ...")
                replays = run_replay_for_episode(task, result, agent_fn, lesson_store_a, replay_out)
                all_replays.extend(replays)
                n_lessons = len(lesson_store_a.to_dict())
                _log(f"    -> {len(replays)} replays done  lessons={n_lessons}")
                ctx_log.log_replays(replays, task.task_id, task.family)
                ctx_log.log_lesson_store(
                    lesson_store_a,
                    f"after task {i}/{len(train_tasks)} ({task.task_id})",
                )

        # Save the lesson store so future runs can skip Phase 1 entirely.
        lesson_store_a.save(lesson_store_path)
        _log(f"Phase 1 done --{len(all_replays)} replays, {len(lesson_store_a.to_dict())} lessons learned")
        _log(f"Lesson store saved -> {lesson_store_path}")
        ctx_log.log_lesson_store(lesson_store_a, "END OF PHASE 1 --full lesson store summary")

    # ------------------------------------------------------------------
    # Phase 2 --Evaluate Agent A on validation + test tasks (4 conditions)
    # ------------------------------------------------------------------
    conditions_a = [
        ("full_context",         FullContextManager()),
        ("non_learning_manager", NonLearningManager()),
        ("lesson_aware_agent_a", LessonAwareManager(lesson_store_a, model_client)),
    ]
    test_results_a: list[dict] = []
    n_eval = len(test_tasks)
    _log(f"Phase 2 --test split ({n_eval} tasks x {len(conditions_a)} conditions: full_context, non_learning, lesson_aware_a)")
    for condition, manager in conditions_a:
        for i, (task, inc) in enumerate(zip(test_tasks, test_incidents), 1):
            cond_label = f"{condition}_test"
            if (task.task_id, cond_label) in eval_completed:
                continue
            _log(f"  [{i}/{n_eval}] {cond_label}  {task.task_id}")
            result = run_episode(task, manager, cond_label, agent_fn,
                                 model_client=model_client)
            _log(f"    -> success={result['evaluation']['task_success']}  words={result['active_context_words']}  obs_source={result['obs_source']}")
            _log_episode_detail(result)
            ctx_log.log(result, phase="eval_test")
            test_results_a.append(result)
            with episode_out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(serializable_result(result)) + "\n")

    _log("Phase 2 done")

    # ------------------------------------------------------------------
    # Phase 3 --Cross-agent transfer: Agent B receives Agent A's lessons
    # ------------------------------------------------------------------
    transferable = lesson_store_a.get_transferable()
    lesson_store_b = LessonStore()
    for lesson in transferable:
        lesson_store_b.add(lesson)
    _log(f"Phase 3 --{len(transferable)} transferable lessons given to Agent B")
    ctx_log.log_lesson_store(
        lesson_store_b,
        f"PHASE 3 START --{len(transferable)} transferable lessons given to Agent B"
        + (" (none reached threshold >=5)" if len(transferable) == 0 else ""),
    )

    test_results_b: list[dict] = []
    n_eval = len(test_tasks)
    _log(f"Phase 3 --test split ({n_eval} tasks)")
    for i, (task, inc) in enumerate(zip(test_tasks, test_incidents), 1):
        cond_label = "lesson_aware_agent_b_test"
        if (task.task_id, cond_label) in eval_completed:
            continue
        _log(f"  [{i}/{n_eval}] {cond_label}  {task.task_id}")
        result = run_episode(
            task,
            LessonAwareManager(lesson_store_b, model_client),
            cond_label,
            agent_fn,
            model_client=model_client,
        )
        _log(f"    -> success={result['evaluation']['task_success']}  words={result['active_context_words']}")
        _log_episode_detail(result)
        ctx_log.log(result, phase="transfer_test")
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
    ctx_log.close()
    _log("Summary written to results/incident/processed/summary.json")
    print(json.dumps(summary, indent=2))
    print_analysis(summary)


if __name__ == "__main__":
    main()
